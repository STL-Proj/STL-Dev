# optimize_scattering_core
# optimize_from_maps
# optimize_from_stats

import time

import numpy as np
import torch
import torch.nn as nn
from torch import device, nn
from torch.optim import LBFGS


# === Learnable field model ===
class ScatteringMatchModel(nn.Module):
    def __init__(
        self,
        st_op,
        DataClass,
        pbc,
        init_shape,
        init_map,
        device,
        dtype,
        has_fewer_convolutions,
        compute_cross_matrix,
        compute_PS,
        keep_batch_dim,
        mean_field,
        prefilter_Nyquist,
        adhoc_weights,
    ):
        super().__init__()

        # === Field configuration ===
        self.st_op = st_op
        self.DataClass = DataClass
        self.pbc = pbc
        self.init_shape = init_shape
        self.init_map = init_map
        self.device = device
        self.dtype = dtype

        # === Stats configuration ===
        self.has_fewer_convolutions = has_fewer_convolutions
        self.compute_cross_matrix = compute_cross_matrix
        self.compute_PS = compute_PS
        self.keep_batch_dim = keep_batch_dim
        self.mean_field = mean_field
        self.adhoc_weights = adhoc_weights

        # === Initialize learnable field u ===
        if self.init_map is None:
            self.u = torch.randn(init_shape, device=device, dtype=dtype)
        else:
            self.u = (
                torch.tensor(self.init_map)
                .to(device=device, dtype=dtype)
                .expand(init_shape)
            )

        if prefilter_Nyquist:
            print("Prefiltering initial map to remove frequencies above Nyquist")
            assert (
                not self.u.isnan().any()
            ), "Cannot apply Nyquist filter on intial map with NaNs. Either remove NaNs from the initial map or specify prefilter_Nyquist=False."
            self.u = apply_nyquist_filter(self.u)

        # === Apply mask constraints ===
        self.mask_full_res = st_op.wavelet_op.mask_full_res
        if self.mask_full_res is not None:
            # for security put large values which should raise abberant values if actually used
            self.u[..., self.mask_full_res.array] = 1e10

        self.u.requires_grad_()

        if self.mask_full_res is not None:

            def freeze_hook(grad):
                return grad * (~self.mask_full_res.array)

            self.u.register_hook(freeze_hook)

            print(
                "NaN detected in the running synthesis mask, the synthesis takes it into account"
            )

    def forward(self):
        # === Build data class ===
        DC_u = self.DataClass(self.u, pbc=self.pbc)

        # === Compute scattering statistics ===
        st_u = self.st_op.apply(
            DC_u,
            has_fewer_convolutions=self.has_fewer_convolutions,
            compute_cross_matrix=self.compute_cross_matrix,
            compute_PS=self.compute_PS,
            norm="load_ref",
        )

        # === Re-weight statistics ===
        if self.adhoc_weights is not None:
            reweight(st_u, self.adhoc_weights)

        # === Flatten statistics ===
        s_flat_u = st_u.to_flatten(
            keep_batch_dim=self.keep_batch_dim,
            mean_along_batch=self.mean_field,
            keepnans=False,
        )

        return s_flat_u


def reweight(stats, weights):
    for coeff_label, weight in weights.items():
        if hasattr(stats, coeff_label):
            coeff = getattr(stats, coeff_label)
            coeff *= weight
        else:
            raise ValueError(f"Unsupported coefficient label: {coeff_label}")


# === LBFGS optimization (low level)===
def optimize_lbfgs(model, loss_fn, lr, max_iter, history_size, verbose, print_iter):
    optimizer = LBFGS(
        [model.u],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )

    loss_history = []

    def closure():
        optimizer.zero_grad()

        output = model()
        loss = loss_fn(output)

        loss.backward()
        loss_history.append(loss.item())

        if verbose and len(loss_history) % print_iter == 0:
            print(f"[LBFGS] iter {len(loss_history)}, loss = {loss.item():.6e}")

        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    end = time.perf_counter()

    torch.cuda.empty_cache() if model.device.type == "cuda" else None

    print(f"{len(loss_history)} iterations of synthesis.")
    print(f"Execution time: {end - start:.3f} s")

    u_opt = model.u.detach()

    return u_opt


# === Optimization function for synthesis from target maps (mid level) ===
def optimize_from_maps(
    target,
    st_op_target,
    st_op_running,
    nbatch=1,
    pbc_running=True,
    running_shape=None,
    init_running=None,
    has_fewer_convolutions=False,
    compute_cross_matrix=None,
    compute_PS=False,
    mean_field=True,
    lr=1.0,
    max_iter=100,
    history_size=50,
    print_iter=10,
    verbose=True,
    seed=None,
    prefilter_Nyquist=True,
    adhoc_weights={"S3": 3.5, "S4": 3.5**2},
):
    # Set random seed
    torch.manual_seed(seed) if seed is not None else None

    # ------- Set homogeneous configuration for device and dtype -------
    device = st_op_running.wavelet_op.device
    dtype = st_op_running.wavelet_op.dtype
    print("Running synthesis on device:", device, "dtype:", dtype)

    if target.array.isnan().any():
        print("NaN detected in the target, the synthesis takes it into account")

    # ------- Determine initial shape for u (from target) -------
    input_dim = target.array.ndim

    if input_dim == 2:
        target_shape = (1, 1, *target.array.shape)
        if running_shape is None:
            init_shape = (nbatch, 1, *target.array.shape)
        else:
            assert len(running_shape) == 2, "running_shape should be a tuple of (H,W)"
            init_shape = (nbatch, 1, *running_shape)

    elif input_dim == 3:
        target_shape = (1, *target.array.shape)
        if running_shape is None:
            init_shape = (nbatch, *target.array.shape)
        else:
            assert len(running_shape) == 2, "running_shape should be a tuple of (H,W)"
            init_shape = (nbatch, target.array.shape[0], *running_shape)

    elif input_dim == 4:
        target_shape = target.array.shape
        if running_shape is None:
            init_shape = (nbatch, *target.array.shape[-3:])
        else:
            assert len(running_shape) == 2, "running_shape should be a tuple of (H,W)"
            init_shape = (nbatch, target.array.shape[1], *running_shape)
    else:
        raise ValueError("target.array must be 2D, 3D or 4D tensor")
    print("Initial shape for u:", init_shape)

    if not mean_field and target.array.shape[0] != init_shape[0]:
        raise ValueError(
            "If mean_field is False, target and running batch sizes should match"
        )

    with torch.no_grad():

        # ------- Standardize target -------
        l_target = target.copy(empty=False)

        l_target.array = l_target.array.reshape(target_shape)

        if prefilter_Nyquist:
            if l_target.array.isnan().any():
                print(
                    "WARNING: prefiltering target above Nyquist is asked but target has NaNs. Only initial noise will be filtered."
                )
            else:
                print("Prefiltering target to remove frequencies above Nyquist")
                l_target.array = apply_nyquist_filter(l_target.array)

        l_target, mean_target, std_target = st_op_target.wavelet_op.standardize(
            l_target, mean_field=mean_field, inplace=True
        )  # [Nb, Nc] if mean_field else [1, Nc]

        # ------- Compute target stats -------
        target_stats = st_op_target.apply(
            l_target,
            has_fewer_convolutions=has_fewer_convolutions,
            compute_cross_matrix=compute_cross_matrix,
            compute_PS=compute_PS,
            norm="store_ref",
            norm_batch_mean=mean_field,
        )

        if adhoc_weights is not None:
            reweight(target_stats, adhoc_weights)

        target_stats = target_stats.to_flatten(
            mean_along_batch=mean_field, keepnans=False
        )  # [n_stats] if mean_field else [Nb, n_stats]

    target_stats = target_stats.detach()
    print("Synthesis on {:} ST coefficients".format(target_stats.nelement()))

    # ------- Transfer reference normalization from target to running operator -------
    st_op_running.S2_ref_sqrt_chan_diag = st_op_target.S2_ref_sqrt_chan_diag
    st_op_running.var_ref = st_op_target.var_ref
    if compute_PS:
        st_op_running.PS_ref_sqrt_chan_diag = st_op_target.PS_ref_sqrt_chan_diag

    # ------- Build model -------
    model = ScatteringMatchModel(
        st_op=st_op_running,
        DataClass=target.__class__,
        pbc=pbc_running,
        init_shape=init_shape,
        init_map=init_running,
        has_fewer_convolutions=has_fewer_convolutions,
        compute_cross_matrix=compute_cross_matrix,
        compute_PS=compute_PS,
        keep_batch_dim=False,
        mean_field=mean_field,
        device=device,
        dtype=dtype,
        prefilter_Nyquist=prefilter_Nyquist,
        adhoc_weights=adhoc_weights,
    )

    # ------- Launch optimization -------
    loss_fn = lambda s_flat_u: ((s_flat_u - target_stats).abs() ** 2).sum()

    u_opt = optimize_lbfgs(
        model=model,
        loss_fn=loss_fn,
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        verbose=verbose,
        print_iter=print_iter,
    )

    # ------- Post-process optimized u: unstandardize, apply mask constraints, reshape -------
    DC_u_opt = target.__class__(array=u_opt, pbc=pbc_running)
    st_op_running.wavelet_op.unstandardize(
        DC_u_opt, mean=mean_target, std=std_target, inplace=True
    )
    u_opt = DC_u_opt.array

    if st_op_running.wavelet_op.mask_full_res is not None:
        u_opt[..., st_op_running.wavelet_op.mask_full_res.array] = torch.nan

    if input_dim == 2:
        u_opt = u_opt[:, 0, ...]  # remove channel dim
    if nbatch == 1:
        u_opt = u_opt[0]  # remove batch dim

    return u_opt


# === Optimization function for synthesis from target stats (mid level) ===
def optimize_from_stats(
    target_stats,
    st_op_running,
    nbatch,
    running_shape=None,
    pbc_running=True,
    init_running=None,
    mean_field=True,
    lr=1.0,
    max_iter=100,
    history_size=50,
    print_iter=10,
    verbose=True,
    seed=None,
    prefilter_Nyquist=True,
    adhoc_weights={"S3": 3.5, "S4": 3.5**2},
):
    """
    Notes:
    - Since the loss function is computed on a per-map basis.
        - The number of maps (Nb) to be synthesized must match the number of maps in the target statistics,
        - mean_field (comparing averaged over batch dimension statistics to handle synthesis with different batch sizes) is not relevant anymore.
    - Since statistics are already computed, one can not specify a running shape different from the target shape
    """
    # Set random seed
    torch.manual_seed(seed) if seed is not None else None

    # ------- Set homogeneous configuration for device and dtype -------
    device = st_op_running.wavelet_op.device
    dtype = st_op_running.wavelet_op.dtype
    print("Running synthesis on device:", device, "dtype:", dtype)

    # ------- Determine initial shape for u (from target stats) -------
    Nb, Nc, N, M = target_stats.Nb, target_stats.Nc, *target_stats.N0

    if nbatch != Nb and not mean_field:
        raise ValueError(
            f"If mean_field is False, target batch size (Nb={Nb}) should match running batch size (nbatch={nbatch})"
        )

    if running_shape is not None:
        init_shape = (nbatch, Nc, *running_shape)
    else:
        init_shape = (nbatch, Nc, N, M)

    print(f"Initial shape for u: {init_shape}")

    # ------- Target stats Processing -------
    if adhoc_weights is not None:
        reweight(target_stats, adhoc_weights)

    target_stats_flat = target_stats.to_flatten(
        keep_batch_dim=True,
        mean_along_batch=mean_field,
        keepnans=False,
    )  # [1, n_stats] if mean_field else [Nb, n_stats]

    # Average pre-standardization stats over batch if mean_field is True (for unstandardization)
    target_stats.mean_pre_std = (
        target_stats.mean_pre_std.mean(dim=0)
        if mean_field
        else target_stats.mean_pre_std
    )  # [1, Nc] if mean_field else [Nc]
    target_stats.std_pre_std = (
        target_stats.std_pre_std.mean(dim=0) if mean_field else target_stats.std_pre_std
    )  # [1, Nc] if mean_field else [Nc]

    # ------- Transfer reference normalization from target stats to running operator -------
    # Those reference normalization attributes have been stored during normalisation of the target stats.
    assert (
        target_stats.S2_ref_sqrt_chan_diag is not None
    ), "target_stats should be normalized to perform reference normalization attributes transfer"
    st_op_running.S2_ref_sqrt_chan_diag = target_stats.S2_ref_sqrt_chan_diag
    st_op_running.var_ref = target_stats.var_ref
    if st_op_running.compute_PS:
        st_op_running.PS_ref_sqrt_chan_diag = target_stats.PS_ref_sqrt_chan_diag

    # ------- Build model -------
    model = ScatteringMatchModel(
        st_op=st_op_running,
        DataClass=target_stats.DataClass,
        pbc=pbc_running,
        init_shape=init_shape,
        init_map=init_running,
        has_fewer_convolutions=target_stats.has_fewer_convolutions,
        compute_cross_matrix=target_stats.compute_cross_matrix,
        compute_PS=st_op_running.compute_PS,
        keep_batch_dim=True,
        mean_field=mean_field,
        device=device,
        dtype=dtype,
        prefilter_Nyquist=prefilter_Nyquist,
        adhoc_weights=adhoc_weights,
    )

    # ------- Launch optimization -------
    def loss_fn(s_flat_u):
        loss = ((s_flat_u - target_stats_flat).abs() ** 2).sum()
        return loss if not mean_field else loss / Nb

    u_opt = optimize_lbfgs(
        model=model,
        loss_fn=loss_fn,
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        verbose=verbose,
        print_iter=print_iter,
    )

    # ------- Post-process optimized u: unstandardize, apply mask constraints -------
    if target_stats.standardized:
        DC_u_opt = target_stats.DataClass(u_opt, pbc=pbc_running)
        st_op_running.wavelet_op.unstandardize(
            DC_u_opt,
            mean=target_stats.mean_pre_std,
            std=target_stats.std_pre_std,
            inplace=True,
        )
        u_opt = DC_u_opt.array

    if st_op_running.wavelet_op.mask_full_res is not None:
        u_opt[..., st_op_running.wavelet_op.mask_full_res.array] = torch.nan

    if Nc == 1:
        u_opt = u_opt[:, 0, ...]  # remove channel dim
    if nbatch == 1:
        u_opt = u_opt[0]  # remove batch dim

    return u_opt


#######################################################################################
# ------- Pre/Post-processing functions -------
def apply_nyquist_filter(tensor, plot=False):
    """
    Apply a low-pass filter to an input tensor, keeping only frequencies within the Nyquist radius.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in real space of shape (..., N, M) where N and M are the spatial dimensions

    Returns
    -------
    torch.Tensor
        Filtered tensor in real space of the same shape as input, with high frequencies removed
    """
    dim = (-2, -1)

    # Compute frequency grids
    N, M = tensor.shape[-2:]
    fx = N * torch.fft.fftfreq(N, d=1.0, device=tensor.device)
    fy = M * torch.fft.fftfreq(M, d=1.0, device=tensor.device)
    FX, FY = torch.meshgrid(fx, fy, indexing="ij")

    # Create low-pass mask based on Nyquist radius
    r2 = FX**2 + FY**2
    nyquist_radius = min(N, M) / 2
    mask = r2 <= nyquist_radius**2

    # Apply mask in Fourier space
    tensor_fft = torch.fft.fft2(tensor, dim=dim)
    tensor_fft[..., ~mask] = 0.0

    # Inverse Fourier transform to get the filtered tensor in real space
    tensor_filtered = torch.fft.ifft2(tensor_fft, dim=dim)

    if not tensor.is_complex():
        tensor_filtered = tensor_filtered.real

    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(
            np.abs(torch.fft.fftshift(torch.fft.fft2(tensor[0, 0])).cpu().numpy()),
            cmap="plasma",
            norm="log",
        )
        plt.colorbar()
        plt.title("Nyquist filtered tensor (Fourier)")
        plt.show()

    return tensor_filtered


# === User-friendly wrapper for synthesis from target maps (high level) ===
def synthesize_from_maps(
    data_target,
    nbatch,
    pbc_running,
    running_shape=None,
    init_running=None,
    running_mask=None,
    has_fewer_convolutions=False,
    compute_cross_matrix=None,
    mean_field=True,
    **optim_kwargs,
):
    """
    User-friendly wrapper to synthesize field maps from target maps.

    Parameters
    ----------
    running_shape : tuple of int, optional
        Default is None. If None, the running field has the same shape as the target.

    running_mask : torch.BoolTensor, optional
        Default is None. If None, the same mask as the target is used.
        If a new mask is provided, it must be a boolean tensor with shape matching the running field.

    mean_field : bool, optional
        Default is True. Default value allows one to perform synthesis between N target samples and M running samples (with N different from or
        equal to M) while matching statistics computed from the batch-averaged field.

    Notes
    -----
    -   The Power Spectrum is optimized by default whenever possible (i.e., when no NaN values are present in either the target or the running data).

    -   Within this user-level wrapper, an ST operator is created for both the target and the running map, and then passed to the mid-level wrapper.
        This is particularly useful for syntheses involving NaN values, as it allows the use of two distinct masks: one for the target and one for the running map.
        For other types of syntheses, the same ST operator is used for both.
    """

    if running_mask is None:
        # Same mask for running and target
        array = (
            np.zeros(running_shape) if running_shape is not None else data_target.array
        )
        data_running = data_target.__class__(array=array, pbc=pbc_running)
    else:
        if running_shape is None and data_target.array.shape[-2:] != running_mask.shape:
            raise ValueError("running_mask shape should match target array shape")
        elif running_shape is not None and running_shape != running_mask.shape:
            raise ValueError("running_mask shape should match running_shape")

        data_running = data_target.__class__(array=running_mask, pbc=pbc_running)

    # Select J used for synthesis
    J_target = data_target.get_wavelet_op().J - (not data_target.pbc)
    J_running = data_running.get_wavelet_op().J - (not data_running.pbc)
    J = min(J_target, J_running)

    n_bins_target = data_target.get_CS_op().n_bins
    n_bins_running = data_running.get_CS_op().n_bins
    n_bins = min(n_bins_target, n_bins_running)

    if J_target != J_running:
        print(
            f"Warning: target.J = {J_target}, running.J = {J_running}. Synthesis will use J = {J}."
        )

    # Get scattering operators for target and running data with selected J
    st_op_target = data_target.get_ST_op(
        J=J, n_bins=n_bins, has_fewer_convolutions=has_fewer_convolutions
    )

    st_op_running = data_running.get_ST_op(
        J=J,
        n_bins=n_bins,
        has_fewer_convolutions=has_fewer_convolutions,
        replace_nan_value=None,
    )

    # Disable power spectrum optimization if NaN values are present in target and/or running data
    target_has_nan = data_target.array.isnan().any()
    running_has_nan = data_running.array.isnan().any()

    if target_has_nan or running_has_nan:
        print(
            "⚠️ Warning: NaN detected in target and/or running data.\n"
            "Power spectrum optimization is disabled because its computation is not yet implemented for NaN values in any dataclass. \n"
        )

    compute_PS = not (target_has_nan or running_has_nan)

    # Set default optimization parameters and update with user-provided values
    optim_params = dict(
        max_iter=100,
        lr=1.0,
        history_size=50,
        print_iter=10,
        verbose=True,
        seed=26,
        prefilter_Nyquist=(
            True if init_running is None else not init_running.isnan.any()
        ),
        adhoc_weights={"S3": 3.5, "S4": 3.5**2},
    )
    optim_params.update(optim_kwargs)

    # Run optimization
    u_opt = optimize_from_maps(
        target=data_target,
        st_op_target=st_op_target,
        st_op_running=st_op_running,
        nbatch=nbatch,
        pbc_running=pbc_running,
        running_shape=running_shape,
        init_running=init_running,
        mean_field=mean_field,
        has_fewer_convolutions=has_fewer_convolutions,
        compute_cross_matrix=compute_cross_matrix,
        compute_PS=compute_PS,
        **optim_params,
    )

    return u_opt


# === User-friendly wrapper for synthesis from target statistics (high level) ===
def synthesize_from_stats(
    target_stats,
    nbatch,
    pbc_running,
    running_shape=None,
    init_running=None,
    running_mask=None,
    mean_field=True,
    **optim_kwargs,
):
    """
    notes
        - Parameters such as `compute_cross_matrix` and `has_fewer_convolutions` are not specified as arguments of this wrapper,
    but rather during the computation of the target statistics.
    """
    if running_mask is None:
        if running_shape is not None:
            array = torch.zeros(running_shape)
        else:
            if target_stats.mask_full_res is None:
                array = torch.zeros(target_stats.N0)
            else:
                array = torch.where(target_stats.mask_full_res.array, torch.nan, 0.0)
        array = array.to(device=target_stats.device, dtype=target_stats.dtype)

        data_running = target_stats.DataClass(array=array, pbc=pbc_running)
    else:
        if running_mask.shape != target_stats.N0:
            raise ValueError("running_mask shape should match target_stats N0")
        data_running = target_stats.DataClass(array=running_mask, pbc=pbc_running)

    running_has_nan = data_running.array.isnan().any()

    if not target_stats.compute_PS or running_has_nan:
        print(
            "⚠️ Warning: Power spectrum optimization is disabled because it is not implemented for NaN values in any dataclass"
            " and/or because Power spectrum computation has not been included in target_stats.\n"
        )

        # Remove power spectrum from target_stats if not optimizable on the running side
        if target_stats.compute_PS:
            target_stats.compute_PS = False

    compute_PS = target_stats.compute_PS and not running_has_nan

    # Get scattering operator for running data
    st_op_running = data_running.get_ST_op(
        J=target_stats.J,
        n_bins=target_stats.n_bins,
        has_fewer_convolutions=target_stats.has_fewer_convolutions,
        compute_PS=compute_PS,
        replace_nan_value=None,
    )

    # Set default optimization parameters and update with user-provided values
    optim_params = dict(
        max_iter=100,
        lr=1.0,
        history_size=50,
        print_iter=10,
        verbose=True,
        seed=26,
        prefilter_Nyquist=(
            True if init_running is None else not init_running.isnan.any()
        ),
        adhoc_weights={"S3": 3.5, "S4": 3.5**2},
    )
    optim_params.update(optim_kwargs)

    # Run optimization
    u_opt = optimize_from_stats(
        target_stats=target_stats,
        st_op_running=st_op_running,
        nbatch=nbatch,
        running_shape=running_shape,
        pbc_running=pbc_running,
        init_running=init_running,
        mean_field=mean_field,
        **optim_params,
    )

    return u_opt
