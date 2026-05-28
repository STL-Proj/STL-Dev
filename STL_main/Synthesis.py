import time

import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS

# Suppose:
# - DataClass is your data wrapper class (e.g. STL_2D_Kernel_Torch or STL_2D_FFT_Torch)
# - st_op is an operator such that st_op.apply(DC).to_flatten()
#   returns a 1D tensor of scattering coefficients.
# - target.array is your reference map of shape (1,1,128,128)
#   and target_stats is the corresponding scattering vector.


class ScatteringMatchModel(nn.Module):
    def __init__(
        self,
        st_op,
        DataClass,
        pbc,
        init_shape,
        compute_cross_matrix,
        compute_PS,
        mean_field,
        device,
        dtype,
        prefilter_Nyquist,
        adhoc_weights,
    ):
        super().__init__()
        self.st_op = st_op
        self.DataClass = DataClass
        self.pbc = pbc
        self.init_shape = init_shape
        self.mask_full_res = st_op.wavelet_op.mask_full_res
        self.compute_cross_matrix = compute_cross_matrix
        self.compute_PS = compute_PS
        self.mean_field = mean_field
        self.adhoc_weights = adhoc_weights

        # Learnable field u
        self.u = torch.randn(
            init_shape, device=device, dtype=dtype
        )  # WARNING: if filtered to match PS constraints in the future, think about sampling non PBC fields when asked!!!

        if prefilter_Nyquist:
            print("Prefiltering initial noise to remove frequencies above Nyquist")
            self.u = apply_prefilter_Nyquist(self.u)

        # for security put large values which should raise abberant values if actually used
        if self.mask_full_res is not None:
            self.u[..., self.mask_full_res.array] = (
                1e10  ##################################### IMPORTANT TO KEEP IN MIND
            )

        self.u.requires_grad_()

        if False:  # self.mask_full_res is not None:

            def freeze_hook(grad):
                return grad * (~self.mask_full_res.array)

            self.u.register_hook(freeze_hook)
            print(
                "NaN detected in the running synthesis mask, the synthesis takes it into account"
            )

    def forward(self):
        DC_u = self.DataClass(self.u, pbc=self.pbc)
        st_u = self.st_op.apply(
            DC_u,
            compute_cross_matrix=self.compute_cross_matrix,
            compute_PS=self.compute_PS,
            norm="load_ref",
            norm_batch_mean=self.mean_field,
        )
        if self.adhoc_weights is not None:
            reweight(st_u, self.adhoc_weights)
        s_flat_u = st_u.to_flatten(mean_along_batch=self.mean_field, keepnans=True)
        return s_flat_u


def reweight(stats, weights):
    for coeff_label, weight in weights.items():
        if hasattr(stats, coeff_label):
            coeff = getattr(stats, coeff_label)
            coeff *= weight
        else:
            print(coeff_label)
            raise


def apply_prefilter_Nyquist(array, plot=False):
    dim = (-2, -1)
    target_fft = torch.fft.fftshift(torch.fft.fft2(array, dim=dim), dim=dim)

    # filter out high frequencies to see better the low frequency content
    X, Y = torch.meshgrid(
        torch.arange(target_fft.shape[-2]), torch.arange(target_fft.shape[-1])
    )
    center_x, center_y = target_fft.shape[-2] // 2, target_fft.shape[-1] // 2
    radius = min(target_fft.shape[-2:]) // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius**2
    target_fft[:, :, ~mask] = 0.0
    if not array.is_complex():
        array = torch.fft.ifft2(torch.fft.ifftshift(target_fft, dim=dim), dim=dim).real
    else:
        array = torch.fft.ifft2(torch.fft.ifftshift(target_fft, dim=dim), dim=dim)

    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(
            np.abs(torch.fft.fftshift(torch.fft.fft2(array[0, 0])).cpu().numpy()),
            cmap="plasma",
            norm="log",
        )
        plt.colorbar()
        plt.title("Nyquist filtered array (Fourier)")
        plt.show()

    return array


def optimize_scattering_LBFGS(
    target,
    st_op_target,
    st_op_running,
    pbc_running=True,
    running_shape=None,
    compute_cross_matrix=None,
    compute_PS=False,
    nbatch=1,
    mean_field=True,
    max_iter=100,
    lr=1.0,
    history_size=50,
    print_iter=10,
    verbose=True,
    seed=None,
    prefilter_Nyquist=True,
    adhoc_weights={"S3": 3.5, "S4": 3.5**2},
):
    device = st_op_running.wavelet_op.device
    dtype = st_op_running.wavelet_op.dtype
    print("Running synthesis on device :", device, "dtype :", dtype)

    torch.manual_seed(seed) if seed is not None else None

    if target.array.isnan().any():
        print("NaN detected in the target, the synthesis takes it into account")

    target.array = target.array.to(
        device=device, dtype=dtype
    )  # dtype is now float64 on cuda but was previously float32, if it slows down your synthesis raise us this issue !!!!!

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

    # Reference scattering
    with torch.no_grad():

        # Standardize target before computing stats
        l_target = target.copy(empty=False)

        l_target.array = l_target.array.reshape(target_shape)

        if prefilter_Nyquist:
            if l_target.array.isnan().any():
                print(
                    "WARNING: prefiltering target above Nyquist is asked but target has NaNs. Only initial noise will be filtered."
                )
            else:
                print("Prefiltering target to remove frequencies above Nyquist")
                l_target.array = apply_prefilter_Nyquist(l_target.array)

        l_target, mean_target, std_target = st_op_target.wavelet_op.standardize(
            l_target, mean_field=mean_field, inplace=True
        )

        target_stats = st_op_target.apply(
            l_target,
            compute_cross_matrix=compute_cross_matrix,
            compute_PS=compute_PS,
            norm="store_ref",
            norm_batch_mean=mean_field,
        )

        if adhoc_weights is not None:
            reweight(target_stats, adhoc_weights)

        target_stats = target_stats.to_flatten(
            mean_along_batch=mean_field, keepnans=True
        )

    target_stats = target_stats.detach()
    target_coeffs_mask = ~target_stats.isnan()
    target_stats = target_stats[target_coeffs_mask]
    print("Synthesis on {:} ST coefficients".format(target_coeffs_mask.sum().item()))

    # reference to running normalization
    st_op_running.S2_ref_sqrt_chan_diag = st_op_target.S2_ref_sqrt_chan_diag
    st_op_running.var_ref = st_op_target.var_ref
    if compute_PS:
        st_op_running.PS_ref_sqrt_chan_diag = st_op_target.PS_ref_sqrt_chan_diag

    # Model with learnable u
    model = ScatteringMatchModel(
        st_op=st_op_running,
        DataClass=target.__class__,
        pbc=pbc_running,
        init_shape=init_shape,
        compute_cross_matrix=compute_cross_matrix,
        compute_PS=compute_PS,
        mean_field=mean_field,
        device=device,
        dtype=dtype,
        prefilter_Nyquist=prefilter_Nyquist,
        adhoc_weights=adhoc_weights,
    )

    optimizer = LBFGS(
        [model.u],
        lr=lr,
        max_iter=max_iter,  # <-- le nombre d'itérations internes LBFGS
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )

    loss_history = []

    def closure():
        optimizer.zero_grad()
        s_flat_u = model()  # forward pass (call model.forward())
        # assert not torch.any(s_flat_u[target_coeffs_mask].isnan()) ####################### sanity check that can be removed
        loss = ((s_flat_u[target_coeffs_mask] - target_stats).abs() ** 2).sum()
        loss.backward()

        if len(loss_history) < 2:
            if False:  ####################### set to True to debug backprop
                import matplotlib.pyplot as plt

                plt.imshow(model.u[0, 0].detach().cpu().numpy()), plt.title(
                    "running u"
                ), plt.colorbar(), plt.show()
                plt.imshow(model.u.grad[0, 0].cpu().numpy()), plt.title(
                    "running u grad"
                ), plt.colorbar(), plt.show()
                plt.imshow(model.u.grad[0, 0].cpu().numpy() == 0), plt.title(
                    "running u grad == 0"
                ), plt.colorbar(), plt.show()

        # assert model.u.grad.isnan().sum().cpu().item() == 0 ####################### sanity check that can be removed

        # Log à chaque appel interne
        loss_val = loss.item()
        loss_history.append(loss_val)
        if verbose:
            if len(loss_history) % print_iter == 0:
                print(f"[LBFGS] inner iter {len(loss_history)}, loss = {loss_val:.6e}")

        return loss

    start = time.perf_counter()
    # Un seul appel : toutes les itérations LBFGS internes sont faites ici
    optimizer.step(closure)
    end = time.perf_counter()

    print(
        "{:} iterations of synthesis done with nbatch={:} and {:} ST coefficients".format(
            len(loss_history), nbatch, target_coeffs_mask.sum().item()
        )
    )
    print(f"Execution time: {end - start:.3f} s")

    u_opt = model.u.detach()

    # Unstandardize u_opt with the computed mean and std of the target
    DC_u_opt = target.__class__(u_opt, pbc=pbc_running)
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

    return u_opt, loss_history


#######################################################################################
def synthesize_from_maps(
    data_target,
    pbc_running,
    nbatch,
    running_shape=None,
    running_mask=None,
    mean_field=True,
    compute_cross_matrix=None,
    has_fewer_convolutions=False,
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
        J=J, has_fewer_convolutions=has_fewer_convolutions, n_bins=n_bins
    )

    st_op_running = data_running.get_ST_op(
        J=J,
        has_fewer_convolutions=has_fewer_convolutions,
        n_bins=n_bins,
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
        max_iter=50, lr=1.0, history_size=50, print_iter=10, verbose=True, seed=26
    )
    optim_params.update(optim_kwargs)

    # Run optimization
    u_opt, _ = optimize_scattering_LBFGS(
        target=data_target,
        st_op_target=st_op_target,
        st_op_running=st_op_running,
        running_shape=running_shape,
        pbc_running=pbc_running,
        compute_cross_matrix=compute_cross_matrix,
        compute_PS=compute_PS,
        nbatch=nbatch,
        mean_field=mean_field,
        **optim_params,
    )

    return u_opt
