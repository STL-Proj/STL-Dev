# optimize_scattering_core
# optimize_from_maps
# optimize_from_stats

import gc
import math
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from torch import device, nn
from torch.optim import LBFGS


def _healpix_patch_domains(nside, patch_nside, halo_rings):
    """Return disjoint NESTED cores and their overlapping halo domains."""
    from healpix_geo.nested import kth_neighbourhood

    nside = int(nside)
    patch_nside = int(patch_nside)
    halo_rings = int(halo_rings)
    if nside < 1 or nside & (nside - 1):
        raise ValueError("nside must be a positive power of two")
    if patch_nside < 1 or patch_nside & (patch_nside - 1):
        raise ValueError("patch_nside must be a positive power of two")
    if patch_nside > nside or nside % patch_nside:
        raise ValueError("patch_nside must divide nside")
    if halo_rings < 0:
        raise ValueError("halo_rings must be non-negative")

    descendants = (nside // patch_nside) ** 2
    patch_depth = int(math.log2(patch_nside))
    domains = []
    for parent in range(12 * patch_nside**2):
        support_parents = kth_neighbourhood(
            np.asarray([parent], dtype=np.uint64), patch_depth, halo_rings
        ).reshape(-1)
        support_parents = np.unique(
            support_parents[support_parents >= 0].astype(np.int64, copy=False)
        )

        core = parent * descendants + np.arange(descendants, dtype=np.int64)
        support = (
            support_parents[:, None] * descendants
            + np.arange(descendants, dtype=np.int64)[None, :]
        ).reshape(-1)
        support.sort()
        core_positions = np.searchsorted(support, core)
        domains.append((core, support, core_positions))
    return domains


def _start_rss_sampler(interval=0.01):
    """Start an optional process-RSS sampler without making psutil mandatory."""
    try:
        import psutil
    except ImportError:
        return None, None, None

    process = psutil.Process()
    stop = threading.Event()
    sample = {"start": process.memory_info().rss, "peak": 0}
    sample["peak"] = sample["start"]

    def run():
        while not stop.wait(interval):
            sample["peak"] = max(sample["peak"], process.memory_info().rss)
        sample["peak"] = max(sample["peak"], process.memory_info().rss)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return stop, thread, sample


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
        # `DataClass` may be the data class itself (legacy) or an existing data
        # object used as a prototype. The prototype form is what makes the
        # synthesis data-type agnostic: geometries that are not described by the
        # array shape alone -- HEALPix cell_ids, for instance -- are inherited
        # from it instead of being rebuilt from scratch.
        self.st_op = st_op
        self.DataClass = DataClass
        self.proto = None if isinstance(DataClass, type) else DataClass
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
            assert (
                not self.u.isnan().any()
            ), "Cannot band-limit an initial map with NaNs. Either remove NaNs from the initial map or specify prefilter_Nyquist=False."
            self.u = self._bandlimit(self.u)

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

    def _bandlimit(self, array):
        """Band-limit through the data type, falling back on the planar filter."""
        if self.proto is not None:
            return _bandlimit_and_report(self.proto, array, "initial map")

        filtered = apply_nyquist_filter(array)
        print("Prefiltering initial map to remove sub-pixel frequencies")
        return filtered

    def _make_data(self, array):
        """Wrap `array` in a data object, keeping the prototype geometry."""
        if self.proto is not None:
            return self.proto.new_like(array, pbc=self.pbc)
        return self.DataClass(array, pbc=self.pbc)

    def forward(self):
        # === Build data class ===
        DC_u = self._make_data(self.u)

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


def _bandlimit_and_report(data_like, array, what):
    """
    Band-limit `array` through the data type and say so only if it changed.

    Not every data type can define a band limit -- a partial HEALPix sky, for
    one -- and announcing a filtering that did not happen is worse than saying
    nothing.
    """
    filtered = data_like.apply_bandlimit(array)
    if filtered is not array and not torch.equal(filtered, array):
        print("Prefiltering %s to remove sub-pixel frequencies" % what)
    return filtered


def _contains_nan(array):
    """True if `array` (numpy or torch, on any device) holds a NaN."""
    tensor = array if torch.is_tensor(array) else torch.as_tensor(np.asarray(array))
    return bool(torch.isnan(tensor).any())


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
    ndim_pix = target.NDIM_PIX  # 2 for planar maps, 1 for HEALPix

    def _check_running_shape(shape):
        assert (
            len(shape) == ndim_pix
        ), f"running_shape should have {ndim_pix} entries for {target.DT}"

    if input_dim == ndim_pix:  # a single map
        target_shape = (1, 1, *target.array.shape)
        if running_shape is None:
            init_shape = (nbatch, 1, *target.array.shape)
        else:
            _check_running_shape(running_shape)
            init_shape = (nbatch, 1, *running_shape)

    elif input_dim == ndim_pix + 1:  # channels
        target_shape = (1, *target.array.shape)
        if running_shape is None:
            init_shape = (nbatch, *target.array.shape)
        else:
            _check_running_shape(running_shape)
            init_shape = (nbatch, target.array.shape[0], *running_shape)

    elif input_dim == ndim_pix + 2:  # batch and channels
        target_shape = target.array.shape
        if running_shape is None:
            init_shape = (nbatch, *target.array.shape[1:])
        else:
            _check_running_shape(running_shape)
            init_shape = (nbatch, target.array.shape[1], *running_shape)
    else:
        raise ValueError(
            f"target.array should have {ndim_pix}, {ndim_pix + 1} or "
            f"{ndim_pix + 2} dimensions, got {input_dim}"
        )
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
                l_target.array = _bandlimit_and_report(target, l_target.array, "target")

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
        DataClass=target,  # prototype: carries the geometry as well as the class
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
    DC_u_opt = target.new_like(u_opt, pbc=pbc_running)
    st_op_running.wavelet_op.unstandardize(
        DC_u_opt, mean=mean_target, std=std_target, inplace=True
    )
    u_opt = DC_u_opt.array

    if st_op_running.wavelet_op.mask_full_res is not None:
        u_opt[..., st_op_running.wavelet_op.mask_full_res.array] = torch.nan

    if input_dim == ndim_pix:
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
    Nb, Nc = target_stats.Nb, target_stats.Nc
    # the pixel grid, which is not the resolution descriptor for every data type
    # (HEALPix stores nside in N0 but Npix pixels)
    pix_shape = tuple(getattr(target_stats, "pix_shape", None) or target_stats.N0)

    if nbatch != Nb and not mean_field:
        raise ValueError(
            f"If mean_field is False, target batch size (Nb={Nb}) should match running batch size (nbatch={nbatch})"
        )

    if running_shape is not None:
        init_shape = (nbatch, Nc, *running_shape)
    else:
        init_shape = (nbatch, Nc, *pix_shape)

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
        DataClass=getattr(target_stats, "data_example", None) or target_stats.DataClass,
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
        proto = getattr(target_stats, "data_example", None)
        DC_u_opt = (
            proto.new_like(u_opt, pbc=pbc_running)
            if proto is not None
            else target_stats.DataClass(u_opt, pbc=pbc_running)
        )
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


def synthesize_healpix_from_patches(
    data_target,
    patch_nside,
    nbatch=1,
    halo_rings=1,
    J=None,
    L=None,
    init_running=None,
    has_fewer_convolutions=False,
    compute_cross_matrix=None,
    mean_field=True,
    lr=5e-2,
    max_iter=20,
    optimizer="adam",
    print_iter=1,
    verbose=True,
    seed=26,
    adhoc_weights={"S3": 3.5, "S4": 3.5**2},
    wavelet_op_kwargs=None,
    track_memory=True,
    return_diagnostics=False,
):
    """Synthesize a full HEALPix map with one autograd graph per patch.

    The full NESTED sphere is split into disjoint core patches.  Each local
    ScatCov loss is evaluated on the core plus ``halo_rings`` neighbouring
    patches, but only the gradient on the core is copied into the full map.
    The local graph is released before the next patch is processed.

    This deliberately optimizes an average of overlapping *local* ScatCov
    losses.  It is therefore a memory experiment and an approximation of the
    global ScatCov objective used by :func:`synthesize_from_maps`.  Increasing
    the halo reduces boundary bias at the cost of a larger peak graph.

    Parameters
    ----------
    data_target : STL_Healpix_Kernel_Torch
        Full-sky, finite, NESTED target map.
    patch_nside : int
        Resolution of the parent grid defining the patches.  A patch contains
        ``(nside / patch_nside)**2`` fine pixels.
    halo_rings : int
        Number of neighbouring parent-pixel rings included as context.
    optimizer : {"adam", "sgd"}
        Optimizer for the full map.  SGD has no full-map moment buffers and is
        useful when the purpose of the run is strictly to measure memory.
    track_memory : bool
        Record CUDA allocator peaks and, when psutil is installed, process RSS.
        The measurement starts after the target patch statistics are cached.
    return_diagnostics : bool
        If True, return ``(map, diagnostics)`` instead of the map alone.

    Notes
    -----
    The optimizer state and the assembled gradient still have full-map size;
    only the scattering autograd graph is bounded by the largest halo patch.
    The angular power spectrum is intentionally excluded because it is global.
    """
    if getattr(data_target, "DT", None) != "HealpixKernel_torch":
        raise TypeError("data_target must be an STL_Healpix_Kernel_Torch instance")
    if not bool(data_target.nest):
        raise ValueError("patch synthesis requires NESTED HEALPix ordering")
    if data_target.dg != 0:
        raise ValueError("data_target must be at its native resolution (dg=0)")
    if not bool(torch.isfinite(data_target.array).all()):
        raise ValueError("patch synthesis currently requires a finite target map")

    nside = int(data_target.N0[0])
    npix = 12 * nside**2
    expected_ids = torch.arange(
        npix, device=data_target.cell_ids.device, dtype=torch.long
    )
    if data_target.array.shape[-1] != npix or not torch.equal(
        data_target.cell_ids, expected_ids
    ):
        raise ValueError("patch synthesis currently requires a complete full-sky map")

    patch_nside = int(patch_nside)
    if patch_nside < 1 or patch_nside & (patch_nside - 1):
        raise ValueError("patch_nside must be a positive power of two")
    if patch_nside > nside or nside % patch_nside:
        raise ValueError("patch_nside must divide nside")
    J = max(1, int(math.log2(nside)) - 1) if J is None else int(J)
    pixels_across_core = nside // patch_nside
    if pixels_across_core < 2 ** (J - 1):
        raise ValueError(
            "patch cores become empty before the coarsest scattering scale; "
            "decrease patch_nside or J"
        )
    if nbatch < 1:
        raise ValueError("nbatch must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if print_iter < 1:
        raise ValueError("print_iter must be positive")

    torch.manual_seed(seed) if seed is not None else None
    device = data_target.device
    dtype = data_target.dtype
    source = data_target.array
    input_dim = source.ndim
    if input_dim == 1:
        target = source[None, None, :]
    elif input_dim == 2:
        target = source[None, ...]
    elif input_dim == 3:
        target = source
    else:
        raise ValueError(
            "target array must have shape [Npix], [Nc,Npix] or [Nb,Nc,Npix]"
        )

    if not mean_field and target.shape[0] != nbatch:
        raise ValueError(
            "target and running batch sizes must match when mean_field is False"
        )
    mean_target = target.mean(dim=-1)
    if mean_field:
        mean_target = mean_target.mean(dim=0, keepdim=True)
    centred = target - mean_target[..., None]
    var_target = (centred * centred.conj()).real.mean(dim=-1)
    if mean_field:
        var_target = var_target.mean(dim=0, keepdim=True)
    std_target = torch.sqrt(var_target)
    if bool((std_target <= torch.finfo(std_target.dtype).eps).any()):
        raise ValueError("every target channel must have non-zero variance")
    target_standardized = centred / std_target[..., None]

    nc = target.shape[1]
    if init_running is None:
        running = torch.randn((nbatch, nc, npix), device=device, dtype=dtype)
    else:
        running = torch.as_tensor(init_running, device=device, dtype=dtype)
        if running.ndim == 1:
            running = running[None, None, :]
        elif running.ndim == 2:
            running = running[None, ...]
        elif running.ndim != 3:
            raise ValueError("init_running has an unsupported number of dimensions")
        if running.shape[0] == 1 and nbatch != 1:
            running = running.expand(nbatch, -1, -1).clone()
        if tuple(running.shape) != (nbatch, nc, npix):
            raise ValueError(
                f"init_running must expand to {(nbatch, nc, npix)}, got "
                f"{tuple(running.shape)}"
            )
        running = (running - mean_target[..., None]) / std_target[..., None]

    domains = _healpix_patch_domains(nside, patch_nside, halo_rings)
    wavelet_op_kwargs = dict(wavelet_op_kwargs or {})

    def make_patch(array, ids):
        return data_target.__class__(
            array=array,
            nside=nside,
            cell_ids=ids,
            nest=True,
            pbc=False,
        )

    def make_operator(patch):
        return patch.get_ST_op(
            J=J,
            L=L,
            compute_PS=False,
            has_fewer_convolutions=has_fewer_convolutions,
            replace_nan_value=None,
            wavelet_op_kwargs=wavelet_op_kwargs,
        )

    # Cache only compact coefficient vectors and normalization references.  The
    # geometry operators themselves are rebuilt so their per-patch caches cannot
    # accumulate and hide the graph-memory reduction being measured.
    target_cache = []
    with torch.no_grad():
        for _, support_np, _ in domains:
            support = torch.as_tensor(support_np, device=device, dtype=torch.long)
            patch = make_patch(target_standardized.index_select(-1, support), support)
            st_op = make_operator(patch)
            stats = st_op.apply(
                patch,
                has_fewer_convolutions=has_fewer_convolutions,
                compute_cross_matrix=compute_cross_matrix,
                compute_PS=False,
                norm="store_ref",
                norm_batch_mean=mean_field,
            )
            if adhoc_weights is not None:
                reweight(stats, adhoc_weights)
            flat = (
                stats.to_flatten(mean_along_batch=mean_field, keepnans=False)
                .detach()
                .cpu()
            )
            refs = {}
            for name in ("S2_ref_sqrt_chan_diag", "var_ref"):
                value = getattr(st_op, name)
                refs[name] = None if value is None else value.detach().cpu()
            target_cache.append((flat, refs))
            del stats, st_op, patch, support

    running = nn.Parameter(running)
    optimizer_name = str(optimizer).lower()
    if optimizer_name == "adam":
        optim = torch.optim.Adam([running], lr=lr)
    elif optimizer_name == "sgd":
        optim = torch.optim.SGD([running], lr=lr)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    gc.collect()
    if device.type == "cuda" and track_memory:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    stop = thread = rss = None
    if track_memory:
        stop, thread, rss = _start_rss_sampler()

    loss_history = []
    max_support_pixels = max(len(support) for _, support, _ in domains)
    start = time.perf_counter()
    try:
        for iteration in range(max_iter):
            grad_full = torch.zeros_like(running)
            loss_value = 0.0
            for patch_index, (core_np, support_np, core_pos_np) in enumerate(domains):
                support = torch.as_tensor(support_np, device=device, dtype=torch.long)
                core = torch.as_tensor(core_np, device=device, dtype=torch.long)
                core_pos = torch.as_tensor(core_pos_np, device=device, dtype=torch.long)
                local = (
                    running.detach()
                    .index_select(-1, support)
                    .clone()
                    .requires_grad_(True)
                )
                patch = make_patch(local, support)
                st_op = make_operator(patch)
                target_flat_cpu, refs = target_cache[patch_index]
                for name, value in refs.items():
                    setattr(
                        st_op,
                        name,
                        None if value is None else value.to(device=device),
                    )
                stats = st_op.apply(
                    patch,
                    has_fewer_convolutions=has_fewer_convolutions,
                    compute_cross_matrix=compute_cross_matrix,
                    compute_PS=False,
                    norm="load_ref",
                )
                if adhoc_weights is not None:
                    reweight(stats, adhoc_weights)
                flat = stats.to_flatten(mean_along_batch=mean_field, keepnans=False)
                target_flat = target_flat_cpu.to(device=device)
                patch_loss = ((flat - target_flat).abs() ** 2).sum()
                local_grad = torch.autograd.grad(patch_loss / len(domains), local)[0]
                grad_full.index_copy_(-1, core, local_grad.index_select(-1, core_pos))
                loss_value += patch_loss.detach().item() / len(domains)
                del (
                    local_grad,
                    patch_loss,
                    target_flat,
                    flat,
                    stats,
                    st_op,
                    patch,
                    local,
                    core_pos,
                    core,
                    support,
                )

            optim.zero_grad(set_to_none=True)
            running.grad = grad_full
            optim.step()
            loss_history.append(loss_value)
            if verbose and (
                (iteration + 1) % print_iter == 0 or iteration + 1 == max_iter
            ):
                print(
                    f"[patch-{optimizer_name}] iter {iteration + 1}/{max_iter}, "
                    f"mean local loss = {loss_value:.6e}"
                )
    finally:
        if stop is not None:
            stop.set()
            thread.join()
    elapsed = time.perf_counter() - start

    output = running.detach() * std_target[..., None] + mean_target[..., None]
    if input_dim == 1:
        output = output[:, 0]
    if nbatch == 1:
        output = output[0]

    diagnostics = {
        "objective": "mean overlapping local ScatCov losses",
        "loss_history": loss_history,
        "elapsed_seconds": elapsed,
        "patch_count": len(domains),
        "core_pixels": len(domains[0][0]),
        "max_graph_pixels": max_support_pixels,
        "full_sky_pixels": npix,
        "halo_rings": halo_rings,
        "patch_nside": patch_nside,
        "optimizer": optimizer_name,
        "rss_start_bytes": None if rss is None else rss["start"],
        "rss_peak_bytes": None if rss is None else rss["peak"],
        "rss_peak_increase_bytes": (
            None if rss is None else max(0, rss["peak"] - rss["start"])
        ),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" and track_memory
            else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" and track_memory
            else None
        ),
    }
    if verbose:
        ratio = max_support_pixels / npix
        print(
            f"Patch synthesis: {elapsed:.3f} s; largest graph domain "
            f"{max_support_pixels}/{npix} pixels ({100 * ratio:.1f}%)."
        )
    return (output, diagnostics) if return_diagnostics else output


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
    compute_PS=None,
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

    ndim_pix = data_target.NDIM_PIX

    if running_mask is None:
        # Same mask for running and target
        if running_shape is not None:
            data_running = data_target.__class__(
                array=np.zeros(running_shape), pbc=pbc_running
            )
        else:
            data_running = data_target.new_like(data_target.array, pbc=pbc_running)
    else:
        if running_shape is None and tuple(
            data_target.array.shape[-ndim_pix:]
        ) != tuple(running_mask.shape):
            raise ValueError("running_mask shape should match target array shape")
        elif running_shape is not None and tuple(running_shape) != tuple(
            running_mask.shape
        ):
            raise ValueError("running_mask shape should match running_shape")

        data_running = data_target.new_like(running_mask, pbc=pbc_running)

    # Select J used for synthesis
    J_target = data_target.get_wavelet_op().J - (not data_target.pbc)
    J_running = data_running.get_wavelet_op().J - (not data_running.pbc)
    J = min(J_target, J_running)

    if J_target != J_running:
        print(
            f"Warning: target.J = {J_target}, running.J = {J_running}. Synthesis will use J = {J}."
        )

    # Decide about the power spectrum *before* touching the spectrum operators:
    # building one is expensive (a spherical harmonic transform at high
    # resolution), so it must not happen when the spectrum will not be used.
    target_has_nan = data_target.array.isnan().any()
    running_has_nan = data_running.array.isnan().any()

    if compute_PS is None:
        compute_PS = not (target_has_nan or running_has_nan)
        if target_has_nan or running_has_nan:
            print(
                "⚠️ Warning: NaN detected in target and/or running data.\n"
                "Power spectrum optimization is disabled because its computation is not yet implemented for NaN values in any dataclass. \n"
            )
    elif compute_PS and (target_has_nan or running_has_nan):
        raise ValueError(
            "compute_PS=True was requested but the data contain NaNs, on which "
            "the power spectrum is undefined."
        )

    st_op_kwargs = {"has_fewer_convolutions": has_fewer_convolutions}
    if compute_PS:
        n_bins = min(data_target.get_CS_op().n_bins, data_running.get_CS_op().n_bins)
        st_op_kwargs["n_bins"] = n_bins

    # Get scattering operators for target and running data with selected J
    st_op_target = data_target.get_ST_op(J=J, compute_PS=compute_PS, **st_op_kwargs)

    st_op_running = data_running.get_ST_op(
        J=J,
        compute_PS=compute_PS,
        replace_nan_value=None,
        **st_op_kwargs,
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
            True if init_running is None else not _contains_nan(init_running)
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
                array = torch.zeros(
                    tuple(getattr(target_stats, "pix_shape", None) or target_stats.N0)
                )
            else:
                array = torch.where(target_stats.mask_full_res.array, torch.nan, 0.0)
        array = array.to(device=target_stats.device, dtype=target_stats.dtype)

        proto = getattr(target_stats, "data_example", None)
        data_running = (
            proto.new_like(array, pbc=pbc_running)
            if proto is not None
            else target_stats.DataClass(array=array, pbc=pbc_running)
        )
    else:
        pix_shape = tuple(getattr(target_stats, "pix_shape", None) or target_stats.N0)
        if tuple(running_mask.shape) != pix_shape:
            raise ValueError("running_mask shape should match the target pixel grid")
        proto = getattr(target_stats, "data_example", None)
        data_running = (
            proto.new_like(running_mask, pbc=pbc_running)
            if proto is not None
            else target_stats.DataClass(array=running_mask, pbc=pbc_running)
        )

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
