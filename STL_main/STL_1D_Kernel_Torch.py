import math
from dataclasses import dataclass

import numpy as np
import torch

import STL_main.torch_backend as bk
from STL_main.STL_2D_Kernel_Torch import (
    CS_operator_2D_Kernel_Torch,
    STL_2D_Kernel_Torch,
    WaveletOperator2Dkernel_torch,
)
from STL_main.torch_backend import (
    _DEFAULT_DEVICE,
    _DEFAULT_DTYPE,
    _get_device,
    _get_dtype,
)


class WaveletOperator1Dkernel_torch(WaveletOperator2Dkernel_torch):
    ndim = 1

    @staticmethod
    def bump_steerable_1d(omega_grid, xi0, eps=1e-12):
        """
        Generate a 1D bump steerable wavelet in Fourier space.

        Parameters
        ----------
        omega_grid : torch.Tensor
            Grid of frequencies in Fourier space, of shape [Nx], where the last dimension corresponds to (omega_x).
        xi0 : float
            Center frequency xi = (xi0, 0) in Fourier space.
        eps : float
            Small constant to avoid division by zero in the bump window.

        Returns
        -------
        torch.Tensor
            A 1D bump steerable wavelet in Fourier space, of shape [Nx].
        """

        r = abs(omega_grid - xi0) / xi0

        # apply bump window over r: g(r) = exp(-r^2 / (1 - r^2)) * 1_{0<r<1}
        r2 = r**2
        support_r = (r >= 0.0) & (r < 1.0)
        denom = (1.0 - r2).clamp_min(eps)
        bump = torch.where(support_r, torch.exp(-r2 / denom), torch.zeros_like(r))

        return bump

    @classmethod
    def bump_steerable_bank(cls, J, size):
        """
        Generate a bank of 1D bump steerable wavelets in Fourier space.

        Parameters
        ----------
        J : int
            Number of dyadic scales.
        size : tuple of int
            Grid size (Nx,).

        Returns
        -------
        torch.Tensor
            A tensor of shape [J, 1, Nx]
        """
        (Nx,) = size
        L = 1  # For 1D, we only have one orientation
        filters_bank = torch.empty((J, L, Nx))
        xi0 = Nx / 2.0  # same convention as 1D scattering spectra

        # Create the frequency grid in Fourier space, with the zero frequency at (0,)
        omega_x = torch.fft.fftfreq(Nx) * Nx
        omega_x = torch.fft.fftshift(omega_x)  # Shift zero frequency to center

        for j in range(J):
            scale_factor = 2**j
            for l_idx, l in enumerate(range(L)):
                filters_bank[j, l_idx] = torch.fft.fftshift(
                    cls.bump_steerable_1d(scale_factor * omega_x, xi0=xi0)
                )

        return filters_bank

    def __init__(
        self,
        J,
        L=1,
        kernel_size=None,
        WType="Bump-Steerable",
        DT="Planar1D_kernel_torch",
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        mask_full_res=None,
        sigma_smooth=1.0,
        downsample_nan_weight_threshold=0.5,  # higher threshold needed for 1D compared to 2D
        get_crop_border_size_method=None,
    ):

        if J is None:
            raise ValueError(
                "J must be specified for WaveletOperator1Dkernel_torch class."
            )
        self.WType = WType

        self.J = J

        if L is None:
            L = 1
        if L != 1:
            raise ValueError(
                f"L must be 1 for WaveletOperator1Dkernel_torch but got {L}"
            )
        self.L = L
        self.KERNELSZ = kernel_size if kernel_size is not None else 5
        self.DT = DT
        if self.DT == "Planar1D_kernel_torch":
            self.data_class = STL_1D_Kernel_Torch
        else:
            raise ValueError(
                f"DT {self.DT} not recognized. Available option: 'Planar1D_kernel_torch'."
            )

        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        if self.WType == "Morlet":
            self._wav_kernel = self._build_morlet_wavelet_kernel()  # [1, L, K]
        elif self.WType == "Bump-Steerable":
            self._wav_kernel = self._build_bump_steerable_wavelet_kernel()  # [1, L, K]
        else:
            raise ValueError(
                f"WType {self.WType} not recognized. Available options: 'Bump-Steerable' or 'Morlet'."
            )

        self.sigma_smooth = (
            sigma_smooth  # to build smoothing kernel used in downsampling
        )
        # raise
        # build low pass kernel?

        # PBC dependant parameters
        if get_crop_border_size_method is not None:
            self._get_crop_border_size_method = get_crop_border_size_method
        else:
            self._get_crop_border_size_method = (
                self.__class__._get_crop_border_size_fully_flexible
            )

        # NaNs handling
        self.mask_full_res = (
            mask_full_res  # None if no NaN in the data. Is True where the data is NaN.
        )
        if self.mask_full_res is not None:
            if (
                self.mask_full_res.array.ndim != self.__class__.ndim
            ):  # TODO: check if this is correct for 1D kernel data
                raise NotImplementedError(
                    "For now, mask_full_res.array must be 1D if provided."
                )

        self.downsample_nan_weight_threshold = downsample_nan_weight_threshold
        (
            self._reweighting_maps_smooth,
            self._reweighting_maps_wav,
            self._layer1_mask,
            self._layer2_mask,
        ) = self._build_reweighting_maps_and_scattering_layer_masks()
        self.j_to_dg = range(J)

    def _build_wavelet_kernel_from_ifft_crop(
        self, fft_wavelet_builder, N=256
    ):  # TODO: improve in the future
        assert (
            self.KERNELSZ % 2 == 1
        ), "KERNELSZ must be odd to have a well-defined center."
        w = self.KERNELSZ // 2
        kernel = fft_wavelet_builder(self.J, size=(N,))[
            0
        ]  # [1, N] # TODO: improve in the future
        kernel = torch.fft.fftshift(torch.fft.ifft(kernel, dim=(-1)), dim=(-1))
        kernel = kernel[:, N // 2 - w : N // 2 + w + 1]  # [1, K]
        kernel -= kernel.mean(dim=(-1,), keepdims=True)
        # Cast to the operator's working precision so the kernel matches the data
        # dtype in the convolution (e.g. complex128 for float64 data on CPU).
        complex_dtype = (
            torch.complex128 if self.dtype == torch.float64 else torch.complex64
        )
        kernel = kernel.to(device=self.device, dtype=complex_dtype)
        kernel = kernel.unsqueeze(0)  # (1, 1, K)

        return kernel

    def _build_bump_steerable_wavelet_kernel(self):
        return self._build_wavelet_kernel_from_ifft_crop(
            fft_wavelet_builder=WaveletOperator1Dkernel_torch.bump_steerable_bank
        )  # (1, L, K)

    def _get_smooth_kernel(self, device, dtype):
        """
        Build and cache a normalized smoothing kernel on (device, dtype)
        for antialiasing 1D filter used in downsampling.

        Returns
        -------
        kernel : torch.Tensor
            Shape (2,) or (3,) or (6,)
        """
        if (
            not hasattr(self, "_smooth_kernel")
            or self._smooth_kernel.device != device
            or self._smooth_kernel.dtype != dtype
        ):
            if False:
                # Precomputed 2-elelement kernel -> no padding needed for striding 2 downsampling, but more aliasing
                self._smooth_kernel = torch.tensor(
                    [0.5, 0.5],
                    device=device,
                    dtype=dtype,
                )
            elif False:
                # Precomputed 5-element Gaussian kernel
                self._smooth_kernel = torch.tensor(
                    [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                    device=device,
                    dtype=dtype,
                )
                self._smooth_kernel /= self._smooth_kernel.sum()
            elif False:
                # Precomputed 3-element kernel -> each pixel has the same weight with striding 2 downsampling
                self._smooth_kernel = torch.tensor(
                    [0.25, 0.5, 0.25],
                    device=device,
                    dtype=dtype,
                )
            else:
                # Precomputed 6-element cropped raised cosine kernel with bandwidth=.71, beta=.35
                # to have smaller loss of energy as possible and avoid aliasing while keeping a small kernel size for efficiency
                self._smooth_kernel = torch.tensor(
                    [
                        [
                            0.00238073,
                            -0.00275097,
                            -0.02402609,
                            -0.02402609,
                            -0.00275097,
                            0.00238073,
                        ],
                        [
                            -0.00275097,
                            0.0031788,
                            0.0277626,
                            0.0277626,
                            0.0031788,
                            -0.00275097,
                        ],
                        [
                            -0.02402609,
                            0.0277626,
                            0.2424694,
                            0.2424694,
                            0.0277626,
                            -0.02402609,
                        ],
                        [
                            -0.02402609,
                            0.0277626,
                            0.2424694,
                            0.2424694,
                            0.0277626,
                            -0.02402609,
                        ],
                        [
                            -0.00275097,
                            0.0031788,
                            0.0277626,
                            0.0277626,
                            0.0031788,
                            -0.00275097,
                        ],
                        [
                            0.00238073,
                            -0.00275097,
                            -0.02402609,
                            -0.02402609,
                            -0.00275097,
                            0.00238073,
                        ],
                    ],
                    device=device,
                    dtype=dtype,
                )
                self._smooth_kernel = self._smooth_kernel[
                    2, :
                ]  # TODO: check if this is correct for 1D kernel data
                self._smooth_kernel /= self._smooth_kernel.sum()
        return self._smooth_kernel


class CS_operator_1D_Kernel_Torch(CS_operator_2D_Kernel_Torch):
    """
    Class whose instances correspond to a cross spectrum operator for 1D Kernel data.
    The operator is applied through apply method and is DT-dependent.
    """

    # class constant
    ndim = 1
    default_DT = "Planar1D_kernel_torch"

    def _build_legacy_bin_masks(self):
        raise NotImplementedError

    def _build_log_gaussian_bin_masks(self):
        (N,) = self.shape

        FX = torch.fft.fftfreq(N, d=1.0, device=self.device)
        self.radial_freq = torch.fft.fftshift(torch.abs(FX)).to(dtype=self.dtype)

        self.min_freq = 1 / (2.0**self.J)
        self.max_freq = 0.5 / (2.0**self.Jmin)

        log_edges = torch.linspace(
            np.log(self.min_freq),
            np.log(self.max_freq),
            self.n_bins + 1,
            device=self.device,
            dtype=self.dtype,
        )
        self.bin_edges = torch.exp(log_edges)

        log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        self.bin_centers = torch.exp(log_centers)

        log_sigma = torch.abs(log_edges[1:] - log_edges[:-1])
        log_sigma = torch.where(log_sigma > 0, log_sigma, torch.ones_like(log_sigma))
        log_radial = torch.log(
            torch.where(
                self.radial_freq > 0,
                self.radial_freq,
                torch.ones_like(self.radial_freq),
            )
        )
        self.bin_masks = torch.exp(
            -0.5
            * ((log_radial[None, :] - log_centers[:, None]) ** 2)
            / (log_sigma[:, None] ** 2)
        )  # [n_bins, N]
        self.bin_masks = torch.where(
            (self.radial_freq == 0)[None, :],
            torch.zeros_like(self.bin_masks),
            self.bin_masks,
        )
        self.lam = None

    def estimate_crop_borders(self):

        (N,) = self.shape

        # Create impulse at right border, centered vertically
        impulse = torch.zeros((N,), device=self.device, dtype=self.dtype)
        impulse[N // 2] = 1.0

        # FFT of impulse
        impulse_ft = torch.fft.fftshift(torch.fft.fft(impulse, norm="ortho"))  # [N]

        # Apply all masks in batch
        impulse_ft = impulse_ft.unsqueeze(0)  # [1, N] for broadcasting
        psfs = torch.fft.ifft(
            torch.fft.ifftshift(impulse_ft * self.bin_masks, dim=(-1,)),
            norm="ortho",
            dim=-1,
        ).real  # [n_bins, N]

        # Extract horizontal traces from pixel source
        traces = psfs[:, : N // 2].abs()  # [n_bins, N//2]

        # Determine border where PSF drops below threshold_percent of the trace at the source pixel (maximum value)
        threshold_percent = 0.1
        threshold = threshold_percent * traces[:, -1].unsqueeze(1)  # [n_bins, 1]
        above_thresh = traces > threshold  # [n_bins, N//2]
        self.crop_borders = math.ceil(N / 2) - (
            above_thresh.float().argmax(dim=1) + 1
        )  # [n_bins]

    ###########################################################################
    def _build_smooth_mask_crop(self, array, border):
        raise NotImplementedError

    def _build_hard_mask_crop(self, array, border):
        n_bins_dim, N = array.shape[-2], array.shape[-1]
        if border.shape[0] != n_bins_dim:
            raise ValueError(
                f"border length ({border.shape[0]}) does not match n_bins ({n_bins_dim})"
            )

        rows = torch.arange(N, device=array.device).view(1, N)
        border_broadcast = border.to(device=array.device).view(n_bins_dim, 1)
        mask = (rows >= border_broadcast) & (rows < (N - border_broadcast))

        return mask

    ###########################################################################
    def apply_fft(
        self, data, compute_cross_spectrum_matrix=None, get_crop_border_size_method=None
    ):
        """
        Compute the power spectrum of the input data array attribute.

        Parameters
        ----------
        - data : STL_2D_Kernel_Torch
            Input data whose array attribute's power spectrum is to be computed.
        - compute_cross_spectrum_matrix : torch.BoolTensor of shape [Nc, Nc]
            Boolean matrix indicating which cross-spectra to compute. If None, only auto-spectra are computed.
        - get_crop_border_size_method : str or None
            Method to determine crop border size for non-PBC data. If None, uses the default method specified in the operator initialization.

        Returns
        -------
        torch.Tensor
            Cross spectrum values of shape [..., Nc, Nc, n_bins]
        """
        # consistency check
        if data.DT != self.DT:
            raise Exception(f"Data Type should be {self.DT}, got {data.DT}.")
        if self.shape != data.N0:
            raise Exception("Data shape does not match operator shape")
        if data.dg != 0:
            raise Exception("Data dg must be 0 for power spectrum computation")
        if data.array.isnan().any():
            raise ValueError(
                "Data array contains NaN values, cannot compute power spectrum with fft method."
            )
        if self.device != data.device:
            raise Exception("Data device does not match operator device")

        get_crop_border_size_method = (
            self.get_crop_border_size_method
            if get_crop_border_size_method is None
            else get_crop_border_size_method
        )

        # Ensure data is in Fourier space
        l_data = data.copy(empty=False)
        l_data.array = torch.fft.fft(
            l_data.array, norm="ortho", dim=tuple(range(-self.ndim, 0))[0]  # [..., N]
        )  # copy of data in Fourier space
        l_data.array = torch.fft.fftshift(
            l_data.array, dim=tuple(range(-self.ndim, 0))
        )  # [..., N]

        # Put in the expected shape if not already (should be already done in ST_op apply)
        if l_data.array.ndim < l_data.ndim + 2:
            raise ValueError(
                f"Data array has unexpected shape {l_data.array.shape}. Expected at least {l_data.ndim + 2} dimensions, because batch and channel dimensions are expected."
            )
        elif l_data.array.ndim == l_data.ndim + 2:
            l_data.array = l_data.array[:, :, None, ...]  # [Nb=Nb, Nc=Nc, Nc=1, ..., N]
        else:
            pass

        if False:
            if l_data.array.ndim == l_data.ndim:
                l_data.array = l_data.array[
                    None, None, None, ...
                ]  # [Nb=1, Nc=1, Nc=1, ..., N]
            elif l_data.array.ndim == l_data.ndim + 1:
                l_data.array = l_data.array[
                    None, :, None, ...
                ]  # [Nb=1, Nc=Nc, Nc=1, ..., N]
            elif l_data.array.ndim == l_data.ndim + 2:
                l_data.array = l_data.array[
                    :, :, None, ...
                ]  # [Nb=Nb, Nc=Nc, Nc=1, ..., N]
            else:
                raise NotImplementedError(
                    f"Data array has unexpected shape {l_data.array.shape}."
                )

        Nb, Nc = l_data.array.shape[:2]
        n_bins = self.n_bins

        cross_spectrum = (
            bk.zeros(
                (Nb, Nc, Nc) + l_data.array.shape[3 : -l_data.ndim] + (n_bins,),
                dtype=bk._DEFAULT_COMPLEX_DTYPE,
            )
            + bk.nan
        )

        compute_cross_spectrum_matrix = (
            bk.eye(Nc, dtype=bool)
            if compute_cross_spectrum_matrix is None
            else compute_cross_spectrum_matrix
        )

        bin_norm = self.bin_masks.sum(dim=tuple(range(-self.ndim, 0)))  # [nbins]

        if l_data.pbc:

            cross_vals = (
                torch.einsum(
                    "bcd...x,bcd...x,jx,j->bcd...j",
                    l_data.array,
                    torch.conj(l_data.array),
                    self.bin_masks.to(dtype=l_data.array.dtype),
                    1 / bin_norm,
                )
            ).to(dtype=bk._DEFAULT_COMPLEX_DTYPE)

            # Symetric part is redundant and then not filled as cross_spectrum(c1, c2) and cross_spectrum(c2, c1) are conjugates
            cross_spectrum[:, compute_cross_spectrum_matrix, :] = cross_vals[
                :, compute_cross_spectrum_matrix, :
            ]

            return cross_spectrum  # [Nb, Nc, Nc, ..., n_bins]

        if get_crop_border_size_method == "flexible_crop":
            border = self.crop_borders  # [n_bins]
        elif get_crop_border_size_method == "largest_crop":
            # border = torch.zeros(self.n_bins)
            border = torch.full_like(
                self.crop_borders, self.crop_borders.max()
            )  # [n_bins]
        else:
            raise ValueError(
                f"Invalid get_crop_border_size_method: {get_crop_border_size_method}"
            )

        l_data_bin = (
            l_data.array[:, :, :, ..., None, :] * self.bin_masks[None, None, None, ...]
        )  # [Nb, Nc, Nc, ..., n_bins, N]

        ifft_l_data_bin = torch.fft.ifft(
            l_data_bin, norm="ortho", dim=tuple(range(-self.ndim, 0))[0]
        )  # [Nb, Nc, Nc, ..., n_bins, N]
        l_data.array = torch.fft.ifft(
            l_data.array, norm="ortho", dim=tuple(range(-self.ndim, 0))[0]
        )  # [Nb, Nc, Nc, ..., N]
        mask_source = (
            ifft_l_data_bin[0, 0, 0]
            if self.power_spectrum_method == "gaussian_rings"
            else l_data.array
        )
        mask_crop = self.build_mask_crop(mask_source, border=border)  # [n_bins, N]
        mask_crop_c = mask_crop.to(dtype=ifft_l_data_bin.dtype)
        prefactor = np.prod(l_data.N0) / mask_crop.sum(
            dim=tuple(range(-self.ndim, 0))
        )  # [n_bins]

        cross_vals = (
            torch.einsum(
                "bcd...jx,bcd...x,jx,j->bcd...j",
                ifft_l_data_bin,
                torch.conj(l_data.array),
                mask_crop_c,
                prefactor / bin_norm,
            )
        ).to(dtype=bk._DEFAULT_COMPLEX_DTYPE)

        cross_spectrum[:, compute_cross_spectrum_matrix, :] = cross_vals[
            :, compute_cross_spectrum_matrix, :
        ]

        return cross_spectrum  # [Nb, Nc, Nc, ..., n_bins]


###############################################################################
###############################################################################
@dataclass
class STL_1D_Kernel_Torch(STL_2D_Kernel_Torch):
    """
    STL_1D_Kernel_Torch child class for 1D planar STL Kernel using PyTorch

    Inherits STL_2D_Kernel_Torch.

    See STL_2D_Kernel_Torch for parameter descriptions.

    Additional comments
    -------------------
    The initial resolution N0 is fixed, but the maps can be downgraded. The
    downgrading factor is the power of 2 that is used. A map of initial
    resolution N0=256 and with dg = 3 is thus at resolution 256/2^3 = 32.
    The downgraded resolutions are called N0, N1, N2, ...

    Can store array at a given downgradind dg:
        - attribute MR is False
        - attribute N0 gives the initial resolution
        - attribute dg gives the downgrading level
        - array is an array of size (..., N) with N = N0 // 2^dg
    Or at multi-resolution (MR):
        - attribute MR is True
        - attribute N0 gives the initial resolution
        - attribute dg is None
        - array is a list of array of sizes (..., N1), (..., N2), etc.,
        with the same dimensions excepts N.

    Method usages if MR=True.
        - mean, cov give a single vector or last dim len(list_N)
        - downsample gives an output of size (..., len(list_N), Nout). Only
          possible if all resolution are downsampled this way.

    The class initialization is the frontend one, which can work from DT and
    data only. It enforces MR=False and dg=0. Two backend init functions for
    MR=False and MR=True also exist.

    Attributes
    ----------
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N0 : tuple of int
        Initial size of array (can be multiple dimensions)
    - dg : int
        2^dg is the downgrading level w.r.t. N0.
    - array : array (..., N)
          array(s) to store
    """

    # child class constant
    DT = "Planar1D_kernel_torch"
    ndim = 1
    wavelet_op_class = WaveletOperator1Dkernel_torch
    cs_op_class = CS_operator_1D_Kernel_Torch
