#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEALPix kernel-based data class for STL.

HEALPix analogue of STL_2D_Kernel_Torch, with the same public interface so that
the DT-independent machinery (ST_Operator, ST_Statistics, Synthesis) can be used
unchanged:

  - data live on HEALPix pixels, the array is shaped (..., Npix)
  - convolutions and downsampling are performed with the `healpix-analyse`
    package (HealPixConv / HealPixDown)
  - the power spectrum is the angular power spectrum C_ell, computed with the
    differentiable spherical harmonic transform of the same package
    (HEALPixSHT: map2alm / alm2map / anafast)
  - the statistics (mean, square_mean, cov, standardize, ...) live on the
    *wavelet operator*, not on the data class, exactly as in the planar kernel

Assumptions
-----------
- Data are real PyTorch tensors (complex after a wavelet convolution).
- The last dimension is always the pixel axis (NDIM_PIX = 1).
- Pixel indexing is HEALPix NESTED by default; downsampling requires NESTED
  ordering.

Status w.r.t. the planar kernel
-------------------------------
Implemented: data class, wavelet convolution, smoothing, downsampling, the full
statistics interface consumed by ST_Operator, and the angular cross-spectrum
operator.
Masked data are supported: passing `mask_full_res` (or simply feeding a map
with NaNs) precomputes the invalid-pixel mask of every (layer, resolution) the
chain visits, together with the reweighting maps that keep the decimation
amplitude-correct. `nan_aware_stats=True` remains as a lighter, mask-free
substitute based on nanmean.
Not implemented yet:
  - deconvolution of the mask-induced mode coupling in the partial-sky power
    spectrum: CS_operator_Healpix_Torch returns a pseudo-C_ell, with no MASTER
    matrix, exactly as the planar operator does not deconvolve its crop window.
"""

import inspect
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down import HealPixDown
from healpix_analyse.healpix_sht import HEALPixSHT

from STL_main.Base_DataClass import Base_DataClass
from STL_main.ST_Operator import ST_Operator
from STL_main.torch_backend import (
    _DEFAULT_DEVICE,
    _DEFAULT_DTYPE,
    _get_device,
    _get_dtype,
    maskmean,
    nan,
)


###############################################################################
def _build_hpa_operator(cls, nside, level, required, optional=None):
    """
    Build a healpix-analyse operator, adapting to the signature of the installed
    release.

    Across versions the resolution argument has been spelled `level`, `nside` or
    `nside_in`, and some options only exist in some releases. `required` entries
    must be present in the signature -- dropping one silently would change the
    result -- while `optional` ones are skipped when the installed version does
    not know them.

    Parameters
    ----------
    cls : type
        The healpix-analyse class to instantiate.
    nside, level : int
        The resolution, in both spellings (level = log2(nside)).
    required : dict
        Keyword arguments that must be accepted by the constructor.
    optional : dict or None
        Keyword arguments passed only when the constructor knows them.

    Returns
    -------
    An instance of `cls`.
    """
    params = inspect.signature(cls.__init__).parameters

    call = {}
    if "nside" in params:
        call["nside"] = int(nside)
    elif "nside_in" in params:
        call["nside_in"] = int(nside)
    if "level" in params:
        call["level"] = int(level)

    if not call:
        raise TypeError(
            f"{cls.__name__} exposes none of the expected resolution arguments "
            "('level', 'nside', 'nside_in'): this version of healpix-analyse is "
            "not supported."
        )

    for name, value in required.items():
        if name not in params:
            raise TypeError(
                f"{cls.__name__} has no '{name}' argument: this version of "
                "healpix-analyse is not supported."
            )
        call[name] = value

    for name, value in (optional or {}).items():
        if name in params:
            call[name] = value

    return cls(**call)


###############################################################################
###############################################################################
@dataclass
class STL_Healpix_Kernel_Torch(Base_DataClass):
    """
    STL_Healpix_Kernel_Torch child class for HEALPix maps using PyTorch.

    Inherits Base_DataClass. See Base_DataClass for the shared parameters
    (array, pbc, dg, N0, conv_history).

    Additional comments
    -------------------
    The initial resolution N0 = (nside,) is fixed, and maps can be downgraded.
    The downgrading factor is the power of 2 that is used, so that a map with
    N0 = (32,) and dg = 3 lives at nside = 32 // 2**3 = 4.

    Parameters
    ----------
    - array : np.ndarray or torch.Tensor
        Input data of shape (..., Npix).
    - pbc : bool or None
        Kept for interface compatibility with the planar kernel. On the sphere,
        True means "full sky" (no boundary, hence no border effect) and False
        means "partial sky described by cell_ids". If None, it is set to True
        when the map covers the full sky and False otherwise.
    - dg : int
        2**dg is the downgrading level w.r.t. N0.
    - N0 : tuple of int
        (nside,) at dg = 0. Inferred from Npix for a full-sky map.
    - conv_history : list of int
        Scales at which the data have been convolved so far.
    - cell_ids : array-like or None
        HEALPix pixel indices of the last axis. If None, full sky [0..Npix-1].
    - nest : bool
        NESTED indexing (required by the downsampling).
    - nside : int or None
        Convenience argument to give the resolution explicitly (mandatory for a
        partial sky). After construction, the attribute holds the *current*
        resolution, i.e. N0[0] // 2**dg.

    Attributes
    ----------
    - DT : str
        Data type identifier.
    - NDIM_PIX : int
        1 -- a single pixel axis.
    """

    # child class constants
    DT = "HealpixKernel_torch"
    NDIM_PIX = 1

    # child class fields (come after the Base_DataClass ones)
    cell_ids: Optional[torch.Tensor] = None
    nest: bool = True
    nside: Optional[int] = None

    ###########################################################################
    @classmethod
    def _infer_N0(cls, array):
        """
        Infer (nside,) from a full-sky array of Npix = 12 * nside**2 pixels.

        For a partial sky the resolution cannot be read from the array shape,
        and `nside` must be passed explicitly to the constructor.
        """
        npix = int(array.shape[-1])
        nside = int(round(math.sqrt(npix / 12.0)))
        if 12 * nside**2 != npix or nside < 1:
            raise ValueError(
                f"Cannot infer nside from Npix={npix} (not a full-sky HEALPix map). "
                "Please pass nside=... (and cell_ids=...) explicitly."
            )
        return (nside,)

    ###########################################################################
    def __post_init__(self):

        if isinstance(self.array, list):
            raise ValueError(
                "Only single-resolution arrays are accepted at construction."
            )

        # An explicitly given nside seeds N0 before the base class fills it in.
        if self.N0 is None and self.nside is not None:
            self.N0 = (int(self.nside),)

        super().__post_init__()

        self.N0 = tuple(int(n) for n in self.N0)
        self.nest = bool(self.nest)

        # From now on, `nside` is the resolution of the stored array.
        self.nside = self.N0[0] // (2**self.dg)

        # Cell ids
        npix = int(self.array.shape[-1])
        if self.cell_ids is None:
            self.cell_ids = torch.arange(npix, device=self.device, dtype=torch.long)
        else:
            self.cell_ids = self._to_cell_ids_tensor(self.cell_ids, npix)

        # Full sky <-> periodic-like behaviour (no boundary to worry about)
        if self.pbc is None:
            self.pbc = npix == 12 * self.nside**2

    ###########################################################################
    def _to_cell_ids_tensor(self, cell_ids, Npix_expected=None):
        """
        Convert any cell_ids-like (list/np/tensor) to a 1D LongTensor on
        self.device, optionally checking its length against Npix_expected.
        """
        if isinstance(cell_ids, torch.Tensor):
            cid = cell_ids.to(device=self.device, dtype=torch.long).view(-1)
        else:
            cid = torch.as_tensor(cell_ids, device=self.device, dtype=torch.long).view(
                -1
            )

        if (Npix_expected is not None) and (cid.numel() != Npix_expected):
            raise ValueError(
                f"cell_ids length {cid.numel()} does not match Npix={Npix_expected}."
            )
        return cid

    ###########################################################################
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the array attribute of data.

        Parameters
        ----------
        - inplace : bool
            If True, acts in-place and returns self.
            If False, returns a new STL_Healpix_Kernel_Torch instance.

        Returns
        -------
        STL_Healpix_Kernel_Torch
            Instance whose array attribute is the modulus.
        """
        data = self.copy(empty=False) if not inplace else self

        data.array = data.array.abs()
        data.dtype = data.array.dtype

        return data

    ###########################################################################
    def divide(self, data2, epsilon=1e-8, pow=1.0, inplace=False):
        """
        Divide self.array by data2.array raised to a power, with a small epsilon
        added to the denominator for numerical stability.

        Parameters
        ----------
        data2 : STL_Healpix_Kernel_Torch
            Instance whose array is used as the denominator.
        epsilon : float, optional
            Small constant added to the denominator (default 1e-8).
        pow : float, optional
            Exponent applied to the denominator (default 1).
        inplace : bool
            If True, performs the operation in-place and returns self.

        Returns
        -------
        STL_Healpix_Kernel_Torch
        """
        data1 = self.copy(empty=False) if not inplace else self

        data1.array = data1.array / (data2.array + epsilon) ** pow
        data1.dtype = data1.array.dtype

        return data1

    ###########################################################################
    def apply_bandlimit(self, array):
        """
        Project the map onto the multipoles the pixelisation can carry, by a
        round trip through the spherical harmonic transform.

        Only defined on the full sky: on a partial sky the transform would need
        the mask deconvolution the spectrum operator does not do either, so the
        array is returned untouched.
        """
        npix_full = 12 * self.N0[0] ** 2
        if array.shape[-1] != npix_full:
            return array

        sht = _build_hpa_operator(
            HEALPixSHT,
            nside=self.N0[0],
            level=int(round(math.log2(self.N0[0]))),
            required={},
            optional={"dtype": self.dtype, "device": self.device},
        )
        return sht.alm2map(sht.map2alm(array, nest=self.nest), nest=self.nest)

    ###########################################################################
    def get_wavelet_op(
        self,
        J=None,
        L=None,
        kernel_size=None,
        mask_full_res=None,
        *args,
        **kwargs,
    ):
        """
        Build the HEALPix wavelet operator, analogous to
        STL_2D_Kernel_Torch.get_wavelet_op.

        The default number of scales stops at nside = 4, mirroring the planar
        convention which stops well before the map becomes a handful of pixels.

        When the data carry NaNs and no mask is given, the mask is built from
        them, exactly as the planar kernel does. Pass `mask_full_res=False` to
        opt out of that inference and keep an unmasked operator.
        """
        J = J if J is not None else max(1, int(np.log2(self.N0[0])) - 1)

        if mask_full_res is False:
            mask_full_res = None
        elif mask_full_res is None and bool(torch.any(self.array.isnan())):
            if self.array.ndim != 1:
                raise NotImplementedError(
                    "The NaN mask can only be inferred from a single map; pass "
                    "mask_full_res explicitly for batched or multi-channel data."
                )
            mask_full_res = STL_Healpix_Kernel_Torch(
                array=self.array.isnan(),
                nside=self.N0[0],
                cell_ids=self.cell_ids,
                nest=self.nest,
                pbc=self.pbc,
            )

        return WaveletOperatorHealpixKernel_torch(
            J=J,
            L=L,
            kernel_size=kernel_size,
            DT=self.DT,
            nside=self.N0[0],
            nest=self.nest,
            cell_ids=self.cell_ids,
            device=self.device,
            dtype=self.dtype,
            mask_full_res=mask_full_res,
            *args,
            **kwargs,
        )

    ###########################################################################
    def get_ST_op(self, *args, **kwargs):
        """
        Build the scattering transform operator for this data type.
        """
        return ST_Operator(data_example=self, *args, **kwargs)

    ###########################################################################
    def get_CS_op(self, *args, **kwargs):
        """
        Build the angular cross-spectrum operator, analogous to
        STL_2D_Kernel_Torch.get_CS_op.

        On the sphere the power spectrum is the angular power spectrum C_ell,
        computed with the spherical harmonic transform of healpix-analyse.
        """
        return CS_operator_Healpix_Torch(
            nside=self.N0[0],
            nest=self.nest,
            cell_ids=self.cell_ids,
            device=self.device,
            dtype=self.dtype,
            *args,
            **kwargs,
        )


###############################################################################
###############################################################################
class WaveletOperatorHealpixKernel_torch:
    """
    HEALPix wavelet operator built on the `healpix-analyse` package.

    Mirrors WaveletOperator2Dkernel_torch:
      - `apply(data, j)` convolves with the L-oriented wavelet bank,
      - `downsample(data, dg_out, ...)` moves to a coarser resolution,
      - `mean`, `square_mean`, `cov`, `standardize`, `unstandardize` and
        `_compute_and_store_cross_cov` provide the statistics used by
        ST_Operator.

    Geometry backend
    ----------------
    `healpix_analyse.convol.HealPixConv` performs the gauge-equivariant
    convolution, and `healpix_analyse.down.HealPixDown` the anti-aliased
    decimation. Both are differentiable torch modules, work on a partial sky
    through `cell_ids`, and precompute their geometry once per resolution --
    which is why they are cached here per (resolution, grid, gauge count).

    Orientation convention
    ----------------------
    A single complex Morlet-like kernel is defined on a KxK stencil, and the L
    orientations come from the L *gauges* of HealPixConv (the stencil is
    rotated, not the kernel). Hence L == n_gauges. The real and imaginary parts
    are carried as two output channels of a single convolution, so one call
    produces the complex answer.

    TODO: align the radial profile and the angle convention
    (a = (L-1-i)/L * pi, normalization by the mean of the smoothing envelope)
    with the planar kernel, so that the coefficients are directly comparable
    between the two data types.
    """

    ###########################################################################
    def __init__(
        self,
        J,
        L=None,
        kernel_size=None,
        WType="HealpixWavelet",
        DT="HealpixKernel_torch",
        nside=None,
        nest=True,
        cell_ids=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        mask_full_res=None,
        sigma_smooth=1.0,
        downsample_nan_weight_threshold=0.33,
        get_crop_border_size_method=None,
        nan_aware_stats=False,
        gauge_type="cosmo",
        ellipsoid="WGS84",
        down_kwargs=None,
    ):
        if J is None:
            raise ValueError(
                "J must be specified for WaveletOperatorHealpixKernel_torch class."
            )
        if nside is None:
            raise ValueError(
                "nside must be specified for WaveletOperatorHealpixKernel_torch class."
            )

        self.WType = WType
        if self.WType not in ("HealpixWavelet", "Morlet"):
            raise ValueError(
                f"WType {self.WType} not recognized. Available options: "
                "'HealpixWavelet' (default) or 'Morlet'."
            )

        self.J = int(J)
        self.L = L if L is not None else 4
        self.KERNELSZ = kernel_size if kernel_size is not None else 5
        self.DT = DT

        self.nside = int(nside)
        self.nest = bool(nest)
        self.gauge_type = gauge_type
        self.ellipsoid = ellipsoid
        self.down_kwargs = dict(down_kwargs or {})

        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        # --- caches of healpix-analyse operators (geometry is expensive) ---
        self._conv_cache = {}
        self._down_cache = {}

        # --- kernels ---
        # complex wavelet on the KxK stencil, kept for inspection/plotting
        self.kernel = self._wavelet_kernel(self.KERNELSZ, self.L).reshape(
            1, 1, self.KERNELSZ * self.KERNELSZ
        )
        # HealPixConv weights: [C_in=1, C_out=2, P] carrying (real, imag)
        self._wav_weights = torch.cat(
            [self.kernel.real, self.kernel.imag], dim=1
        )  # (1, 2, P)

        # real low-pass used for same-resolution smoothing, [C_in=1, C_out=1, P]
        self.sigma_smooth = sigma_smooth
        self.smooth_kernel = self._smooth_kernel(self.KERNELSZ, sigma=sigma_smooth)

        # --- scale <-> resolution mapping (scale j lives at resolution dg = j) ---
        self.j_to_dg = range(self.J)

        # --- NaNs / masks handling ---
        self.mask_full_res = mask_full_res
        self.downsample_nan_weight_threshold = downsample_nan_weight_threshold
        self.nan_aware_stats = bool(nan_aware_stats)
        self._build_masks(cell_ids)

        # No border to crop on the sphere: kept so that the planar and spherical
        # operators expose the same attribute.
        self._get_crop_border_size_method = (
            get_crop_border_size_method
            if get_crop_border_size_method is not None
            else self.__class__._get_crop_border_size_zero
        )

    ###########################################################################
    @staticmethod
    def _get_crop_border_size_zero(data, wavelet_op):
        """No border on the sphere: the planar crop has no counterpart here."""
        return 0

    ###########################################################################
    def _wavelet_kernel(self, kernel_size: int, n_orientation: int, sigma=1.0):
        """
        Create the complex directional wavelet on a KxK tangent grid.

        Note: `n_orientation` is accepted for signature compatibility but is not
        used -- the orientations come from the convolution gauges (see the class
        docstring). A single kernel is returned.

        Returns
        -------
        kernel : torch.Tensor
            Complex tensor of shape (1, 1, K, K).
        """
        coords = (
            torch.arange(kernel_size, device=self.device, dtype=self.dtype)
            - (kernel_size - 1) / 2.0
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        # Isotropic Gaussian envelope
        mother_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))[None, :, :]

        # Orientations done in the gauges paradigm
        angles_proj = 0.5 * torch.pi * (xx[None, ...])

        kernel = torch.complex(
            torch.cos(angles_proj) * mother_kernel,
            torch.sin(angles_proj) * mother_kernel,
        )

        # Zero-mean and L1 normalization
        kernel = kernel - torch.mean(kernel, dim=(1, 2), keepdim=True)
        kernel = kernel / torch.sum(kernel.abs(), dim=(1, 2), keepdim=True)

        return kernel.reshape(1, 1, kernel_size, kernel_size)

    ###########################################################################
    def _smooth_kernel(self, kernel_size: int, sigma=1.0):
        """
        Build the low-pass kernel used for same-resolution smoothing, in the
        [C_in=1, C_out=1, P] layout expected by HealPixConv.

        A Gaussian envelope rather than the modulus of the wavelet, so that the
        smoothing is a controlled low-pass.

        Returns
        -------
        torch.Tensor
            Real tensor of shape (1, 1, P) summing to 1.
        """
        coords = (
            torch.arange(kernel_size, device=self.device, dtype=self.dtype)
            - (kernel_size - 1) / 2.0
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        w = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        w = w / w.sum()

        return w.reshape(1, 1, kernel_size * kernel_size)

    ###########################################################################
    #                     healpix-analyse operator cache                      #
    ###########################################################################
    @staticmethod
    def _grid_key(cell_ids_t):
        """Cheap but discriminating fingerprint of a target grid."""
        return (
            int(cell_ids_t.numel()),
            int(cell_ids_t[0].item()),
            int(cell_ids_t[-1].item()),
            int(cell_ids_t.sum().item()),
        )

    def _grid_spec(self, dg, cell_ids):
        """
        Return (nside, level, cell_ids_numpy_or_None) for the given resolution.

        A full sky in canonical order is passed as cell_ids=None, which lets
        healpix-analyse take its fast path.
        """
        nside = self.nside // (2**dg)
        if nside < 1:
            raise ValueError(f"dg={dg} is too deep for nside={self.nside}.")
        level = int(round(math.log2(nside)))

        cid = torch.as_tensor(cell_ids).view(-1)
        npix = int(cid.numel())
        full_sky = npix == 12 * nside**2 and bool(
            (cid.cpu() == torch.arange(npix)).all()
        )
        cid_np = None if full_sky else cid.detach().cpu().numpy().astype(np.int64)
        return nside, level, cid_np

    def _get_conv(self, dg, cell_ids, n_gauges, weights):
        """
        Return a cached HealPixConv for this resolution / grid / gauge count,
        with `weights` ([C_in, C_out, P]) already installed.
        """
        cid_t = torch.as_tensor(cell_ids).view(-1)
        c_out = int(weights.shape[1])
        key = (
            int(dg),
            int(self.KERNELSZ),
            int(n_gauges),
            int(c_out),
            str(self.gauge_type),
            bool(self.nest),
            self._grid_key(cid_t),
        )

        conv = self._conv_cache.get(key, None)
        if conv is None:
            nside, level, cid_np = self._grid_spec(dg, cid_t)
            conv = _build_hpa_operator(
                HealPixConv,
                nside=nside,
                level=level,
                required=dict(
                    in_channels=1,
                    out_channels=c_out,
                    kernel_sz=self.KERNELSZ,
                    n_gauges=n_gauges,
                    gauge_type=self.gauge_type,
                    cell_ids=cid_np,
                    nest=self.nest,
                ),
                optional=dict(
                    device=self.device,
                    dtype=self.dtype,
                    ellipsoid=self.ellipsoid,
                ),
            )
            conv.set_kernel(weights.detach().cpu().numpy())
            self._conv_cache[key] = conv

        return conv

    def _get_down(self, dg, cell_ids):
        """Return a cached HealPixDown taking this resolution one level down."""
        cid_t = torch.as_tensor(cell_ids).view(-1)
        key = (int(dg), bool(self.nest), self._grid_key(cid_t))

        down = self._down_cache.get(key, None)
        if down is None:
            nside, level, cid_np = self._grid_spec(dg, cid_t)
            down = _build_hpa_operator(
                HealPixDown,
                nside=nside,
                level=level,
                required=dict(mode="smooth", cell_ids=cid_np),
                optional=dict(
                    device=self.device,
                    dtype=self.dtype,
                    ellipsoid=self.ellipsoid,
                    **self.down_kwargs,
                ),
            )
            self._down_cache[key] = down

        return down

    ###########################################################################
    def get_L(self):
        return self.L

    ###########################################################################
    def _check_data(self, data):
        if type(data).__name__ != "STL_Healpix_Kernel_Torch":
            raise Exception(
                f"Data should be a STL_Healpix_Kernel_Torch instance, got {type(data)}"
            )
        if self.DT != data.DT:
            raise Exception("Data and wavelet transform should have same DT")

    ###########################################################################
    #                              CONVOLUTIONS                               #
    ###########################################################################
    def apply(self, data, j: int):
        """
        Apply the wavelet convolution to data.array [..., Npix] and return
        [..., L, Npix].

        Parameters
        ----------
        data : STL_Healpix_Kernel_Torch
            Input data, at downgrading level dg == j.
        j : int
            Scale index.

        Returns
        -------
        STL_Healpix_Kernel_Torch
            New object with array shape [..., L, Npix], same nside/cell_ids, and
            an updated conv_history.
        """
        self._check_data(data)

        if j != data.dg:
            raise ValueError(
                "j is not equal to data.dg; convolution not consistent with scale."
            )

        x = data.array  # [..., Npix]
        if torch.is_complex(x):
            raise NotImplementedError(
                "Convolution of complex data is not supported (real input expected)."
            )

        cid = data.cell_ids
        *leading, K = x.shape
        B = int(np.prod(leading)) if leading else 1

        conv = self._get_conv(data.dg, cid, n_gauges=self.L, weights=self._wav_weights)

        # (B, 1, K) -> (B, L*2, K): output channel o = g * C_out + o
        y = conv(x.reshape(B, 1, K).to(dtype=self.dtype))
        y = y.reshape(B, self.L, 2, y.shape[-1])
        y = torch.complex(y[:, :, 0], y[:, :, 1])  # (B, L, K)

        out = data.copy(empty=True)
        out.array = y.reshape(*leading, self.L, y.shape[-1])
        out.dtype = out.array.dtype
        out.cell_ids = cid.clone()
        out.conv_history = list(data.conv_history) + [j]
        return out

    ###########################################################################
    def apply_smooth(self, data, inplace: bool = True):
        """
        Smooth the data with the low-pass kernel, preserving the resolution.

        Parameters
        ----------
        data : STL_Healpix_Kernel_Torch
        inplace : bool
            If False, return a new instance instead of modifying `data`.

        Returns
        -------
        STL_Healpix_Kernel_Torch
        """
        self._check_data(data)

        x = data.array  # [..., Npix]
        cid = data.cell_ids
        *leading, K = x.shape
        B = int(np.prod(leading)) if leading else 1

        conv = self._get_conv(data.dg, cid, n_gauges=1, weights=self.smooth_kernel)

        y = self._apply_linear(conv, x.reshape(B, 1, K)).reshape(*leading, K)

        out = data.copy(empty=True) if not inplace else data
        out.array = y
        if not inplace:
            out.cell_ids = cid
        return out

    ###########################################################################
    @staticmethod
    def _apply_linear(op, x):
        """
        Apply a healpix-analyse linear operator to (possibly complex) data,
        keeping only the tensor part of the answer.

        Complex data are split into real and imaginary parts, since the sparse
        operators are defined on real tensors.
        """
        if torch.is_complex(x):
            re = op(x.real)
            im = op(x.imag)
            re = re[0] if isinstance(re, tuple) else re
            im = im[0] if isinstance(im, tuple) else im
            return torch.complex(re, im)

        y = op(x)
        return y[0] if isinstance(y, tuple) else y

    ###########################################################################
    def _smooth_with_nan(self, data, inplace: bool = True):
        """
        NaN-aware smoothing: the map and the validity mask are both smoothed,
        and the result is normalized by the sum of the valid weights.

        Pixels with no valid neighbour in the kernel support come back as NaN.
        """
        self._check_data(data)

        x = data.array  # [..., Npix]
        cid = data.cell_ids
        *leading, K = x.shape
        B = int(np.prod(leading)) if leading else 1

        conv = self._get_conv(data.dg, cid, n_gauges=1, weights=self.smooth_kernel)

        x_bc = x.reshape(B, 1, K)
        valid = self._valid_mask(x_bc)
        x_filled = torch.where(valid, x_bc, torch.zeros_like(x_bc))
        mask_f = valid.to(x.real.dtype if torch.is_complex(x) else x.dtype)

        num = self._apply_linear(conv, x_filled)
        w_sum = self._apply_linear(conv, mask_f)

        y = num / (w_sum + 1e-8)
        y = torch.where(w_sum <= 0, torch.full_like(y, float("nan")), y)
        y = y.reshape(*leading, K)

        out = data if inplace else data.copy(empty=False)
        out.array = y
        if not inplace:
            out.cell_ids = cid
        return out

    ###########################################################################
    #                             DOWNSAMPLING                                #
    ###########################################################################
    @staticmethod
    def _valid_mask(x):
        """True where the sample is not NaN (either part, for complex data)."""
        if torch.is_complex(x):
            return ~(torch.isnan(x.real) | torch.isnan(x.imag))
        return ~torch.isnan(x)

    def _down_one_level(self, data, nan_aware):
        """
        Anti-aliased decimation by one level with healpix-analyse.

        HealPixDown("smooth") is a linear, l1-normalized sparse operator, so the
        NaN-aware variant is obtained by applying it to the zero-filled data and
        to the validity mask, then dividing.
        """
        down = self._get_down(data.dg, data.cell_ids)

        x = data.array
        *leading, K = x.shape
        B = int(np.prod(leading)) if leading else 1
        x_flat = x.reshape(B, K)

        def apply_down(t):
            """Run the operator once, keeping the coarse ids it returns."""
            if torch.is_complex(t):
                re, ids = down(t.real)
                im, _ = down(t.imag)
                return torch.complex(re, im), ids
            return down(t)

        if nan_aware:
            valid = self._valid_mask(x_flat)
            mask_f = valid.to(x.real.dtype if torch.is_complex(x) else x.dtype)
            x_filled = torch.where(valid, x_flat, torch.zeros_like(x_flat))

            num, out_ids = apply_down(x_filled)
            den, _ = apply_down(mask_f)
            out = num / (den + 1e-8)
            out = torch.where(den <= 0, torch.full_like(out, float("nan")), out)
        else:
            out, out_ids = apply_down(x_flat)

        data.array = out.reshape(*leading, out.shape[-1])
        data.cell_ids = torch.as_tensor(
            np.asarray(out_ids), device=data.array.device, dtype=torch.long
        )
        data.dg += 1
        data.nside = data.N0[0] // (2**data.dg)
        return data

    ###########################################################################
    def _bin_to_parents(self, data, dg_out, nan_aware):
        """
        Plain NESTED binning of the pixel axis onto the parent pixels at dg_out,
        without any anti-aliasing (used when smooth=False).

        parent_id = cell_id // 4**(dg_out - dg)
        """
        if not data.nest:
            raise ValueError(
                "Downsampling requires NESTED pixel ordering (data.nest must be True)."
            )

        delta_g = dg_out - data.dg
        factor_pix = 4**delta_g  # children per parent in NESTED

        cid = data.cell_ids
        parent_ids = torch.div(cid, factor_pix, rounding_mode="floor")

        x = data.array
        *leading, K = x.shape
        B = int(np.prod(leading)) if leading else 1
        x_flat = x.reshape(B, K)

        parent_unique, inv = torch.unique(parent_ids, return_inverse=True)
        Kc = parent_unique.numel()
        idx = inv.unsqueeze(0).expand(B, -1)

        if nan_aware:
            valid = self._valid_mask(x_flat)
            mask_f = valid.to(
                x_flat.real.dtype if torch.is_complex(x_flat) else x_flat.dtype
            )
            x_filled = torch.where(valid, x_flat, torch.zeros_like(x_flat))

            out_sum = torch.zeros(B, Kc, device=x_flat.device, dtype=x_flat.dtype)
            out_sum = out_sum.scatter_add(1, idx, x_filled)

            out_count = torch.zeros(B, Kc, device=mask_f.device, dtype=mask_f.dtype)
            out_count = out_count.scatter_add(1, idx, mask_f)

            out = out_sum / (out_count + 1e-8)
            out = torch.where(out_count <= 0, torch.full_like(out, float("nan")), out)
        else:
            out = torch.zeros(B, Kc, device=x_flat.device, dtype=x_flat.dtype)
            out = out.scatter_add(1, idx, x_flat)
            counts = torch.bincount(inv, minlength=Kc)
            out = out / counts.to(
                x_flat.real.dtype if torch.is_complex(x_flat) else x_flat.dtype
            )

        data.array = out.reshape(*leading, Kc)
        data.cell_ids = parent_unique.to(device=data.array.device, dtype=torch.long)
        data.dg = dg_out
        data.nside = data.N0[0] // (2**dg_out)
        return data

    ###########################################################################
    def downsample(
        self,
        data,
        dg_out,
        inplace=True,
        replace_nan_value=nan,
        smooth=True,
        nan_aware=None,
    ):
        """
        Downsample the data to the dg_out resolution.

        With `smooth=True` (default) the decimation is done one level at a time
        with healpix-analyse's anti-aliased operator; with `smooth=False` the
        four NESTED children are simply averaged. Same signature as the planar
        `downsample`, so that ST_Operator can call it unchanged.

        Parameters
        ----------
        data : STL_Healpix_Kernel_Torch
        dg_out : int
            Target downgrading level (dg_out >= data.dg >= 0).
        inplace : bool
            If False, work on a copy.
        replace_nan_value : float or None
            Accepted for interface parity with the planar kernel. Only used when
            the NaN-aware path is active.
        smooth : bool
            Anti-aliased decimation (True) or plain child averaging (False).
        nan_aware : bool or None
            If None, NaN-aware processing is used when a mask is declared on the
            operator or when nan_aware_stats is True.

        Returns
        -------
        STL_Healpix_Kernel_Torch
        """
        self._check_data(data)

        dg_out = int(dg_out)
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out < data.dg:
            raise ValueError(
                "Requested dg_out < current dg; upsampling not supported by "
                "downsampling method."
            )
        if dg_out == data.dg:
            return data if inplace else data.copy(empty=False)

        data = data if inplace else data.copy(empty=False)

        if self.mask_full_res is not None:
            return self._downsample_masked(data, dg_out, replace_nan_value, smooth)

        if nan_aware is None:
            nan_aware = self.nan_aware_stats

        if smooth:
            while data.dg < dg_out:
                data = self._down_one_level(data, nan_aware=nan_aware)
        else:
            data = self._bin_to_parents(data, dg_out, nan_aware=nan_aware)

        if nan_aware and replace_nan_value is not None:
            # WARNING: as in the planar kernel, replacing NaNs breaks backprop
            # through those pixels.
            if not (
                isinstance(replace_nan_value, float) and math.isnan(replace_nan_value)
            ):
                data.array = torch.where(
                    torch.isnan(data.array),
                    torch.full_like(data.array, replace_nan_value),
                    data.array,
                )

        return data

    ###########################################################################
    def _downsample_masked(self, data, dg_out, replace_nan_value, smooth):
        """
        Decimation driven by the precomputed masks, mirroring the planar kernel.

        The invalid pixels are zeroed, the map is taken down one level at a
        time, and each step is rescaled by the reweighting map so that the
        remaining valid pixels keep their amplitude. Pixels that become invalid
        are finally set to `replace_nan_value`.
        """
        layer = len(data.conv_history)
        if layer == 0:
            convolved_at = None
        elif layer == 1:
            convolved_at = int(data.conv_history[0])
        else:
            raise ValueError("data must be at layer 0 or 1 to be downsampled.")

        if not smooth:
            raise NotImplementedError(
                "The masked decimation follows the anti-aliased path; "
                "smooth=False is not supported when a mask is declared."
            )

        input_mask = self._find_mask(data)
        data.array = torch.where(~input_mask, data.array, torch.zeros_like(data.array))

        while data.dg < dg_out:
            data = self._down_one_level(data, nan_aware=False)
            reweight = (
                self._reweight_smooth[data.dg]
                if convolved_at is None
                else self._reweight_wav[(data.dg, convolved_at)]
            )
            data.array = data.array * reweight.to(dtype=data.array.real.dtype)

        if replace_nan_value is not None:
            # WARNING: as in the planar kernel, writing a finite value here
            # would let the invalid pixels contribute to the backward pass.
            output_mask = self._find_mask(data)
            data.array = torch.where(
                ~output_mask,
                data.array,
                torch.full_like(data.array, replace_nan_value),
            )

        return data

    ###########################################################################
    def nandownsample(self, data, dg_out, inplace=True, smooth=True):
        """
        NaN-aware downsampling. Kept for backward compatibility: this is
        `downsample(..., nan_aware=True, replace_nan_value=None)`.
        """
        return self.downsample(
            data,
            dg_out,
            inplace=inplace,
            replace_nan_value=None,
            smooth=smooth,
            nan_aware=True,
        )

    ###########################################################################
    #                     MASKS AND REWEIGHTING (NaN handling)                #
    ###########################################################################
    def _build_masks(self, cell_ids):
        """
        Precompute, once per operator, the invalid-pixel mask of every
        (layer, resolution) the scattering chain will visit, together with the
        reweighting maps that compensate the missing pixels when decimating.

        This is the spherical counterpart of the planar
        `_build_reweighting_maps_and_scattering_layer_masks`. Two things differ:

        * there is no padding mode to enumerate -- the geometry is fixed by
          `cell_ids`, so a single set of maps is enough;
        * "convolving with the wavelet support" becomes a convolution with a
          kernel of ones over the stencil, taken over all L gauges, so the mask
          is eroded by the exact union of the supports actually used.

        Layers follow the planar convention, indexed by len(conv_history):

        * layer 0 -- the data themselves, possibly decimated;
        * layer 1 -- |I * psi_j|, at resolution dg >= j;
        * layer 2 -- |I * psi_j2| * psi_j3, at resolution dg = j3.

        Everything is stored as boolean masks (True where the pixel is invalid)
        on the operator's own grid, plus float reweighting maps that are zero on
        the invalid pixels.
        """
        self._cell_ids_at = None
        self._mask_smooth = None
        self._reweight_smooth = None
        self._layer1_mask = None
        self._mask_wav = None
        self._reweight_wav = None
        self._layer2_mask = None

        if self.mask_full_res is None:
            return

        m = self.mask_full_res.array
        if m.ndim != 1:
            raise NotImplementedError(
                "For now, mask_full_res.array must be 1D (a single map of "
                f"Npix pixels), got {tuple(m.shape)}."
            )

        invalid = m.to(torch.bool) if m.dtype == torch.bool else (m != 0)
        invalid = invalid.to(self.device)

        if cell_ids is None:
            cell_ids = getattr(self.mask_full_res, "cell_ids", None)
        if cell_ids is None:
            raise ValueError(
                "cell_ids are needed to build the masks; pass them to the "
                "operator or carry them on mask_full_res."
            )
        cid0 = torch.as_tensor(cell_ids).view(-1).to(self.device)
        if invalid.numel() != cid0.numel():
            raise ValueError(
                f"mask_full_res has {invalid.numel()} pixels but the operator "
                f"grid has {cid0.numel()}."
            )

        threshold = float(self.downsample_nan_weight_threshold)
        J = self.J

        # ---- layer 0: successive decimations of the invalid mask ----------
        cid = {0: cid0}
        mask_smooth = {0: invalid}
        reweight_smooth = {}

        for dg in range(1, J):
            frac, ids = self._down_field(
                dg - 1, cid[dg - 1], mask_smooth[dg - 1].to(self.dtype)
            )
            cid[dg] = ids
            mask_smooth[dg] = frac > threshold
            reweight_smooth[dg] = self._reweight_from_fraction(frac, mask_smooth[dg])

        # ---- layer 1: erosion by the wavelet support at every scale -------
        layer1_mask = {
            j: self._support_mask(j, cid[j], mask_smooth[j]) for j in range(J)
        }

        # ---- layer 1 decimated: |I * psi_j| taken down to dg > j ----------
        mask_wav, reweight_wav = {}, {}
        for j in range(J - 1):
            previous = layer1_mask[j]
            for dg in range(j + 1, J):
                frac, _ = self._down_field(dg - 1, cid[dg - 1], previous.to(self.dtype))
                mask_wav[(dg, j)] = frac > threshold
                reweight_wav[(dg, j)] = self._reweight_from_fraction(
                    frac, mask_wav[(dg, j)]
                )
                previous = mask_wav[(dg, j)]

        # ---- layer 2: second erosion, at the resolution of the second scale
        layer2_mask = {}
        for j3 in range(J):
            for j2 in range(j3 + 1):
                source = layer1_mask[j3] if j2 == j3 else mask_wav[(j3, j2)]
                layer2_mask[(j3, j2)] = self._support_mask(j3, cid[j3], source)

        self._cell_ids_at = cid
        self._mask_smooth = mask_smooth
        self._reweight_smooth = reweight_smooth
        self._layer1_mask = layer1_mask
        self._mask_wav = mask_wav
        self._reweight_wav = reweight_wav
        self._layer2_mask = layer2_mask

    ###########################################################################
    @staticmethod
    def _reweight_from_fraction(fraction, invalid, eps=1e-8):
        """
        Turn a local invalid fraction f into the factor 1/(1-f) that restores
        the amplitude of a zero-filled average, and zero it where the output
        pixel is itself declared invalid.
        """
        weight = 1.0 / (1.0 - fraction).clamp_min(eps)
        return torch.where(invalid, torch.zeros_like(weight), weight)

    ###########################################################################
    def _down_field(self, dg, cell_ids, field):
        """
        Take one real field down by one level with the same operator the data
        use, and return it together with the coarse pixel indices.
        """
        down = self._get_down(dg, cell_ids)
        out, ids = down(field.reshape(1, -1))
        return (
            out.reshape(-1),
            torch.as_tensor(np.asarray(ids), device=field.device, dtype=torch.long),
        )

    ###########################################################################
    def _support_mask(self, dg, cell_ids, invalid):
        """
        Erode an invalid mask by the stencil support.

        A convolution with a kernel of ones, evaluated on the L gauges, is
        non-zero exactly on the pixels whose support touches an invalid pixel --
        that is, on the pixels the wavelet convolution would contaminate.
        """
        ones = torch.ones_like(self.smooth_kernel)
        conv = self._get_conv(dg, cell_ids, n_gauges=self.L, weights=ones)

        touched = conv(invalid.to(self.dtype).reshape(1, 1, -1))  # (1, L, Npix)
        return (touched.abs() > 1e-12).any(dim=1).reshape(-1)

    ###########################################################################
    #                              STATISTICS                                 #
    ###########################################################################
    def _find_mask(self, data):
        """
        Return the boolean mask (True where the pixel is invalid) matching the
        layer and the resolution of `data`, or None when the operator carries no
        mask.

        The layer is read off `len(data.conv_history)`, exactly as in the planar
        kernel: 0 for the data themselves, 1 after one wavelet convolution,
        2 after two.
        """
        if self.mask_full_res is None:
            return None

        layer = len(data.conv_history)
        dg = int(data.dg)

        if layer == 0:
            mask = self._mask_smooth[dg]
        elif layer == 1:
            j = int(data.conv_history[0])
            mask = self._layer1_mask[j] if dg == j else self._mask_wav[(dg, j)]
        elif layer == 2:
            if dg != int(data.conv_history[-1]):
                raise ValueError(
                    "Layer-2 data are expected at the resolution of their last "
                    f"convolution (dg={dg}, conv_history={data.conv_history})."
                )
            mask = self._layer2_mask[(dg, int(data.conv_history[0]))]
        else:
            raise ValueError("len(data.conv_history) must be 0, 1 or 2.")

        if mask.shape[-1] != data.array.shape[-1]:
            raise ValueError(
                f"The stored mask has {mask.shape[-1]} pixels but the data have "
                f"{data.array.shape[-1]}: the data are not on the grid the "
                "operator was built for."
            )
        return mask

    @staticmethod
    def _nanmean(x, dim):
        """
        NaN-ignoring mean that also works for complex tensors (torch.nanmean
        does not support them). A sample is discarded when either its real or
        its imaginary part is NaN.
        """
        if not torch.is_complex(x):
            return x.nanmean(dim=dim)

        valid = ~(torch.isnan(x.real) | torch.isnan(x.imag))
        count = valid.sum(dim=dim)
        x_filled = torch.where(valid, x, torch.zeros_like(x))
        out = x_filled.sum(dim=dim) / count.clamp_min(1).to(x.real.dtype)
        return torch.where(count > 0, out, torch.full_like(out, float("nan")))

    def _reduce(self, x, dim, mask):
        """Masked (or NaN-aware) mean over the pixel axis."""
        if mask is None and self.nan_aware_stats:
            return self._nanmean(x, dim=dim)
        return maskmean(x=x, dim=dim, mask=mask)

    ###########################################################################
    def mean(self, data, dim=None):
        """
        Compute the mean over the pixel dimension.
        """
        dim = dim if dim is not None else -1
        return self._reduce(data.array, dim=dim, mask=self._find_mask(data))

    ###########################################################################
    def square_mean(self, data, dim=-1, **kwargs):
        """
        Compute the mean of |x|^2 over the pixel dimension.
        """
        return self._reduce(
            data.array * data.array.conj(), dim=dim, mask=self._find_mask(data)
        )

    ###########################################################################
    def cov(self, data1, data2, remove_mean=None, dim=None, specific_channel_pair=None):
        """
        Compute the covariance between data1 and data2 over the pixel dimension.
        """
        assert data1.dg == data2.dg, "data1 and data2 must have the same resolution."

        dim = dim if dim is not None else -1
        remove_mean = remove_mean if remove_mean is not None else False

        if remove_mean:
            raise NotImplementedError(
                "remove_mean is not yet implemented. Think about giving the right "
                "mask when doing it."
            )

        # Choose the mask covering both operands. A deeper layer has been
        # convolved more often, so its invalid region contains the shallower
        # one; at equal depth but different scales -- S4 pairs |I*psi_j1|*psi_j3
        # with |I*psi_j2|*psi_j3 -- neither contains the other and the union is
        # required.
        if self.mask_full_res is None:
            mask = None
        elif len(data1.conv_history) > len(data2.conv_history):
            mask = self._find_mask(data1)
        elif len(data1.conv_history) < len(data2.conv_history):
            mask = self._find_mask(data2)
        elif list(data1.conv_history) == list(data2.conv_history):
            mask = self._find_mask(data1)
        else:
            mask = self._find_mask(data1) | self._find_mask(data2)

        return self._reduce(data1.array * torch.conj(data2.array), dim=dim, mask=mask)

    ###########################################################################
    def standardize(self, data, mean_field, inplace=False, dim=None):
        """
        Standardize the data by removing the mean and scaling to unit variance
        over the pixel dimension.

        Returns
        -------
        - STL_Healpix_Kernel_Torch
            Standardized data.
        - torch.Tensor
            Mean used for standardization.
        - torch.Tensor
            Standard deviation used for standardization.
        """
        if dim is None:
            dim = -1

        l_data = data.copy(empty=False) if not inplace else data

        mean = self.mean(l_data)  # [Nb,Nc]
        if mean_field:
            mean = mean.mean(dim=0, keepdim=True)  # [1,Nc]

        l_data.array = l_data.array - mean[..., None]

        var = self.cov(l_data, l_data)
        if mean_field:
            var = var.mean(dim=0, keepdim=True)

        std = torch.sqrt(var)

        l_data.array = l_data.array / std[..., None]

        return l_data, mean, std

    ###########################################################################
    def unstandardize(self, data, mean, std, inplace=False):
        """
        Undo `standardize` using the provided mean and std.
        """
        l_data = data.copy(empty=False) if not inplace else data

        l_data.array = l_data.array * std[..., None] + mean[..., None]

        return l_data

    ###########################################################################
    def _compute_and_store_cross_cov(
        self,
        data1,
        data2,
        output,
        compute_cross_matrix,
        redundant_channels,
        remove_mean=False,
        dim=-1,
    ):
        """
        Fill `output[:, c1, c2, ...]` with cov(data1[:, c1], data2[:, c2]) for
        every channel pair flagged in compute_cross_matrix.
        """
        assert (
            data1.array.shape[1] == data2.array.shape[1]
        ), "data1 and data2 arrays must have the same number of channels."
        assert (
            data1.array.ndim == data2.array.ndim
        ), "data1 and data2 arrays must have the same number of dimensions."
        assert (
            data1.array.shape[1] == output.shape[1]
        ), "output and data must have the same number of channels."
        assert (
            output.shape[1] == output.shape[2]
        ), "output must have shape (Nb, Nc, Nc, ...)."

        Nc = output.shape[1]

        for c1 in range(Nc):
            for c2 in range(c1, Nc):
                if compute_cross_matrix[c1, c2]:

                    output[:, c1, c2, ...] = self.cov(
                        data1=data1[:, c1, ...],
                        data2=data2[:, c2, ...],
                        remove_mean=remove_mean,
                        dim=dim,
                        specific_channel_pair=(c1, c2),
                    )

                    if not redundant_channels and c1 != c2:
                        output[:, c2, c1, ...] = self.cov(
                            data1=data1[:, c2, ...],
                            data2=data2[:, c1, ...],
                            remove_mean=remove_mean,
                            dim=dim,
                            specific_channel_pair=(c2, c1),
                        )
        return


###############################################################################
###############################################################################
class CS_operator_Healpix_Torch:
    """
    Angular cross-spectrum operator for HEALPix data.

    Spherical counterpart of CS_operator_2D_Kernel_Torch: same public contract
    (`n_bins`, `bin_centers`, `apply(...) -> [Nb, Nc, Nc, n_bins]`,
    `plot_cross_spectrum`), but the estimator is the angular power spectrum
    C_ell rather than an FFT ring binning.

    Everything rests on `healpix_analyse.healpix_sht.HEALPixSHT`, whose
    `map2alm`, `alm2map` and `anafast` are differentiable, so the spectrum can
    be used inside a synthesis loss.

    Binning
    -------
    The C_ell are averaged in `n_bins` logarithmically spaced multipole bins,
    each weighted by (2l+1) times a window W_b(l):

        C_b = sum_l (2l+1) W_b(l) C_l  /  sum_l (2l+1) W_b(l)

    With `power_spectrum_method="gaussian_rings"` the window is a Gaussian in
    log(l), which mirrors the planar `_build_log_gaussian_bin_masks`; with
    `"tophat"` it is a plain indicator over the bin.

    Estimators
    ----------
    Full sky (`data.pbc` True) -- the C_ell come straight from the harmonic
    coefficients, then are binned. Two equivalent routes are available:

    * ``cross_spectrum_method="alm"`` (default) computes `map2alm` once per
      channel and forms every requested pair from those coefficients: Nc
      transforms instead of Nc^2;
    * ``cross_spectrum_method="anafast"`` calls `HEALPixSHT.anafast(im, map2=)`
      for each requested pair. Slower, kept as the reference route.

    Partial sky (`data.pbc` False, or `use_band_maps=True`) -- the map is
    zero-padded outside `cell_ids`, band-filtered with `alm2map`, and the
    cross-spectrum is estimated in pixel space over the observed pixels only:

        C_b = 4 pi * < f_b . g >_observed  /  sum_l (2l+1) W_b(l)

    which reduces exactly to the full-sky expression when every pixel is
    observed. This is a pseudo-C_ell estimator: the mode coupling induced by
    the mask is *not* deconvolved (no MASTER matrix), exactly as the planar
    operator does not deconvolve its crop window.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the data at dg = 0.
    n_bins : int or None
        Number of multipole bins. Defaults to roughly three bins per octave.
    J, Jmin : int or None
        Kept for signature parity with the planar operator. `Jmin` sets the
        lowest multipole kept, l_min = 2**Jmin; `J` is accepted and ignored
        (on the sphere l_max is set by the resolution).
    lmax : int or None
        Highest multipole. Defaults to the transform's own l_max (3*nside-1).
    nest : bool
        NESTED pixel indexing.
    cell_ids : array-like or None
        Pixel indices of the data. None means full sky.
    power_spectrum_method : {"gaussian_rings", "tophat"}
        Shape of the bin windows.
    cross_spectrum_method : {"alm", "anafast"}
        Route used for the full-sky estimator.
    device, dtype : torch device / dtype
    ellipsoid : str
        Passed through to HEALPixSHT when that version accepts it.
    """

    ###########################################################################
    def __init__(
        self,
        nside,
        n_bins=None,
        J=None,
        Jmin=0,
        lmax=None,
        nest=True,
        cell_ids=None,
        power_spectrum_method="gaussian_rings",
        cross_spectrum_method="alm",
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        ellipsoid="sphere",
    ):
        self.nside = int(nside)
        self.level = int(round(math.log2(self.nside)))
        if 2**self.level != self.nside:
            raise ValueError(f"nside={nside} is not a power of 2.")

        self.nest = bool(nest)
        self.ellipsoid = ellipsoid
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        self.J = J  # accepted for parity with the planar operator, unused
        self.Jmin = int(Jmin)

        self.power_spectrum_method = str(power_spectrum_method).lower().strip()
        if self.power_spectrum_method not in {"gaussian_rings", "tophat"}:
            raise ValueError(
                "power_spectrum_method must be either 'gaussian_rings' or 'tophat'"
            )
        self.cross_spectrum_method = str(cross_spectrum_method).lower().strip()
        if self.cross_spectrum_method not in {"alm", "anafast"}:
            raise ValueError("cross_spectrum_method must be either 'alm' or 'anafast'")

        self._cell_ids = (
            None
            if cell_ids is None
            else torch.as_tensor(cell_ids).view(-1).to(torch.long)
        )

        # --- spherical harmonic transform ---
        # Built lazily: its Legendre precompute is very expensive at high
        # resolution, and callers routinely construct this operator only to read
        # n_bins. lmax is therefore derived from the resolution here and checked
        # against the transform the first time one is actually needed.
        self._sht = None
        self._requested_lmax = None if lmax is None else int(lmax)
        self.lmax = (
            self._requested_lmax
            if self._requested_lmax is not None
            else 3 * self.nside - 1
        )

        # --- multipole binning ---
        self.ell = torch.arange(self.lmax + 1, device=self.device, dtype=self.dtype)
        self.n_bins = n_bins
        self._build_bin_windows()

        # --- flat (l, m) bookkeeping used to form the cross-spectra ---
        self._build_alm_index()

    ###########################################################################
    def _get_sht(self):
        """Return the cached HEALPixSHT, building it on first use."""
        if self._sht is None:
            required = {}
            optional = {"dtype": self.dtype, "device": self.device}
            if self._requested_lmax is not None:
                optional["lmax"] = self._requested_lmax
            optional["ellipsoid"] = self.ellipsoid
            self._sht = _build_hpa_operator(
                HEALPixSHT,
                nside=self.nside,
                level=self.level,
                required=required,
                optional=optional,
            )
            built = int(getattr(self._sht, "lmax"))
            if built != self.lmax:
                raise ValueError(
                    f"HEALPixSHT reports lmax={built} where this operator assumed "
                    f"{self.lmax}; pass lmax={built} explicitly."
                )
        return self._sht

    ###########################################################################
    def _build_bin_windows(self):
        """
        Build the [n_bins, lmax+1] window matrix and its (2l+1) weighting.
        """
        l_min = max(1.0, float(2**self.Jmin))
        l_max = float(self.lmax)
        if l_max <= l_min:
            raise ValueError(
                f"lmax={self.lmax} is too small for Jmin={self.Jmin}: no bin left."
            )

        if self.n_bins is None:
            # about three bins per octave, the density the planar operator uses
            self.n_bins = max(4, int(round(3 * math.log2(l_max / l_min))))
        self.n_bins = int(self.n_bins)

        log_edges = torch.linspace(
            math.log(l_min),
            math.log(l_max),
            self.n_bins + 1,
            device=self.device,
            dtype=self.dtype,
        )
        self.bin_edges = torch.exp(log_edges)
        log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        self.bin_centers = torch.exp(log_centers)

        ell = self.ell
        safe_ell = torch.where(ell > 0, ell, torch.ones_like(ell))
        log_ell = torch.log(safe_ell)

        if self.power_spectrum_method == "gaussian_rings":
            log_sigma = (log_edges[1:] - log_edges[:-1]).abs()
            log_sigma = torch.where(
                log_sigma > 0, log_sigma, torch.ones_like(log_sigma)
            )
            windows = torch.exp(
                -0.5
                * (log_ell[None, :] - log_centers[:, None]) ** 2
                / log_sigma[:, None] ** 2
            )
        else:  # tophat
            lo = self.bin_edges[:-1][:, None]
            hi = self.bin_edges[1:][:, None]
            windows = ((ell[None, :] >= lo) & (ell[None, :] < hi)).to(self.dtype)
            # the last bin is closed so that l = lmax is never dropped
            windows[-1] = torch.where(
                ell >= lo[-1, 0], torch.ones_like(ell), windows[-1]
            )

        # the monopole carries no information and l < l_min is excluded
        windows = torch.where(
            (ell >= l_min)[None, :], windows, torch.zeros_like(windows)
        )

        self.bin_windows = windows  # [n_bins, lmax+1]
        self.bin_weights = windows * (2.0 * ell + 1.0)[None, :]
        self.bin_norm = self.bin_weights.sum(dim=-1)  # [n_bins]

        if bool((self.bin_norm <= 0).any()):
            raise ValueError(
                "Some multipole bins are empty; reduce n_bins or lower Jmin."
            )

    ###########################################################################
    def _build_alm_index(self):
        """
        Map the flat a_lm layout of HEALPixSHT onto multipoles.

        The layout is [m=0: l=0..lmax | m=1: l=1..lmax | ... | m=lmax], so the
        multipole of every flat entry and the m>0 doubling factor are built
        once and reused at every call.
        """
        lmax = self.lmax
        ell_of_k, weight_of_k = [], []
        for m in range(lmax + 1):
            ell_of_k.append(np.arange(m, lmax + 1, dtype=np.int64))
            weight_of_k.append(np.full(lmax + 1 - m, 1.0 if m == 0 else 2.0))

        self._ell_of_k = torch.as_tensor(
            np.concatenate(ell_of_k), device=self.device, dtype=torch.long
        )
        self._weight_of_k = torch.as_tensor(
            np.concatenate(weight_of_k), device=self.device, dtype=self.dtype
        )
        self._n_alm = int(self._ell_of_k.numel())

    ###########################################################################
    def _cross_cl(self, alm1, alm2):
        """
        Angular cross-spectrum of two sets of harmonic coefficients.

        C_l = 1/(2l+1) * Re[ a_l0^1 conj(a_l0^2)
                             + 2 sum_{m>0} a_lm^1 conj(a_lm^2) ]

        which is the estimator HEALPixSHT.anafast implements for the auto case.

        Parameters
        ----------
        alm1, alm2 : torch.Tensor
            Complex coefficients of shape (..., K).

        Returns
        -------
        torch.Tensor
            Real C_l of shape (..., lmax+1).
        """
        contrib = (alm1 * alm2.conj()).real * self._weight_of_k

        shape = contrib.shape[:-1] + (self.lmax + 1,)
        cl = torch.zeros(shape, device=contrib.device, dtype=contrib.dtype)
        cl = cl.index_add(-1, self._ell_of_k, contrib)

        return cl / (2.0 * self.ell + 1.0)

    ###########################################################################
    def _bin_cl(self, cl):
        """Bin C_l into the n_bins multipole bands. (..., lmax+1) -> (..., n_bins)."""
        weights = self.bin_weights.to(dtype=cl.dtype)
        return torch.einsum("...l,bl->...b", cl, weights) / self.bin_norm.to(
            dtype=cl.dtype
        )

    ###########################################################################
    def _to_full_sky(self, x, cell_ids):
        """
        Scatter partial-sky data onto the full HEALPix grid, zero elsewhere.

        Parameters
        ----------
        x : torch.Tensor
            (..., Npix_observed)
        cell_ids : torch.LongTensor or None

        Returns
        -------
        torch.Tensor
            (..., 12*nside**2)
        """
        npix_full = 12 * self.nside**2
        if cell_ids is None or (
            x.shape[-1] == npix_full
            and bool((cell_ids.cpu() == torch.arange(npix_full)).all())
        ):
            return x

        full = torch.zeros(x.shape[:-1] + (npix_full,), device=x.device, dtype=x.dtype)
        return full.index_copy(-1, cell_ids.to(x.device), x)

    ###########################################################################
    def apply(
        self,
        data,
        compute_cross_spectrum_matrix=None,
        cross_spectrum_method=None,
        use_band_maps=None,
        **kwargs,
    ):
        """
        Compute the binned angular cross-spectrum of the data.

        Parameters
        ----------
        data : STL_Healpix_Kernel_Torch
            Input data at dg = 0.
        compute_cross_spectrum_matrix : torch.BoolTensor of shape [Nc, Nc]
            Which channel pairs to compute. None means auto-spectra only.
        cross_spectrum_method : {"alm", "anafast"} or None
            Overrides the operator default for this call.
        use_band_maps : bool or None
            Force (True) or forbid (False) the pixel-space band-filtered
            estimator. None picks it automatically for a partial sky.

        Returns
        -------
        torch.Tensor
            Cross-spectra of shape [Nb, Nc, Nc, n_bins]. Entries that were not
            requested are NaN, as in the planar operator.
        """
        if type(data).__name__ != "STL_Healpix_Kernel_Torch":
            raise Exception(
                f"Data should be a STL_Healpix_Kernel_Torch instance, got {type(data)}"
            )
        if tuple(data.N0) != (self.nside,):
            raise Exception(
                f"Data resolution {tuple(data.N0)} does not match operator "
                f"nside {self.nside}."
            )
        if data.dg != 0:
            raise Exception("Data dg must be 0 for power spectrum computation")
        if data.nest != self.nest:
            raise Exception("Data and operator must share the same pixel ordering")
        if bool(data.array.isnan().any()):
            raise ValueError(
                "Data array contains NaN values; the angular power spectrum "
                "cannot be computed on them. Restrict the map through "
                "cell_ids, or build the ST operator with compute_PS=False."
            )

        method = (
            self.cross_spectrum_method
            if cross_spectrum_method is None
            else str(cross_spectrum_method).lower().strip()
        )

        # --- put the data in the expected (Nb, Nc, Npix) shape ---
        x = data.array
        if x.ndim == 1:
            x = x[None, None, :]
        elif x.ndim == 2:
            x = x[None, :, :]
        elif x.ndim != 3:
            raise ValueError(f"Expected data of dimension 1, 2 or 3, got {x.ndim}.")
        Nb, Nc, _ = x.shape

        if compute_cross_spectrum_matrix is None:
            compute_cross_spectrum_matrix = torch.eye(
                Nc, dtype=torch.bool, device=x.device
            )
        else:
            compute_cross_spectrum_matrix = compute_cross_spectrum_matrix.to(
                device=x.device
            )

        full_sky = bool(data.pbc)
        if use_band_maps is None:
            use_band_maps = not full_sky

        maps_full = self._to_full_sky(x.to(dtype=self.dtype), data.cell_ids)

        out = torch.full(
            (Nb, Nc, Nc, self.n_bins),
            float("nan"),
            device=x.device,
            dtype=torch.promote_types(self.dtype, torch.complex64),
        )

        pairs = [
            (c1, c2)
            for c1 in range(Nc)
            for c2 in range(c1, Nc)
            if bool(compute_cross_spectrum_matrix[c1, c2])
        ]
        if not pairs:
            return out

        if use_band_maps:
            values = self._apply_band_maps(maps_full, data, pairs)
        elif method == "anafast":
            values = self._apply_anafast(maps_full, pairs)
        else:
            values = self._apply_alm(maps_full, pairs)

        for (c1, c2), cs in zip(pairs, values):
            # C_l is real and symmetric in the two channels
            out[:, c1, c2, :] = cs.to(dtype=out.dtype)

        return out  # [Nb, Nc, Nc, n_bins]

    ###########################################################################
    def _apply_alm(self, maps_full, pairs):
        """Full-sky estimator: one map2alm per channel, then every pair."""
        sht = self._get_sht()
        alm = sht.map2alm(maps_full, nest=self.nest)  # (Nb, Nc, K)

        return [
            self._bin_cl(self._cross_cl(alm[:, c1], alm[:, c2])) for c1, c2 in pairs
        ]

    ###########################################################################
    def _apply_anafast(self, maps_full, pairs):
        """Full-sky estimator going through HEALPixSHT.anafast pair by pair."""
        sht = self._get_sht()

        values = []
        for c1, c2 in pairs:
            if c1 == c2:
                cl = sht.anafast(maps_full[:, c1], nest=self.nest)
            else:
                cl = sht.anafast(
                    maps_full[:, c1], map2=maps_full[:, c2], nest=self.nest
                )
            values.append(self._bin_cl(cl))
        return values

    ###########################################################################
    def _apply_band_maps(self, maps_full, data, pairs):
        """
        Partial-sky estimator: band-filter with alm2map, then take the
        cross-covariance over the observed pixels.

            C_b = 4 pi * < f_b . g >_observed / sum_l (2l+1) W_b(l)

        `alm2map` is called once for every (batch, channel, bin), in a single
        batched call, because its cost is dominated by the transform setup
        rather than by the batch size.
        """
        sht = self._get_sht()
        alm = sht.map2alm(maps_full, nest=self.nest)  # (Nb, Nc, K)
        Nb, Nc, K = alm.shape

        # window every set of coefficients by W_b(l): (Nb, Nc, n_bins, K)
        window_k = self.bin_windows[:, self._ell_of_k]  # (n_bins, K)
        alm_binned = alm[:, :, None, :] * window_k[None, None, :, :].to(alm.dtype)

        band = sht.alm2map(
            alm_binned.reshape(Nb * Nc * self.n_bins, K), nest=self.nest
        ).reshape(Nb, Nc, self.n_bins, -1)

        # restrict to the observed pixels
        cell_ids = data.cell_ids.to(band.device)
        observed = maps_full.index_select(-1, cell_ids)  # (Nb, Nc, Npix_obs)
        band = band.index_select(-1, cell_ids)  # (Nb, Nc, n_bins, Npix_obs)

        prefactor = 4.0 * math.pi / self.bin_norm  # (n_bins,)

        values = []
        for c1, c2 in pairs:
            cs = (band[:, c1] * observed[:, c2][:, None, :]).mean(dim=-1)
            values.append(cs * prefactor)
        return values

    ###########################################################################
    def plot_cross_spectrum(self, cs_tensor, b=0, c1=0, c2=0, label=None, color="b"):
        """
        Plot a binned angular cross-spectrum.

        Parameters
        ----------
        cs_tensor : torch.Tensor of shape [Nb, Nc, Nc, n_bins]
            Cross-spectra as returned by `apply`.
        b : int
            Batch index.
        c1, c2 : int
            Channel indices.
        label, color : passed through to matplotlib.
        """
        import matplotlib.pyplot as plt

        cs_values = cs_tensor[b, c1, c2, :].real.detach().cpu().numpy()
        ell = self.bin_centers.detach().cpu().numpy()

        if cs_values.shape != ell.shape:
            raise ValueError(
                f"cs values shape {cs_values.shape} and bin_centers shape "
                f"{ell.shape} must match."
            )

        plt.plot(ell, cs_values, "-", marker="o", label=label, color=color)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"multipole $\ell$")
        plt.ylabel(r"$C_\ell$")
        plt.title(f"Angular cross spectrum c{c1 + 1}-c{c2 + 1} for map {b + 1}")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if label is not None:
            plt.legend()


###############################################################################
# Backward-compatible alias (the class used to be spelled "Wavelate").
WavelateOperatorHealpixKernel_torch = WaveletOperatorHealpixKernel_torch
