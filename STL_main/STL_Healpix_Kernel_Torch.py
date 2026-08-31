#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEALPix kernel-based data class for STL.

HEALPix analogue of STL_2D_Kernel_Torch, with the same public interface so that
the DT-independent machinery (ST_Operator, ST_Statistics, Synthesis) can be used
unchanged:

  - data live on HEALPix pixels, the array is shaped (..., Npix)
  - convolutions and downsampling are performed with SphericalStencil
  - the statistics (mean, square_mean, cov, standardize, ...) live on the
    *wavelet operator*, not on the data class, exactly as in the planar kernel

Assumptions
-----------
- Data are real PyTorch tensors (complex after a wavelet convolution).
- The last dimension is always the pixel axis (NDIM_PIX = 1).
- Pixel indexing is HEALPix NESTED by default (must be consistent with
  SphericalStencil.nest); downsampling requires NESTED ordering.

Status w.r.t. the planar kernel
-------------------------------
Implemented: data class, wavelet convolution, smoothing, downsampling, and the
full statistics interface consumed by ST_Operator.
Not implemented yet:
  - get_CS_op(): on the sphere the power spectrum is anafast, not an FFT
    binning. Use ST_Operator(..., compute_PS=False) until it lands.
  - mask/NaN bookkeeping (mask_full_res): the planar kernel precomputes layer
    masks and reweighting maps; the spherical equivalent (mask erosion by the
    stencil support) is not written yet. `nan_aware_stats=True` gives a
    lightweight substitute based on nanmean.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from STL_main.Base_DataClass import Base_DataClass
from STL_main.SphericalStencil import SphericalStencil
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
        """
        J = J if J is not None else max(1, int(np.log2(self.N0[0])) - 1)

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

        Note: compute_PS defaults to False here, because the spherical
        cross-spectrum operator (anafast) is not implemented yet. Pass
        compute_PS=True explicitly once get_CS_op is available.
        """
        kwargs.setdefault("compute_PS", False)
        return ST_Operator(data_example=self, *args, **kwargs)

    ###########################################################################
    def get_CS_op(self, *args, **kwargs):
        raise NotImplementedError(
            "The HEALPix cross-spectrum operator is not implemented yet. On the "
            "sphere the power spectrum is anafast (see foscat.alm.anafast and "
            "foscat.alm_loc.anafast_loc), not an FFT ring binning. In the "
            "meantime, build the ST operator with compute_PS=False."
        )


###############################################################################
###############################################################################
class WaveletOperatorHealpixKernel_torch:
    """
    HEALPix wavelet operator using SphericalStencil.

    Mirrors WaveletOperator2Dkernel_torch:
      - `apply(data, j)` convolves with the L-oriented wavelet bank,
      - `downsample(data, dg_out, ...)` moves to a coarser resolution,
      - `mean`, `square_mean`, `cov`, `standardize`, `unstandardize` and
        `_compute_and_store_cross_cov` provide the statistics used by
        ST_Operator.

    Orientation convention
    ----------------------
    A single complex Morlet-like kernel is built on a KxK tangent stencil, and
    the L orientations are obtained from the L *gauges* of SphericalStencil
    (rotated stencils) rather than from L rotated kernels. Hence L == n_gauges.

    TODO (parity with FOSCAT): align the radial profile and the angle
    convention (a = (L-1-i)/L * pi, normalization by mean(w_smooth)) so that the
    coefficients are directly comparable with foscat.scat_cov.
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

        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        # --- cache for SphericalStencil objects (one per geometry) ---
        # key: (dg, kernel_sz, n_gauges, gauge_type, nest, cell_ids fingerprint)
        self._stencil_cache = {}
        self._default_cell_ids = (
            None if cell_ids is None else torch.as_tensor(cell_ids).view(-1).clone()
        )

        # --- kernels ---
        # (1, 1, P) complex wavelet, flattened for SphericalStencil
        kernel_2d = self._wavelet_kernel(self.KERNELSZ, self.L)  # (1, 1, K, K)
        self.kernel = kernel_2d.reshape(1, 1, self.KERNELSZ * self.KERNELSZ)

        # (1, 1, P) real low-pass used for the anti-aliasing before decimation
        self.sigma_smooth = sigma_smooth
        self.smooth_kernel = self._smooth_kernel(self.KERNELSZ, sigma=sigma_smooth)

        # --- scale <-> resolution mapping (scale j lives at resolution dg = j) ---
        self.j_to_dg = range(self.J)

        # --- NaNs / masks handling ---
        # Kept for interface parity with the planar kernel. The spherical
        # equivalent of the layer masks and reweighting maps is not written yet.
        self.mask_full_res = mask_full_res
        if self.mask_full_res is not None:
            raise NotImplementedError(
                "mask_full_res is not supported yet by the HEALPix kernel. The "
                "spherical counterpart of the planar layer masks (erosion of the "
                "validity mask by the stencil support) still has to be written. "
                "Use nan_aware_stats=True for a lightweight nanmean-based "
                "substitute."
            )
        self.downsample_nan_weight_threshold = downsample_nan_weight_threshold
        self.nan_aware_stats = bool(nan_aware_stats)

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
        used -- the orientations come from the stencil gauges (see the class
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
        Build the low-pass kernel used before decimation, flattened for
        SphericalStencil.

        A Gaussian envelope (as FOSCAT's w_smooth) rather than the modulus of
        the wavelet, so that the decimation is a controlled anti-aliasing.

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
    def _get_stencil(
        self,
        dg: int,
        cell_ids,
        kernel_sz: int,
        n_gauges: int = 1,
        gauge_type=None,
    ):
        """
        Return a cached SphericalStencil for the given geometry.

        The cache key includes a fingerprint of `cell_ids`: two different
        partial skies at the same dg do NOT share a stencil.
        """
        gauge_type = self.gauge_type if gauge_type is None else gauge_type

        cid_t = (
            cell_ids
            if isinstance(cell_ids, torch.Tensor)
            else torch.as_tensor(cell_ids)
        )
        # cheap but discriminating fingerprint of the target grid
        fingerprint = (
            int(cid_t.numel()),
            int(cid_t[0].item()),
            int(cid_t[-1].item()),
            int(cid_t.sum().item()),
        )

        key = (
            int(dg),
            int(kernel_sz),
            int(n_gauges),
            str(gauge_type),
            bool(self.nest),
            fingerprint,
        )

        stencil = self._stencil_cache.get(key, None)
        if stencil is None:
            cid_np = cid_t.detach().cpu().numpy().astype(np.int64)
            stencil = SphericalStencil(
                nside=self.nside // (2**dg),
                kernel_sz=kernel_sz,
                nest=self.nest,
                cell_ids=cid_np,
                n_gauges=n_gauges,
                gauge_type=gauge_type,
                device=self.device,
                dtype=self.dtype,
            )
            self._stencil_cache[key] = stencil
        else:
            # Rebind device/dtype if they changed (the geometry stays valid)
            stencil.device = self.device
            stencil.dtype = self.dtype

        return stencil

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

        # Flatten leading dims into a batch dimension: (B, Ci=1, K)
        B = int(np.prod(leading)) if leading else 1
        x_bc = x.reshape(B, 1, K)

        # Kernel for SphericalStencil: (Ci=1, Co=1, P), applied on L gauges
        wr = torch.real(self.kernel).to(device=x.device, dtype=x.dtype)
        wi = torch.imag(self.kernel).to(device=x.device, dtype=x.dtype)

        l_stencil = self._get_stencil(data.dg, cid, self.KERNELSZ, n_gauges=self.L)
        cid_np = cid.detach().cpu().numpy().astype(np.int64)

        # Convolution on the sphere -> (B, L, K)
        y_bc = torch.complex(
            l_stencil.Convol_torch(x_bc, wr, cell_ids=cid_np),
            l_stencil.Convol_torch(x_bc, wi, cell_ids=cid_np),
        )

        _, L, K_out = y_bc.shape
        y = y_bc.reshape(*leading, L, K_out)  # [..., L, Npix]

        out = data.copy(empty=True)
        out.array = y
        out.dtype = y.dtype
        out.cell_ids = cid.clone()
        out.conv_history = list(data.conv_history) + [j]
        return out

    ###########################################################################
    def apply_smooth(self, data, inplace: bool = True):
        """
        Smooth the data with the low-pass kernel, preserving the shape.

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
        x_bc = x.reshape(B, 1, K)

        w_smooth = self.smooth_kernel.to(device=x.device, dtype=x.dtype)

        l_stencil = self._get_stencil(data.dg, cid, self.KERNELSZ, n_gauges=1)
        cid_np = cid.detach().cpu().numpy().astype(np.int64)

        y_bc = l_stencil.Convol_torch(x_bc, w_smooth, cell_ids=cid_np)  # (B, 1, K)
        y = y_bc.reshape(*leading, K)

        out = data.copy(empty=True) if not inplace else data
        out.array = y
        if not inplace:
            out.cell_ids = cid
        return out

    ###########################################################################
    def _smooth_with_nan(self, data, inplace: bool = True):
        """
        NaN-aware smoothing: the map and the validity mask are both convolved,
        and the result is normalized by the sum of the valid weights.

        Pixels with no valid neighbour in the kernel support come back as NaN.
        """
        self._check_data(data)

        x = data.array  # [..., Npix]
        cid = data.cell_ids
        *leading, K = x.shape

        B = int(np.prod(leading)) if leading else 1
        x_bc = x.reshape(B, 1, K)

        mask_valid = ~torch.isnan(x_bc)
        mask_f = mask_valid.to(x_bc.dtype)
        x_filled = torch.where(mask_valid, x_bc, torch.zeros_like(x_bc))

        w_smooth = self.smooth_kernel.to(device=x.device, dtype=x.dtype)

        stencil = self._get_stencil(data.dg, cid, self.KERNELSZ, n_gauges=1)
        cid_np = cid.detach().cpu().numpy().astype(np.int64)

        num = stencil.Convol_torch(x_filled, w_smooth, cell_ids=cid_np)  # (B,1,K)
        w_sum = stencil.Convol_torch(mask_f, w_smooth, cell_ids=cid_np)  # (B,1,K)

        eps = 1e-8
        y_bc = num / (w_sum + eps)

        # Pixels with no valid neighbour -> NaN
        y_bc = torch.where(w_sum <= 0, torch.full_like(y_bc, float("nan")), y_bc)
        y = y_bc.reshape(*leading, K)

        out = data if inplace else data.copy(empty=False)
        out.array = y
        if not inplace:
            out.cell_ids = cid
        return out

    ###########################################################################
    #                             DOWNSAMPLING                                #
    ###########################################################################
    def _bin_to_parents(self, data, dg_out, nan_aware):
        """
        NESTED binning of the pixel axis onto the parent pixels at dg_out.

        parent_id = cell_id // 4**(dg_out - dg)
        """
        if not data.nest:
            raise ValueError(
                "Downsampling requires NESTED pixel ordering (data.nest must be True)."
            )

        delta_g = dg_out - data.dg
        factor_pix = 4**delta_g  # children per parent in NESTED

        cid = data.cell_ids
        parent_ids = cid // factor_pix

        x = data.array
        *leading, K = x.shape
        B = int(np.prod(leading)) if leading else 1
        x_flat = x.reshape(B, K)

        parent_unique, inv = torch.unique(parent_ids, return_inverse=True)
        Kc = parent_unique.numel()
        idx = inv.unsqueeze(0).expand(B, -1)

        if nan_aware:
            mask_valid = ~torch.isnan(x_flat)
            mask_f = mask_valid.to(x_flat.dtype)
            x_filled = torch.where(mask_valid, x_flat, torch.zeros_like(x_flat))

            out_sum = torch.zeros(B, Kc, device=x_flat.device, dtype=x_flat.dtype)
            out_sum.scatter_add_(1, idx, x_filled)

            out_count = torch.zeros(B, Kc, device=x_flat.device, dtype=x_flat.dtype)
            out_count.scatter_add_(1, idx, mask_f)

            out = out_sum / (out_count + 1e-8)
            out = torch.where(out_count <= 0, torch.full_like(out, float("nan")), out)
        else:
            out = torch.zeros(B, Kc, device=x_flat.device, dtype=x_flat.dtype)
            out.scatter_add_(1, idx, x_flat)
            counts = torch.bincount(inv, minlength=Kc).to(x_flat.dtype)
            out = out / counts.unsqueeze(0)

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

        The map is first low-pass filtered on the sphere, then the pixels are
        averaged over their NESTED parent. Same signature as the planar
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
            Apply the low-pass filter before decimating.
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

        if nan_aware is None:
            nan_aware = (self.mask_full_res is not None) or self.nan_aware_stats

        data = data if inplace else data.copy(empty=False)

        if smooth:
            if nan_aware:
                data = self._smooth_with_nan(data, inplace=True)
            else:
                data = self.apply_smooth(data, inplace=True)

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
    #                              STATISTICS                                 #
    ###########################################################################
    def _find_mask(self, data):
        """
        Return the boolean mask (True where the data is invalid) associated with
        the layer and the resolution of `data`.

        Always None for now -- see the class docstring: the spherical
        counterpart of the planar layer masks is not implemented yet.
        """
        return None

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

        # finding the appropriate mask: the deepest layer carries the largest one
        if self.mask_full_res is None:
            mask = None
        elif len(data1.conv_history) >= len(data2.conv_history):
            mask = self._find_mask(data1)
        else:
            mask = self._find_mask(data2)

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
# Backward-compatible alias (the class used to be spelled "Wavelate").
WavelateOperatorHealpixKernel_torch = WaveletOperatorHealpixKernel_torch
