#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Nov 26 2025

Example methods for a test data type.

2D planar maps with convolution using kernel.

This class makes all computations in torch.

Characteristics:
    - in pytorch
    - assume real maps
    - N0 gives x and y sizes for array shaped (..., Nx, Ny).
    - masks are supported in convolutions
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from STL_main.torch_backend import to_torch_tensor


###############################################################################
###############################################################################
class STL_2D_Kernel_Torch:
    """
    Class which contain the different types of data used in STL.
    Store important parameters, such as DT, N0, and the Fourier type.
    Also allow to convert from numpy to pytorch (or other type).
    Allow to transfer internally these parameters.

    Has different standard functions as methods (
    modulus, mean, cov, downsample)

    The initial resolution N0 is fixed, but the maps can be downgraded. The
    downgrading factor is the power of 2 that is used. A map of initial
    resolution N0=256 and with dg = 3 is thus at resolution 256/2^3 = 32.
    The downgraded resolutions are called N0, N1, N2, ...

    Can store array at a given downgradind dg:
        - attribute MR is False
        - attribute N0 gives the initial resolution
        - attribute dg gives the downgrading level
        - attribute list_dg is None
        - array is an array of size (..., N) with N = N0 // 2^dg
    Or at multi-resolution (MR):
        - attribute MR is True
        - attribute N0 gives the initial resolution
        - attribute dg is None
        - attribute list_dg is the list of downgrading
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

    ###########################################################################
    def __init__(self, array, dg=None, N0=None):
        """
        Constructor, see details above. Frontend version, which assume the
        array is at N0 resolution with dg=0.
        """

        # Check that a signle array is given (not a list of multiple resolutions)
        if isinstance(array, list):
            raise ValueError("Only single resolution array are accepted.")

        # Main
        self.DT = "Planar2D_kernel_torch"
        if dg is None:
            self.dg = 0
            self.N0 = array.shape[-2:]
        else:
            self.dg = dg
            if N0 is None:
                raise ValueError("dg is given, N0 should not be None")
            self.N0 = N0

        self.array = self.to_array(array)

        self.list_dg = None

        # Find N0 value
        self.device = self.array.device
        self.dtype = self.array.dtype

    ###########################################################################
    def to_array(self, array):
        """
        Transform input array (NumPy or PyTorch) into a PyTorch tensor.
        Should return None if None.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input array to be converted.

        Returns
        -------
        torch.Tensor
            Converted PyTorch tensor.
        """

        if array is None:
            return None
        elif isinstance(array, list):
            return array
        else:
            # Choose device: use GPU if available, otherwise CPU
            return to_torch_tensor(array)

    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a STL_2D_Kernel_Torch instance.
        Array is put to None if empty==True.

        Parameters
        ----------
        - empty : bool
            If True, set array to None.

        Output
        ----------
        - STL_2D_Kernel_Torch
           copy of self
        """
        new = object.__new__(STL_2D_Kernel_Torch)

        # Copy metadata
        new.N0 = self.N0
        new.dg = self.dg
        new.device = self.device
        new.dtype = self.dtype

        # Copy array
        if empty:
            new.array = None
        else:
            new.array = (
                self.array.clone() if isinstance(self.array, torch.Tensor) else None
            )

        return new

    ###########################################################################
    def __getitem__(self, key):
        """
        To slice directly the array attribute. Produce a view of array, to
        match with usual practices, allowing to conveniently pass only part
        of an instance.
        """
        new = self.copy(empty=True)
        new.array = self.array[key]

        return new

    ###########################################################################
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the data.
        """
        data = self.copy(empty=False) if not inplace else self

        if data.MR:
            data.array = [torch.abs(a) for a in data.array]
        else:
            data.array = torch.abs(data.array)

        data.dtype = data.array.dtype

        return data

    ###########################################################################
    def mean(self, square=False):
        """
        Compute the mean on the last two dimensions (Nx, Ny).
        """
        arr_use = torch.abs(self.array) ** 2 if square else self.array
        mean = arr_use.nanmean(dim=(-2, -1))

        return mean

    ###########################################################################
    def cov(self, data2=None, remove_mean=False):
        """
        Compute the covariance between data1=self and data2 on the last two
        dimensions (Nx, Ny).
        """
        x = self.array
        if data2 is None:
            y = x
        else:
            if not isinstance(data2, STL_2D_Kernel_Torch):
                raise TypeError("data2 must be a Planar2D_kernel_torch instance.")
            if data2.dg != self.dg:
                raise ValueError("data2 must have the same dg as self.")
            y = data2.array

        if remove_mean:
            x_c = x - x.mean(dim=(-2, -1), keepdim=True)
            y_c = y - y.mean(dim=(-2, -1), keepdim=True)
        else:
            x_c = x
            y_c = y
        cov = (x_c * y_c.conj()).nanmean(dim=dims)

        return cov

    def get_wavelet_op(self, J=None, L=None, kernel_size=None):
        if L is None:
            L = 4
        if kernel_size is None:
            kernel_size = 5
        if J is None:
            J = np.min([int(np.log2(self.N0[0])), int(np.log2(self.N0[1]))]) - 2

        return WavelateOperator2Dkernel_torch(
            kernel_size, L, J, device=self.array.device, dtype=self.array.dtype
        )


class WavelateOperator2Dkernel_torch:
    @staticmethod
    def _conv2d_circular(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Backend-style 2D convolution mirroring FoCUS/BkTorch strategy.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., Nx, Ny].
        w : torch.Tensor
            Kernel tensor of shape [O_c, wx, wy].

        Returns
        -------
        torch.Tensor
            Convolved tensor with shape [..., O_c, Nx, Ny].
        """

        *leading_dims, Nx, Ny = x.shape
        O_c, wx, wy = w.shape

        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        x4d = x.reshape(B, 1, Nx, Ny)

        weight = w[:, None, :, :]
        pad_x = wx // 2
        pad_y = wy // 2

        x_padded = F.pad(x4d, (pad_y, pad_y, pad_x, pad_x), mode="circular")
        y = F.conv2d(x_padded, weight)

        return y.reshape(*leading_dims, O_c, Nx, Ny)

    @classmethod
    def _complex_conv2d_circular(cls, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Complex-aware wrapper around ``_conv2d_circular``."""

        xr = torch.real(x) if torch.is_complex(x) else x
        xi = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(xr)

        wr = torch.real(w) if torch.is_complex(w) else w
        wi = torch.imag(w) if torch.is_complex(w) else torch.zeros_like(wr)

        real_part = cls._conv2d_circular(xr, wr) - cls._conv2d_circular(xi, wi)
        imag_part = cls._conv2d_circular(xr, wi) + cls._conv2d_circular(xi, wr)

        if torch.is_complex(x) or torch.is_complex(w):
            return torch.complex(real_part, imag_part)
        return real_part

    def __init__(
        self,
        kernel_size: int,
        L: int,
        J: int,
        device="cuda",
        mask_full_res=None,
        dtype=torch.float,
    ):
        self.KERNELSZ = kernel_size
        self.L = L
        self.J = J
        self.device = torch.device(device)
        self.dtype = dtype

        self._wav_kernel = self._build_wavelet_kernel()
        self.WType = "simple"

        self.mask_full_res = mask_full_res
        self._reweighting_maps = self._build_reweighting_maps()

    def _build_reweighting_maps(self):
        # mask_full_res None or not ?
        # recup le smooth kernel
        # convol le smooth kernel avec le nan mask
        raise

    def _build_wavelet_kernel(self, sigma=1):
        """Create a 2D Wavelet kernel."""

        # Morlay wavelet
        coords = (
            torch.arange(self.KERNELSZ, device=self.device, dtype=self.dtype)
            - (self.KERNELSZ - 1) / 2.0
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        # Gaussian envelope
        gaussian_envelope = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Orientations
        angles = (
            torch.arange(self.L, device=self.device, dtype=self.dtype)
            / self.L
            * torch.pi
        )

        # Morlet wavelet: exp(i*k0*x_rot) * gaussian_envelope
        # x_rot is the coordinate along the orientation direction
        x_rot = xx[None, :, :] * torch.cos(angles[:, None, None]) + yy[
            None, :, :
        ] * torch.sin(angles[:, None, None])

        # Complex Morlet wavelet
        kernel = torch.exp(1j * 0.75 * np.pi * x_rot) * gaussian_envelope[None, :, :]

        # Remove DC component (admissibility condition)
        kernel = kernel - torch.mean(kernel, dim=(1, 2))[:, None, None]

        # L2 normalization
        kernel = (
            kernel
            / torch.sqrt(torch.sum(torch.abs(kernel) ** 2, dim=(1, 2)))[:, None, None]
        )

        return kernel.reshape(1, self.L, self.KERNELSZ, self.KERNELSZ)

    def apply(self, data, j):
        """
        Apply the convolution kernel to data.array [..., Nx, Ny]
        and return cdata [..., L, Nx, Ny].

        Parameters
        ----------
        data : object
            Object with an attribute `array` storing the data as a tensor
            or numpy array with shape [..., Nx, Ny].

        Returns
        -------
        torch.Tensor
            Convolved data with shape [..., L, Nx, Ny].
        """
        if j != data.dg:
            raise "j is not equal to dg, convolution not possible"

        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device / dtype as the _wav_kernel
        x = torch.as_tensor(
            x, device=self._wav_kernel.device, dtype=self._wav_kernel.dtype
        )

        weight = self._wav_kernel.squeeze(0)  # [L, K, K]

        convolved = self.__class__._complex_conv2d_circular(x, weight)

        return STL_2D_Kernel_Torch(convolved, dg=data.dg, N0=data.N0)

    def apply_smooth(self, data: STL_2D_Kernel_Torch, inplace: bool = False):
        """
        Smooth the data by convolving with a smooth kernel derived from the
        wavelet orientation 0. The data shape is preserved.
        """
        x = data.array  # [..., Nx, Ny]
        *leading, Nx, Ny = x.shape

        # Build a real, positive smoothing kernel from orientation 0
        # self.kernel: (1, L, K, K) complex
        k0 = torch.abs(self._wav_kernel[0, 0])  # (K, K)
        k0 = k0 / k0.sum()
        w_smooth = k0.unsqueeze(0).to(device=data.device, dtype=data.dtype)  # (1, K, K)

        # Convolution is circular through _conv2d_circular
        y = _conv2d_circular(x, w_smooth)  # [..., 1, Nx, Ny]

        # Remove the extra output-channel dim
        y = y.squeeze(-3)  # from [..., 1, Nx, Ny] -> [..., Nx, Ny]

        out = data.copy(empty=True) if not inplace else data
        out.array = y
        return out

    @staticmethod
    def _downsample_tensor(x: torch.Tensor, dg_inc: int) -> torch.Tensor:
        """
        Downsample a tensor by a factor 2**dg_inc along the last two
        dimensions using 2x2 mean pooling (FoCUS ud_grade strategy).

        Requires that both spatial dimensions be divisible by 2**dg_inc.
        """
        if dg_inc < 0:
            raise ValueError("dg_inc must be non-negative")
        if dg_inc == 0:
            return x

        scale = 2**dg_inc
        H, W = x.shape[-2:]
        if H % scale != 0 or W % scale != 0:
            raise ValueError(
                f"Cannot downsample from ({H},{W}) by 2^{dg_inc}: "
                "dimensions must be divisible."
            )

        # Create Gaussian kernel for anti-aliasing
        def get_gaussian_kernel(sigma=1.0, dtype=torch.float32):
            """Create a 2D Gaussian kernel with kernel_size based on sigma"""
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            x_coord = torch.arange(kernel_size, dtype=dtype) - kernel_size // 2
            gauss_1d = torch.exp(-(x_coord**2) / (2 * sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
            return gauss_2d.unsqueeze(0).unsqueeze(0), kernel_size

        leading_dims = x.shape[:-2]
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        y = x.reshape(B, 1, H, W)

        for _ in range(dg_inc):
            h, w = y.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(
                    "Downsampling requires even spatial dimensions at each step."
                )
            sigma = 1.0
            kernel, kernel_size = get_gaussian_kernel(sigma, dtype=y.dtype)
            kernel = kernel.to(y.device)
            # Add circular padding for periodic boundaries
            pad = kernel_size // 2
            y_padded = F.pad(y, (pad, pad, pad, pad), mode="circular")
            y = F.conv2d(y_padded, kernel)
            # Now downsample with 2x2 mean pooling
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

        H2, W2 = y.shape[-2:]
        return y.reshape(*leading_dims, H2, W2)

    ###########################################################################
    def downsample(self, data, dg_out, inplace=True):
        """
        Downsample the data to the dg_out resolution.

        Downsampling is done in real space by



        average pooling,



        with factor
        2^(dg_out - dg) on both spatial axes.
        """
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out == data.dg and inplace:
            return data
        if dg_out < data.dg:
            raise ValueError(
                "Requested dg_out < current dg; upsampling not supported by downsampling method."
            )

        data = data.copy(empty=False) if not inplace else data
        dg_inc = dg_out - data.dg

        if True:  # nan or not:
            raise
            if dg_inc > 0:
                data.array = self._downsample_tensor(data.array, dg_inc)
                data.dg = dg_out

        return data

    def _gaussian_kernel_5x5(self, device, dtype, sigma: float = 1.0):
        """
        Build and cache a normalized 5x5 Gaussian kernel on (device, dtype)
        for circular convolution with _conv2d_circular.

        Returns
        -------
        kernel : torch.Tensor
            Shape (1, 5, 5): [O_c, wx, wy]
        """
        if (
            not hasattr(self, "_smooth_kernel_5x5")
            or self._smooth_kernel_5x5.device != device
            or self._smooth_kernel_5x5.dtype != dtype
        ):
            size = 5
            coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            # _conv2d_circular expects w shape (O_c, wx, wy)
            self._smooth_kernel_5x5 = kernel.view(1, size, size)
        return self._smooth_kernel_5x5

    def _nandownsample_tensor(self, x: torch.Tensor, dg_inc: int) -> torch.Tensor:
        """
        Smooth x with a 5x5 Gaussian ignoring NaNs, then downsample by a factor
        2**dg_inc using a nan-aware mean (nanmean) on each block.

        x : tensor of shape (..., H, W)
        returns : tensor of shape (..., H/2**dg_inc, W/2**dg_inc)
        """
        if dg_inc <= 0:
            return x

        device = x.device
        dtype = x.dtype
        factor = 2**dg_inc

        # ---- reshape to (B, H, W) for _conv2d_circular ----
        *leading, H, W = x.shape
        if leading:
            B = math.prod(leading)
        else:
            B = 1
        x_flat = x.reshape(B, H, W)  # (B, H, W)

        # ---- Gaussian smoothing with NaN handling (circular) ----
        mask_valid = ~torch.isnan(x_flat)  # (B, H, W) bool
        mask_f = mask_valid.to(dtype)  # float

        # replace NaN with 0 so they don't contribute
        x_filled = torch.where(mask_valid, x_flat, torch.zeros_like(x_flat))

        # kernel 5x5, shape (1, 5, 5) for _conv2d_circular
        sigma = getattr(self, "sigma_smooth", 1.0)
        kernel = self._gaussian_kernel_5x5(device=device, dtype=dtype, sigma=sigma)

        # Convolution on data and on mask (circular)
        num = _conv2d_circular(x_filled, kernel)  # (B, 1, H, W)
        w_sum = _conv2d_circular(mask_f, kernel)  # (B, 1, H, W)

        eps = 1e-8
        x_smooth = num / (w_sum + eps)  # (B, 1, H, W)
        x_smooth = x_smooth.squeeze(1)  # (B, H, W)

        # Put NaN where no valid pixel at all in the 5x5 window
        no_valid = w_sum.squeeze(1) <= 0
        x_smooth = torch.where(
            no_valid, torch.full_like(x_smooth, float("nan")), x_smooth
        )

        # ---- Downsample with nan-aware mean via avg_pool2d ----
        # Treat (B,1,H,W) again
        x_smooth_4d = x_smooth.unsqueeze(1)  # (B,1,H,W)
        mask_valid2 = ~torch.isnan(x_smooth_4d)
        mask2_f = mask_valid2.to(dtype)

        # replace NaN by 0 for pooling
        x2_filled = torch.where(mask_valid2, x_smooth_4d, torch.zeros_like(x_smooth_4d))

        # average pooling of data and mask (non-overlapping blocks)
        pool_num = F.avg_pool2d(
            x2_filled, kernel_size=factor, stride=factor
        )  # (B,1,H',W')
        pool_mask = F.avg_pool2d(
            mask2_f, kernel_size=factor, stride=factor
        )  # (B,1,H',W')

        y = pool_num / (pool_mask + eps)  # nan-aware mean
        no_valid_block = pool_mask <= 0
        y = torch.where(no_valid_block, torch.full_like(y, float("nan")), y)

        # reshape back to original leading dims + downsampled spatial dims
        H_out, W_out = y.shape[-2:]
        y = y.view(*leading, H_out, W_out)
        return y

    ###########################################################################
    def nandownsample(self, data, dg_out, mask_MR=None, inplace=True):
        """
        Nan-aware downsample of `data` to resolution `dg_out`:

          1) Circular Gaussian smoothing with a 5x5 kernel ignoring NaNs
             (normalized by sum of valid weights),
          2) nan-aware average pooling by factor 2^(dg_out - dg).
        """
        if data.MR:
            raise ValueError("downsample only supports MR == False.")
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out == data.dg:
            return data
        if dg_out < data.dg:
            raise ValueError("Requested dg_out < current dg; upsampling not supported.")

        data = data.copy(empty=False) if not inplace else data
        dg_inc = dg_out - data.dg

        if dg_inc > 0:
            data.array = self._nandownsample_tensor(data.array, dg_inc)
            data.dg = dg_out

        if mask_MR is not None:
            mask = self._get_mask_at_dg(mask_MR, data.dg)
            if mask.shape[-2:] != data.array.shape[-2:]:
                raise ValueError("Mask and data have incompatible spatial shapes.")
            data.array = data.array * mask

        return data
