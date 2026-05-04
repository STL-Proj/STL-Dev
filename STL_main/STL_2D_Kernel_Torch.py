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
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from STL_main.Base_DataClass import Base_DataClass
from STL_main.ST_Operator import ST_Operator
from STL_main.torch_backend import (
    _DEFAULT_DEVICE,
    _DEFAULT_DTYPE,
    _get_device,
    _get_dtype,
    maskmean,
    nan,
    to_torch_tensor,
)


###############################################################################
###############################################################################
@dataclass
class STL_2D_Kernel_Torch(Base_DataClass):
    """
    STL_2D_FFT_torch child class for 2D planar STL Kernel using PyTorch

    Inherits Base_DataClass.

    See Base_DataClass for parameter descriptions.

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
    DT = "Planar2D_kernel_torch"

    def __post_init__(self):
        super().__post_init__()

    ###########################################################################
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the array attribute of data.

        Parameters
        ----------
        - inplace : bool
            If True, acts in-place and returns self.
            If False, returns a new STL_2D_Kernel_Torch instance.

        Returns
        -------
        STL_2D_Kernel_Torch
            STL_2D_Kernel_Torch instance whose array attribute is the modulus
        """
        data = self.copy(empty=False) if not inplace else self

        data.array = data.array.abs()

        data.dtype = data.array.dtype

        return data

    def get_wavelet_op(
        self,
        J=None,
        mask_full_res=None,
        *args,
        **kwargs,
    ):

        J = J if J is not None else int(np.log2(min(self.N0))) - 2

        if mask_full_res is None:
            if torch.any(self.array.isnan()):
                mask_full_res = STL_2D_Kernel_Torch(array=self.array.isnan())

        return WaveletOperator2Dkernel_torch(
            J=J,
            DT=self.DT,
            device=self.device,
            dtype=self.dtype,
            mask_full_res=mask_full_res,
            *args,
            **kwargs,
        )

    def get_ST_op(self, *args, **kwargs):

        return ST_Operator(data_example=self, *args, **kwargs)

    ###############################################################################
    def get_PS_op(self, *args, **kwargs):

        return PS_operator_2D_Kernel_torch(
            shape=self.N0, device=self.device, dtype=self.dtype, *args, **kwargs
        )


class WaveletOperator2Dkernel_torch:
    @staticmethod
    def _get_padding_mode(pbc: bool = True) -> str:
        assert pbc is not None, "pbc must be specified"
        return (
            "circular" if pbc else "replicate"
        )  # most suited option for non-PBC, better than 'constant' and 'reflect'

    @staticmethod
    def _conv2d_circular(
        x: torch.Tensor, w: torch.Tensor, padding_mode: str
    ) -> torch.Tensor:
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

        x_padded = F.pad(x4d, (pad_y, pad_y, pad_x, pad_x), mode=padding_mode)
        y = F.conv2d(x_padded, weight)

        return y.reshape(*leading_dims, O_c, Nx, Ny)

    @classmethod
    def _semicomplex_conv2d_circular(
        cls, x: torch.Tensor, w: torch.Tensor, padding_mode: str
    ) -> torch.Tensor:
        """
        Perform a 2D convolution with a real input and complex kernel.
        This method decomposes the complex kernel ``w`` into its real and
        imaginary parts, applies ``_conv2d_circular`` separately to each part
        using the real-valued input ``x``, and combines the two real-valued
        results into a complex-valued output tensor.
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor of shape ``[..., Nx, Ny]``. The tensor
            must not be complex (``torch.is_complex(x)`` is expected to be
            ``False``).
        w : torch.Tensor
            Complex-valued convolution kernel of shape ``[O_c, wx, wy]``. The
            tensor must be complex (``torch.is_complex(w)`` is expected to be
            ``True``), and its real and imaginary parts are convolved with
            ``x`` separately.
        padding_mode : str
            Padding mode passed through to ``torch.nn.functional.pad`` in
            ``_conv2d_circular``. Typically ``"circular"`` for periodic
            boundary conditions or ``"replicate"`` for non-periodic padding,
            but any mode supported by ``torch.nn.functional.pad`` may be used.
        Returns
        -------
        torch.Tensor
            Complex-valued output tensor of shape ``[..., O_c, Nx, Ny]``,
            where ``O_c`` is the number of output channels defined by the
            kernel ``w``.
        """

        assert not torch.is_complex(x), "Input tensor x must be real-valued"
        assert torch.is_complex(w), "Kernel w must be complex-valued"

        wr = torch.real(w)  # if torch.is_complex(w) else w
        wi = torch.imag(w)  # if torch.is_complex(w) else torch.zeros_like(wr)

        real_part = cls._conv2d_circular(
            x, wr, padding_mode=padding_mode
        )  # - cls._conv2d_circular(xi, wi)
        imag_part = cls._conv2d_circular(
            x, wi, padding_mode=padding_mode
        )  # + cls._conv2d_circular(xi, wr)

        return torch.complex(real_part, imag_part)

    @staticmethod
    def _get_crop_border_size_largest_scale_second_layer(data, wavelet_op):
        if data.pbc:
            return 0
        else:
            deepest_layer = 2
            return (
                deepest_layer
                * 2 ** (wavelet_op.J - 1 - data.dg)
                * (wavelet_op.KERNELSZ // 2)
            )

    @staticmethod
    def _get_crop_border_size_largest_scale_layer_flexible(data, wavelet_op):
        if data.pbc or len(data.conv_history) == 0:
            return 0
        else:
            return (
                len(data.conv_history)
                * 2 ** (wavelet_op.J - 1 - data.dg)
                * (wavelet_op.KERNELSZ // 2)
            )

    @staticmethod
    def _get_crop_border_size_fully_flexible(data, wavelet_op):
        if data.pbc or len(data.conv_history) == 0:
            return 0
        elif len(data.conv_history) == 1:
            return math.ceil(
                2 ** (data.conv_history[0] - data.dg) * (wavelet_op.KERNELSZ // 2)
            )
        elif len(data.conv_history) == 2:
            first_conv_border_downgraded = math.ceil(
                2 ** (data.conv_history[0] - data.conv_history[-1])
                * (wavelet_op.KERNELSZ // 2)
            )
            return math.ceil(
                2 ** (data.conv_history[-1] - data.dg)
                * (first_conv_border_downgraded + wavelet_op.KERNELSZ // 2)
            )
        else:
            raise ValueError("Invalid data conv_history.")

    @staticmethod
    def _get_crop_border_size_zero(data, wavelet_op):
        return 0

    def __init__(
        self,
        J,
        L=None,
        kernel_size=None,
        DT="Planar2D_kernel_torch",
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        mask_full_res=None,
        sigma_smooth=1.0,
        downsample_nan_weight_threshold=0.33,
        get_crop_border_size_method=None,
    ):
        if J is None:
            raise ValueError(
                "J must be specified for WaveletOperator2Dkernel_torch class."
            )
        self.J = J
        self.L = L if L is not None else 4
        self.KERNELSZ = kernel_size if kernel_size is not None else 5
        self.DT = DT

        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        self.sigma_smooth = (
            sigma_smooth  # to build smoothing kernel used in downsampling
        )
        #simag_smooth should be defined before build wavelet kernel for dg=0
        self._wav_kernel,self._wav_kernel_0 = self._build_wavelet_kernel()
        # raise
        # build low pass kernel?
        self.WType = "simple"

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
        self.downsample_nan_weight_threshold = downsample_nan_weight_threshold
        (
            self._reweighting_maps_smooth,
            self._reweighting_maps_wav,
            self._layer1_mask,
            self._layer2_mask,
        ) = self._build_reweighting_maps_and_scattering_layer_masks()
        self.j_to_dg = range(J)

    def _build_reweighting_maps_and_scattering_layer_masks(self):
        if self.mask_full_res is None:
            return None, None, None, None
        else:
            (
                reweighting_maps_smooth_dict,
                reweighting_maps_wav_dict,
                layer1_mask_dict,
                layer2_mask_dict,
            ) = ({}, {}, {}, {})

            for pbc in [False, True]:
                padding_mode = self.__class__._get_padding_mode(pbc=pbc)

                # 1) reweighting maps needed in downsampling of layer 0 data (no wavelet convolution, only smoothing kernel convolution)
                local_nan_weight_maps_smooth = {}
                smooth_kernel = self._gaussian_kernel_5x5(
                    device=self.mask_full_res.array.device, dtype=self.dtype
                )
                assert torch.isclose(
                    smooth_kernel.sum(), torch.tensor(1.0, dtype=smooth_kernel.dtype)
                )

                # no need for reweighting at resolution dg=0.
                for dg in range(1, self.J):
                    parent_array = (
                        local_nan_weight_maps_smooth[dg - 1]
                        .array.isnan()
                        .to(dtype=self.dtype)
                        if dg > 1
                        else self.mask_full_res.array.to(dtype=self.dtype)
                    )
                    local_nan_weight_maps_smooth[dg] = STL_2D_Kernel_Torch(
                        array=self._downsample_tensor(
                            x=parent_array,
                            smooth_kernel=smooth_kernel,
                            dg_inc=1,
                            padding_mode=padding_mode,
                        ),
                        dg=dg,
                        N0=self.mask_full_res.N0,
                    )  # local nan fraction

                    local_nan_weight_maps_smooth[dg].array = torch.where(
                        condition=local_nan_weight_maps_smooth[dg].array
                        <= self.downsample_nan_weight_threshold,
                        input=local_nan_weight_maps_smooth[dg].array,
                        other=nan,
                    )  # replace with nan where above threshold

                # 2) reweighting maps needed in downsampling of layer 1 data (convolved once with wavelets)
                wav_kernels_envelope = torch.ones(
                    self._wav_kernel.shape[-2:], dtype=self.dtype, device=self.device
                ).unsqueeze(
                    0
                )  # (1,K,K) assumes identical wavelet support for all angles
                local_nan_weight_maps_wav = {}

                # Stores at every dg=j3 NaNs position of layer 1 data (convolved once with wavelets at j3) in a mask.
                layer1_mask = {  # {J: (N3)} one mask per scale j at resolution dg=j, same for all angles
                    dg: STL_2D_Kernel_Torch(
                        array=torch.abs(
                            self.__class__._conv2d_circular(
                                x=(
                                    self.mask_full_res.array
                                    if dg == 0
                                    else local_nan_weight_maps_smooth[dg].array.isnan()
                                ).to(dtype=self.dtype),
                                w=wav_kernels_envelope,  # assumes identical wavelet support for all angles
                                padding_mode=padding_mode,
                            ).squeeze(0)
                        )
                        > 0.0,
                        dg=dg,
                        N0=self.mask_full_res.N0,
                        conv_history=[dg],
                    )
                    for dg in range(self.J)
                }

                # no need for reweighting at resolution dg=0
                local_nan_weight_maps_wav = {
                    dg: {} for dg in range(1, self.J)
                }  # j in range(dg-1)
                for j in range(self.J - 1):  # level at which the map was convolved
                    for dg in range(j + 1, self.J):  # target level of downsampling
                        if (
                            dg == j + 1
                        ):  # needs to convolve with wavelets' support before downsampling
                            parent_array = layer1_mask[j].array.to(dtype=self.dtype)
                        else:  # dg > j+1, needs only to downsample with a smoothing from previous level
                            parent_array = (
                                local_nan_weight_maps_wav[dg - 1][j]
                                .array.isnan()
                                .to(dtype=self.dtype)
                            )

                        local_nan_weight_maps_wav[dg][j] = STL_2D_Kernel_Torch(
                            array=self._downsample_tensor(
                                x=parent_array,
                                smooth_kernel=smooth_kernel,
                                dg_inc=1,
                                padding_mode=padding_mode,
                            ),
                            dg=dg,
                            N0=self.mask_full_res.N0,
                            conv_history=[j],
                        )  # (Ndg,Ndg) local nan fraction

                        local_nan_weight_maps_wav[dg][j].array = torch.where(
                            condition=local_nan_weight_maps_wav[dg][j].array
                            <= self.downsample_nan_weight_threshold,
                            input=local_nan_weight_maps_wav[dg][j].array,
                            other=nan,
                        )  # (Ndg,Ndg) replace with nan where above threshold

                # 3) Stores at every dg=j3 and every j2 NaNs position of layer 2 data (convolved first with wavelets at j2, then possibly local operations such as modulus, and then convolved a second time with wavelets at j3) in a mask.
                layer2_mask = {
                    j3: {
                        j2: STL_2D_Kernel_Torch(
                            array=(
                                self.__class__._conv2d_circular(  # convolve with wavelet support at resolution j3
                                    x=(
                                        local_nan_weight_maps_wav[j3][j2].array.isnan()
                                        if j2 < j3
                                        else layer1_mask[j3].array
                                    ).to(dtype=self.dtype),
                                    w=wav_kernels_envelope,
                                    padding_mode=padding_mode,
                                )
                                .squeeze(0)
                                .squeeze(0)
                                > 0.0  # back to bool
                            ),
                        )
                        for j2 in range(j3 + 1)
                    }
                    for j3 in range(self.J)
                }

                # 4) final reweighting maps
                reweighting_maps_smooth = local_nan_weight_maps_smooth
                reweighting_maps_wav = local_nan_weight_maps_wav
                for dg in range(1, self.J):
                    reweighting_maps_smooth[dg].array = 1.0 / (
                        1.0 - reweighting_maps_smooth[dg].array
                    )
                    for j in range(dg):
                        reweighting_maps_wav[dg][j].array = 1.0 / (
                            1.0 - reweighting_maps_wav[dg][j].array
                        )

                reweighting_maps_smooth_dict[padding_mode] = reweighting_maps_smooth
                reweighting_maps_wav_dict[padding_mode] = reweighting_maps_wav
                layer1_mask_dict[padding_mode] = layer1_mask
                layer2_mask_dict[padding_mode] = layer2_mask

            return (
                reweighting_maps_smooth_dict,
                reweighting_maps_wav_dict,
                layer1_mask_dict,
                layer2_mask_dict,
            )

    def _find_mask(self, data):
        if self.mask_full_res is None:
            return None
        else:
            layer = len(data.conv_history)
            if layer == 0:
                # For mean computation at layer 0 and full resolution, use full res mask
                # TODO: implement for downgraded resolution at layer 0 if needed
                assert data.dg == 0
                return self.mask_full_res.array

                # raise NotImplementedError(
                #     "So far, data mask should not be called for data at layer 0."
                # )
            assert data.dg == data.conv_history[-1]
            padding_mode = self.__class__._get_padding_mode(pbc=data.pbc)
            if layer == 1:
                return self._layer1_mask[padding_mode][data.conv_history[-1]].array
            elif layer == 2:
                return self._layer2_mask[padding_mode][data.conv_history[-1]][
                    data.conv_history[0]
                ].array
            else:
                raise ValueError("len(data.conv_history) must be 0, 1 or 2.")

    def _build_wavelet_kernel(self, sigma=1):
        """Create a 2D Wavelet kernel."""

        # Morlay wavelet
        coords = (
            torch.arange(self.KERNELSZ, device=self.device, dtype=self.dtype)
            - (self.KERNELSZ - 1) / 2.0
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        # Gaussian envelope
        gaussian_envelope = torch.exp(-2*(xx**2 + yy**2) / (self.L*sigma**2))
        gaussian_envelope_0 = torch.exp(-8*(xx**2 + yy**2) / (self.L*sigma**2))

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
        
        # y: (L, 3K, 3K)
        y = torch.zeros([self.L, self.KERNELSZ*3, self.KERNELSZ*3],
                        device=self.device, dtype=kernel.dtype)
        y[:, self.KERNELSZ:self.KERNELSZ*2, self.KERNELSZ:self.KERNELSZ*2] = kernel

        # conv2d expects 4D input: (N, C, H, W)
        y4 = y.unsqueeze(1)  # (L, 1, 3K, 3K)

        # weight: (C_out=1, C_in=1, 5, 5)
        w = gaussian_envelope_0.unsqueeze(0).unsqueeze(0)  # (1,1,5,5)
        w = w.to(dtype=kernel.dtype)  # optional: cast to complex if you want

        # No padding needed because you already embedded into a larger array
        y4 = F.conv2d(input=y4, weight=w, stride=1, padding=0)  # (L,1,3K-4,3K-4)

        # Back to (L, H, W)
        y = y4.squeeze(1)  # (L, 3K-4, 3K-4)

        # IMPORTANT: indices shift because output is smaller by 4 pixels
        # You want the central KxK block corresponding to original center
        kernel_0 = y[:, (self.KERNELSZ-2):(self.KERNELSZ-2)+self.KERNELSZ,
                       (self.KERNELSZ-2):(self.KERNELSZ-2)+self.KERNELSZ]
            
        # Remove DC component (admissibility condition)
        kernel = kernel - torch.mean(kernel, dim=(1, 2))[:, None, None]
        kernel_0 = kernel_0 - torch.mean(kernel_0, dim=(1, 2))[:, None, None]

        # L2 normalization
        # tune the normalisation 
        kernel_0 /= 2*self.L
        kernel   /= self.L
        '''
        if self.L==4:
            kernel[1]*=1.5
            kernel[3]*=1.5
            kernel_0[1]*=2
            kernel_0[3]*=2
        
        kernel_0 = (
            kernel_0
            / torch.sqrt(torch.sum(torch.abs(kernel) ** 2, dim=(1, 2)))[:, None, None]
        )
        kernel = (
            kernel
            / torch.sqrt(torch.sum(torch.abs(kernel) ** 2, dim=(1, 2)))[:, None, None]
        )
        '''
        return kernel.reshape(1, self.L, self.KERNELSZ, self.KERNELSZ), \
            kernel_0.reshape(1, self.L, self.KERNELSZ, self.KERNELSZ)

    def _crop(self, array, border):
        """
        Crops an array by removing 'border' pixels from each side
        along the last two dimensions.

        Parameters
        ----------
        array : torch.Tensor
            Input array to be cropped.
        border : int
            Number of pixels to remove from each side.
        Returns
        -------
        torch.Tensor
            Cropped array.
        """
        if array is None:
            return None
        elif border == 0:
            return array
        else:
            # handling of borders larger than array can be adapted depending on desired behavior
            if False:  # conservative handling of borders larger than array
                assert array.shape[-2] > 2 * border
                assert array.shape[-1] > 2 * border
            elif True:  # flexible handling of borders larger than array
                if min(array.shape[-2:]) <= 2 * border:
                    if not getattr(
                        self, "_border_warning_raised", False
                    ):  # warns the user only once per wavelet operator
                        print(
                            "Warning! Data with shape {:} too small to be cropped with border {:}. Using border={:} instead.".format(
                                array.detach().cpu().numpy().shape[-2:],
                                border,
                                (min(array.shape[-2:]) - 1) // 2,
                            )
                        )
                        self._border_warning_raised = True
                    border = (min(array.shape[-2:]) - 1) // 2
            else:  # simple handling of borders larger than array: maskmean will return nan
                pass
            return array[..., border:-border, border:-border]

    def mean(self, data, square=False, dim=None):
        """
        Compute the mean on the last two dimensions (Nx, Ny).
        """
        if data.pbc is None and len(data.conv_history) > 0:
            raise ValueError("data.pbc should be specified (True or False).")

        border = self._get_crop_border_size_method(data=data, wavelet_op=self)
        cropped_array = self._crop(array=data.array, border=border)
        cropped_mask = self._crop(array=self._find_mask(data), border=border)

        dim = dim if dim is not None else (-2, -1)

        return maskmean(
            x=cropped_array,
            dim=dim,
            mask=cropped_mask,
        )

    def square_mean(self, data, dim=(-2, -1), **kwargs):

        if data.pbc is None and len(data.conv_history) > 0:
            raise ValueError("data.pbc should be specified (True or False).")

        border = self._get_crop_border_size_method(data=data, wavelet_op=self)
        cropped_array = self._crop(array=data.array * data.array.conj(), border=border)
        cropped_mask = self._crop(array=self._find_mask(data), border=border)

        return maskmean(x=cropped_array, dim=dim, mask=cropped_mask)

    def cov(self, data1, data2, remove_mean=None, dim=None):
        """
        Compute the covariance between data1=self and data2 on the last two
        dimensions (Nx, Ny).
        """

        if (data1.pbc is None and len(data1.conv_history) > 0) or (
            data2.pbc is None and len(data2.conv_history) > 0
        ):
            raise ValueError(
                "data1.pbc and data2.pbc should be specified (True or False)."
            )

        assert data1.dg == data2.dg, "data1 and data2 must have the same resolution."
        dim = dim if dim is not None else (-2, -1)
        remove_mean = remove_mean if remove_mean is not None else False

        # finding the appropriate mask
        if self.mask_full_res is None:
            mask = None
        else:
            if len(data1.conv_history) > len(
                data2.conv_history
            ):  # mask for |I*psi2|*psi3 contains the one for I*psi3
                mask = self._find_mask(data1)
            elif len(data1.conv_history) < len(
                data2.conv_history
            ):  # mask for |I*psi2|*psi3 contains the one for I*psi3
                mask = self._find_mask(data2)
            else:
                if data1.conv_history == data2.conv_history:  # same mask for both
                    mask = self._find_mask(data1)
                else:
                    # mask for |I*psi2|*psi3 does not necessarily contains the one for |I*psi1|*psi3, and vice-versa
                    mask = self._find_mask(data1) + self._find_mask(data2)

        border = max(
            self._get_crop_border_size_method(data=data1, wavelet_op=self),
            self._get_crop_border_size_method(data=data2, wavelet_op=self),
        )

        x = data1.array
        y = data2.array

        if remove_mean:
            raise NotImplementedError(
                "remove_mean is not yet implemented. think about giving the right mask when doing it"
            )
            # x_c = x - x.mean(dim=dim, keepdim=True)
            # y_c = y - y.mean(dim=dim, keepdim=True)
        else:
            x_c = x
            y_c = y

        cropped_array = self._crop(array=x_c * torch.conj(y_c), border=border)
        cov = maskmean(
            x=cropped_array,
            dim=dim,
            mask=self._crop(array=mask, border=border),
        )

        return cov

    ###########################################################################
    def standardize(self, data, inplace=False, dim=None):
        """
        Standardize the data by removing the mean and scaling to unit variance
        on the last two dimensions (Nx, Ny) in real space.

        Parameters
        ----------
        - data : STL_2D_Kernel_Torch
            Input data whose array attribute has to be standardized.

        Returns
        -------
        - STL_2D_Kernel_Torch
            Standardized data.
        """

        if dim is None:
            dim = (-2, -1)

        l_data = data.copy(empty=False) if not inplace else data

        mean = self.mean(l_data)  # [Nb,Nc]
        l_data.array = (
            l_data.array - mean[..., None, None]
        )  # centering first because no remove_mean in cov

        var = self.cov(l_data, l_data)
        std = torch.sqrt(var)

        l_data.array = l_data.array / std[..., None, None]

        return l_data, mean, std

    ###########################################################################
    def unstandardize(self, data, mean, std, inplace=False):
        """
        Unstandardize the data by scaling back using the provided mean and std.

        Parameters
        ----------
        - data : STL_2D_Kernel_Torch
            Input data whose array attribute has to be unstandardized.
        - mean : torch.Tensor
            Mean used for standardization.
        - std : torch.Tensor
            Standard deviation used for standardization.

        Returns
        -------
        - STL_2D_Kernel_Torch
            Unstandardized data.
        """
        l_data = data.copy(empty=False) if not inplace else data

        l_data.array = l_data.array * std[..., None, None] + mean[..., None, None]

        return l_data

    def _compute_and_store_cross_cov(
        self,
        data1,
        data2,
        output,
        compute_cross_matrix,
        redundant_channels,
        remove_mean=False,
        dim=(-2, -1),
    ):
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
        Nc = output.shape[1]  # number of channels

        for c1 in range(Nc):
            for c2 in range(c1, Nc):
                if compute_cross_matrix[c1, c2]:

                    output[:, c1, c2, ...] = self.cov(
                        data1=data1[:, c1, ...],
                        data2=data2[:, c2, ...],
                        remove_mean=remove_mean,
                        dim=dim,
                    )

                    if not redundant_channels and c1 != c2:
                        output[:, c2, c1, ...] = self.cov(
                            data1=data1[:, c2, ...],
                            data2=data2[:, c1, ...],
                            remove_mean=remove_mean,
                            dim=dim,
                        )
        return

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
        # Check coherence of input data.
        if not isinstance(data, STL_2D_Kernel_Torch):
            raise Exception("Data should be a STL_2D_Kernel_Torch instance")
        if self.DT != data.DT:
            raise Exception("Data and wavelet transform should have same DT")

        if j != data.dg:
            raise ValueError("j is not equal to dg, convolution not possible")

        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device as the _wav_kernel
        x = torch.as_tensor(x, device=self._wav_kernel.device)

        if j==0:
            weight = self._wav_kernel_0.squeeze(0)  # [L, K, K]
        else:
            weight = self._wav_kernel.squeeze(0)  # [L, K, K]

        convolved = self.__class__._semicomplex_conv2d_circular(
            x, weight, padding_mode=self.__class__._get_padding_mode(pbc=data.pbc)
        )

        return STL_2D_Kernel_Torch(
            convolved,
            dg=data.dg,
            N0=data.N0,
            pbc=data.pbc,
            conv_history=data.conv_history + [j],
        )

    @staticmethod
    def _downsample_tensor(
        x: torch.Tensor, smooth_kernel: torch.Tensor, dg_inc: int, padding_mode: str
    ) -> torch.Tensor:
        """
        Downsample a tensor by a factor 2**dg_inc along the last two
        dimensions using (successive iterations of, if dg_inc > 1) torch.conv2d with stride=2.

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
        if len(smooth_kernel.shape) != 2:
            raise ValueError("Smooth kernel must be of dimension 2.")
        if smooth_kernel.shape[0] != smooth_kernel.shape[1]:
            raise ValueError("Smooth kernel must be a square.")
        if smooth_kernel.shape[-1] % 2 == 0:
            raise ValueError("Smooth kernel side length must be odd.")

        leading_dims = x.shape[:-2]
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        y = x.reshape(B, 1, H, W)

        for _ in range(dg_inc):
            h, w = y.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(
                    "Downsampling requires even spatial dimensions at each step."
                )
            # Add circular padding for periodic boundaries
            pad = smooth_kernel.shape[-1] // 2
            y_padded = F.pad(y, (pad, pad, pad, pad), mode=padding_mode)
            y = F.conv2d(
                input=y_padded, weight=smooth_kernel.unsqueeze(0).unsqueeze(0), stride=2
            )

        H2, W2 = y.shape[-2:]
        return y.reshape(*leading_dims, H2, W2)

    @staticmethod
    def _upsample_tensor(
        x: torch.Tensor,
        smooth_kernel: torch.Tensor,
        dg_inc: int,
        padding_mode: str,  # kept for symmetry; not used here
    ) -> torch.Tensor:
        """
        Upsample by factor 2**dg_inc using conv_transpose2d so that
        downsample->upsample recovers exact spatial sizes.

        Assumes the corresponding downsample used:
            y_padded = F.pad(y, pad, mode=padding_mode)
            y = F.conv2d(y_padded, weight, stride=2)
        with kernel size k and pad = k//2.
        """
        if dg_inc < 0:
            raise ValueError("dg_inc must be non-negative")
        if dg_inc == 0:
            return x

        if len(smooth_kernel.shape) != 2:
            raise ValueError("Smooth kernel must be of dimension 2.")
        if smooth_kernel.shape[0] != smooth_kernel.shape[1]:
            raise ValueError("Smooth kernel must be a square.")
        if smooth_kernel.shape[-1] % 2 == 0:
            raise ValueError("Smooth kernel side length must be odd.")

        k = smooth_kernel.shape[-1]
        pad = k // 2
        stride = 2

        leading_dims = x.shape[:-2]
        Hc, Wc = x.shape[-2:]
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        y = x.reshape(B, 1, Hc, Wc)

        weight = smooth_kernel.unsqueeze(0).unsqueeze(0)

        for _ in range(dg_inc):
            # target size after one upsampling step
            target_h = y.shape[-2] * 2
            target_w = y.shape[-1] * 2

            # Base output size with output_padding=0
            base_h = (y.shape[-2] - 1) * stride - 2 * pad + k
            base_w = (y.shape[-1] - 1) * stride - 2 * pad + k

            # output_padding must be 0 or 1 for stride=2
            out_pad_h = target_h - base_h
            out_pad_w = target_w - base_w
            if out_pad_h not in (0, 1) or out_pad_w not in (0, 1):
                raise ValueError(
                    f"Invalid output_padding computed: ({out_pad_h},{out_pad_w}). "
                    f"Check kernel/padding assumptions."
                )

            y = F.conv_transpose2d(
                input=y,
                weight=weight,
                stride=stride,
                padding=pad,
                output_padding=(out_pad_h, out_pad_w),
            )

            # Optional: compensate amplitude dilution for 2D stride=2
            y = y * 4.0

        Hf, Wf = y.shape[-2:]
        return y.reshape(*leading_dims, Hf, Wf)
    
    ###########################################################################
    def downsample(self, data, dg_out, inplace=True, replace_nan_value=nan):
        """
        Downsample the data to the dg_out resolution.
        Downsampling is done in real space along the last two dimensions using (successive iterations of, if dg_out - dg > 1) torch.conv2d with stride=2.
        If a mask is provided at full resolution, the downsampling is nan-aware, and sufficiently isolated NaNs can be removed through local averaging.
        """
        if data.pbc is None:
            raise ValueError(
                "data.pbc must be specified to perform downsampling (for adequate padding mode)."
            )

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

        if dg_inc > 0:
            smooth_kernel = self._gaussian_kernel_5x5(
                device=data.array.device, dtype=data.array.dtype
            )
            padding_mode = self.__class__._get_padding_mode(pbc=data.pbc)

            if self.mask_full_res is None:  # no mask
                data.array = self._downsample_tensor(
                    x=data.array,
                    smooth_kernel=smooth_kernel,
                    dg_inc=dg_inc,
                    padding_mode=padding_mode,
                )
                data.dg = dg_out
            else:  # mask
                if len(data.conv_history) == 0:
                    convolved_at = None
                else:
                    assert (
                        len(data.conv_history) < 2
                    ), "data must be at layer 0 or 1 to be downsampled."
                    convolved_at = data.conv_history[0]

                if convolved_at is None:
                    if data.dg == 0:
                        input_data_mask = self.mask_full_res.array
                    else:
                        input_data_mask = self._reweighting_maps_smooth[padding_mode][
                            data.dg
                        ].array.isnan()
                else:
                    if data.dg < convolved_at:
                        raise ValueError(
                            "convolved_at level must be greater than or equal to input data resolution."
                        )
                    if data.dg == convolved_at:
                        input_data_mask = self._layer1_mask[padding_mode][data.dg].array
                    else:
                        input_data_mask = self._reweighting_maps_wav[padding_mode][
                            data.dg
                        ][convolved_at].array.isnan()

                data.array = torch.where(
                    condition=~input_data_mask,
                    input=data.array,
                    other=0.0,
                )

                for _ in range(
                    dg_inc
                ):  # downsampling is done step by step to apply reweighting at each step
                    data.array = self._downsample_tensor(
                        x=data.array,
                        smooth_kernel=smooth_kernel,
                        dg_inc=1,
                        padding_mode=padding_mode,
                    )
                    data.dg += 1

                    reweighting_map = (
                        self._reweighting_maps_smooth[padding_mode][data.dg]
                        if convolved_at is None
                        else self._reweighting_maps_wav[padding_mode][data.dg][
                            convolved_at
                        ]
                    )

                    data.array *= torch.where(
                        condition=~reweighting_map.array.isnan(),
                        input=reweighting_map.array,
                        other=0.0,
                    )  # reweighting while avoiding to thrwow NaNs into data.attay for backprop

                if replace_nan_value is not None:
                    data.array = torch.where(
                        condition=~reweighting_map.array.isnan(),
                        input=data.array,
                        other=replace_nan_value,
                    )  # put a large value instead of NaNs WARNING: if applied, this breaks the backprop!!!
        return data

    ###########################################################################
    def upsample(self, data, dg_out, inplace=True, replace_nan_value=nan):
        """
        Upsample the data to the dg_out resolution.

        Upsampling is performed in real space along the last two dimensions
        using successive applications (if dg - dg_out >1) of
        torch.conv_transpose2d with stride=2 and the same 5x5 smoothing kernel.

        This corresponds to the adjoint (transpose) of the downsampling operator.

        If a full-resolution mask is defined, a mask-aware behavior is applied.
        Note that exact adjointness of the full masked + reweighted pipeline
        depends on the interpretation of the reweighting maps.
        """

        # Periodic boundary conditions must be defined
        if data.pbc is None:
            raise ValueError(
                "data.pbc must be specified to perform upsampling "
                "(for adequate padding mode)."
            )

        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")

        # No change in resolution
        if dg_out == data.dg and inplace:
            return data

        # Upsampling only (downsampling must use downsample())
        if dg_out >= data.dg:
            raise ValueError(
                "Requested dg_out <= current dg; "
                "downsampling not supported by upsampling method."
            )

        # Work on a copy if not inplace
        data = data.copy(empty=False) if not inplace else data

        dg_inc = data.dg - dg_out
        if dg_inc == 0:
            return data

        # Build the same smoothing kernel used for downsampling
        smooth_kernel = self._gaussian_kernel_5x5(
            device=data.array.device,
            dtype=data.array.dtype,
        )

        padding_mode = self.__class__._get_padding_mode(pbc=data.pbc)

        # ============================================================
        # Case A — No mask defined
        # ============================================================
        if self.mask_full_res is None:

            # Apply transpose operator step-by-step
            data.array = self._upsample_tensor(
                x=data.array,
                smooth_kernel=smooth_kernel,
                dg_inc=dg_inc,
                padding_mode=padding_mode,
            )

            data.dg = dg_out
            return data

        # ============================================================
        # Case B — Mask-aware upsampling
        # ============================================================

        # Determine convolution history (same logic as in downsample)
        if len(data.conv_history) == 0:
            convolved_at = None
        else:
            assert len(data.conv_history) < 2, (
                "data must be at layer 0 or 1 to be upsampled."
            )
            convolved_at = data.conv_history[0]

        # Replace NaNs by zeros before applying transpose convolution
        input_mask = data.array.isnan()
        data.array = torch.where(
            ~input_mask,
            data.array,
            torch.zeros_like(data.array),
        )

        # Perform step-by-step upsampling
        for _ in range(dg_inc):

            # Apply one level of transpose convolution (factor 2)
            data.array = self._upsample_tensor(
                x=data.array,
                smooth_kernel=smooth_kernel,
                dg_inc=1,
                padding_mode=padding_mode,
            )

            data.dg -= 1

            # Retrieve corresponding reweighting map at new resolution
            reweighting_map = (
                self._reweighting_maps_smooth[padding_mode][data.dg]
                if convolved_at is None
                else self._reweighting_maps_wav[padding_mode][data.dg][convolved_at]
            )

            # --------------------------------------------------------
            # Scaling strategy
            #
            # In downsampling, data was multiplied by reweighting_map.
            #
            # Two possible interpretations here:
            #
            # 1) Strict adjoint of scaling:
            #       multiply again by reweighting_map
            #
            # 2) Reconstruction-oriented behavior (default here):
            #       divide by reweighting_map (undo normalization)
            #
            # The second option is typically preferred for field
            # reconstruction rather than gradient consistency.
            # --------------------------------------------------------

            denom = torch.where(
                ~reweighting_map.array.isnan(),
                reweighting_map.array,
                torch.ones_like(reweighting_map.array),
            )

            # Avoid division by very small values
            eps = torch.tensor(1e-12, device=denom.device, dtype=denom.dtype)
            denom = torch.where(
                denom.abs() > eps,
                denom,
                torch.ones_like(denom),
            )

            # Reconstruction-oriented scaling
            data.array = data.array / denom

            # If strict adjoint behavior is desired instead:
            # data.array *= torch.where(
            #     ~reweighting_map.array.isnan(),
            #     reweighting_map.array,
            #     0.0,
            # )

            # Restore NaN (or large placeholder) if requested
            if replace_nan_value is not None:
                data.array = torch.where(
                    ~reweighting_map.array.isnan(),
                    data.array,
                    replace_nan_value,
                )

        return data
        
    def _gaussian_kernel_5x5(self, device, dtype):
        """
        Build and cache a normalized 5x5 Gaussian kernel on (device, dtype)
        for antialiasing 2D filter used in downsampling.

        Returns
        -------
        kernel : torch.Tensor
            Shape (5, 5)
        """
        if (
            not hasattr(self, "_smooth_kernel_5x5")
            or self._smooth_kernel_5x5.device != device
            or self._smooth_kernel_5x5.dtype != dtype
        ):
            # force real dtype for arange/meshgrid
            if dtype == torch.complex128:
                rdtype = torch.float64
            elif dtype == torch.complex64:
                rdtype = torch.float32
            else:
                rdtype = dtype

            ax = torch.arange(-2, 3, device=device, dtype=rdtype)
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")

            sigma = torch.tensor(1.0, device=device, dtype=rdtype)
            kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
            kernel = kernel / kernel.sum()

            # _conv2d_circular expects w shape (O_c, wx, wy)
            self._smooth_kernel_5x5 = kernel.to(dtype=dtype)
        return self._smooth_kernel_5x5


class PS_operator_2D_Kernel_torch:
    """
    Class whose instances correspond to a power spectrum operator for 2D FFT data.
    The operator is applied through apply method and is DT-dependent.
    """

    ###########################################################################
    def __init__(
        self,
        shape,
        n_bins=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        get_crop_border_size_method="flexible_crop",
    ):
        """
        Initialize a frequency binning object.

        Args:
            N0 (tuple): Image size (N, M)
            n_bins (int): Number of radial frequency bins
            device: torch device
            dtype: torch dtype
            get_crop_border_size_method : str ("flexible_crop" or "largest_crop")
        """
        self.shape = shape
        self.n_bins = (
            int(2 ** (np.log2(min(shape)) - 4)) if n_bins is None else n_bins
        )  # adaptive number of bins
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)
        self.get_crop_border_size_method = get_crop_border_size_method

        # --- Build frequency bin masks ---
        self._build()

        # --- Estimate crop borders for each bin (for non-PBC data apply) ---
        self.estimate_crop_borders()

    ###########################################################################
    def _build(self):
        N, M = self.shape

        # --- frequency grids ---
        freq_y = torch.fft.fftfreq(N, d=1.0, device=self.device)
        freq_x = torch.fft.fftfreq(M, d=1.0, device=self.device)

        FY, FX = torch.meshgrid(freq_y, freq_x, indexing="ij")

        # --- radial frequency ---
        self.radial_freq = torch.fft.fftshift(
            torch.sqrt(FX**2 + FY**2).to(self.dtype)
        )  # [N, M]

        # TODO: think about computing max spatial scale w.r.t pbc (-2 for periodic and -3 for non-periodic?)
        J = int(np.log2(min(N, M))) - 2  # max spatial scale
        self.min_freq = 1 / (2.0**J)
        self.max_freq = 0.5  # Nyquist

        # Linear regular binning
        self.bin_edges = torch.linspace(
            self.min_freq,
            self.max_freq,
            self.n_bins + 1,
            device=self.device,
            dtype=self.dtype,
        )

        # --- bin masks ---
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])  # [n_bins]
        sigma = 0.5 * (self.bin_edges[:-1] - self.bin_edges[1:])  # [n_bins]
        self.bin_masks = torch.exp(
            -0.5
            * ((self.radial_freq[None, :, :] - self.bin_centers[:, None, None]) ** 2)
            / (sigma[:, None, None] ** 2)
        )  # [n_bins, N, M]

    def estimate_crop_borders(self):

        N, M = self.shape

        # Create impulse at right border, centered vertically
        impulse = torch.zeros((N, M), device=self.device, dtype=self.dtype)
        impulse[N // 2, M // 2] = 1.0

        # FFT of impulse
        impulse_ft = torch.fft.fftshift(torch.fft.fft2(impulse, norm="ortho"))  # [N, M]

        # Apply all masks in batch
        impulse_ft = impulse_ft.unsqueeze(0)  # [1, N, M] for broadcasting
        psfs = torch.fft.ifft2(
            torch.fft.ifftshift(impulse_ft * self.bin_masks, dim=(-2, -1)),
            norm="ortho",
            dim=(-2, -1),
        ).real  # [n_bins, N, M]

        # Extract horizontal traces from pixel source
        traces = psfs[:, N // 2, : M // 2].abs()  # [n_bins, M//2]

        # Determine border where PSF drops below threshold_percent of the trace at the source pixel (maximum value)
        threshold_percent = 0.1
        threshold = threshold_percent * traces[:, -1].unsqueeze(1)  # [n_bins, 1]
        above_thresh = traces > threshold  # [n_bins, M//2]
        self.crop_borders = math.ceil(M / 2) - (
            above_thresh.float().argmax(dim=1) + 1
        )  # [n_bins]

    ###########################################################################
    def buid_mask_crop(self, array, border):
        """
        Crops an array by removing 'border' pixels from each side
        along the last two dimensions. Pads with zeros for each
        cropped side (border may be different for each bin) to keep
        the same output shape.

        Parameters
        ----------
        array : torch.Tensor
            Input array to be cropped. Shape [Nb, Nc, n_bins, N, M].
        border : torch.Tensor
            Number of pixels to remove from each side. Shape [n_bins].

        Returns
        -------
        torch.Tensor
            Cropped array. Shape [Nb, Nc, n_bins, N, M].
        """

        if array.ndim < 3:
            raise ValueError(
                "Input tensor must have at least 3 dimensions to apply per-bin crop."
            )

        n_bins_dim = array.shape[-3]  # dimension corresponding to bins
        N, M = array.shape[-2], array.shape[-1]

        # consistency check
        if border.numel() != n_bins_dim:
            raise ValueError(
                f"border tensor length ({border.numel()}) "
                f"does not match number of bins ({n_bins_dim})"
            )

        rows = torch.arange(N, device=array.device).view(1, N, 1)
        cols = torch.arange(M, device=array.device).view(1, 1, M)
        border_broadcast = border.view(n_bins_dim, 1, 1)

        mask = (
            (rows >= border_broadcast)
            & (rows < (N - border_broadcast))
            & (cols >= border_broadcast)
            & (cols < (M - border_broadcast))
        )  # [n_bins, N, M]

        return mask

    ###########################################################################
    def apply(self, data, get_crop_border_size_method=None):
        """
        Compute the power spectrum of the input data array attribute.

        Parameters
        ----------
        - data : STL_2D_Kernel_Torch
            Input data whose array attribute's power spectrum is to be computed. Array should be in real space.

        Returns
        -------
        torch.Tensor
            Power spectrum values of shape [..., n_bins].
        """
        # consistency check
        if type(data).__name__ != "STL_2D_Kernel_Torch":
            raise Exception(
                f"Data should be a STL_2D_Kernel_Torch instance, got {type(data)}"
            )
        if self.shape != data.N0:
            raise Exception("Data shape does not match operator shape")
        if data.dg != 0:
            raise Exception("Data dg must be 0 for power spectrum computation")
        if self.device != data.device:
            raise Exception("Data device does not match operator device")

        get_crop_border_size_method = (
            self.get_crop_border_size_method
            if get_crop_border_size_method is None
            else get_crop_border_size_method
        )

        # Copy data and put its array in Fourier space
        l_data = data.copy(empty=False)
        l_data.array = torch.fft.fft2(l_data.array, norm="ortho")  # [Nb, Nc, N, M]
        l_data.array = torch.fft.fftshift(l_data.array, dim=(-2, -1))  # [Nb, Nc, N, M]

        # Put in the expected shape if not already (should be already done in ST_op apply)
        if l_data.array.ndim == 2:
            l_data.array = l_data.array[None, None, :, :]  # [1, 1, N, M]
        elif l_data.array.ndim == 3:
            l_data.array = l_data.array[None, :, :, :]  # [1, Nc, N, M]

        # Apply bin masks
        l_data.array = (
            l_data.array[:, :, None, :, :] * self.bin_masks[None, None, :, :, :]
        )  # [Nb, Nc, n_bins, N, M]

        # Compute power spectrum
        if l_data.pbc:
            power_spectrum = (l_data.array.abs() ** 2).sum(
                dim=(-2, -1)
            ) / self.bin_masks.sum(
                dim=(-2, -1)
            )  # [Nb, Nc, n_bins]
            return power_spectrum

        if get_crop_border_size_method == "flexible_crop":
            border = self.crop_borders
        elif get_crop_border_size_method == "largest_crop":
            border = torch.full_like(self.crop_borders, self.crop_borders.max())
        else:
            raise ValueError(
                f"Invalid get_crop_border_size_method: {get_crop_border_size_method}"
            )

        l_data.array = torch.fft.ifft2(
            l_data.array, norm="ortho"
        )  # [Nb, Nc, n_bins, N, M]
        l_data.array = l_data.array.abs() ** 2  # [Nb, Nc, n_bins, N, M]
        mask_crop = self.buid_mask_crop(l_data.array, border=border)  # [n_bins, N, M]
        prefactor = (l_data.N0[0] * l_data.N0[1]) / (mask_crop).sum(dim=(-2, -1))

        power_spectrum = (
            prefactor
            * (l_data.array * (mask_crop)).sum(dim=(-2, -1))
            / self.bin_masks.sum(dim=(-2, -1))
        )
        return power_spectrum  # [Nb, Nc, n_bin]

    ###########################################################################
    def plot_PS(self, ps_tensor, b=0, c=0, label="Power Spectrum", color="b"):
        """
        Plot the power spectrum.
        Parameters
        ----------
        b : int
            Batch index (0<=b<Nb)
        c : int
            Channel index (0<=c<Nc)
        ps_tensor: torch.Tensor of shape [Nb, Nc, n_bins]
            Power spectrum values to plot

        Returns
        -------
        None
        """

        ps_values = ps_tensor[b, c, :].cpu().numpy()
        freqs = self.bin_centers.cpu().numpy()

        if ps_values.shape != freqs.shape:
            raise ValueError(
                f"ps_values shape: {ps_values.shape} and freqs shape: {freqs.shape} must have the same shape."
            )

        plt.plot(freqs, ps_values, "-", marker="o", label=label, color=color)

        plt.yscale("log")
        plt.xlabel("frequency")
        plt.ylabel("Power Spectrum")
        plt.title("Radial Power Spectrum")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()


###############################################################################
# Minkowski / Peak / Betti — internal helpers (self-contained, no external dep)
###############################################################################

# ── Gaussian wavelet helpers ──────────────────────────────────────────────────

def _gaussian_kernel_2d(
    sigma: float,
    truncate: float = 3.5,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return a [1, 1, K, K] isotropic Gaussian kernel."""
    radius = max(1, int(truncate * sigma + 0.5))
    size   = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]     # [K, K]
    return kernel[None, None]            # [1, 1, K, K]


def _gaussian_filter_2d(
    img: torch.Tensor,
    sigma: float,
    padding_mode: str = "circular",
) -> torch.Tensor:
    """Gaussian-filter a [B, N, M] image batch; returns same shape."""
    B, N, M = img.shape
    kernel = _gaussian_kernel_2d(sigma, device=img.device, dtype=img.dtype)
    radius = kernel.shape[-1] // 2
    x = F.pad(img.unsqueeze(1), [radius] * 4, mode=padding_mode)
    return F.conv2d(x, kernel).squeeze(1)   # [B, N, M]


def _extract_scale_flat(
    flat: torch.Tensor,
    j: int,
    pbc: bool = True,
    wavelet_op=None,
    scale_mode: str = "smooth",
) -> torch.Tensor:
    """
    Return the scale-j representation of a [B, N, M] batch.

    Parameters
    ----------
    j : int
        Dyadic scale index (0 = finest).  Gaussian sigma = 2^(j+1).
    pbc : bool
        Periodic boundary → circular padding.
    scale_mode : str
        ``'smooth'`` (default) — Gaussian low-pass at sigma=2^(j+1).
            Minkowski / Betti computed on the smoothed field; thresholds
            retain their physical meaning.
        ``'dog'`` — Difference-of-Gaussians band-pass |G(σ/2) − G(σ)|.
            Isolates detail at scale j (wavelet-style).
    wavelet_op : callable or None
        Reserved for future custom wavelet.
    """
    if wavelet_op is not None:
        raise NotImplementedError(
            "Custom wavelet_op is not yet implemented. Use wavelet_op=None."
        )
    padding_mode = "circular" if pbc else "replicate"
    sigma = 2.0 ** (j + 1)   # sigma = 2, 4, 8, 16, ...
    if scale_mode == "smooth":
        return _gaussian_filter_2d(flat, sigma, padding_mode=padding_mode)
    elif scale_mode == "dog":
        g_lo = _gaussian_filter_2d(flat, sigma / 2, padding_mode=padding_mode)
        g_hi = _gaussian_filter_2d(flat, sigma,     padding_mode=padding_mode)
        return (g_lo - g_hi).abs()
    else:
        raise ValueError(f"scale_mode must be 'smooth' or 'dog', got '{scale_mode}'")


# ── Minkowski helpers ─────────────────────────────────────────────────────────

def _mink2d_as_threshold(threshold, img: torch.Tensor) -> torch.Tensor:
    """Cast threshold to a tensor broadcastable on img [B, N, N]."""
    B = img.shape[0]
    t = (threshold if isinstance(threshold, torch.Tensor)
         else torch.tensor(threshold, dtype=img.dtype, device=img.device))
    if t.ndim == 1 and t.shape[0] == B:   # [B] → [B, 1, 1]
        t = t[:, None, None]
    torch.broadcast_shapes(t.shape, img.shape)  # raises if incompatible
    return t


def _mink2d_functionals(
    img: torch.Tensor,
    threshold=None,
    temperature: float = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Three 2D Minkowski functionals for a batch of square images [B, N, N].

    Returns {'W0': [B], 'W1': [B], 'W2': [B]}, fully differentiable.
    """
    B, N, M = img.shape
    if threshold is not None:
        img = torch.sigmoid(temperature * (img - _mink2d_as_threshold(threshold, img)))

    W0 = img.mean(dim=(-2, -1))

    dh = (img[:, :,  1:] - img[:, :, :-1]).abs()
    dv = (img[:, 1:, :]  - img[:, :-1, :]).abs()
    W1 = (dh.sum(dim=(-2, -1)) + dv.sum(dim=(-2, -1))) / (N * N)

    Q1 = img.sum(dim=(-2, -1))
    Qh = (img[:, :,  :-1] * img[:, :,  1:]).sum(dim=(-2, -1))
    Qv = (img[:, :-1, :]  * img[:, 1:, :] ).sum(dim=(-2, -1))
    Qf = (img[:, :-1, :-1] * img[:, :-1, 1:]
        * img[:,  1:, :-1] * img[:,  1:,  1:]).sum(dim=(-2, -1))
    W2 = (Q1 - Qh - Qv + Qf) / (N * N)

    return {"W0": W0, "W1": W1, "W2": W2}


def _mink2d_curves(
    img: torch.Tensor,
    thresholds: torch.Tensor,
    temperature: float = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Minkowski functionals at multiple thresholds for [B, N, N].

    thresholds : [T] or [B, T].  Returns {'W0','W1','W2'} each [B, T].
    """
    B, N, _ = img.shape
    t = thresholds
    if t.ndim == 1:
        t = t.unsqueeze(0).expand(B, t.shape[0])
    T = t.shape[1]
    soft = torch.sigmoid(temperature * (img.unsqueeze(1) - t.view(B, T, 1, 1)))
    mf = _mink2d_functionals(soft.view(B * T, N, N))
    return {k: v.view(B, T) for k, v in mf.items()}


###############################################################################
class MinkowskiOperator2D:
    """
    Minkowski functional operator for 2D planar STL data.

    Computes the three 2D Minkowski functionals — area (W0), perimeter (W1),
    and Euler characteristic (W2) — in a differentiable manner via soft
    sigmoid thresholding.

    Follows the same interface as ``PS_operator_2D_Kernel_torch``:
    build the operator once, then call ``minkowski()`` on any compatible
    ``STL_2D_Kernel_Torch`` instance.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape ``(Nx, Ny)`` of the input maps.  Must match ``data.N0``.
    thresholds : Tensor [T], Tensor [B, T], or None
        Default threshold levels.  ``None`` = no thresholding.
    temperature : float
        Sigmoid sharpness.  Higher → closer to hard binary thresholding.
    J : int
        Number of dyadic scales.

        - ``J=1`` (default) — works at full resolution.
        - ``J>1`` — applies a Difference-of-Gaussians (DoG) bandpass filter
          at each scale j=0..J-1 (finest to coarsest) and stacks results
          along a new leading ``J`` dimension.

          Output shape becomes ``[Nb, Nc, J]`` (no thresholds) or
          ``[Nb, Nc, J, T]`` (with thresholds).
    wavelet_op : callable or None
        Reserved for future use.  Pass ``None`` to use the default DoG filter.
    device : device
    dtype  : dtype

    Examples
    --------
    >>> op = MinkowskiOperator2D(shape=(64, 64))
    >>> mf = op.minkowski(data)           # {'W0','W1','W2'} each [Nb, Nc]

    >>> t = torch.linspace(0.1, 0.9, 16)
    >>> op = MinkowskiOperator2D(shape=(64, 64), thresholds=t, J=3)
    >>> curves = op.minkowski(data)       # {'W0','W1','W2'} each [Nb, Nc, 3, 16]
    """

    def __init__(
        self,
        shape,
        thresholds=None,
        temperature: float = 20.0,
        J: int = 1,
        scale_mode: str = "smooth",
        wavelet_op=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
    ):
        self.shape = shape
        self.thresholds = thresholds
        self.temperature = temperature
        self.J = J
        self.scale_mode = scale_mode
        self.wavelet_op = wavelet_op
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

    ###########################################################################
    def minkowski(
        self,
        data,
        thresholds=None,
        temperature: float = None,
    ) -> "dict[str, torch.Tensor]":
        """
        Compute the three Minkowski functionals of 2D planar data.

        Parameters
        ----------
        data : STL_2D_Kernel_Torch
            Input data with array of shape ``[..., Nx, Ny]``.
            Complex arrays are reduced to their modulus before computation.
        thresholds : Tensor [T] | Tensor [B, T] | None
            Overrides the operator-level default if provided.
        temperature : float or None
            Overrides the operator-level default if provided.

        Returns
        -------
        dict[str, Tensor]
            Keys ``'W0'``, ``'W1'``, ``'W2'``.

            - J=1, no thresholds : ``[Nb, Nc]``
            - J=1, thresholds T  : ``[Nb, Nc, T]``
            - J>1, no thresholds : ``[Nb, Nc, J]``
            - J>1, thresholds T  : ``[Nb, Nc, J, T]``
        """
        if not isinstance(data, STL_2D_Kernel_Torch):
            raise TypeError(
                f"data must be STL_2D_Kernel_Torch, got {type(data)}"
            )
        if self.shape != data.N0:
            raise ValueError(
                f"Operator shape {self.shape} != data.N0 {data.N0}"
            )

        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature

        arr = data.array
        if torch.is_complex(arr):
            arr = arr.abs()
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[None, :, :, :]
        Nb, Nc, Nx, Ny = arr.shape
        flat = arr.reshape(Nb * Nc, Nx, Ny)   # [B, Nx, Ny]
        pbc  = getattr(data, "pbc", True)

        def _compute_one(band: torch.Tensor):
            """Return mink dict for a single [B, N, M] band."""
            if thresholds is None:
                mf = _mink2d_functionals(band, temperature=temperature)
                return {k: v.view(Nb, Nc) for k, v in mf.items()}
            t = torch.as_tensor(thresholds, dtype=band.dtype, device=band.device)
            if t.ndim == 1:
                t = t.unsqueeze(0).expand(Nb * Nc, t.shape[0])
            elif t.ndim == 2:
                assert t.shape[0] == Nb * Nc, (
                    f"thresholds dim-0 ({t.shape[0]}) must equal Nb*Nc ({Nb*Nc})"
                )
            else:
                raise ValueError(f"thresholds must be 1-D or 2-D, got {t.ndim}-D")
            T = t.shape[1]
            curves = _mink2d_curves(band, t, temperature=temperature)
            return {k: v.view(Nb, Nc, T) for k, v in curves.items()}

        if self.J == 1:
            return _compute_one(flat)

        # ── Multi-scale ──────────────────────────────────────────────────────
        scale_results = [
            _compute_one(_extract_scale_flat(flat, j, pbc=pbc, wavelet_op=self.wavelet_op, scale_mode=self.scale_mode))
            for j in range(self.J)
        ]
        # stack along J dimension (inserted after Nc)
        return {
            k: torch.stack([sr[k] for sr in scale_results], dim=2)
            for k in scale_results[0]
        }



###############################################################################
# Peak counts & Betti curves — internal helpers
###############################################################################

def _neighborhood_extrema(
    img: torch.Tensor,
    mode: str = "max",
    connectivity: int = 8,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """
    Max or min over the local K-connectivity neighborhood, excluding center.

    img          : [B, N, M]
    mode         : 'max' or 'min'
    connectivity : 4 (N/E/S/W) or 8 (+ diagonals)
    Returns      : [B, N, M]
    """
    B, N, M = img.shape
    padded = F.pad(img.unsqueeze(1), (1, 1, 1, 1), mode=padding_mode).squeeze(1)

    shifts = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]
    if connectivity == 4:
        shifts = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    neighbors = torch.stack(
        [padded[:, 1+di:1+di+N, 1+dj:1+dj+M] for di, dj in shifts], dim=0
    )   # [K, B, N, M]

    return neighbors.amax(dim=0) if mode == "max" else neighbors.amin(dim=0)


def _soft_peaks(
    img: torch.Tensor,
    temperature: float,
    connectivity: int = 8,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """
    Soft local-maximum indicator.
    is_peak[i] = σ(τ * (f_i − max_neighbor_i)).  Returns [B, N, M].
    """
    max_nbr = _neighborhood_extrema(img, "max", connectivity, padding_mode)
    return torch.sigmoid(temperature * (img - max_nbr))


def _soft_valleys(
    img: torch.Tensor,
    temperature: float,
    connectivity: int = 8,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """
    Soft local-minimum indicator.
    is_valley[i] = σ(τ * (min_neighbor_i − f_i)).  Returns [B, N, M].
    """
    min_nbr = _neighborhood_extrema(img, "min", connectivity, padding_mode)
    return torch.sigmoid(temperature * (min_nbr - img))


def _threshold_weighted_sum(
    img: torch.Tensor,          # [B, N, M]
    indicator: torch.Tensor,    # [B, N, M]  soft binary mask (peaks / valleys)
    thresholds: torch.Tensor,   # [B, T]
    temperature: float,
    above: bool = True,         # True → count pixels ABOVE t, False → BELOW t
) -> torch.Tensor:              # [B, T]
    """
    Weighted sum: for each threshold t, accumulate indicator over active pixels.
    Active = soft excursion-set mask.  Result is normalised by N*M.
    """
    B, N, M = img.shape
    T = thresholds.shape[1]
    sign = 1.0 if above else -1.0
    active = torch.sigmoid(
        sign * temperature * (img.unsqueeze(1) - thresholds.view(B, T, 1, 1))
    )   # [B, T, N, M]
    return (active * indicator.unsqueeze(1)).sum(dim=(-2, -1)) / (N * M)


def _normalise_thresholds(
    thresholds,
    B: int,
    device,
    dtype,
) -> torch.Tensor:
    """Cast thresholds to [B, T] on the correct device/dtype."""
    t = torch.as_tensor(thresholds, dtype=dtype, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(0).expand(B, t.shape[0])
    if t.ndim != 2 or t.shape[0] != B:
        raise ValueError(
            f"thresholds must be shape [T] or [B={B}, T], got {tuple(t.shape)}"
        )
    return t


###############################################################################
class PeakCountOperator2D:
    """
    Soft local-extrema count operator for 2D planar STL data.

    Detects peaks (local maxima) and valleys (local minima) using a
    differentiable sigmoid approximation, optionally conditioned on
    intensity thresholds and evaluated at multiple dyadic scales.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape ``(Nx, Ny)`` matching ``data.N0``.
    thresholds : Tensor [T] | Tensor [B, T] | None
        Default threshold grid.  ``None`` = unconditional count.
    temperature : float
        Sigmoid sharpness.
    connectivity : int
        4 (cardinal) or 8 (incl. diagonals).
    J : int
        Number of dyadic scales.

        - ``J=1`` (default) — full resolution.
        - ``J>1`` — DoG band-pass at each scale j=0..J-1, results stacked
          along a new ``J`` dimension.
    wavelet_op : callable or None
        Reserved for future custom wavelet.  Use ``None`` (DoG).
    device, dtype

    Examples
    --------
    >>> t  = torch.linspace(0.1, 0.9, 16)
    >>> op = PeakCountOperator2D(shape=(128, 128), thresholds=t)
    >>> out = op.peaks(data)    # {'peaks': [Nb, Nc, 16]}

    >>> op3 = PeakCountOperator2D(shape=(128, 128), thresholds=t, J=3)
    >>> out3 = op3.peaks(data)  # {'peaks': [Nb, Nc, 3, 16]}
    """

    def __init__(
        self,
        shape,
        thresholds=None,
        temperature: float = 20.0,
        connectivity: int = 8,
        J: int = 1,
        scale_mode: str = "smooth",
        wavelet_op=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
    ):
        self.shape        = shape
        self.thresholds   = thresholds
        self.temperature  = temperature
        self.connectivity = connectivity
        self.J            = J
        self.scale_mode   = scale_mode
        self.wavelet_op   = wavelet_op
        self.device = _get_device(torch.device(device))
        self.dtype  = _get_dtype(dtype=dtype, device=self.device)

    # ------------------------------------------------------------------
    def _prepare(self, data):
        """Validate data, return (flat [B, Nx, Ny], padding_mode, Nb, Nc)."""
        if not isinstance(data, STL_2D_Kernel_Torch):
            raise TypeError(f"Expected STL_2D_Kernel_Torch, got {type(data)}")
        if self.shape != data.N0:
            raise ValueError(f"Shape mismatch: {self.shape} vs {data.N0}")
        arr = data.array.abs() if torch.is_complex(data.array) else data.array
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        pbc  = getattr(data, "pbc", True)
        pmode = "circular" if pbc else "replicate"
        return arr.reshape(Nb * Nc, Nx, Ny), pmode, Nb, Nc

    # ------------------------------------------------------------------
    def _count_one(self, band, mode, thresholds, temperature, pmode, Nb, Nc):
        """Compute count dict for a single [B, N, M] band."""
        B, N, M = band.shape
        key = "peaks" if mode == "peaks" else "valleys"
        if mode == "peaks":
            indicator = _soft_peaks(band, temperature, self.connectivity, pmode)
            above = True
        else:
            indicator = _soft_valleys(band, temperature, self.connectivity, pmode)
            above = False

        if thresholds is None:
            count = indicator.mean(dim=(-2, -1))   # [B]
            return {key: count.view(Nb, Nc)}

        t = _normalise_thresholds(thresholds, B, band.device, band.dtype)
        T = t.shape[1]
        counts = _threshold_weighted_sum(band, indicator, t, temperature, above=above)
        return {key: counts.view(Nb, Nc, T)}

    # ------------------------------------------------------------------
    def peaks(self, data, thresholds=None, temperature=None):
        """
        Soft peak (local-maximum) count, optionally conditioned on threshold.

        Returns
        -------
        dict['peaks']:

        - J=1, no thresholds : ``[Nb, Nc]``
        - J=1, thresholds T  : ``[Nb, Nc, T]``
        - J>1, no thresholds : ``[Nb, Nc, J]``
        - J>1, thresholds T  : ``[Nb, Nc, J, T]``
        """
        flat, pmode, Nb, Nc = self._prepare(data)
        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature
        pbc = pmode == "circular"

        if self.J == 1:
            return self._count_one(flat, "peaks", thresholds, temperature, pmode, Nb, Nc)

        scales = [
            self._count_one(
                _extract_scale_flat(flat, j, pbc=pbc, wavelet_op=self.wavelet_op, scale_mode=self.scale_mode),
                "peaks", thresholds, temperature, pmode, Nb, Nc,
            )
            for j in range(self.J)
        ]
        return {"peaks": torch.stack([s["peaks"] for s in scales], dim=2)}

    # ------------------------------------------------------------------
    def valleys(self, data, thresholds=None, temperature=None):
        """
        Soft valley (local-minimum) count, optionally conditioned on threshold.

        Same return-shape convention as :meth:`peaks`.
        """
        flat, pmode, Nb, Nc = self._prepare(data)
        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature
        pbc = pmode == "circular"

        if self.J == 1:
            return self._count_one(flat, "valleys", thresholds, temperature, pmode, Nb, Nc)

        scales = [
            self._count_one(
                _extract_scale_flat(flat, j, pbc=pbc, wavelet_op=self.wavelet_op, scale_mode=self.scale_mode),
                "valleys", thresholds, temperature, pmode, Nb, Nc,
            )
            for j in range(self.J)
        ]
        return {"valleys": torch.stack([s["valleys"] for s in scales], dim=2)}


###############################################################################
class BettiCurveOperator2D:
    """
    Differentiable Betti curves for 2D planar STL data.

    Approximates topological Betti numbers as functions of an intensity
    threshold using soft / differentiable surrogates:

    - **β₀(t)** ≈ soft peak count above t  (Morse theory: one max per component)
    - **χ(t)**  = Minkowski W2(t)           (exact pixel-complex Euler char.)
    - **β₁(t)** = β₀(t) − χ(t)

    Parameters
    ----------
    shape : tuple of int
        Spatial shape ``(Nx, Ny)`` matching ``data.N0``.
    thresholds : Tensor [T] | Tensor [B, T]
        Threshold levels (required; there is no single-threshold mode).
    temperature : float
        Sigmoid sharpness.
    connectivity : int
        4 or 8 — neighbourhood for peak detection.
    J : int
        Number of dyadic scales (1 = full resolution; >1 = multi-scale DoG).
    wavelet_op : callable or None
        Reserved for custom wavelet.
    device, dtype

    Examples
    --------
    >>> t  = torch.linspace(0.1, 0.9, 16)
    >>> op = BettiCurveOperator2D(shape=(128, 128), thresholds=t)
    >>> out = op.betti(data)
    >>> # out['beta0'].shape == out['beta1'].shape == out['chi'].shape == [Nb, Nc, 16]

    >>> op3 = BettiCurveOperator2D(shape=(128, 128), thresholds=t, J=3)
    >>> out3 = op3.betti(data)
    >>> # out3['beta0'].shape == [Nb, Nc, 3, 16]
    """

    def __init__(
        self,
        shape,
        thresholds,
        temperature: float = 20.0,
        connectivity: int = 8,
        J: int = 1,
        scale_mode: str = "smooth",
        wavelet_op=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
    ):
        self.shape        = shape
        self.thresholds   = thresholds
        self.temperature  = temperature
        self.connectivity = connectivity
        self.J            = J
        self.scale_mode   = scale_mode
        self.wavelet_op   = wavelet_op
        self.device = _get_device(torch.device(device))
        self.dtype  = _get_dtype(dtype=dtype, device=self.device)

    # ------------------------------------------------------------------
    def _betti_one(self, band, thresholds, temperature, pmode, Nb, Nc):
        """Compute Betti dict for a single [B, N, M] band."""
        B, N, M = band.shape
        t = _normalise_thresholds(thresholds, B, band.device, band.dtype)
        T = t.shape[1]

        # β0 : soft peak count above each threshold
        indicator = _soft_peaks(band, temperature, self.connectivity, pmode)
        beta0 = _threshold_weighted_sum(band, indicator, t, temperature, above=True)

        # χ : Minkowski W2 curves
        chi = _mink2d_curves(band, t, temperature=temperature)["W2"]   # [B, T]

        # β1 = β0 − χ
        beta1 = beta0 - chi

        return {
            "beta0": beta0.view(Nb, Nc, T),
            "beta1": beta1.view(Nb, Nc, T),
            "chi":   chi.view(Nb, Nc, T),
        }

    # ------------------------------------------------------------------
    def betti(self, data, thresholds=None, temperature=None):
        """
        Compute Betti curves β₀(t), β₁(t) and Euler characteristic χ(t).

        Returns
        -------
        dict with keys ``'beta0'``, ``'beta1'``, ``'chi'``:

        - J=1 : each of shape ``[Nb, Nc, T]``
        - J>1 : each of shape ``[Nb, Nc, J, T]``

        All outputs are fully differentiable w.r.t. ``data.array``.
        """
        if not isinstance(data, STL_2D_Kernel_Torch):
            raise TypeError(f"Expected STL_2D_Kernel_Torch, got {type(data)}")
        if self.shape != data.N0:
            raise ValueError(f"Shape mismatch: {self.shape} vs {data.N0}")

        arr = data.array.abs() if torch.is_complex(data.array) else data.array
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        flat = arr.reshape(Nb * Nc, Nx, Ny)
        pbc  = getattr(data, "pbc", True)
        pmode = "circular" if pbc else "replicate"

        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature

        if thresholds is None:
            raise ValueError("thresholds must be provided for Betti curves.")

        if self.J == 1:
            return self._betti_one(flat, thresholds, temperature, pmode, Nb, Nc)

        scale_results = [
            self._betti_one(
                _extract_scale_flat(flat, j, pbc=pbc, wavelet_op=self.wavelet_op, scale_mode=self.scale_mode),
                thresholds, temperature, pmode, Nb, Nc,
            )
            for j in range(self.J)
        ]
        return {
            k: torch.stack([sr[k] for sr in scale_results], dim=2)
            for k in scale_results[0]
        }
