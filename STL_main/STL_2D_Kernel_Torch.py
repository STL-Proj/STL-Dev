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
from scipy.integrate import quad

import STL_main.torch_backend as bk
from STL_main.Base_DataClass import Base_DataClass
from STL_main.ST_Operator import ST_Operator
from STL_main.STL_2D_FFT_Torch import WaveletOperator2D_FFT_torch
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
    STL_2D_Kernel_Torch child class for 2D planar STL Kernel using PyTorch

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

    def divide(self, data2, epsilon=1e-8, pow=1.0, inplace=False):
        """
        Divide self.array by data2.array raised to a power, with a small epsilon added to the denominator for numerical stability.

        Division is still performed in real space

        Parameters
        ----------
        data2 : STL_2D_FFT_Torch
            Another instance whose array is used as the denominator. Its Fourier
            status determines the computation domain.
        epsilon : float, optional
            Small constant added to the denominator for numerical stability
            (default is 1e-8).
        power : float, optional
            Exponent applied to the denominator (default is 1).
        inplace : bool
            If True, performs the operation in-place and returns self.
            If False, returns a new instance.

        Returns
        -------
        STL_2D_FFT_Torch
            Result of the division in the appropriate domain.
        """

        data1 = self.copy(empty=False) if not inplace else self

        # Apply the division in real space
        data1.array = data1.array / (data2.array + epsilon) ** pow
        data1.dtype = data1.array.dtype

        return data1

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
    def get_CS_op(self, *args, **kwargs):

        return CS_operator_2D_Kernel_Torch(
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

        # TODO: stack real and imag kernels and convolve once instead of twice for efficiency
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
        WType="Bump-Steerable",
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
        self.WType = WType

        self.J = J
        self.L = L if L is not None else 4
        self.KERNELSZ = kernel_size if kernel_size is not None else 5
        self.DT = DT

        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        if self.WType == "Morlet":
            self._wav_kernel = self._build_morlet_wavelet_kernel()  # [1, L, K, K]
        elif self.WType == "Bump-Steerable":
            self._wav_kernel = (
                self._build_bump_steerable_wavelet_kernel()
            )  # [1, L, K, K]
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
            if self.mask_full_res.array.ndim != 2:
                raise NotImplementedError(
                    "For now, mask_full_res.array must be 2D if provided."
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
                smooth_kernel = self._get_smooth_kernel(
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

    def _build_wavelet_kernel_from_ifft_crop(self, fft_wavelet_builder, N=256):
        assert (
            self.KERNELSZ % 2 == 1
        ), "KERNELSZ must be odd to have a well-defined center."
        w = self.KERNELSZ // 2
        kernel = fft_wavelet_builder(self.J, self.L, size=(N, N))[0]  # [L, N, N]
        kernel = torch.fft.fftshift(torch.fft.ifft2(kernel, dim=(-2, -1)), dim=(-2, -1))
        kernel = kernel[
            :, N // 2 - w : N // 2 + w + 1, N // 2 - w : N // 2 + w + 1
        ]  # [L, K, K]
        kernel -= kernel.mean(dim=(-2, -1), keepdims=True)
        kernel = kernel.to(device=self.device)
        return kernel.unsqueeze(0)  # (1, L, K, K)

    def _build_bump_steerable_wavelet_kernel(self):
        return self._build_wavelet_kernel_from_ifft_crop(
            fft_wavelet_builder=WaveletOperator2D_FFT_torch.bump_steerable_bank
        )  # (1, L, K, K)

    def _build_morlet_wavelet_kernel(self):
        return self._build_wavelet_kernel_from_ifft_crop(
            fft_wavelet_builder=WaveletOperator2D_FFT_torch.gaussian_bank
        )  # (1, L, K, K)

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

    def mean(self, data, dim=None):
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

    def cov(self, data1, data2, remove_mean=None, dim=None, specific_channel_pair=None):
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

            # if specific_channel_pair is not None:
            #    c1, c2 = specific_channel_pair
            #    if c1 == c2:
            #        mask = mask[c1, ...]
            #    else:
            #        mask = self._find_mask(data1)[c1, ...] + self._find_mask(data2)[c2, ...]

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
    def standardize(self, data, mean_field, inplace=False, dim=None):
        """
        Standardize the data by removing the mean and scaling to unit variance
        on the last two dimensions (Nx, Ny) in real space.

        Parameters
        ----------
        - data : STL_2D_Kernel_Torch
            Input data whose array attribute has to be standardized.
        - mean_field : bool
            If True, compute mean/std averaged over the batch dimension.
        - inplace : bool
            If True, perform the operation in-place on the input data.
        - dim : tuple
            Dimensions over which to compute the mean and standard deviation.

        Returns
        -------
        - STL_2D_Kernel_Torch
            Standardized data.
        - torch.Tensor
            Mean used for standardization.
        - torch.Tensor
            Standard deviation used for standardization.
        """

        if dim is None:
            dim = (-2, -1)

        l_data = data.copy(empty=False) if not inplace else data

        mean = self.mean(l_data)  # [Nb,Nc]
        if mean_field:
            mean = mean.mean(dim=0, keepdim=True)  # [1,Nc]

        l_data.array = (
            l_data.array - mean[..., None, None]
        )  # centering first because no remove_mean in cov

        var = self.cov(l_data, l_data)
        if mean_field:
            var = var.mean(dim=0, keepdim=True)  # [1,Nc]

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

        # No need for mean_field parameter here as the folloowing operation is the same whether mean and std are averaged over the batch dimension or not.
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
        if type(data).__name__ != "STL_2D_Kernel_Torch":
            raise Exception(
                f"Data should be a STL_2D_Kernel_Torch instance, got {type(data)}"
            )
        if self.DT != data.DT:
            raise Exception("Data and wavelet transform should have same DT")

        if j != data.dg:
            raise ValueError("j is not equal to dg, convolution not possible")

        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device as the _wav_kernel
        x = torch.as_tensor(x, device=self._wav_kernel.device)

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
        assert not torch.is_complex(x), "Input tensor x must be real-valued"
        assert not torch.is_complex(smooth_kernel), "smooth_kernel must be real-valued"

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

        leading_dims = x.shape[:-2]
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        y = x.reshape(B, 1, H, W)

        for _ in range(dg_inc):
            h, w = y.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(
                    "Downsampling requires even spatial dimensions at each step."
                )
            if smooth_kernel.shape == (2, 2) and torch.allclose(
                smooth_kernel,
                0.25
                * torch.ones(
                    (2, 2), device=smooth_kernel.device, dtype=smooth_kernel.dtype
                ),
            ):
                # Fast path for 2x2 average pooling (equivalent to convolution with a 2x2 kernel of 0.25)
                y = 0.25 * (
                    y[..., ::2, ::2]
                    + y[..., 1::2, ::2]
                    + y[..., ::2, 1::2]
                    + y[..., 1::2, 1::2]
                )
            else:
                # if smooth_kernel.shape[-1] % 2 == 0:
                #    raise ValueError("Smooth kernel side length must be odd.")
                # TODO: check if below fix using pad_before != pad_after for even kernel size is correct

                # Add circular padding for periodic boundaries
                pad_before = (smooth_kernel.shape[-1] - 1) // 2
                pad_after = smooth_kernel.shape[-1] // 2
                y_padded = F.pad(
                    y, (pad_before, pad_after, pad_before, pad_after), mode=padding_mode
                )
                # TODO: rely on self.__class__._conv2d_circular instead
                y = F.conv2d(
                    input=y_padded,
                    weight=smooth_kernel.unsqueeze(0).unsqueeze(0),
                    stride=2,
                )

        H2, W2 = y.shape[-2:]
        return y.reshape(*leading_dims, H2, W2)

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
            smooth_kernel = self._get_smooth_kernel(
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

    def _get_smooth_kernel(self, device, dtype):
        """
        Build and cache a normalized smoothing kernel on (device, dtype)
        for antialiasing 2D filter used in downsampling.

        Returns
        -------
        kernel : torch.Tensor
            Shape (2, 2) or (3,3) or (6, 6)
        """
        if (
            not hasattr(self, "_smooth_kernel")
            or self._smooth_kernel.device != device
            or self._smooth_kernel.dtype != dtype
        ):
            if False:
                # Precomputed 2x2 square -> no padding needed for striding 2 downsampling, but more aliasing
                self._smooth_kernel = torch.tensor(
                    [[0.25, 0.25], [0.25, 0.25]],
                    device=device,
                    dtype=dtype,
                )
            elif False:
                # Precomputed 5x5 Gaussian
                self._smooth_kernel = torch.tensor(
                    [
                        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                        [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                    ],
                    device=device,
                    dtype=dtype,
                )
                self._smooth_kernel /= self._smooth_kernel.sum()
            elif False:
                # Precomputed 3x3 square -> each pixel has the same weight with striding 2 downsampling
                self._smooth_kernel = torch.tensor(
                    [
                        [0.0625, 0.125, 0.0625],
                        [0.125, 0.25, 0.125],
                        [0.0625, 0.125, 0.0625],
                    ],
                    device=device,
                    dtype=dtype,
                )
            else:
                # Precomputed 6x6 cropped raised cosine kernel with bandwidth=.71, beta=.35
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
        return self._smooth_kernel


class CS_operator_2D_Kernel_Torch:
    """
    Class whose instances correspond to a cross spectrum operator for 2D Kernel data.
    The operator is applied through apply method and is DT-dependent.
    """

    # Useful functions for the bin mask wavelet bank construction
    @staticmethod
    def s(t):
        if -1 < t < 1:
            return np.exp(-1.0 / (1.0 - t**2))
        return 0.0

    @classmethod
    def s_lambda(cls, t, lam):
        return cls.s((2.0 * lam / (lam - 1.0)) * (t - 1.0 / lam) - 1.0)

    @classmethod
    def k_lambda(cls, t, lam):
        if t <= 1.0 / lam:
            return 1.0
        if t >= 1.0:
            return 0.0

        # Integrals
        num, _ = quad(lambda tp: (cls.s_lambda(tp, lam) ** 2) / tp, t, 1.0)
        den, _ = quad(lambda tp: (cls.s_lambda(tp, lam) ** 2) / tp, 1.0 / lam, 1.0)
        return num / den

    @classmethod
    def kappa_lambda(cls, t, lam):
        val = cls.k_lambda(t / lam, lam) - cls.k_lambda(t, lam)
        return np.sqrt(max(val, 0))

    ###########################################################################
    def __init__(
        self,
        shape,
        n_bins=None,
        J=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        get_crop_border_size_method="flexible_crop",
        cross_spectrum_method="fft",
    ):
        """
        Initialize a frequency binning object.

        Args:
            N0 (tuple): Image size (N, M)
            n_bins (int): Number of radial frequency bins
            device: torch device
            dtype: torch dtype
            get_crop_border_size_method : str ("flexible_crop" or "largest_crop")
            cross_spectrum_method : str ("fft" or "kernel")
                    Method to compute the cross spectrum.
                    "fft": Estimate cross spectrum via Fourier product
                    "kernel": Estimate cross spectrum via convolution with kernel in real space (has to be designed and implemented, not yet available)
        """
        self.shape = shape
        self.n_bins = (
            int(2 ** (np.log2(min(shape)) - 4)) if n_bins is None else n_bins
        )  # adaptive number of bins
        self.J = int(np.log2(min(shape))) - 2 if J is None else J
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)
        self.get_crop_border_size_method = get_crop_border_size_method
        self.cross_spectrum_method = cross_spectrum_method

        # --- Build frequency bin masks ---
        self._build_bin_masks()

        # --- Estimate crop borders for each bin (for non-PBC data apply) ---
        self.estimate_crop_borders()

    ###########################################################################
    def _build_bin_masks(self):

        N, M = self.shape

        self.min_freq = 1 / (2.0**self.J)
        self.max_freq = 0.5  # Nyquist frequency

        # get radial profil at high resolution
        lam = (self.max_freq / self.min_freq) ** (1 / (self.n_bins + 1))

        k_vals = torch.linspace(self.min_freq, self.max_freq, 1000)
        scales_j = torch.arange(1, self.n_bins + 1)

        psi_kernels = []
        for j in scales_j:
            psi_j = np.array(
                [self.kappa_lambda(k / (self.min_freq * lam**j), lam) for k in k_vals]
            )
            psi_kernels.append(psi_j)

        # go from 1D radial profile to 2D bin masks
        freq_y = torch.fft.fftfreq(N)
        freq_x = torch.fft.fftfreq(M)
        FY, FX = torch.meshgrid(freq_y, freq_x, indexing="ij")

        radial_freq = torch.fft.fftshift(torch.sqrt(FX**2 + FY**2))

        diff = torch.abs(radial_freq.unsqueeze(-1) - k_vals)
        idx = torch.argmin(diff, dim=-1)

        psi_kernels = torch.from_numpy(np.array(psi_kernels))  # shape [n_bins, 1000]
        self.bin_masks = torch.zeros(
            (self.n_bins, N, M), device=self.device, dtype=self.dtype
        )
        for j in range(self.n_bins):
            self.bin_masks[j] = psi_kernels[j][idx]

        self.bin_centers = self.min_freq * lam**scales_j
        self.lam = lam

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
    def build_mask_crop(self, array, border):
        """
        Crops an array by removing 'border' pixels from each side
        along the last two dimensions. Pads with zeros for each
        cropped side (border may be different for each bin) to keep
        the same output shape.

        Parameters
        ----------
        array : torch.Tensor
            Input array to be cropped.
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
        N, M = array.shape[-2:]

        rows = torch.arange(N, device=array.device).view(1, N, 1)
        cols = torch.arange(M, device=array.device).view(1, 1, M)
        border_broadcast = border.view(self.n_bins, 1, 1)

        mask = (
            (rows >= border_broadcast)
            & (rows < (N - border_broadcast))
            & (cols >= border_broadcast)
            & (cols < (M - border_broadcast))
        )  # [n_bins, N, M]

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
        if type(data).__name__ != "STL_2D_Kernel_Torch":
            raise Exception(
                f"Data should be a STL_2D_Kernel_Torch instance, got {type(data)}"
            )
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
        l_data.array = torch.fft.fft2(
            l_data.array, norm="ortho"
        )  # copy of data in Fourier space
        l_data.array = torch.fft.fftshift(l_data.array, dim=(-2, -1))  # [Nb, Nc, N, M]

        # Put in the expected shape if not already (should be already done in ST_op apply)
        if l_data.array.ndim == 2:
            l_data.array = l_data.array[None, None, :, :]  # [1, 1, N, M]
        elif l_data.array.ndim == 3:
            l_data.array = l_data.array[None, :, :, :]  # [1, Nc, N, M]

        Nb, Nc, N, M = l_data.array.shape
        n_bins = self.n_bins

        cross_spectrum = (
            bk.zeros((Nb, Nc, Nc, n_bins), dtype=bk._DEFAULT_COMPLEX_DTYPE) + bk.nan
        )

        compute_cross_spectrum_matrix = (
            bk.eye(Nc, dtype=bool)
            if compute_cross_spectrum_matrix is None
            else compute_cross_spectrum_matrix
        )

        l_data_bin = (
            l_data.array[:, :, None, :, :] * self.bin_masks[None, None, :, :, :]
        )  # [Nb, Nc, Nbin, N, M]

        cross_product_bin = l_data_bin[:, :, None, :, :, :] * torch.conj(
            l_data.array[:, None, :, None, :, :]
        )  # [Nb, Nc, Nc, n_bins, N, M]

        if l_data.pbc:

            cross_vals = (
                cross_product_bin.sum(dim=(-2, -1))
                / self.bin_masks.sum(dim=(-2, -1))[None, None, None, :]
            ).to(dtype=bk._DEFAULT_COMPLEX_DTYPE)

            # Symetric part is redundant and then not filled as cross_spectrum(c1, c2) and cross_spectrum(c2, c1) are conjugates
            cross_spectrum[:, compute_cross_spectrum_matrix, :] = cross_vals[
                :, compute_cross_spectrum_matrix, :
            ]

            return cross_spectrum  # [Nb, Nc, Nc, n_bins]

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

        ifft_l_data_bin = torch.fft.ifft2(
            l_data_bin, norm="ortho", dim=(-2, -1)
        )  # [Nb, Nc, n_bins, N, M]
        l_data.array = torch.fft.ifft2(l_data.array, norm="ortho")  # [Nb, Nc, N, M]
        mask_crop = self.build_mask_crop(l_data.array, border=border)  # [n_bins, N, M]
        prefactor = (l_data.N0[0] * l_data.N0[1]) / mask_crop.sum(
            dim=(-2, -1)
        )  # [n_bins]

        cross_product_bin_real = ifft_l_data_bin[:, :, None, :, :, :] * torch.conj(
            l_data.array[:, None, :, None, :, :]
        )  # [Nb, Nc, Nc, n_bins, N, M]

        cross_vals = (
            prefactor
            * (cross_product_bin_real * mask_crop[None, None, None, :, :, :]).sum(
                dim=(-2, -1)
            )
            / self.bin_masks.sum(dim=(-2, -1))[None, None, None, :]
        ).to(dtype=bk._DEFAULT_COMPLEX_DTYPE)

        cross_spectrum[:, compute_cross_spectrum_matrix, :] = cross_vals[
            :, compute_cross_spectrum_matrix, :
        ]

        return cross_spectrum  # [Nb, Nc, Nc, n_bins]

    ############################################################################
    def apply(
        self,
        data,
        compute_cross_spectrum_matrix=None,
        get_crop_border_size_method=None,
        cross_spectrum_method=None,
        **kwargs,
    ):

        method = (
            self.cross_spectrum_method
            if cross_spectrum_method is None
            else cross_spectrum_method
        )

        if method == "fft":
            return self.apply_fft(
                data=data,
                compute_cross_spectrum_matrix=compute_cross_spectrum_matrix,
                get_crop_border_size_method=get_crop_border_size_method,
                **kwargs,
            )

        elif method == "kernel":
            raise NotImplementedError(
                "Cross spectrum computation via kernel convolution is not yet implemented."
            )

        else:
            raise ValueError(f"Invalid cross_spectrum_method: {method}")

    ###########################################################################
    def plot_cross_spectrum(self, cs_tensor, b=0, c1=0, c2=0, label=None, color="b"):
        """
        Plot the power spectrum.
        Parameters
        ----------
        b : int
            Batch index (0<=b<Nb)
        c1, c2 : int
            Channel indices (0<=c1,c2<Nc)
        cs_tensor: torch.Tensor of shape [Nb, Nc, Nc, n_bins]
            Cross spectrum values to plot

        Returns
        -------
        None
        """

        cs_values = cs_tensor[b, c1, c2, :].cpu().numpy()
        freqs = self.bin_centers.cpu().numpy()

        if cs_values.shape != freqs.shape:
            raise ValueError(
                f"ps_values shape: {cs_values.shape} and freqs shape: {freqs.shape} must have the same shape."
            )

        plt.plot(freqs, cs_values, "-", marker="o", label=label, color=color)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("frequency")
        plt.ylabel("Cross Spectra")
        plt.title(f"Radial Cross Spectra c{c1+1}-c{c2+1} for image {b+1}")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
