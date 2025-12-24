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

    ###########################################################################
    def __init__(self, array, dg=None, N0=None, conv_history=[]):
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

        self.device = self.array.device
        self.dtype = self.array.dtype

        self.conv_history = conv_history

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
            # matches the input tensor dtype to the device
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
        new.conv_history = self.conv_history

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

        data.array = data.array.abs()

        data.dtype = data.array.dtype

        return data

    def get_wavelet_op(
        self,
        J=None,
        L=None,
        kernel_size=None,
        mask_full_res=None,
        pbc=True,
        *args,
        **kwargs,
    ):
        if L is None:
            L = 4
        if kernel_size is None:
            kernel_size = 5
        if J is None:
            J = np.min([int(np.log2(self.N0[0])), int(np.log2(self.N0[1]))]) - 2
        if mask_full_res is None:
            if torch.any(self.array.isnan()):
                mask_full_res = self.array.isnan()
        return WaveletOperator2Dkernel_torch(
            kernel_size,
            L,
            J,
            device=self.array.device,
            dtype=self.array.dtype,
            mask_full_res=mask_full_res,
            pbc=pbc,
            *args,
            **kwargs,
        )


class WaveletOperator2Dkernel_torch:
    @staticmethod
    def _get_padding_mode(pbc: bool = True) -> str:
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
    def _get_crop_border_size_largest_scale_second_layer(data, pbc, wavelet_op):
        if pbc:
            return 0
        else:
            deepest_layer = 2
            return (
                deepest_layer
                * 2 ** (wavelet_op.J - 1 - data.dg)
                * (wavelet_op.KERNELSZ // 2)
            )

    @staticmethod
    def _get_crop_border_size_largest_scale_layer_flexible(data, pbc, wavelet_op):
        if pbc or len(data.conv_history) == 0:
            return 0
        else:
            return (
                len(data.conv_history)
                * 2 ** (wavelet_op.J - 1 - data.dg)
                * (wavelet_op.KERNELSZ // 2)
            )

    @staticmethod
    def _get_crop_border_size_fully_flexible(data, pbc, wavelet_op):
        if pbc or len(data.conv_history) == 0:
            return 0
        elif len(data.conv_history) == 1:
            return int(
                np.ceil(
                    2 ** (data.conv_history[0] - data.dg) * (wavelet_op.KERNELSZ // 2)
                )
            )
        elif len(data.conv_history) == 2:
            first_conv_border_downgraded = int(
                np.ceil(
                    2 ** (data.conv_history[0] - data.conv_history[-1])
                    * (wavelet_op.KERNELSZ // 2)
                )
            )
            return int(
                np.ceil(
                    2 ** (data.conv_history[-1] - data.dg)
                    * (first_conv_border_downgraded + wavelet_op.KERNELSZ // 2)
                )
            )
        else:
            raise ValueError("Invalid data conv_history.")

    @staticmethod
    def _get_crop_border_size_zero(data, pbc, wavelet_op):
        return 0

    def __init__(
        self,
        kernel_size: int,
        L: int,
        J: int,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        mask_full_res=None,
        pbc=True,
        sigma_smooth=1.0,
        downsample_nan_weight_threshold=0.33,
        get_crop_border_size_method=None,
    ):
        self.KERNELSZ = kernel_size
        self.L = L
        self.J = J
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        self._wav_kernel = self._build_wavelet_kernel()
        self.sigma_smooth = (
            sigma_smooth  # to build smoothing kernel used in downsampling
        )
        # raise
        # build low pass kernel?
        self.WType = "simple"

        # PBC dependant parameters
        self.pbc = pbc
        if get_crop_border_size_method is not None:
            self._get_crop_border_size_method = get_crop_border_size_method
        else:
            self._get_crop_border_size_method = (
                self.__class__._get_crop_border_size_fully_flexible
            )

        # NaNs handling
        self.mask_full_res = (
            STL_2D_Kernel_Torch(
                array=mask_full_res.to(device=self.device, dtype=torch.bool)
            )
            if mask_full_res is not None
            else None
        )  # None if no NaN in the data. Is True where the data is NaN. TODO: add a more flexible constructor later..!!!
        self.downsample_nan_weight_threshold = downsample_nan_weight_threshold
        (
            self._reweighting_maps_smooth,
            self._reweighting_maps_wav,
            self._layer1_mask,
            self._layer2_mask,
        ) = self._build_reweighting_maps_and_scattering_layer_masks(
            pbc=self.pbc
        )  # behavior depending on PBC!!

    def _build_reweighting_maps_and_scattering_layer_masks(self, pbc):
        if self.mask_full_res is None:
            return None, None, None, None
        else:
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

            return (
                reweighting_maps_smooth,
                reweighting_maps_wav,
                layer1_mask,
                layer2_mask,
            )

    def _find_mask(self, data):
        if self.mask_full_res is None:
            return None
        else:
            layer = len(data.conv_history)
            if layer == 0:
                raise NotImplementedError(
                    "So far, data mask should not be called for data at layer 0."
                )
            assert data.dg == data.conv_history[-1]
            if layer == 1:
                return self._layer1_mask[data.conv_history[-1]].array
            elif layer == 2:
                return self._layer2_mask[data.conv_history[-1]][
                    data.conv_history[0]
                ].array

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

    def mean(self, data, square=False, dim=(-2, -1), pbc=True):
        """
        Compute the mean on the last two dimensions (Nx, Ny).
        """
        border = self._get_crop_border_size_method(data=data, pbc=pbc, wavelet_op=self)
        return maskmean(
            x=self._crop(array=data.array, border=border),
            square=square,
            dim=dim,
            mask=self._crop(array=self._find_mask(data), border=border),
        )

    def cov(self, data1, data2=None, remove_mean=False, dim=(-2, -1), pbc=True):
        """
        Compute the covariance between data1=self and data2 on the last two
        dimensions (Nx, Ny).
        """
        if data2 is None:
            data2 = data1
            mask = self._find_mask(data1)
            border = self._get_crop_border_size(data1, pbc)
        else:
            assert (
                data1.dg == data2.dg
            ), "data1 and data2 must have the same resolution."

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
                else:  # mask for |I*psi2|*psi3 does not necessarily contains the one for |I*psi1|*psi3, and vice-versa
                    mask = self._find_mask(data1) + self._find_mask(data2)

            border = max(
                self._get_crop_border_size_method(data=data1, pbc=pbc, wavelet_op=self),
                self._get_crop_border_size_method(data=data2, pbc=pbc, wavelet_op=self),
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

        cov = maskmean(
            x=self._crop(array=x_c * y_c.conj(), border=border),
            square=False,
            dim=dim,
            mask=self._crop(array=mask, border=border),
        )

        return cov

    def apply(self, data, j, pbc=True):
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
            raise ValueError("j is not equal to dg, convolution not possible")

        assert (
            pbc == self.pbc
        ), "pbc flag must match the one used to build the wavelet operator."

        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device as the _wav_kernel
        x = torch.as_tensor(x, device=self._wav_kernel.device)

        weight = self._wav_kernel.squeeze(0)  # [L, K, K]

        convolved = self.__class__._semicomplex_conv2d_circular(
            x, weight, padding_mode=self.__class__._get_padding_mode(pbc=pbc)
        )

        return STL_2D_Kernel_Torch(
            convolved, dg=data.dg, N0=data.N0, conv_history=data.conv_history + [j]
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

    ###########################################################################
    def downsample(self, data, dg_out, inplace=True, replace_nan_value=nan, pbc=True):
        """
        Downsample the data to the dg_out resolution.
        Downsampling is done in real space along the last two dimensions using (successive iterations of, if dg_out - dg > 1) torch.conv2d with stride=2.
        If a mask is provided at full resolution, the downsampling is nan-aware, and sufficiently isolated NaNs can be removed through local averaging.
        """
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out == data.dg and inplace:
            return data
        if dg_out < data.dg:
            raise ValueError(
                "Requested dg_out < current dg; upsampling not supported by downsampling method."
            )

        assert (
            pbc == self.pbc
        ), "pbc flag must match the one used to build the wavelet operator."

        data = data.copy(empty=False) if not inplace else data
        dg_inc = dg_out - data.dg

        if dg_inc > 0:
            smooth_kernel = self._gaussian_kernel_5x5(
                device=data.array.device, dtype=data.array.dtype
            )
            padding_mode = self.__class__._get_padding_mode(pbc=pbc)

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
                        input_data_mask = self._reweighting_maps_smooth[
                            data.dg
                        ].array.isnan()
                else:
                    if data.dg < convolved_at:
                        raise ValueError(
                            "convolved_at level must be greater than or equal to input data resolution."
                        )
                    if data.dg == convolved_at:
                        input_data_mask = self._layer1_mask[data.dg].array
                    else:
                        input_data_mask = self._reweighting_maps_wav[data.dg][
                            convolved_at
                        ].array.isnan()

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
                        self._reweighting_maps_smooth[data.dg]
                        if convolved_at is None
                        else self._reweighting_maps_wav[data.dg][convolved_at]
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
            size = 5
            coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma_smooth**2))
            kernel = kernel / kernel.sum()
            # _conv2d_circular expects w shape (O_c, wx, wy)
            self._smooth_kernel_5x5 = kernel
        return self._smooth_kernel_5x5
