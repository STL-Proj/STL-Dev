"""
Created on Wed Nov 14:07 2018
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from STL_main.ST_Operator import ST_Operator
from STL_main.torch_backend import (
    _DEFAULT_DEVICE,
    _DEFAULT_DTYPE,
    _get_device,
    _get_dtype,
    maskmean,
    to_torch_tensor,
)


###############################################################################
###############################################################################
class STL_2D_FFT_Torch:
    """
    Class for 2D planar STL FFT using PyTorch
    """

    def __init__(
        self, array, dg=None, N0=None, pbc=False, conv_history=[], fourier_status=False
    ):
        """
        Initialize the STL_2D_FFT_torch class.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input 2D array (NumPy or PyTorch tensor).
        dg : int, optional
            Data resolution. If None, set to 0.
        N0 : tuple of int, optional
            Original size of the array. Required if dg is provided.
        conv_history : list of int, optional
            History of convolutions applied to the data. e.g., [j1, j2] if data has been convolved
            successively with wavelets at scales j1 and j2.
        fourier_status : bool, optional
            Indicates if the data is in Fourier space (True) or real space (False).
        """

        # Main
        self.DT = "Planar2D_FFT_torch"
        if dg is None:
            self.dg = 0
            self.N0 = array.shape[-2:]
        else:
            self.dg = dg
            if N0 is None:
                raise ValueError("dg is given, N0 should not be None")
            self.N0 = N0

        self.array = self._to_array(array)
        self.fourier_status = fourier_status

        self.device = self.array.device
        self.dtype = self.array.dtype

        self.pbc = pbc
        self.conv_history = conv_history

    ###########################################################################
    def _to_array(self, array):
        """
        Transform input array (NumPy or PyTorch) into a PyTorch tensor.

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
            raise ValueError("Input array should not be None")
        else:
            # Choose device: use GPU if available, otherwise CPU
            # matches the input tensor dtype to the device
            return to_torch_tensor(array)

    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a STL_2D_FFT_Torch instance.
        Array is put to None if empty==True.

        Parameters
        ----------
        - empty : bool
            If True, set array to None.

        Returns
        ----------
        - STL_2D_FFT_Torch
            Copied instance.
        """
        new = object.__new__(STL_2D_FFT_Torch)

        # Copy metadata
        for k, v in self.__dict__.items():
            if k != "array":
                setattr(new, k, v)

        # Copy array
        if empty:
            new.array = None
        else:
            new.array = (
                self.array.clone() if isinstance(self.array, torch.Tensor) else None
            )

        return new

    ###########################################################################
    ### Note: slicing should be on MR=True data, to be removed
    def __getitem__(self, key):
        """
        To slice directly the array attribute.

        Parameters
        ----------
        - key : int or slice
            Slicing key.

        Returns
        -------
        - STL_2D_FFT_Torch
            New STL_2D_FFT_Torch instance with sliced array.
        """
        new = self.copy(empty=False)
        new.array = self.array[key]

        return new

    ###########################################################################
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the array attribute of data.

        Parameters
        ----------
        - inplace : bool
            If True, acts in-place and returns self.
            If False, returns a new STL_2D_FFT_Torch instance.

        Returns
        -------
        STL_2D_FFT_Torch
            STL_2D_FFT_Torch instance whose array attribute is the modulus
        """
        data = self.copy(empty=False) if not inplace else self

        data.array = torch.abs(data.array)

        return data

    ###########################################################################
    def fourier(self, inplace=False):
        """
        Compute the Fourier Transform on the last two dimensions of the input tensor.

        Parameters
        ----------
        - inplace : bool
            If True, acts in-place and returns self.
            If False, returns a new STL_2D_FFT_Torch instance.

        Returns
        -------
        STL_2D_FFT_Torch
            STL_2D_FFT_Torch instance whose array attribute is the Fourier
        """
        data = self.copy(empty=False) if not inplace else self

        if data.fourier_status:
            return data
        else:
            data.array = torch.fft.fft2(data.array, norm="ortho")
            data.fourier_status = True
            return data

    ###########################################################################
    def ifourier(self, inplace=False):
        """
        Compute the inverse Fourier Transform on the last two dimensions of the input
        tensor.

        Parameters
        ----------
        - inplace : bool
            If True, acts in-place and returns self.
            If False, returns a new STL_2D_FFT_Torch instance.

        Returns
        -------
        STL_2D_FFT_Torch
            STL_2D_FFT_Torch instance whose array attribute is the Fourier
        """
        data = self.copy(empty=False) if not inplace else self

        if not data.fourier_status:
            return data
        else:
            data.array = torch.fft.ifft2(data.array, norm="ortho")
            data.fourier_status = False
            return data

    ###########################################################################
    def set_fourier_status(self, target_fourier_status, inplace=False):
        """
        Put the  in the desired Fourier status (target_fourier_status).

        Parameters
        ----------
        - target_fourier_status : bool
            Desired Fourier status: True = Fourier space, False = real space.
        - inplace : bool
            If True, acts in-place and returns self.
            If False, returns a new STL_2D_FFT_Torch instance.

        Returns
        -------
        STL_2D_FFT_Torch
            STL_2D_FFT_Torch instance in the desired Fourier status.
        """
        data = self.copy(empty=False) if not inplace else self

        # If current status differs from desired
        if data.fourier_status != target_fourier_status:
            if target_fourier_status:
                data.fourier(inplace=True)
            else:
                data.ifourier(inplace=True)

        return data

    ###########################################################################
    def get_wavelet_op(self, J=None, L=None, pbc=None, **kwargs):
        if L is None:
            L = 4
        if J is None:
            J = np.min([int(np.log2(self.N0[0])), int(np.log2(self.N0[1]))]) - 2
        return CrappyWavelateOperator2D_FFT_torch(
            L,
            J,
            self.N0,
            self.DT,
            device=self.array.device,
            dtype=self.array.dtype,
            **kwargs
        )

    ###############################################################################
    def get_ST_op(
        self,
        J=None,
        L=None,
        jmin=None,
        jmax=None,
        dj=None,
        get_crop_border_size_method=None,
    ):

        return ST_Operator(
            self,
            J=J,
            L=L,
            jmin=jmin,
            jmax=jmax,
            dj=dj,
            get_crop_border_size_method=get_crop_border_size_method,
        )

    ###############################################################################
    def get_PS_op(self, n_bins=None):

        size = min(self.N0)

        if n_bins is None:
            n_bins = 16

        return PS_operator_2D_FFT_torch(
            size=size,
            n_bins=n_bins,
            device=self.device,
        )


class CrappyWavelateOperator2D_FFT_torch:
    """
    Class whose instances correspond to a wavelet transform operator.
    The wavelet set and the operator is built during the initilization.
    The operator is applied through apply method.
    This method is DT-dependent, and actually calls independent iterations,
    but with common method and attribute structure.

    The multi-resolution is dealt with several parameters:
        dg_max, which indicates the maximum dg resolution
        j_to_dg, which indicate the dg_j resolution associated to each j scale.

    For example, if you work with J=6 wavelets and N0=256, you can have:
        - dg_max=4, with associated dg factors (0, 1, 2, 3, 4)
        - true downsampling factors (1, 2, 4, 8, 16)
        - actual resolutions (256, 128, 64, 32, 16)
        - a j_to_dg list (0, 0, 1, 1, 2, 3) associated to j in range(J=6)

    The wavelet convolution is DT-dependent, an is performed either in real or
    Fourier space.

    They are two main types of wavelet arrays.
        - Single_Kernel==True: In this case, a set of L oriented wavelets is
        defined at a single pixellation. Convolutions at different scales are
        then down by subsequent susampling and convolution in pixel space with
        this single set of L oriented wavelets, and convolutions at all scales
        can not be done at the initial N0 resolution.
        - SR_Kernel==False: In this case, a set of J*L dilated and rotated
        wavelets is defined at the initial N0 resolution, and a convolution at
        all scales at this initial resolution can be performed. Convolution in
        a multi-resolution scheme can also be done, where convolution at each
        scale is done at the proper downsampling factor.

    This mean that different wavelets need to be stored:
        - Single_Kernel==True: a single set of L wavelets, stored in the
        wavelet_array attribute.
        - SR_Kernel==False: a set of J*L wavelets, store both at the
        initial N0 resolution in wavelet_array, and in a multi-resolution
        framework in wavelet_array_MR.
    => The wavelet_j method then allows to call for the correct quantity when
        a convolution is performed.

    Rq: when Single_Kernel==True, j_to_dg = range(J). This is not necessarily
        the case when False.

    See apply for more details.

    Parameters
    ----------
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N0 : tuple
        initial size of array (can be multiple dimensions)
    - J : int
        number of scales
    - L : int
        number of orientations
    - WType : str
        type of wavelets (e.g., "Morlet" or "Bump-Steerable")

    Attributes
    ----------
    - parent parameters (DT,N0,J,L,WType)
    - dg_max: int  (DT- and WType-dependent)
        maximum dg resolution of the Wavelet Transform
        (DT- and WType-dependent)
    - j_to_dg : list of int
        list of actual dg_j resolutions at each j scale
    - wavelet_array : torch tensor
        * array of wavelets at L orientation if Single_Kernel==True.
        * array of wavelets at J*L scales and orientation at N0 resolution
        if Single_Kernel==False
    - wavelet_array_MR : list (len J) of arrays
        list of arrays of L wavelets at all J scales and at Nj resolution
        Only if if Single_Kernel==False.
    - Single_Kernel : bool
        if convolution done at all scales with the same L oriented wavelets

    Questions and to do
    ----------
    - Is Single_Kernel sufficient in itself? We could separate the fact to do
    the convolution with a kernel in real space and to be able to do all
    convolution at the initial N0 resolution, using two different attribute.
    - We could similarly attach the fact to use mask to the fact to have
    Single_Kernel==True.
    - Do we add a low-pass filter per default for j=J ?
    - Do we impose j_to_dg = range(J) for simplicity and efficiency?
    - I propose to anyway only have dyadic wavelets for the "main set".
        -> the inclusion of P00' power spectrum terms can be done differently.
    - for __init__, we could ask either for DT and N, or for a stl_array
      instance, from which DT and N are obtained. Could be added if useful.
    - A proper "by the book" set of Wavelets should be implemented, with proper
    Littelwood-Paley and co conditions.

    """

    @staticmethod
    def gaussian_2d_rotated(mu, sigma, angle, size):
        """
        Generate a rotated 2D Gaussian centered at an offset mu along the rotated
        axis from image center.

        Parameters
        ----------
        mu : float
            Offset along the rotated axis from the image center (in pixels).
        sigma : float
            Isotropic standard deviation (spread).
        angle : float
            Rotation angle in radians (0 to pi).
        size : tuple of int

        Returns
        -------
        torch.Tensor
            A 2D Gaussian of shape [Nx, Ny].
        """

        M, N = size
        x = torch.linspace(0, M - 1, M)
        y = torch.linspace(0, N - 1, N)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Image center
        cx = M / 2
        cy = N / 2

        # Compute offset from center along rotated axis
        cos_a = torch.cos(torch.tensor(angle))
        sin_a = torch.sin(torch.tensor(angle))
        center_x = cx - mu * sin_a
        center_y = cy + mu * cos_a

        # Gaussian centered at (center_x, center_y)
        G = torch.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma**2))

        # Threshold
        eps = 10**-1
        G[G < eps] = 0

        return G

    @classmethod
    def gaussian_bank(cls, J, L, size, base_mu=None, base_sigma=None):
        """
        Generate a bank of rotated and scaled 2D Gaussians.

        Parameters
        ----------
        J : int
            Number of dyadic scales.
        L : int
            Number of orientations.
        base_sigma : float
            Smallest sigma (spread).
        base_mu : float
            Base offset along the rotated axis.
        size : tuple of int
            Grid size (M, N).

        Returns
        -------
        torch.Tensor
            A tensor of shape [J, L, Nx, Ny], each entry L2-normalized.
        """
        Nx, Ny = size
        filters_bank = torch.empty((J, L, Nx, Ny))

        if base_mu is None:
            base_mu = min(Nx, Ny) / (2 * torch.sqrt(torch.tensor(2.0)))
        if base_sigma is None:
            base_sigma = base_mu / (2 * torch.sqrt(torch.tensor(2.0)))

        for j in range(J):
            sigma = base_sigma / (2**j)
            mu = base_mu / (2**j)
            for l in range(L):
                angle = float(l) * torch.pi / L
                filters_bank[j, l] = cls.gaussian_2d_rotated(mu, sigma, angle, size)

        # Return the zero frequency to (0,0), and put it to zero
        filters_bank = torch.fft.fftshift(filters_bank, dim=(-2, -1))
        filters_bank[:, :, 0, 0] = 0

        return filters_bank

    @staticmethod
    def _calculate_base_sigma0(wavelet_op):
        """
        Calculate the base sigma value used in gaussian_bank.

        Parameters
        ----------
        wavelet_op : CrappyWavelateOperator2D_FFT_torch
            Wavelet operator instance.

        Returns
        -------
        float
            Base sigma value calculated as min(wavelet_op.N0) / 8.
        """
        return min(wavelet_op.N0) / 8

    @staticmethod
    def _get_crop_border_size_largest_scale_second_layer(data, wavelet_op):
        sigma0 = CrappyWavelateOperator2D_FFT_torch._calculate_base_sigma0(wavelet_op)
        if data.pbc:
            return 0
        else:
            deepest_layer = 2
            safety_prefactor = math.sqrt(
                4 * math.log(10)
            )  # point where Gaussian is ~1% of max
            return math.ceil(
                safety_prefactor
                * deepest_layer
                * (2 ** (wavelet_op.J - 1 - data.dg))
                / (2 * math.pi * sigma0)
            )

    @staticmethod
    def _get_crop_border_size_largest_scale_layer_flexible(data, wavelet_op):
        sigma0 = CrappyWavelateOperator2D_FFT_torch._calculate_base_sigma0(wavelet_op)
        if data.pbc or len(data.conv_history) == 0:
            return 0
        else:
            safety_prefactor = math.sqrt(
                4 * math.log(10)
            )  # point where Gaussian is ~1% of max
            return math.ceil(
                safety_prefactor
                * len(data.conv_history)
                * (2 ** (wavelet_op.J - 1 - data.dg))
                / (2 * math.pi * sigma0)
            )

    @staticmethod
    def _get_crop_border_size_fully_flexible(data, wavelet_op):
        sigma0 = CrappyWavelateOperator2D_FFT_torch._calculate_base_sigma0(wavelet_op)
        safety_prefactor = math.sqrt(
            4 * math.log(10)
        )  # point where Gaussian is ~1% of max

        if data.pbc or len(data.conv_history) == 0:
            return 0
        elif len(data.conv_history) == 1:
            return max(
                1,
                math.ceil(
                    safety_prefactor
                    * 2 ** (data.conv_history[0] - data.dg)
                    / (2 * math.pi * sigma0)
                ),
            )
        elif len(data.conv_history) == 2:
            return max(
                1,
                math.ceil(
                    safety_prefactor
                    * (
                        2 ** (data.conv_history[0] - data.dg)
                        + 2 ** (data.conv_history[1] - data.dg)
                    )
                    / (2 * math.pi * sigma0)
                ),
            )
        else:
            raise ValueError("Invalid data conv_history.")

    def __init__(
        self,
        L,
        J,
        N0,
        DT="Planar2D_FFT_torch",
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        get_crop_border_size_method=None,
    ):
        """
        Constructor, see details above.

        Parameters
        ----------
        - L : int
            number of orientations
        - J : int
            number of scales
        - N0 : tuple of int
            initial size of fourier domain array (same as data to be processed)
        - DT : str
            Type of data (1d, 2d planar, HealPix, 3d)
        - device : torch.device
            Device to store the wavelet arrays.
        - dtype : torch.dtype
            Data type to store the wavelet arrays.
        - get_crop_border_size_method : function
            Method to compute the crop border size.
        """
        self.WType = "Crappy"

        # Main parameters
        self.L = L
        self.J = J
        self.N0 = N0
        self.DT = DT
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        self.wavelet_array = None
        self.wavelet_array_MR = None
        self.dg_max = None
        self.j_to_dg = None
        self.build()  # Build all the wavelets-related attributes.

        if get_crop_border_size_method is not None:
            self._get_crop_border_size_method = get_crop_border_size_method
        else:
            self._get_crop_border_size_method = (
                self.__class__._get_crop_border_size_fully_flexible
            )

    ###########################################################################
    def build(self):
        """
        Build attributes related to the wavelet set and in multi-resolution framework:
            - wavelet_array
            - wavelet_array_MR
            - dg_max
            - j_to_dg
        """
        # Create the full resolution Wavelet set (in fourier space plus fftshifted)
        self.wavelet_array = self.__class__.gaussian_bank(
            self.J, self.L, self.N0
        )  # [J, L, N0x, N0y]

        # Find dg_max (with a min size of 16 = 2 * 8)
        # To avoid storing tensors at the same effective resolution
        self.dg_max = int(np.log2(min(self.N0)) - 4)

        # Create the MR list of wavelets
        self.wavelet_array_MR = []
        self.j_to_dg = []
        for j in range(self.J):
            dg = min(j, self.dg_max)
            subsampled_wavelet = self.__class__.downsample(
                data=STL_2D_FFT_Torch(array=self.wavelet_array[j], fourier_status=True),
                dg_out=dg,
                inplace=True,
                target_fourier_status=True,
            )  # [L, Njx, Njy]
            assert subsampled_wavelet.fourier_status
            self.wavelet_array_MR.append(subsampled_wavelet.array)
            self.j_to_dg.append(dg)

    @staticmethod
    def wavelet_conv_full(data, wavelet_set):
        """
        Perform convolutions of data with the entire wavelet set at full resolution.
        WARNING: Sets the data in Fourier space in place if data is in real space.

        Parameters
        ----------
        - data: STL_2D_FFT_Torch instance whose array attribute is a torch.Tensor of size [..., Nx, Ny]
            Data to be convolved with the wavelt_set
        - wavelet_set: torch.Tensor of size [J, L, Nx, Ny]
            Wavelet set in Fourier space at all J scales and L orientations

        Returns
        -------
        - STL_2D_FFT_Torch instance with:
            - array: torch.Tensor [..., J, L, Nx, Ny]
                Convolution in Fourier space between data and wavelet_set
            - fourier_status: bool
                True
        """
        # Set data in Fourier space in place
        data = data.set_fourier_status(target_fourier_status=True, inplace=True)
        return STL_2D_FFT_Torch(
            array=data[..., None, None, :, :].array * wavelet_set, fourier_status=True
        )  # [..., J, L, Nx, Ny]

    @staticmethod
    def wavelet_conv(data, wavelet_j):
        """
        Perform convolutions of data with a set of L wavelets fixed at a given scale and covering all orientations.
        Both the data and the wavelet should be at the Nj resolution.
        WARNING: Sets the data in Fourier space in place if data is in real space.

        Parameters
        ----------
        - data: STL_2D_FFT_Torch instance whose array attribute is a torch.Tensor of size [..., Njx, Njy]
            Data to be convolved with the wavelt_set, at resolution Nj
        - wavelet_j: torch.Tensor of size [L, Njx, Njy]
            Wavelet set in Fourier space at scale j and L orientations

        Returns
        -------
        - STL_2D_FFT_Torch instance with:
            - array: torch.Tensor [..., L, Njx, Njy]
                Convolution in Fourier space between data and wavelet_set at scale j
            - fourier_status: bool
                True
        """
        # Set data in Fourier space in place
        data = data.set_fourier_status(target_fourier_status=True, inplace=True)
        return STL_2D_FFT_Torch(
            array=data[..., None, :, :].array * wavelet_j,
            dg=data.dg,
            N0=data.N0,
            fourier_status=True,
        )  # [..., L, Njx, Njy]

    ###########################################################################
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
            elif False:  # flexible handling of borders larger than array
                if min(array.shape[-2:]) <= 2 * border:
                    if not getattr(
                        self, "_border_warning_raised", False
                    ):  # warns the user only once per wavelate operator
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

    ###########################################################################
    def mean(self, data, square=False, dim=(-2, -1), **kwargs):
        """
        Compute the mean on the last two dimensions (Nx, Ny).
        Parameters
        ----------
        - data : STL_2D_FFT_Torch
            Input data.
        - square : bool
            If True, compute the mean of the square of the data.
        - dim : tuple of int
            Dimensions on which the mean is computed.
        """
        if data.pbc and data.fourier_status:
            if square == False:
                return data.array[..., 0, 0]
            else:
                # Parseval identity
                return maskmean(x=data.array, square=square, dim=dim, mask=None)
        else:
            data = data.set_fourier_status(target_fourier_status=False, inplace=False)
            border = self._get_crop_border_size_method(data=data, wavelet_op=self)
            cropped_array = self._crop(array=data.array, border=border)

            return maskmean(x=cropped_array, square=square, dim=dim)

    ###########################################################################
    def cov(self, data1, data2=None, remove_mean=False, dim=(-2, -1), **kwargs):
        """
        Compute the covariance between data1 and data2 on the last two
        dimensions (Nx, Ny).
        """
        if data2 is None:
            data2 = data1
            border = self._get_crop_border_size_method(data=data1, wavelet_op=self)
        else:
            assert (
                data1.dg == data2.dg
            ), "data1 and data2 must have the same resolution."

            border = max(
                self._get_crop_border_size_method(data=data1, wavelet_op=self),
                self._get_crop_border_size_method(data=data2, wavelet_op=self),
            )

        data1 = data1.set_fourier_status(target_fourier_status=False, inplace=False)
        data2 = data2.set_fourier_status(target_fourier_status=False, inplace=False)

        x = data1.array
        y = data2.array

        if remove_mean:
            raise NotImplementedError("remove_mean is not yet implemented.")
            # x_c = x - x.mean(dim=dim, keepdim=True)
            # y_c = y - y.mean(dim=dim, keepdim=True)
        else:
            x_c = x
            y_c = y

        cropped_array = self._crop(array=x_c * torch.conj(y_c), border=border)

        cov = maskmean(x=cropped_array, square=False, dim=dim)

        return cov

    ###########################################################################
    def apply(self, data, j=None, target_fourier_status=None, **kwargs):
        """
        Compute the Wavelet Transform (WT) of data.
        This method is DT dependent, and calls independent iterations with
        common method and attribute structure.

        Data should be a MR==False StlData instance. The wavelet transform can
        either be computed at all J*L scales and angles (fullJ), or at a given
        j scale (single_j) for all L orientations?

        For code efficiency, this method requires a MR=True StlData instance
        for the masks at all resolution, with list_dg = range(dg_max + 1).

        The different modes are:
        - fullJ (j=None and MR=False): convolution at J*L scales and angles,
            without a MR framework, only if Single_Kernel==False.
            Input data should be a MR=False StlData instance at
            dg=0 resolution. No mask are a priori allowed in this case.
        - fullJ_MR (j=None and MR=True): convolution at J*L scales and angles,
            within a MR framework, always possible.
            Input data should be a MR=True StlData instance at all
            resolution between dg=0 and dg_max [ lids_dg = range(dg_max) ]
        - single_j: at L angles at a given scale j, within a MR framework.
            Input data should be a MR=False StlData instance wtih dg = dg_j.

        Rq: If j = None, the defaut value for MR if j=None is False if
        Single_Kernel==False, and True else. For a single_j convolution, MR
        can only be true.
        Rq: mask_MR is allowed only if mask_opt==True.

        Parameters
        ----------
        - data : STL_2D_FFT_Torch,
            Input data of same DT/N0, can be batched on several dimension.
            -> dg=0 if fullJ
            -> dg=dg_j if single_j
        - j : int
            Scale at which the convolution is done. Done at all scales if None.
        - target_fourier_status : bool or None
            Desired Fourier status of output.
            If None, DT-dependent default is used.

        Output
        ----------
        - WT : StlData:
            -> [..., J, L, N0] if j is None
            -> [..., L, Nj] if j == int

        Questions and to do
        ----------
        - I propose not to deal with the issue of non-periodicity here, but
        only in the mean and cov functions, at the end of the computations.
        - We could think at the possibility to compute WT at fixed (j,l)
        values, if it helps distributing the computations for large batchs.
        - To decide if we impose a condition on mask_MR, like the fact that it
        is on unit mean.
        - I'm a bit skeptical by the fact that an internal Fourier transform
        could be necessary here, since it means that the same transform could
        have to be on multiple call of this method.
        - For the convolution at a fixed scale. Should we accept data that are
        not at Nj resolution and downsample them? It need to be see with usage
        """

        # Check coherence of input data.
        if not isinstance(data, STL_2D_FFT_Torch):
            raise Exception("Data should be a STL_2D_FFT_Torch instance")
        if self.DT != data.DT:
            raise Exception("Data and wavelet transform should have same DT")
        if self.N0 != data.N0:
            raise Exception("Data and wavelet transform should have same N0 attributes")

        # fullJ convolution (j=None)
        if j is None:

            if data.dg != 0:
                raise Exception("Data should be at dg=0 resolution")

            # Convolution at all scales at full resolution N0
            WT = self.__class__.wavelet_conv_full(data, self.wavelet_array)

        # single_j convolution
        elif isinstance(j, int):
            # Check that dg_j resolutions are compatible
            if data.dg != self.j_to_dg[j]:
                raise Exception("Data should be at dg_j resolution")

            # Convolution at scale j at resolution Nj
            WT = self.__class__.wavelet_conv(
                data,
                self.wavelet_array_MR[j],
            )
        else:
            raise Exception("j should be a single int")

        # Transform to correct Fourier status if necessary
        if target_fourier_status is not None:
            WT.set_fourier_status(target_fourier_status, inplace=True)

        return WT

    ###########################################################################
    @staticmethod
    def downsample(data, dg_out, inplace=True, target_fourier_status=True, **kwargs):
        """
        Downsample the data.array to the dg_out resolution.


        Parameters
        ----------
        data : STL_2D_FFT_Torch instance
            Data whose array attribute has to be downsampled.
        dg_out : int
            Desired downsampling factor of the data.
        inplace : bool
            If True, acts in-place and returns data.
            If False, returns a new STL_2D_FFT_Torch instance.
        target_fourier_status : bool
            Desired Fourier status of the output data.
            As downsample is performed in Fourier space, default is True
            to avoid a final inverse Fourier step.

        Returns
        -------
        STL_2D_FFT_Torch instance
            Downsampled data at the desired downgrading factor dg_out.
        """
        data = data.copy(empty=False) if not inplace else data

        if dg_out == data.dg:
            return data

        # Tuning parameter to keep the aspect ratio and a unified resolution
        min_x, min_y = 8, 8
        if data.N0[0] > data.N0[1]:
            min_x = int(min_x * data.N0[0] / data.N0[1])
        elif data.N0[1] > data.N0[0]:
            min_y = int(min_y * data.N0[1] / data.N0[0])

        # Identify the new dimensions
        dx = int(max(min_x, data.N0[0] // 2 ** (dg_out + 1)))
        dy = int(max(min_y, data.N0[1] // 2 ** (dg_out + 1)))

        # Check expected current dimensions
        dx_cur = int(max(min_x, data.N0[0] // 2 ** (data.dg + 1)))
        dy_cur = int(max(min_y, data.N0[1] // 2 ** (data.dg + 1)))

        # Perform downsampling if necessary
        if dx != dx_cur or dy != dy_cur:

            # set data to Fourier space
            data = data.set_fourier_status(target_fourier_status=True, inplace=True)

            # Downsampling in Fourier
            data.array = torch.cat(
                (
                    torch.cat(
                        (data.array[..., :dx, :dy], data.array[..., -dx:, :dy]), -2
                    ),
                    torch.cat(
                        (data.array[..., :dx, -dy:], data.array[..., -dx:, -dy:]), -2
                    ),
                ),
                -1,
            ) * np.sqrt(dx * dy / dx_cur / dy_cur)

        data.dg = dg_out
        data = data.set_fourier_status(
            target_fourier_status=target_fourier_status, inplace=True
        )
        return data


class PS_operator_2D_FFT_torch:
    """
    Class whose instances correspond to a power spectrum operator for 2D FFT data.
    The operator is applied through apply method and is DT-dependent.
    """

    ###########################################################################
    def __init__(self, size, n_bins, device=_DEFAULT_DEVICE):
        self.size = size
        self.n_bins = n_bins
        self.device = device  # device of the data used to instantiate the operator

        # create frequency grid
        freqs = torch.fft.fftfreq(size) * size  # [-size/2, ..., size/2]
        u, v = torch.meshgrid(freqs, freqs, indexing="ij")

        # radial frequencies
        rho = torch.sqrt(u**2 + v**2)
        self.rho = torch.fft.fftshift(rho).to(device)

        # frequency bins
        self.max_freq = size // 2
        self.min_freq = 1  # to be refined if needed
        self.bin_edges = torch.linspace(self.min_freq, self.max_freq, n_bins + 1).to(
            device
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # create masks for each bin
        self.masks = []
        for i in range(n_bins):
            mask = (self.rho >= self.bin_edges[i]) & (self.rho < self.bin_edges[i + 1])
            self.masks.append(mask)

    def apply(self, data):
        """
        Compute the power spectrum of the input data.

        Parameters
        ----------
        - data : STL_2D_FFT_Torch
            Input data whose array attribute power spectrum is to be computed.

        Returns
        -------
        torch.Tensor
            Power spectrum values for each frequency bin.
        """
        # consistency check
        if not isinstance(data, STL_2D_FFT_Torch):
            raise Exception("Data should be a STL_2D_FFT_Torch instance")
        if self.size != min(data.N0):
            raise Exception("Data size does not match operator size")

        # Ensure data is in Fourier space
        l_data = data.set_fourier_status(target_fourier_status=True, inplace=False)
        l_data.array = torch.fft.fftshift(l_data.array, dim=(-2, -1))

        power_spectrum = []
        for mask in self.masks:
            masked_data = l_data.array * mask
            power = torch.mean(torch.abs(masked_data) ** 2)
            power_spectrum.append(power.item())

        return torch.tensor(power_spectrum, device=self.device)

    ###########################################################################
    def plot_PS(self, ps_tensor, label="Power Spectrum", color="b"):
        """
        Plot the power spectrum.
        Parameters
        ----------
        ps_tensor: torch.Tensor
            Power spectrum values to plot

        Returns
        -------
        None
        """
        # Conversion en numpy pour matplotlib
        freqs = self.bin_centers.cpu().numpy()

        ps_values = ps_tensor.cpu().numpy()
        plt.plot(freqs, ps_values, "-", marker="o", label=label, color=color)

        plt.yscale("log")
        plt.xlabel("frequency(cycles per image)")
        plt.ylabel("Power Spectrum")
        plt.title("Power Spectrum Radial")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
