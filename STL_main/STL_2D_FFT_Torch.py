"""
Created on Wed Nov 14:07 2018
"""

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import quad

import STL_main.torch_backend as bk
from STL_main.Base_DataClass import Base_DataClass
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
@dataclass
class STL_2D_FFT_Torch(Base_DataClass):
    """
    STL_2D_FFT_torch child class for 2D planar STL FFT using PyTorch

    Inherits Base_DataClass.

    See Base_DataClass for parameter descriptions.

    Additional parameters
    ---------------------
    fourier_status : bool
        Indicates if the data is in Fourier space (True) or real space (False).
    """

    # child class constant
    DT = "Planar2D_FFT_torch"

    # child instance attributes
    fourier_status: bool = False

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
            If False, returns a new STL_2D_FFT_Torch instance.

        Returns
        -------
        STL_2D_FFT_Torch
            STL_2D_FFT_Torch instance whose array attribute is the modulus
        """
        data = self.copy(empty=False) if not inplace else self

        data = data.set_fourier_status(target_fourier_status=False, inplace=True)

        data.array = torch.abs(data.array)

        data.dtype = data.array.dtype

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
            STL_2D_FFT_Torch instance whose array attribute is Fourier domain
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
    def divide(self, data2, epsilon=1e-8, pow=1.0, inplace=False):
        """
        Divide self.array by data2.array raised to a power, with a small epsilon added to the denominator for numerical stability.

        - If `data2.array` is in real space:
            `self.array` is converted to real space (if needed), and the division
            is performed in real space.

        - If `data2.array` is in Fourier space:
            `self.array` is converted to Fourier space (if needed), and the division
            is performed in Fourier space (i.e., deconvolution in real space).

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

        # convert self to the Fourier status of data2
        data1 = self.set_fourier_status(
            target_fourier_status=data2.fourier_status, inplace=inplace
        )

        # perform the division in the appropriate domain
        data1.array = data1.array / (data2.array + epsilon) ** pow
        data1.dtype = data1.array.dtype

        return data1

    ###########################################################################
    def get_wavelet_op(self, *args, **kwargs):

        return WaveletOperator2D_FFT_torch(
            N0=self.N0,
            DT=self.DT,
            device=self.device,
            dtype=self.dtype,
            *args,
            **kwargs,
        )

    ###############################################################################
    def get_ST_op(self, *args, **kwargs):

        return ST_Operator(data_example=self, *args, **kwargs)

    ###############################################################################
    def get_CS_op(self, *args, **kwargs):

        return CS_operator_2D_FFT_torch(
            shape=self.N0, device=self.device, dtype=self.dtype, *args, **kwargs
        )


class WaveletOperator2D_FFT_torch:
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

        # ad hoc normalisation
        filters_bank /= 0.8

        return filters_bank

    @staticmethod
    def bump_steerable_2d(omega_grid, L, xi0, width_factor=2.5, eps=1e-12):
        """
        Generate a 2D bump steerable wavelet in Fourier space.

        Parameters
        ----------
        omega_grid : torch.Tensor
            Grid of frequencies in Fourier space, of shape [Nx, Ny, 2], where the last dimension corresponds to (omega_x, omega_y).
        L : int
            Number of orientations (steerability order).
        xi0 : float
            Center frequency xi = (xi0, 0) in Fourier space.
        width_factor : float
            Optimized constant to follow at best Littlewood-Paley condition (optimized for L=4).
        eps : float
            Small constant to avoid division by zero in the bump window.

        Returns
        -------
        torch.Tensor
            A 2D bump steerable wavelet in Fourier space, of shape [Nx, Ny].
        """

        # radial part: bump window centered at xi0
        omega_norm = torch.sqrt(omega_grid[..., 0] ** 2 + omega_grid[..., 1] ** 2)

        r = abs(omega_norm - xi0) / xi0

        # apply bump window over r: g(r) = exp(-r^2 / (1 - r^2)) * 1_{0<r<1}
        r2 = r**2
        r2 *= width_factor  # optimized parameter (for L=4) to follow at best Littlewood Paley condition, see numerical integration in the notebook "consistency_Kernel_FFT.ipynb"
        support_r = (r >= 0.0) & (r < width_factor ** (-0.5))
        denom = (1.0 - r2).clamp_min(eps)
        bump = torch.where(support_r, torch.exp(-r2 / denom), torch.zeros_like(r))

        # angular part: cos(theta)^(L-1) where theta is the angle of omega in Fourier space
        theta = torch.atan2(omega_grid[..., 1], omega_grid[..., 0])
        support_theta = abs(theta) < torch.pi / 2
        angular = torch.where(
            support_theta, torch.cos(theta).pow(L - 1), torch.zeros_like(theta)
        )

        weights = bump * angular

        # normalize
        weights /= weights.abs().max()

        # Add a constant to follow Littlewood-Paley condition when L!=4
        c = (
            (2 ** (L - 1))
            * math.factorial(L - 1)
            / math.sqrt(L * math.factorial(2 * (L - 1)))
        )
        c_L4 = (
            (2 ** (4 - 1))
            * math.factorial(4 - 1)
            / math.sqrt(4 * math.factorial(2 * (4 - 1)))
        )

        return weights * (c / c_L4)

    @classmethod
    def bump_steerable_bank(cls, J, L, size):
        """
        Generate a bank of 2D bump steerable wavelets in Fourier space.

        Parameters
        ----------
        J : int
            Number of dyadic scales.
        L : int
            Number of orientations (steerability order).
        size : tuple of int
            Grid size (Nx, Ny).

        Returns
        -------
        torch.Tensor
            A tensor of shape [J, L, Nx, Ny]
        """
        Nx, Ny = size
        filters_bank = torch.empty((J, L, Nx, Ny))
        xi0 = min(Nx, Ny) / (2 * torch.sqrt(torch.tensor(2.0)))

        # Create the frequency grid in Fourier space, with the zero frequency at (0,0)
        omega_x = torch.fft.fftfreq(Nx) * Nx
        omega_x = torch.fft.fftshift(omega_x)  # Shift zero frequency to center

        omega_y = torch.fft.fftfreq(Ny) * Ny
        omega_y = torch.fft.fftshift(omega_y)  # Shift zero frequency to center

        Omega_x, Omega_y = torch.meshgrid(omega_x, omega_y, indexing="ij")
        omega_grid = torch.stack((Omega_x, Omega_y), dim=-1)

        for j in range(J):
            scale_factor = 2**j
            for l_idx, l in enumerate(range(L)):
                theta = math.pi * l / L + math.pi / 2

                cos_theta = torch.cos(torch.tensor(theta))
                sin_theta = torch.sin(torch.tensor(theta))
                R = torch.tensor(
                    [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
                    dtype=omega_grid.dtype,
                    device=omega_grid.device,
                )

                q = (
                    scale_factor * omega_grid @ R
                )  # rotate and dilate the frequency grid
                filters_bank[j, l_idx] = torch.fft.fftshift(
                    cls.bump_steerable_2d(q, L=L, xi0=xi0)
                )

        return filters_bank

    @staticmethod
    def _get_crop_border_size_largest_scale_second_layer(data, wavelet_op):
        if data.pbc:
            return 0
        else:
            deepest_layer = 2
            return math.ceil(
                deepest_layer
                * wavelet_op.crop_borders[-1, :]
                .max()
                .item()  # largest crop at full resolution
                / (2**data.dg)  # adapt to current resolution
            )

    @staticmethod
    def _get_crop_border_size_largest_scale_layer_flexible(data, wavelet_op):
        if data.pbc or len(data.conv_history) == 0:
            return 0
        else:
            return math.ceil(
                len(data.conv_history)
                * wavelet_op.crop_borders[-1, :]
                .max()
                .item()  # largest crop at full resolution
                / (2**data.dg)  # adapt to current resolution
            )

    @staticmethod
    def _get_crop_border_size_fully_flexible(data, wavelet_op):

        if data.pbc or len(data.conv_history) == 0:
            return 0
        elif len(data.conv_history) == 1:
            return math.ceil(
                wavelet_op.crop_borders[data.conv_history[0], :]
                .max()
                .item()  # crop at first convolution scale at full resolution
                / (2**data.dg)  # adapt to current resolution
            )
        elif len(data.conv_history) == 2:
            first_conv_border_downgraded = wavelet_op.crop_borders[
                data.conv_history[0], :
            ].max().item() / (  # crop at first convolution scale at full resolution
                2 ** data.conv_history[-1]
            )  # adapt to second convolution scale
            return math.ceil(
                first_conv_border_downgraded / (2 ** (data.dg - data.conv_history[-1]))
                + wavelet_op.crop_borders[data.conv_history[1], :].max().item()
                / (2**data.dg)
            )

        else:
            raise ValueError("Invalid data conv_history.")

    def __init__(
        self,
        N0,
        J=None,
        L=None,
        WType="Bump-Steerable",
        DT="Planar2D_FFT_torch",
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
        get_crop_border_size_method=None,
    ):
        """
        Constructor, see details above.

        Parameters
        ----------
        - WType : str
            type of wavelets (e.g., "Bump-Steerable" or "Morlet")
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
        self.WType = WType  # type of wavelets (e.g., "Bump-Steerable" or "Morlet")

        # Main parameters
        self.N0 = N0
        self.J = J if J is not None else int(np.log2(min(N0))) - 2
        self.L = L if L is not None else 4
        self.DT = DT
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        self.wavelet_array = None
        self.wavelet_array_MR = None
        self.dg_max = None
        self.j_to_dg = None
        self._build()  # Build all the wavelets-related attributes.

        if get_crop_border_size_method is not None:
            self._get_crop_border_size_method = get_crop_border_size_method
        else:
            self._get_crop_border_size_method = (
                self.__class__._get_crop_border_size_fully_flexible
            )

        # NaNs handling
        self.mask_full_res = None  # Used for NaNs handling in other data types. Must be None for this one which does not handles NaNs.
        assert (
            self.mask_full_res is None
        ), "mask_full_res must be set to None for this DataType that does not handle NaNs."

        self.estimate_crop_borders()

    ###########################################################################
    def _build(self):
        """
        Build attributes related to the wavelet set and in multi-resolution framework:
            - wavelet_array
            - wavelet_array_MR
            - dg_max
            - j_to_dg
        """
        # Create the full resolution Wavelet set (in fourier space plus fftshifted)
        if self.WType == "Morlet":
            self.wavelet_array = self.__class__.gaussian_bank(
                self.J, self.L, self.N0
            ).to(
                device=self.device, dtype=self.dtype
            )  # [J, L, N0x, N0y]

        elif self.WType == "Bump-Steerable":
            self.wavelet_array = self.__class__.bump_steerable_bank(
                self.J, self.L, self.N0
            ).to(
                device=self.device, dtype=self.dtype
            )  # [J, L, N0x, N0y]

        else:
            raise ValueError("Invalid WType.")

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
                normalize=False,
                inplace=True,
                target_fourier_status=True,
            )  # [L, Njx, Njy]
            assert subsampled_wavelet.fourier_status
            self.wavelet_array_MR.append(subsampled_wavelet.array)
            self.j_to_dg.append(dg)

    def estimate_crop_borders(self):

        N, M = self.N0

        # Create impulse at the center of the image
        impulse = torch.zeros((N, M), device=self.device, dtype=self.dtype)
        impulse[N // 2, M // 2] = 1.0

        # FFT of impulse
        impulse_ft = torch.fft.fftshift(torch.fft.fft2(impulse, norm="ortho"))  # [N, M]

        # Apply the wavelet filter at j=J and L=0 to the impulse in Fourier Space (look at the impulse response along this direction)
        psf = torch.fft.ifft2(
            torch.fft.ifftshift(impulse_ft * self.wavelet_array[-1, 0], dim=(-2, -1)),
            norm="ortho",
            dim=(-2, -1),
        ).abs()  # [N, M]

        # Extract horizontal traces from pixel source
        traces = psf[N // 2, : M // 2]  # [M//2]

        # Determine border where PSF drops below threshold_percent of the vertical trace at the source pixel (maximum value)
        threshold_percent = 0.1
        threshold = threshold_percent * traces[-1]  # [N//2]
        above_threshold = traces > threshold  # [N//2]
        crop_border = math.ceil(N / 2) - (
            above_threshold.float().argmax(dim=0) + 1
        )  # scalar

        # Expanding crop borders to all scales and orientations
        crop_borders = torch.tensor(
            [math.ceil(crop_border / (2**j)) for j in range(self.J - 1, -1, -1)]
        )
        crop_borders = crop_borders[:, None].repeat(1, self.L)
        self.crop_borders = crop_borders

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
            array=data[..., None, None, :, :].array * wavelet_set,
            pbc=data.pbc,
            fourier_status=True,
        )  # [..., J, L, Nx, Ny]

    @staticmethod
    def wavelet_conv(data, wavelet_set_MR, j):
        """
        Perform convolutions of data with a set of L wavelets fixed at a given scale and covering all orientations.
        Both the data and the wavelet should be at the Nj resolution.
        WARNING: Sets the data in Fourier space in place if data is in real space.

        Parameters
        ----------
        - data: STL_2D_FFT_Torch instance whose array attribute is a torch.Tensor of size [..., Njx, Njy]
            Data to be convolved with the wavelet_set, at resolution Nj
        - wavelet_set_MR: list (len J) of torch.Tensor of size [L, Njx, Njy]
        - j: int
            Scale index to select the wavelet set at resolution Nj

        Returns
        -------
        - STL_2D_FFT_Torch instance with:
            - array: torch.Tensor [..., L, Njx, Njy]
                Convolution in Fourier space between data and wavelet_set at scale j
            - fourier_status: bool
                True
        """
        # Set data in Fourier space in place
        data.set_fourier_status(target_fourier_status=True, inplace=True)

        wavelet_j = wavelet_set_MR[j]  # [L, Njx, Njy]

        return STL_2D_FFT_Torch(
            array=data[..., None, :, :].array * wavelet_j,
            dg=data.dg,
            N0=data.N0,
            fourier_status=True,
            pbc=data.pbc,
            conv_history=data.conv_history + [j],
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
            elif True:  # flexible handling of borders larger than array
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
    def mean(self, data, dim=(-2, -1), **kwargs):
        """
        Compute the mean on the last two dimensions (Nx, Ny).

        Parameters
        ----------
        - data : STL_2D_FFT_Torch
            Input data. Array should in real space in ST_op workflow
        - dim : tuple of int
            Dimensions on which the mean is computed.
        """

        if data.pbc is None and len(data.conv_history) > 0:
            raise ValueError("data.pbc should be specified (True or False).")

        if data.fourier_status:
            if data.pbc:
                return data.array[..., 0, 0] / np.sqrt(
                    math.prod(data.array.shape[i] for i in dim)
                )
            else:
                raise NotImplementedError(
                    "Mean computation in Fourier space for non-periodic data is not implemented."
                )

        else:
            border = self._get_crop_border_size_method(data=data, wavelet_op=self)
            cropped_array = self._crop(array=data.array, border=border)

            # No prefactor needed for mean in real  space thanks to downsample function
            return maskmean(x=cropped_array, dim=dim)

    def square_mean(self, data, dim=(-2, -1), **kwargs):

        if data.pbc is None and len(data.conv_history) > 0:
            raise ValueError("data.pbc should be specified (True or False).")

        if data.fourier_status:
            if data.pbc:
                return torch.mean(
                    data.array * data.array.conj()
                ).real  # Parseval identity
            else:
                raise NotImplementedError(
                    "Square mean computation in Fourier space for non-periodic data is not implemented."
                )

        else:
            border = self._get_crop_border_size_method(data=data, wavelet_op=self)
            cropped_array = self._crop(
                array=data.array * data.array.conj(), border=border
            )

            # No prefactor needed for mean in real  space thanks to downsample function
            return maskmean(x=cropped_array, dim=dim)

    def cov(self, data1, data2, remove_mean=False, dim=(-2, -1), **kwargs):

        assert data1.dg == data2.dg, "data1 and data2 must have the same resolution."

        if remove_mean:
            raise NotImplementedError("remove_mean is not yet implemented.")

        border = max(
            self._get_crop_border_size_method(data=data1, wavelet_op=self),
            self._get_crop_border_size_method(data=data2, wavelet_op=self),
        )

        if data1.pbc and data1.fourier_status and data2.pbc and data2.fourier_status:
            # Parseval identity
            return maskmean(
                x=data1.array * torch.conj(data2.array),
                dim=dim,
            )

        else:
            data1.set_fourier_status(target_fourier_status=False, inplace=True)
            data2.set_fourier_status(target_fourier_status=False, inplace=True)

            cropped_array = self._crop(
                array=data1.array * torch.conj(data2.array),
                border=border,
            )

            return maskmean(x=cropped_array, dim=dim)

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
    def unstandardize(self, data, mean, std, inplace=False, dim=None):
        """
        Unstandardize the data by scaling back using the provided mean and std.

        Parameters
        ----------
        - data : STL_2D_FFT_Torch
            Input data whose array attribute has to be unstandardized.
        - mean : torch.Tensor
            Mean used for standardization.
        - std : torch.Tensor
            Standard deviation used for standardization.

        Returns
        -------
        - STL_2D_FFT_Torch
            Unstandardized data.
        """
        l_data = data.copy(empty=False) if not inplace else data

        # unstandardization is done in real space
        if l_data.fourier_status:
            l_data = l_data.set_fourier_status(
                target_fourier_status=False, inplace=True
            )

        l_data.array = l_data.array * std[..., None, None] + mean[..., None, None]

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
        if type(data).__name__ != "STL_2D_FFT_Torch":
            raise Exception(
                f"Data should be a STL_2D_FFT_Torch instance, got {type(data)}"
            )
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
            WT = self.__class__.wavelet_conv(data, self.wavelet_array_MR, j)
        else:
            raise Exception("j should be a single int")

        # Transform to correct Fourier status if necessary
        if target_fourier_status is not None:
            WT.set_fourier_status(target_fourier_status, inplace=True)

        return WT

    ###########################################################################
    @staticmethod
    def downsample(
        data, dg_out, normalize=True, inplace=True, target_fourier_status=True, **kwargs
    ):
        """
        Downgrade the data array to dg_out resolution by cropping in Fourier space.

        Parameters
        ----------
        data : STL_2D_FFT_Torch
            Data object to be downgraded (currently at dg_in resolution).
        dg_out : int
            Target resolution after downgrading.
        normalize : bool
            If True, normalize the output in fourier space to keep the same real mean as input.
        inplace : bool
            If True, modifies data in-place. If False, returns a new instance.
        target_fourier_status : bool
            If True, output is in Fourier space.
            If False, output is in real space and normalized to keep the same real mean as input.

        Notes
        -----
        - To remain consistent, if you downgrade an image that was already downgraded,
        it is recommended to keep the output in the same domain (Fourier or real)
        as the previous data. Otherwise, normalization issues may appear.

        Returns
        -------
        STL_2D_FFT_Torch
            Data downgraded to dg_out resolution.
        """
        dg_in = data.dg

        if dg_out < dg_in:
            raise ValueError("dg_out should be greater than or equal to dg_in.")

        # Prepare target data object
        data = data.copy(empty=False) if not inplace else data

        # Compute input and output shapes
        in_shape = data.array.shape
        factor = 2 ** (dg_out - dg_in)
        out_shape = (in_shape[-2] // factor, in_shape[-1] // factor)

        # Ensure data is in Fourier space
        data.set_fourier_status(target_fourier_status=True, inplace=True)
        data_fft = torch.fft.fftshift(data.array, dim=(-2, -1))

        # Determine minimal crop size and keep original image ratio
        min_x, min_y = 8, 8
        if data.N0[0] > data.N0[1]:
            min_x = int(min_x * data.N0[0] / data.N0[1])
        elif data.N0[1] > data.N0[0]:
            min_y = int(min_y * data.N0[1] / data.N0[0])

        dx = max(min_x, out_shape[0])
        dy = max(min_y, out_shape[1])

        # Compute crop indices
        center_x, center_y = in_shape[-2] // 2, in_shape[-1] // 2
        half_dx, half_dy = dx // 2, dy // 2

        # Crop in Fourier space
        cropped_fft = data_fft[
            ...,
            center_x - half_dx : center_x + half_dx,
            center_y - half_dy : center_y + half_dy,
        ]

        # Assign cropped array back, inverse shift
        data.array = torch.fft.ifftshift(cropped_fft, dim=(-2, -1))
        if normalize:
            data.array *= 1 / factor
        data.dg = dg_out

        # Optionally convert back to real space with normalization
        if not target_fourier_status:
            data.set_fourier_status(target_fourier_status=False, inplace=True)

        return data


class CS_operator_2D_FFT_torch:
    """
    Class whose instances correspond to a cross spectrum operator for 2D FFT data.
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
        self.J = int(np.log2(min(shape))) - 2 if J is None else J
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)
        self.get_crop_border_size_method = get_crop_border_size_method

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
    def apply(
        self, data, compute_cross_spectrum_matrix=None, get_crop_border_size_method=None
    ):
        """
        Compute the power spectrum of the input data array attribute.

        Parameters
        ----------
        - data : STL_2D_FFT_Torch
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
        if type(data).__name__ != "STL_2D_FFT_Torch":
            raise Exception(
                f"Data should be a STL_2D_FFT_Torch instance, got {type(data)}"
            )
        if self.shape != data.N0:
            raise Exception("Data shape does not match operator shape")
        if data.dg != 0:
            raise Exception("Data dg must be 0 for power spectrum computation")
        if data.array.isnan().any():
            raise ValueError(
                "Data array contains NaN values, cannot compute power spectrum"
            )
        if self.device != data.device:
            raise Exception("Data device does not match operator device")

        get_crop_border_size_method = (
            self.get_crop_border_size_method
            if get_crop_border_size_method is None
            else get_crop_border_size_method
        )

        # Ensure data is in Fourier space
        l_data = data.set_fourier_status(
            target_fourier_status=True, inplace=False
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
        l_data.set_fourier_status(
            target_fourier_status=False, inplace=True
        )  # [Nb, Nc, N, M]
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

        # return cross_spectrum, cross_product_bin, cross_product_bin_real
        return cross_spectrum  # [Nb, Nc, Nc, n_bins]

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
