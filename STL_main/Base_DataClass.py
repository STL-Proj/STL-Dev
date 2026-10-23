import copy as cp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from hashlib import new
from typing import Any, ClassVar, Optional

import numpy as np
import torch

from STL_main.torch_backend import to_torch_tensor


@dataclass
class Base_DataClass(ABC):
    """
    Base_DataClass parent class.

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Input 2D array (NumPy or PyTorch tensor).
    dg : int, optional
        Data resolution. If None, set to 0.
    N0 : tuple of int, optional
        Original size of the array. Required if dg is provided.
    conv_history : dict, optional
        History of convolutions applied to the data.
    pbc : bool
        Periodic boundary conditions flag. Must be specified.
    """

    # Common class constant(s) (shall be defined in child classes)
    DT: ClassVar[str]
    ndim: ClassVar[int] = (
        2  # number of intrinsic dimensions of the data (ie dimensions on which convolutions are applied, excluding batch and channel dimensions)
    )
    wavelet_op_class: ClassVar[
        Any
    ]  # Class variable to hold the wavelet operator class, to be defined in child classes
    cs_op_class: ClassVar[
        Any
    ]  # Class variable to hold the cross spectrum operator class, to be defined in child classes

    # Common instance attributes
    array: (
        torch.Tensor
    )  # Other types will be converted to torch.Tensor in __post_init__ if possible
    pbc: Optional[bool] = None
    dg: Optional[int] = None
    N0: Optional[tuple[int, ...]] = None
    conv_history: list[int] = field(default_factory=list)  # empty list by default
    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)

    def __post_init__(self):

        if not hasattr(self, "DT"):
            raise ValueError("Child class must define class attribute 'DT'")

        self.array = self._to_array(self.array)

        if self.dg is None:
            self.dg = 0
            self.N0 = self.array.shape[-self.__class__.ndim :]

        if self.dg is not None and self.N0 is None:
            raise ValueError("dg is given, N0 should not be None")

        assert self.__class__.ndim == len(
            self.N0
        ), f"ndim ({self.__class__.ndim}) must match the length of N0 ({len(self.N0)})"

        self.device = self.array.device
        self.dtype = self.array.dtype

    ###########################################################################
    @staticmethod
    def _to_array(array):
        """
        Transform input array (NumPy or PyTorch) into a PyTorch tensor.

        Parameters
        ----------
        array : np.ndarray, torch.Tensor or list
            Input array to be converted.

        Returns
        -------
        torch.Tensor
            Converted PyTorch tensor.
        """
        if array is None:
            raise ValueError("Input array should not be None")
        else:
            # Transformation en torch.Tensor
            return to_torch_tensor(array)

    ###########################################################################
    def copy(self, empty=False):
        """
        Copy an instance of the class.

        If empty=True, the array attribute will be set to None.

        Parameters
        ----------
        empty : bool
            If True, set array to None.

        Returns
        ----------
        instance of self.__class__
            A deep copy of the object, including all attributes.
            If called from a subclass, returns an instance of the subclass.
        """

        new = object.__new__(self.__class__)  # create a new instance of the same class

        # Copy metadata
        for k, v in self.__dict__.items():
            if k != "array":
                if isinstance(v, (list, dict)):
                    setattr(new, k, cp.deepcopy(v))  # deep copy for mutable types
                else:
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
    # TODO: shall we return an instance of self.__class__ or a simple array? (attribute N0 would not be relevant then)
    def __getitem__(self, key):
        """
        To slice directly the array attribute.

        Parameters
        ----------
        - key : int or slice
            Slicing key.

        Returns
        -------
        instance of self.__class__
            A new instance of the class with the sliced array.
            If called from a subclass, returns an instance of the subclass.
        """
        new = self.copy(empty=False)
        new.array = self.array[key]

        return new

    ###########################################################################
    @abstractmethod
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the data.
        Must be implemented in each child class.
        """
        pass

    ###############################################################################
    @abstractmethod
    def get_wavelet_op(self):
        """
        Abstract method.
        Must be implemented by the child class to return the specific
        WaveletOperator class for that child.
        """
        pass

    #######################################################################################
    # ------- Pre/Post-processing functions -------
    def apply_nyquist_filter(self, inplace=False, plot=False):
        """
        Apply a low-pass filter to self.array, keeping only frequencies within the Nyquist radius.
        The filter is applied along the last self.ndim dimensions.

        Parameters
        ----------
        self : Base_DataClass
            The instance of the class to which the method is applied.
        plot : bool, optional
            If True, plot the filtered tensor in the frequency domain.

        Returns
        -------
        instance of self.__class__
            Filtered tensor in real space of the same shape as input, with high frequencies removed
        """
        assert (
            not self.array.isnan().any()
        ), "Cannot apply Nyquist filter on map with NaNs."

        dim = tuple(
            range(-self.__class__.ndim, 0)
        )  # dimensions to apply the Fourier transform
        print(
            f"Applying Nyquist filter on dimensions {dim} of the array with shape {self.array.shape}"
        )

        if dim == (-2, -1):

            # Compute frequency grids
            N, M = self.array.shape[-2:]
            fx = N * torch.fft.fftfreq(N, d=1.0, device=self.array.device)
            fy = M * torch.fft.fftfreq(M, d=1.0, device=self.array.device)
            FX, FY = torch.meshgrid(fx, fy, indexing="ij")

            # Create low-pass mask based on Nyquist radius
            r2 = FX**2 + FY**2
            nyquist_radius = min(N, M) / 2
            mask = r2 <= nyquist_radius**2

            # Apply mask in Fourier space
            tensor_fft = torch.fft.fft2(self.array, dim=dim)
            tensor_fft[..., ~mask] = 0.0

            # Inverse Fourier transform to get the filtered tensor in real space
            tensor_filtered = torch.fft.ifft2(tensor_fft, dim=dim)

            if not self.array.is_complex():
                tensor_filtered = tensor_filtered.real

            if plot:
                import matplotlib.pyplot as plt

                plt.imshow(
                    np.abs(
                        torch.fft.fftshift(torch.fft.fft2(self.array[0, 0]))
                        .cpu()
                        .numpy()
                    ),
                    cmap="plasma",
                    norm="log",
                )
                plt.colorbar()
                plt.title("Nyquist filtered tensor (Fourier)")
                plt.show()

        elif dim == (-1,):
            # Compute frequency grids
            (N,) = self.array.shape[-1:]
            FX = N * torch.fft.fftfreq(N, d=1.0, device=self.array.device)

            # Create low-pass mask based on Nyquist radius
            r2 = FX**2
            nyquist_radius = N / 2
            mask = r2 <= nyquist_radius**2

            # Apply mask in Fourier space
            tensor_fft = torch.fft.fft(self.array, dim=dim[0])
            tensor_fft[..., ~mask] = 0.0

            # Inverse Fourier transform to get the filtered tensor in real space
            tensor_filtered = torch.fft.ifft(tensor_fft, dim=dim[0])

            if not self.array.is_complex():
                tensor_filtered = tensor_filtered.real

            if plot:
                import matplotlib.pyplot as plt

                plt.plot(
                    np.abs(
                        torch.fft.fftshift(torch.fft.fft(self.array[0, 0]))
                        .cpu()
                        .numpy()
                    ),
                )
                plt.yscale("log")
                plt.colorbar()
                plt.title("Nyquist filtered tensor (Fourier)")
                plt.show()
        else:
            raise NotImplementedError("Dim must be for now either (-2,-1) or (-1,)")

        data_filtered = self.copy(empty=True) if not inplace else self
        data_filtered.array = tensor_filtered

        return data_filtered

    # TODO: add get_ST_op and get_PS_op when instance methods are implemented in STL_2D_Kernel_Torch
