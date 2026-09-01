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

    # Number of trailing axes of `array` describing the pixel grid.
    # 2 for planar maps (..., Nx, Ny), 1 for HEALPix maps (..., Npix), 1 for 1D data.
    # It is used to make all the DT-independent code (ST_Operator, ST_Statistics,
    # Synthesis) agnostic to the geometry of the data.
    NDIM_PIX: ClassVar[int] = 2

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

        if self.N0 is None:
            if self.dg != 0:
                raise ValueError("dg is given, N0 should not be None")
            self.N0 = self._infer_N0(self.array)

        self.device = self.array.device
        self.dtype = self.array.dtype

    ###########################################################################
    @classmethod
    def _infer_N0(cls, array):
        """
        Infer the initial resolution descriptor N0 from the stored array.

        The default implementation returns the NDIM_PIX trailing axes, which is
        the right answer whenever the pixel axes *are* the resolution (planar
        maps). Child classes whose resolution is not the array shape (HEALPix,
        where the resolution is nside and not Npix) must override this.

        Returns
        -------
        tuple of int
            Resolution descriptor. len(N0) is always NDIM_PIX, so that
            DT-independent code can use it to recognize the pixel axes.
        """
        return tuple(int(s) for s in array.shape[-cls.NDIM_PIX :])

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
    def new_like(self, array, **overrides):
        """
        Build a new instance carrying `array` on this object's geometry.

        Everything that describes *where* the data live -- resolution, pixel
        indices, boundary convention -- is inherited, and only the values are
        replaced. This is what lets the DT-independent code (Synthesis in
        particular) create companion fields without knowing which data type it
        is dealing with.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Values of the new instance. Its trailing NDIM_PIX axes must describe
            the same pixel grid as this object.
        **overrides
            Attributes to set on the new instance, e.g. pbc=False.

        Returns
        -------
        instance of self.__class__
        """
        new = self.copy(empty=True)
        new.array = new._to_array(array)
        new.conv_history = []

        for name, value in overrides.items():
            setattr(new, name, value)

        new.device = new.array.device
        new.dtype = new.array.dtype
        return new

    ###########################################################################
    def apply_bandlimit(self, array):
        """
        Remove from `array` the content below the pixel scale.

        Used by the synthesis to start from a band-limited random field. The
        default is a no-op; data types that can define a band limit (a Nyquist
        radius in the plane, a maximum multipole on the sphere) override it.
        """
        return array

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

    # TODO: add get_ST_op and get_PS_op when instance methods are implemented in STL_2D_Kernel_Torch
