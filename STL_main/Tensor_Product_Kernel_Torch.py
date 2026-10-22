# TODO:
# - extension to non tensor prod nan mask
# - extension to to tendor prod channels (eg bathy, daylight, etc)
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np
import torch

from STL_main.ST_Operator_Tensor_Product import ST_Operator_TensorProd
from STL_main.STL_2D_Kernel_Torch import (
    CS_operator_2D_Kernel_Torch,
    STL_2D_Kernel_Torch,
    WaveletOperator2Dkernel_torch,
)
from STL_main.torch_backend import (
    maskmean,
    nan,
)


@dataclass
class DimHandler:

    _dim: list[int] = field(init=False)
    _ndim: int = field(init=False)
    _nblock: int = field(init=False)
    _start_dim_buffer: dict[int, list[int]] = field(init=False)

    def __init__(self, dim):
        self._dim = dim
        self._ndim = sum(dim)
        self._nblock = len(dim)
        self._start_dim_buffer = {}  # buffer for start_dim computation

    @property
    def dim(self) -> list[int]:
        return self._dim

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def nblock(self) -> int:
        return self._nblock

    def _compute_start_dim(self, array_ndim) -> list[int]:
        start_dim = [array_ndim - self.ndim]
        for ind in range(1, self.nblock):
            start_dim.append(start_dim[ind - 1] + self.dim[ind - 1])
        return start_dim

    def start_dim(self, array_ndim) -> list[int]:
        if array_ndim not in self._start_dim_buffer:
            self._start_dim_buffer[array_ndim] = self._compute_start_dim(array_ndim)
        return self._start_dim_buffer[array_ndim]


class Wavelet_Operator_TensorProd_Kernel_Torch(WaveletOperator2Dkernel_torch):

    @property
    def dim(self) -> list[int]:
        return self._dim_handler.dim

    @property
    def nblock(self) -> int:
        return self._dim_handler.nblock

    @property
    def ndim(self) -> int:
        return self._dim_handler.ndim

    @staticmethod
    def _get_crop_border_size_from_children(data, wavelet_op):
        return [
            wavelet_op_child._get_crop_border_size_method(
                data._as_data_class_of(block_id), wavelet_op=wavelet_op_child
            )
            for block_id, wavelet_op_child in enumerate(wavelet_op.wavelet_ops)
        ]

    def __init__(
        self,
        data_structure,
        J,
        DT,
        device,
        dtype,
        mask_full_res=None,
        **kwargs,
    ):
        self.data_structure = data_structure
        self._dim_handler = DimHandler(
            dim=[data_class.ndim for data_class in self.data_structure]
        )
        self.start_dim = self._dim_handler.start_dim  # method copy
        self.DT = DT
        self.device = device
        self.dtype = dtype
        self.mask_full_res = mask_full_res

        for key, value in kwargs.items():
            if value is None:
                kwargs[key] = [None] * self.nblock
            assert isinstance(kwargs[key], list), f"{key} must be a list"

        wavelet_ops = []
        for ind, data_class in enumerate(self.data_structure):
            wavelet_op = data_class.wavelet_op_class(
                J[ind],
                device=self.device,
                dtype=self.dtype,
                mask_full_res=self.mask_full_res,
                **{key: kwargs[key][ind] for key in kwargs.keys()},
            )
            # self.DT[ind] is not sent to the wavelet operator, so we check that the DT of the wavelet operator matches the DT of the data class
            assert (
                wavelet_op.DT == self.DT[ind]
            ), f"Wavelet operator DT mismatch: {wavelet_op.DT} != {self.DT[ind]}"
            wavelet_ops.append(wavelet_op)
        self.wavelet_ops = wavelet_ops

        self._get_crop_border_size_method = (
            self.__class__._get_crop_border_size_from_children
        )
        self.J = [wavelet_op.J for wavelet_op in self.wavelet_ops]
        self.L = [wavelet_op.L for wavelet_op in self.wavelet_ops]
        self.WType = [wavelet_op.WType for wavelet_op in self.wavelet_ops]

    def _crop(self, array, border):  # conv_dims dependant
        """
        Crops an array by removing 'border' pixels from each side
        along the last self.__class__.ndim dimensions.

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
        elif max(border) == 0:
            return array
        else:
            slices = [slice(None)] * (array.ndim - self.ndim)

            for block_id in range(self.nblock):
                # handling of borders larger than array can be adapted depending on desired behavior
                block_start_dim = self.start_dim(array.ndim)[block_id]
                block_end_dim = block_start_dim + self.dim[block_id]
                if False:  # conservative handling of borders larger than array
                    assert (
                        min(array.shape[block_start_dim:block_end_dim])
                        > 2 * border[block_id]
                    ), f"Array shape {array.shape[block_start_dim:block_end_dim]} is too small to be cropped with border {border[block_id]}."
                elif True:  # flexible handling of borders larger than array
                    if (
                        min(array.shape[block_start_dim:block_end_dim])
                        <= 2 * border[block_id]
                    ):
                        if not getattr(
                            self, "_border_warning_raised", False
                        ):  # warns the user only once per wavelet operator
                            print(
                                "Warning! Data with shape {:} too small to be cropped with border {:}. Using border={:} instead.".format(
                                    array.detach()
                                    .cpu()
                                    .numpy()
                                    .shape[block_start_dim:block_end_dim],
                                    border[block_id],
                                    (
                                        min(array.shape[block_start_dim:block_end_dim])
                                        - 1
                                    )
                                    // 2,
                                )
                            )
                            self._border_warning_raised = True
                        border[block_id] = (
                            min(array.shape[block_start_dim:block_end_dim]) - 1
                        ) // 2
                else:  # simple handling of borders larger than array: empty maskmean will return nan
                    pass
                # crop the array along the dimensions [block_start_dim:block_end_dim]
                slices += [
                    (
                        slice(border[block_id], -border[block_id])
                        if border[block_id] > 0
                        else slice(None)
                    )
                ] * self.dim[block_id]

            return array[tuple(slices)]

    def cov(
        self,
        data1,
        data2,
        remove_mean=None,
        dim=None,
        keepdim=False,
        specific_channel_pair=None,
    ):  # conv_dims dependant
        """
        Compute the covariance between data1=self and data2 on the last self.ndim
        dimensions.
        """

        for block_id in range(self.nblock):
            if (
                data1.pbc[block_id] is None and len(data1.conv_history[block_id]) > 0
            ) or (
                data2.pbc[block_id] is None and len(data2.conv_history[block_id]) > 0
            ):
                raise ValueError(
                    f"data1.pbc[{block_id}] and data2.pbc[{block_id}] should be specified (True or False)."
                )

        assert data1.dg == data2.dg, "data1 and data2 must have the same resolution."
        dim = (
            dim if dim is not None else tuple(range(-self.ndim, 0))
        )  # default to last self.ndim dimensions
        remove_mean = remove_mean if remove_mean is not None else False

        # finding the appropriate mask
        if self.mask_full_res is None:
            mask = None
        else:
            raise NotImplementedError(
                "mask_full_res is not yet implemented for tensor product wavelet operators."
            )

            # if specific_channel_pair is not None:
            #    c1, c2 = specific_channel_pair
            #    if c1 == c2:
            #        mask = mask[c1, ...]
            #    else:
            #        mask = self._find_mask(data1)[c1, ...] + self._find_mask(data2)[c2, ...]

        border = [
            max(border1, border2)
            for border1, border2 in zip(
                self._get_crop_border_size_method(data=data1, wavelet_op=self),
                self._get_crop_border_size_method(data=data2, wavelet_op=self),
            )
        ]

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
            keepdim=keepdim,
        )

        return cov

    def apply(self, data, j, block_id):
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

        convolved_data = self.wavelet_ops[block_id].apply(
            data=data._as_data_class_of(block_id),
            j=j,
        )

        # convolution inserted a new dimension beteween the spatial dimensions not convolved and convolved,
        # so we need to move it in front of all spatial dimensions to restore the continuity of all spatial dimensions
        # before casting the data back to the original data class
        first_spatial_dim = (
            convolved_data.array.ndim - self.ndim - 1
        )  # -1 because of the new dimension inserted by the convolution
        inserted_dim = convolved_data.array.ndim - self.dim[block_id] - 1

        dims = list(range(convolved_data.array.ndim))
        dims = (
            dims[:first_spatial_dim]
            + dims[inserted_dim : inserted_dim + 1]
            + dims[first_spatial_dim:inserted_dim]
            + dims[inserted_dim + 1 :]
        )

        convolved_data.array = convolved_data.array.permute(dims)

        convolved_data = data._from_data_class_of(
            block_id, convolved_data, inplace=False
        )

        return convolved_data

    def downsample(self, data, dg_out, block_id, inplace=True, replace_nan_value=nan):
        """
        Downsample the data to the dg_out resolution along the specified block.
        Downsampling is done in real space along the specified block dimensions using (successive iterations of, if dg_out - dg > 1) torch.conv2d with stride=2.
        If a mask is provided at full resolution, the downsampling is nan-aware, and sufficiently isolated NaNs can be removed through local averaging.
        """

        downsampled_data = self.wavelet_ops[block_id].downsample(
            data=data._as_data_class_of(block_id),
            dg_out=dg_out,
            inplace=inplace,
            replace_nan_value=replace_nan_value,
        )

        downsampled_data = data._from_data_class_of(
            block_id, downsampled_data, inplace=inplace
        )

        return downsampled_data


class CS_Operator_TensorProd_Kernel_Torch(CS_operator_2D_Kernel_Torch):
    pass


@dataclass
class STL_TensorProd_Kernel_Torch(STL_2D_Kernel_Torch):

    # class attributes
    wavelet_op_class = Wavelet_Operator_TensorProd_Kernel_Torch
    cs_op_class = CS_Operator_TensorProd_Kernel_Torch

    # instance attributes
    #    array: (  # already defined in Base_DataClass
    #        torch.Tensor
    #    )  # Other types will be converted to torch.Tensor in __post_init__ if possible
    data_structure: list[Any] | None = field(
        default=None, metadata={"help": "List of data classes for each block"}
    )
    pbc: Optional[list[bool]] | None = field(
        default=None,
        metadata={"help": "List of periodic boundary condition for each block"},
    )
    dg: Optional[list[int]] | None = field(
        default=None, metadata={"help": "List of downsampling factor for each block"}
    )
    N0: Optional[list[tuple[int, ...]]] | None = field(
        default=None, metadata={"help": "List of shape for each block"}
    )
    conv_history: list[list[int]] | None = field(
        default=None, metadata={"help": "List of convolution history for each block"}
    )

    #    device: torch.device = field(init=False) # already defined in Base_DataClass
    #    dtype: torch.dtype = field(init=False)   # already defined in Base_DataClass

    DT: list[str] = field(init=False)
    _dim_handler: DimHandler = field(init=False)

    @property
    def dim(self) -> list[int]:
        return self._dim_handler.dim

    @property
    def nblock(self) -> int:
        return self._dim_handler.nblock

    @property
    def ndim(self) -> int:
        return self._dim_handler.ndim

    @property
    def start_dim(self) -> list[int]:
        return self._dim_handler.start_dim(self.array.ndim)

    @classmethod
    def _get_frozen_class_from_data_structure(cls, data_structure_arg):
        class FrozenClassFromDataStructure(cls):
            data_structure: ClassVar[list[Any]] = data_structure_arg

        frozen_class_name = f"Frozen[{cls.__name__}]@({' x '.join(data_class.__name__ for data_class in data_structure_arg)})"
        FrozenClassFromDataStructure.__name__ = frozen_class_name
        FrozenClassFromDataStructure.__qualname__ = frozen_class_name

        return FrozenClassFromDataStructure

    def __post_init__(self):
        if self.data_structure is None:
            assert hasattr(
                self.__class__, "data_structure"
            ), "data_structure must be provided either as an instance attribute or as a class attribute"
            self.data_structure = self.__class__.data_structure
        if hasattr(self.__class__, "data_structure"):
            assert (self.__class__.data_structure is None) or (
                self.__class__.data_structure == self.data_structure
            ), f"data_structure at class level: {self.__class__.data_structure} must be None or match the one of instance: {self.data_structure}"
        assert self.data_structure is not None
        assert isinstance(self.data_structure, list)
        assert len(self.data_structure) > 0

        self.array = self._to_array(self.array)

        self.DT = [data_class.DT for data_class in self.data_structure]
        self._dim_handler = DimHandler(
            dim=[data_class.ndim for data_class in self.data_structure]
        )

        if self.pbc is None:
            self.pbc = [None] * self.nblock

        if self.dg is None:
            self.dg = [None] * self.nblock
            self.N0 = [None] * self.nblock

        if self.dg is not None and self.N0 is None:
            raise ValueError("dg is given, N0 should not be None")

        if self.conv_history is None:
            self.conv_history = [[]] * self.nblock

        for ind in range(self.nblock):

            if self.dg[ind] is None:
                self.dg[ind] = 0
                self.N0[ind] = self.array.shape[
                    self.start_dim[ind] : self.start_dim[ind] + self.dim[ind]
                ]

            if self.dg[ind] is not None and self.N0[ind] is None:
                raise ValueError(f"dg[{ind}] is given, N0[{ind}] should not be None")

            assert self.dim[ind] == len(
                self.N0[ind]
            ), f"dim[{ind}]={self.dim[ind]} must match the length of N0[{ind}]={len(self.N0[ind])}"

        self.device = self.array.device
        self.dtype = self.array.dtype

        self.__class__ = self.__class__._get_frozen_class_from_data_structure(
            self.data_structure
        )

    def _as_data_class_of(self, block_id):
        """
        Cast the current instance to the data class of the given block.
        In particular, the data tensor dimensions are permuted in order to send the dimensions of the block to the end.

        Parameters
        ----------
        block_id : int
            The index of the block to convert.

        Returns
        -------
        object
            An instance of the data class of the given block with the same attributes and permuted dimensions.
        """
        data_class = self.data_structure[block_id]

        # Permute the data dimensions of the block to the last dimensions
        block_start_dim = self.start_dim[block_id]
        block_end_dim = block_start_dim + self.dim[block_id]

        dims = list(range(self.array.ndim))
        dims = (
            dims[:block_start_dim]
            + dims[block_end_dim:]
            + dims[block_start_dim:block_end_dim]
        )

        transposed_array = self.array.permute(dims)

        return data_class(
            array=transposed_array,
            pbc=self.pbc[block_id],
            dg=self.dg[block_id],
            N0=self.N0[block_id],
            conv_history=self.conv_history[block_id],
        )

    def _from_data_class_of(self, block_id, data_instance, inplace=False):
        """
        Cast an instance of the data class of the given block to the current instance.
        If inplace is True, the current instance is updated in-place. Otherwise, a new instance is returned.
        In particular, the data tensor dimensions are permuted back to their original order.

        Parameters
        ----------
        block_id : int
            The index of the block to update.
        data_instance : object
            An instance of the data class of the given block with the same attributes and permuted dimensions.
        inplace : bool, optional
            If True, updates the self instance in-place. Default is False.
        """
        assert (
            data_instance.__class__.__name__ == self.data_structure[block_id].__name__
        ), f"data_instance must be of type {self.data_structure[block_id].__name__}, got {data_instance.__class__.__name__}"
        assert (
            data_instance.device == self.device
        ), f"data_instance must be on device {self.device}, got {data_instance.device}"
        # dtype is not checked because the wavelet transform may change the dtype (eg float32 -> complex64)

        # Permute the data dimensions back to their original order
        block_start_dim = data_instance.array.ndim - sum(self.dim[block_id:])

        dims = list(range(data_instance.array.ndim))
        dims = (
            dims[:block_start_dim]
            + dims[-self.dim[block_id] :]
            + dims[block_start_dim : -self.dim[block_id]]
        )

        transposed_array = data_instance.array.permute(dims)

        data = self.copy(empty=True) if not inplace else self

        data.array = transposed_array
        data.dtype = (
            data_instance.dtype
        )  # dtype is updated because the wavelet transform may change the dtype (eg float32 -> complex64)
        data.dg[block_id] = data_instance.dg
        data.conv_history[block_id] = data_instance.conv_history

        return data

    def get_wavelet_op(
        self,
        J=None,
        mask_full_res=None,
        *args,
        **kwargs,
    ):

        if J is None:
            J = [None] * self.nblock

        for block_id in range(self.nblock):
            if J[block_id] is None:
                J[block_id] = int(np.log2(min(self.N0[block_id]))) - 2

        if mask_full_res is None:
            if torch.any(self.array.isnan()):
                mask_full_res = self.__class__(
                    array=self.array.isnan(), data_structure=self.data_structure
                )

        return self.__class__.wavelet_op_class(
            data_structure=self.data_structure,
            J=J,
            DT=self.DT,
            device=self.device,
            dtype=self.dtype,
            mask_full_res=mask_full_res,
            *args,
            **kwargs,
        )

    def get_ST_op(self, *args, **kwargs):

        return ST_Operator_TensorProd(data_example=self, *args, **kwargs)

    ###############################################################################
    def get_CS_op(self, *args, **kwargs):

        raise NotImplementedError


#        return self.__class__.cs_op_class(
#            shape=self.N0, device=self.device, dtype=self.dtype, *args, **kwargs
#        )
