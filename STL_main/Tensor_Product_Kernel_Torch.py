# TODO:
# - extension to non tensor prod nan mask
# - extension to to tendor prod channels (eg bathy, daylight, etc)
from dataclasses import dataclass, field
from itertools import product
from typing import Any, ClassVar, Optional

import numpy as np
import torch

import STL_main.torch_backend as bk
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
        # Blocks may have their own default L inside a tensor product (2 for 1D blocks)
        kwargs["L"] = [
            (
                getattr(data_class.wavelet_op_class, "L_tensor_product", None)
                if L_block is None
                else L_block
            )
            for data_class, L_block in zip(
                self.data_structure, kwargs.get("L", [None] * self.nblock)
            )
        ]

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

    def lowpass_dg(self, block_id):
        """
        Resolution of the low-pass filter phi_J of the given block: one downsampling below its coarsest wavelet.
        """
        return self.wavelet_ops[block_id].j_to_dg[self.J[block_id] - 1] + 1

    def lowpass(self, data, block_id):
        """
        Apply the low-pass filter phi_J of the given block to data already downsampled to lowpass_dg(block_id)
        along this block. phi_J is the cascade of the smoothing kernels of these downsamplings, so the data are
        left unchanged: only a new orientation dimension of size 1 is inserted in front of the spatial dimensions,
        as apply does for the orientations of the wavelets.
        """
        if data.dg[block_id] != self.lowpass_dg(block_id):
            raise ValueError(
                f"data.dg[{block_id}]={data.dg[block_id]} must be {self.lowpass_dg(block_id)} to apply the low-pass filter."
            )

        filtered_data = data.copy(empty=True)
        filtered_data.array = data.array.unsqueeze(data.array.ndim - self.ndim)

        return filtered_data

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

    @property
    def dim(self) -> list[int]:
        return self._dim_handler.dim

    @property
    def nblock(self) -> int:
        return self._dim_handler.nblock

    @property
    def ndim(self) -> int:
        return self._dim_handler.ndim

    def __init__(
        self,
        data_structure,
        shape,
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

        if mask_full_res is not None:
            raise NotImplementedError(
                "mask_full_res is not yet implemented for tensor product cross spectrum operators."
            )

        for key, value in kwargs.items():
            if value is None:
                kwargs[key] = [None] * self.nblock
            assert isinstance(kwargs[key], list), f"{key} must be a list"

        cs_ops = []
        for ind, data_class in enumerate(self.data_structure):
            cs_op = data_class.cs_op_class(
                shape=shape[ind],
                DT=self.DT[ind],
                device=self.device,
                dtype=self.dtype,
                **{key: kwargs[key][ind] for key in kwargs.keys()},
            )
            # self.DT[ind] is not sent to the cross spectrum operator, so we check that the DT of the cross spectrum operator matches the DT of the data class
            assert (
                cs_op.DT == self.DT[ind]
            ), f"Cross spectrum operator DT mismatch: {cs_op.DT} != {self.DT[ind]}"
            cs_ops.append(cs_op)

        self.cs_ops = cs_ops

        self.n_bins = [cs_op.n_bins for cs_op in self.cs_ops]

    def _block_crop_masks(self, data, get_crop_border_size_method):
        """
        Per-bin crop mask of each block, as in CS_operator_2D_Kernel_Torch.apply_fft:
        all ones for a periodic block, interior of the per-bin border otherwise.

        Returns
        -------
        list of torch.Tensor
            Crop mask of each block. Shape [n_bins[b], *N0[b]].
        """
        masks = []
        for block_id, cs_op in enumerate(self.cs_ops):
            if data.pbc[block_id]:
                masks.append(
                    torch.ones_like(cs_op.bin_masks, dtype=cs_op.bin_masks.dtype)
                )
                continue
            method = (
                cs_op.get_crop_border_size_method
                if get_crop_border_size_method[block_id] is None
                else get_crop_border_size_method[block_id]
            )
            if method == "flexible_crop":
                border = cs_op.crop_borders  # [n_bins]
            elif method == "largest_crop":
                border = torch.full_like(cs_op.crop_borders, cs_op.crop_borders.max())
            else:
                raise ValueError(f"Invalid get_crop_border_size_method: {method}")
            masks.append(
                cs_op.build_mask_crop(cs_op.bin_masks, border=border).to(
                    dtype=cs_op.bin_masks.dtype
                )
            )
        return masks

    def apply(
        self,
        data,
        compute_cross_spectrum_matrix=None,
        get_crop_border_size_method=None,
        cross_spectrum_method=None,
        **kwargs,
    ):
        """
        Joint cross spectrum over all blocks, binned with the tensor product of the block bin masks:
        CS[c1, c2, j_0, j_1, ...] = mean of X_c1(f) conj(X_c2(f)) weighted by B_0[j_0](f_0) B_1[j_1](f_1) ...,
        where X is the Fourier transform over the dimensions of all blocks (eg the (k, omega) spectrum
        for 2D+1D data). Non-periodic blocks use the per-bin crop of their own cross spectrum operator.

        Returns
        -------
        torch.Tensor
            Cross spectrum values of shape [Nb, Nc, Nc, n_bins[0], ..., n_bins[nblock-1]]
        """
        if not isinstance(get_crop_border_size_method, list):
            get_crop_border_size_method = [
                get_crop_border_size_method
            ] * self.nblock  # default to the same method for all blocks

        if not isinstance(cross_spectrum_method, list):
            cross_spectrum_method = [cross_spectrum_method] * self.nblock
        for block_id, cs_op in enumerate(self.cs_ops):
            method = (
                cs_op.cross_spectrum_method
                if cross_spectrum_method[block_id] is None
                else cross_spectrum_method[block_id]
            )
            if method != "fft":
                raise NotImplementedError(
                    f"Tensor product cross spectrum is only implemented with the fft method, got {method} for block {block_id}."
                )

        # consistency checks
        if list(self.DT) != list(data.DT):
            raise Exception(f"Data Type should be {self.DT}, got {data.DT}.")
        if any(dg != 0 for dg in data.dg):
            raise Exception("Data dg must be 0 for power spectrum computation")
        if any(
            tuple(cs_op.shape) != tuple(N0) for cs_op, N0 in zip(self.cs_ops, data.N0)
        ):
            raise Exception("Data shape does not match operator shape")
        if data.array.isnan().any():
            raise ValueError(
                "Data array contains NaN values, cannot compute power spectrum with fft method."
            )

        x = data.array
        if x.ndim == self.ndim:
            x = x[None, None]  # [Nb=1, Nc=1, ...]
        elif x.ndim == self.ndim + 1:
            x = x[None]  # [Nb=1, Nc, ...]
        elif x.ndim != self.ndim + 2:
            raise NotImplementedError(
                f"Data array has unexpected shape {x.shape} with ndim={self.ndim}."
            )
        Nb, Nc = x.shape[:2]
        fft_dims = tuple(range(-self.ndim, 0))

        x_hat = torch.fft.fftshift(
            torch.fft.fftn(x, norm="ortho", dim=fft_dims), dim=fft_dims
        )  # [Nb, Nc, *N] with the zero frequency at the center, as the bin masks

        compute_cross_spectrum_matrix = (
            torch.eye(Nc, dtype=torch.bool)
            if compute_cross_spectrum_matrix is None
            else compute_cross_spectrum_matrix
        )
        c1, c2 = torch.nonzero(
            compute_cross_spectrum_matrix.to(device=x.device), as_tuple=True
        )  # channel pairs to compute

        # einsum letters: y for batch, z for channel pairs, one lowercase letter per data
        # dimension (block by block) and one uppercase letter per block of bins
        letters = "abcdefghijklmnopqrstuvwx"
        if self.ndim > len(letters):
            raise NotImplementedError(
                "Too many dimensions for einsum. Can be fixed by adding more letters to the einsum letters string."
            )
        block_letters, start = [], 0
        for dim in self.dim:
            block_letters.append(letters[start : start + dim])
            start += dim
        all_letters = "".join(block_letters)
        bin_letters = "".join(chr(ord("A") + b) for b in range(self.nblock))

        bin_masks = [
            cs_op.bin_masks.to(device=x.device, dtype=x_hat.dtype)
            for cs_op in self.cs_ops
        ]  # [n_bins[b], *N0[b]]
        bin_norm = [
            cs_op.bin_masks.sum(dim=tuple(range(1, cs_op.bin_masks.ndim)))
            for cs_op in self.cs_ops
        ]  # [n_bins[b]]
        n_bins = [masks.shape[0] for masks in bin_masks]

        cross_spectrum = torch.full(
            (Nb, Nc, Nc, *n_bins),
            torch.nan,
            dtype=bk._DEFAULT_COMPLEX_DTYPE,
            device=x.device,
        )

        if all(data.pbc):
            # sum over f of X_c1 conj(X_c2) B_0[j_0] B_1[j_1] ... / (sum B_0[j_0]) / (sum B_1[j_1]) ...
            subscripts = (
                f"yz{all_letters},yz{all_letters},"
                + ",".join(
                    bin_letters[b] + block_letters[b] for b in range(self.nblock)
                )
                + f"->yz{bin_letters}"
            )
            cross_vals = torch.einsum(
                subscripts, x_hat[:, c1], torch.conj(x_hat[:, c2]), *bin_masks
            )  # [Nb, n_pairs, *n_bins]
            for b in range(self.nblock):
                shape = [1] * (2 + self.nblock)
                shape[2 + b] = n_bins[b]
                cross_vals = cross_vals / bin_norm[b].to(device=x.device).view(shape)
            cross_spectrum[:, c1, c2] = cross_vals.to(dtype=bk._DEFAULT_COMPLEX_DTYPE)
            return cross_spectrum  # [Nb, Nc, Nc, *n_bins]

        # Non-periodic blocks: as in the 2D operator, the data filtered in each joint bin is correlated
        # with the data on the interior of the domain, cropped by the support of the bin filter.
        crop_masks = [
            mask.to(device=x.device)
            for mask in self._block_crop_masks(data, get_crop_border_size_method)
        ]  # [n_bins[b], *N0[b]]
        x_conj = torch.conj(x[:, c2]).to(dtype=x_hat.dtype)  # [Nb, n_pairs, *N]
        x_hat_c1 = x_hat[:, c1]  # [Nb, n_pairs, *N]
        N_total = float(np.prod([np.prod(N0) for N0 in data.N0]))

        for bins in product(*[range(n) for n in n_bins]):
            filt = torch.einsum(
                ",".join(block_letters) + "->" + all_letters,
                *[bin_masks[b][j] for b, j in enumerate(bins)],
            )
            crop = torch.einsum(
                ",".join(block_letters) + "->" + all_letters,
                *[crop_masks[b][j] for b, j in enumerate(bins)],
            )
            x_bin = torch.fft.ifftn(
                torch.fft.ifftshift(x_hat_c1 * filt, dim=fft_dims),
                norm="ortho",
                dim=fft_dims,
            )
            norm = (N_total / crop.real.sum()) / np.prod(
                [float(bin_norm[b][j]) for b, j in enumerate(bins)]
            )
            cross_spectrum[(slice(None), c1, c2) + bins] = (
                (x_bin * x_conj * crop).sum(dim=fft_dims) * norm
            ).to(dtype=bk._DEFAULT_COMPLEX_DTYPE)

        return cross_spectrum  # [Nb, Nc, Nc, *n_bins]


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

    def apply_nyquist_filter(self, inplace=False, plot=False):
        """
        Apply a low-pass filter to self.array, keeping only frequencies within the Nyquist radius.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies the instance in place. Default is False.
        plot : bool, optional
            If True, plots the filtered tensor in the frequency domain. Default is False.

        Returns
        -------
        instance of self.__class__
            Filtered tensor in real space of the same shape as input, with high frequencies removed.
        """
        data_filtered = self.copy(empty=False) if not inplace else self

        for block_id in range(self.nblock):
            partial_data_filtered = self._as_data_class_of(
                block_id
            ).apply_nyquist_filter(inplace=True, plot=plot)
            data_filtered = data_filtered._from_data_class_of(
                block_id, partial_data_filtered, inplace=True
            )

        return data_filtered

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
    def get_CS_op(self, mask_full_res=None, *args, **kwargs):

        if mask_full_res is None:
            if torch.any(self.array.isnan()):
                mask_full_res = self.__class__(
                    array=self.array.isnan(), data_structure=self.data_structure
                )

        return self.__class__.cs_op_class(
            data_structure=self.data_structure,
            shape=self.N0,
            DT=self.DT,
            device=self.device,
            dtype=self.dtype,
            mask_full_res=mask_full_res,
            *args,
            **kwargs,
        )
