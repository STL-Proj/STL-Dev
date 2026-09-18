# -*- coding: utf-8 -*-
"""
Torch backend for STL.

Provides a minimal API used across the codebase, with optional device
selection (CPU / GPU) via a `device` argument.

Exposed functions
-----------------
- from_numpy(x, device=None, dtype=None)
- zeros(shape, device=None, dtype=torch.float32)
- mean(x, dim)
- dim(x)
- shape(x, axis=None)
- nan
"""

import numpy as np
import torch

# ---------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------

# Global default device: GPU if available, else CPU
_DEFAULT_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else (torch.device("mps") if torch.backends.mps.is_available() else "cpu")
)
_DEFAULT_DTYPE = (
    torch.float64 if _DEFAULT_DEVICE != torch.device("mps") else torch.float32
)
_DEFAULT_COMPLEX_DTYPE = (
    torch.complex128 if _DEFAULT_DTYPE == torch.float64 else torch.complex64
)


def _get_device(device=None) -> torch.device:
    """
    Resolve a device spec into a torch.device.

    device:
        - None        -> _DEFAULT_DEVICE
        - "gpu"       -> "cuda" if available, else ("mps" if available, else "cpu")
        - "cuda", "cuda:0", "cpu", ... -> passed to torch.device
        - torch.device -> returned as-is
    """
    if device is None:
        return _DEFAULT_DEVICE

    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        d = device.lower()
        if d in ("gpu", "cuda", "mps"):
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else (
                    torch.device("mps") if torch.backends.mps.is_available() else "cpu"
                )
            )
        if d == "cpu":
            return torch.device("cpu")
        # Let torch.device handle strings like "cuda:0"
        return torch.device(device)

    # Fallback: let torch.device figure it out / raise
    return torch.device(device)


def _get_dtype(dtype=None, device=None) -> torch.dtype:
    """
    Adapts input dtype to device.

    Indeed, so far, MPS is not compatible with float64.
    So if input dtype is float64 and device is MPS, return float32.

    :param dtype: input torch dtype
    :param device: input torch device
    :return: closest torch dtype compatible with device
    :rtype: torch dtype
    """
    dtype = _DEFAULT_DTYPE if dtype is None else dtype
    if device == torch.device("mps"):
        if dtype == torch.float64:
            print(
                "WARNING: torch.float64 not supported on MPS device. Switching to torch.float32."
            )
            dtype = torch.float32
        else:
            pass  # will probably need to be handled in the future
    return dtype


# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------


def _from_numpy(x: np.ndarray, device=None, dtype=None):
    """
    Convert a NumPy array to a torch.Tensor on the requested device.

    Parameters
    ----------
    x : np.ndarray
    device : None, str, or torch.device
        If "gpu"/"cuda"/"mps" -> CUDA (if available), else MPS (if available), otherwise CPU.
        If "cpu"       -> CPU.
        If None        -> default device (_DEFAULT_DEVICE).
    dtype : torch.dtype or None
        If not None, cast to this dtype.
    """
    t = torch.from_numpy(x)
    dtype = t.dtype if dtype is None else dtype
    device = _get_device(
        device
    )  # returns device if device is not None else finds a suitable device
    fix_dtype = _get_dtype(
        dtype=dtype, device=device
    )  # matches the input array dtype to the device
    return t.to(device=device, dtype=fix_dtype)


def to_torch_tensor(array):
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
    if isinstance(array, np.ndarray):
        return _from_numpy(array)
    elif isinstance(array, torch.Tensor):
        device = _get_device()  # Choose device: use GPU if available, otherwise CPU
        fix_dtype = _get_dtype(
            dtype=array.dtype, device=device
        )  # matches the input tensor dtype to the device
        return array.to(device=device, dtype=fix_dtype)
    else:
        raise TypeError(f"Unsupported array type: {type(array)}")


def zeros(shape, device=None, dtype=None):
    """
    Return a tensor of zeros on CPU or GPU depending on `device`.

    Parameters
    ----------
    shape : tuple or list
    device : None, str, or torch.device
    dtype : torch.dtype
    """
    device = _get_device(device)
    fix_dtype = _get_dtype(dtype=dtype, device=device)
    return torch.zeros(shape, dtype=fix_dtype, device=device)


def ones(shape, device=None, dtype=None):
    """
    Return a tensor of ones on CPU or GPU depending on `device`.

    Parameters
    ----------
    shape : tuple or list
    device : None, str, or torch.device
    dtype : torch.dtype
    """
    device = _get_device(device)
    fix_dtype = _get_dtype(dtype=dtype, device=device)
    return torch.ones(shape, dtype=fix_dtype, device=device)


def eye(n, device=None, dtype=None):
    """
    Return a tensor of ones on CPU or GPU depending on `device`.

    Parameters
    ----------
    shape : tuple or list
    device : None, str, or torch.device
    dtype : torch.dtype
    """
    device = _get_device(device)
    fix_dtype = _get_dtype(dtype=dtype, device=device)
    return torch.eye(n, dtype=fix_dtype, device=device)


def maskmean(x, dim=(-2, -1), mask=None):
    """
    Compute the mean of x along given dims, optionally masked.
    If mask is given, assumes mask.shape == x.shape[-mask.ndim:],
    flattens the masked dimensions and makes sure these were included in the input: dim.

    Args:
        x: input tensor
        mask: boolean tensor, same shape as the last dimensions of x
        dim: int or tuple of ints along which to compute the mean.

    Returns:
        Tensor with the mean
    """
    if mask is None:
        return x.mean() if dim is None else x.mean(dim=dim)
    else:
        # Accept any set of trailing dimensions, so that the same helper serves
        # planar data (dim=(-2,-1)) and HEALPix data (dim=-1).
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        assert sorted(dims) == list(
            range(-len(dims), 0)
        ), f"masked mean requires trailing dims, got {dim}"
        dim = dims
        assert (
            mask.shape[-len(dim) :] == x.shape[-len(dim) :]
        )  # check mask shape matches x shape on masked dims
        # assert x[..., ~mask].isnan().sum() == 0 ############################################### sanity check to remove in the future
        x_masked = torch.where(~mask, x, 0.0)
        count = (
            (~mask).sum() if dim is None else (~mask).sum(dim=dim)
        )  ########################################## can be improved in the future by normalizing masks once for all beforehand
        # assert count != 0, "mask is full of NaNs, cannot compute mean"
        return (x_masked.sum() if dim is None else x_masked.sum(dim=dim)) / count


def dim(x) -> int:
    """
    Return the number of dimensions of a tensor-like object.
    """
    if hasattr(x, "dim"):
        return x.dim()
    return np.array(x).ndim


def shape(x, axis=None):
    """
    Return the shape of `x`, or the size along a given axis.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
    axis : int or None
        If None, return the full shape tuple.
        Otherwise, return the size along the given axis.
    """
    s = x.shape
    return s if axis is None else s[axis]


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Scalar NaN; e.g. bk.zeros(...)+bk.nan -> NaN-filled tensor
nan = torch.nan
