#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topological / morphological statistics operators for 2D planar STL data.

Classes
-------
MinkowskiOperator2D
    Three 2D Minkowski functionals (area, perimeter, Euler characteristic).
PeakCountOperator2D
    Soft local-extrema (peaks / valleys) counts.
BettiCurveOperator2D
    Differentiable Betti curves β₀(t), β₁(t) and Euler characteristic χ(t).

All operators follow the same interface: build once, then call the main
method on any compatible ``STL_2D_Kernel_Torch`` instance.
"""

import math

import torch
import torch.nn.functional as F

from STL_main.torch_backend import (
    _DEFAULT_DEVICE,
    _DEFAULT_DTYPE,
    _get_device,
    _get_dtype,
)

###############################################################################
# ── Data compatibility helper ────────────────────────────────────────────────

def _check_data(data, shape):
    """Accept any object that exposes .array, .N0, .pbc (duck typing)."""
    for attr in ("array", "N0", "pbc"):
        if not hasattr(data, attr):
            raise TypeError(
                f"data must expose .array, .N0 and .pbc attributes "
                f"(got {type(data).__name__!r} which is missing .{attr})"
            )
    if shape != data.N0:
        raise ValueError(f"Operator shape {shape} != data.N0 {data.N0}")


###############################################################################
# ── Internal helpers ──────────────────────────────────────────────────────────
###############################################################################


def _gaussian_kernel_2d(
    sigma: float,
    truncate: float = 3.5,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return a [1, 1, K, K] isotropic Gaussian kernel."""
    radius = max(1, int(truncate * sigma + 0.5))
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]  # [K, K]
    return kernel[None, None]  # [1, 1, K, K]


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
    return F.conv2d(x, kernel).squeeze(1)  # [B, N, M]


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
        ``'dog'`` — Difference-of-Gaussians band-pass |G(σ/2) − G(σ)|.
    wavelet_op : callable or None
        Reserved for future custom wavelet.
    """
    if wavelet_op is not None:
        raise NotImplementedError(
            "Custom wavelet_op is not yet implemented. Use wavelet_op=None."
        )
    padding_mode = "circular" if pbc else "replicate"
    sigma = 2.0 ** (j + 1)
    if scale_mode == "smooth":
        return _gaussian_filter_2d(flat, sigma, padding_mode=padding_mode)
    elif scale_mode == "dog":
        g_lo = _gaussian_filter_2d(flat, sigma / 2, padding_mode=padding_mode)
        g_hi = _gaussian_filter_2d(flat, sigma, padding_mode=padding_mode)
        return (g_lo - g_hi).abs()
    else:
        raise ValueError(f"scale_mode must be 'smooth' or 'dog', got '{scale_mode}'")


# ── Minkowski helpers ─────────────────────────────────────────────────────────


def _mink2d_as_threshold(threshold, img: torch.Tensor) -> torch.Tensor:
    """Cast threshold to a tensor broadcastable on img [B, N, N]."""
    B = img.shape[0]
    t = (
        threshold
        if isinstance(threshold, torch.Tensor)
        else torch.tensor(threshold, dtype=img.dtype, device=img.device)
    )
    if t.ndim == 1 and t.shape[0] == B:
        t = t[:, None, None]
    torch.broadcast_shapes(t.shape, img.shape)
    return t


def _mink2d_functionals(
    img: torch.Tensor,
    threshold=None,
    temperature: float = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Three 2D Minkowski functionals for a batch [B, N, N].

    Returns {'W0': [B], 'W1': [B], 'W2': [B]}, fully differentiable.
    """
    B, N, M = img.shape
    if threshold is not None:
        img = torch.sigmoid(temperature * (img - _mink2d_as_threshold(threshold, img)))

    W0 = img.mean(dim=(-2, -1))

    dh = (img[:, :, 1:] - img[:, :, :-1]).abs()
    dv = (img[:, 1:, :] - img[:, :-1, :]).abs()
    W1 = (dh.sum(dim=(-2, -1)) + dv.sum(dim=(-2, -1))) / (N * N)

    Q1 = img.sum(dim=(-2, -1))
    Qh = (img[:, :, :-1] * img[:, :, 1:]).sum(dim=(-2, -1))
    Qv = (img[:, :-1, :] * img[:, 1:, :]).sum(dim=(-2, -1))
    Qf = (
        img[:, :-1, :-1] * img[:, :-1, 1:] * img[:, 1:, :-1] * img[:, 1:, 1:]
    ).sum(dim=(-2, -1))
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


# ── Peak-count helpers ────────────────────────────────────────────────────────


def _neighborhood_extrema(
    img: torch.Tensor,
    mode: str = "max",
    connectivity: int = 8,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """
    Max or min over the local K-connectivity neighbourhood, excluding center.

    img          : [B, N, M]
    mode         : 'max' or 'min'
    connectivity : 4 (N/E/S/W) or 8 (+ diagonals)
    Returns      : [B, N, M]
    """
    B, N, M = img.shape
    padded = F.pad(img.unsqueeze(1), (1, 1, 1, 1), mode=padding_mode).squeeze(1)

    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if connectivity == 4:
        shifts = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    neighbors = torch.stack(
        [padded[:, 1 + di : 1 + di + N, 1 + dj : 1 + dj + M] for di, dj in shifts],
        dim=0,
    )  # [K, B, N, M]

    return neighbors.amax(dim=0) if mode == "max" else neighbors.amin(dim=0)


def _soft_peaks(
    img: torch.Tensor,
    temperature: float,
    connectivity: int = 8,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """Soft local-maximum indicator. Returns [B, N, M]."""
    max_nbr = _neighborhood_extrema(img, "max", connectivity, padding_mode)
    return torch.sigmoid(temperature * (img - max_nbr))


def _soft_valleys(
    img: torch.Tensor,
    temperature: float,
    connectivity: int = 8,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """Soft local-minimum indicator. Returns [B, N, M]."""
    min_nbr = _neighborhood_extrema(img, "min", connectivity, padding_mode)
    return torch.sigmoid(temperature * (min_nbr - img))


def _threshold_weighted_sum(
    img: torch.Tensor,       # [B, N, M]
    indicator: torch.Tensor, # [B, N, M]
    thresholds: torch.Tensor,# [B, T]
    temperature: float,
    above: bool = True,
) -> torch.Tensor:           # [B, T]
    """
    Weighted sum: for each threshold t, accumulate indicator over active pixels.
    Active = soft excursion-set mask.  Result normalised by N*M.
    """
    B, N, M = img.shape
    T = thresholds.shape[1]
    sign = 1.0 if above else -1.0
    active = torch.sigmoid(
        sign * temperature * (img.unsqueeze(1) - thresholds.view(B, T, 1, 1))
    )  # [B, T, N, M]
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
# ── Operators ─────────────────────────────────────────────────────────────────
###############################################################################


class MinkowskiOperator2D:
    """
    Minkowski functional operator for 2D planar STL data.

    Computes the three 2D Minkowski functionals — area (W0), perimeter (W1),
    and Euler characteristic (W2) — in a differentiable manner via soft
    sigmoid thresholding.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape ``(Nx, Ny)`` of the input maps.  Must match ``data.N0``.
    thresholds : Tensor [T], Tensor [B, T], or None
        Default threshold levels.  ``None`` = no thresholding.
    temperature : float
        Sigmoid sharpness.  Higher → closer to hard binary thresholding.
    J : int
        Number of dyadic scales (1 = full resolution; >1 = multi-scale DoG).
    scale_mode : str
        ``'smooth'`` or ``'dog'``.
    wavelet_op : callable or None
        Reserved for future use.
    device, dtype

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

    def minkowski(self, data, thresholds=None, temperature=None):
        """
        Compute the three Minkowski functionals of 2D planar data.

        Parameters
        ----------
        data : STL_2D_Kernel_Torch
        thresholds : Tensor [T] | Tensor [B, T] | None
        temperature : float or None

        Returns
        -------
        dict[str, Tensor]
            Keys ``'W0'``, ``'W1'``, ``'W2'``.

            - J=1, no thresholds : ``[Nb, Nc]``
            - J=1, thresholds T  : ``[Nb, Nc, T]``
            - J>1, no thresholds : ``[Nb, Nc, J]``
            - J>1, thresholds T  : ``[Nb, Nc, J, T]``
        """
        _check_data(data, self.shape)

        thresholds = thresholds if thresholds is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature

        arr = data.array
        if torch.is_complex(arr):
            arr = arr.abs()
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[None, :, :, :]
        Nb, Nc, Nx, Ny = arr.shape
        flat = arr.reshape(Nb * Nc, Nx, Ny)
        pbc = getattr(data, "pbc", True)

        def _compute_one(band):
            if thresholds is None:
                mf = _mink2d_functionals(band, temperature=temperature)
                return {k: v.view(Nb, Nc) for k, v in mf.items()}
            t = torch.as_tensor(thresholds, dtype=band.dtype, device=band.device)
            if t.ndim == 1:
                t = t.unsqueeze(0).expand(Nb * Nc, t.shape[0])
            elif t.ndim == 2:
                assert t.shape[0] == Nb * Nc
            else:
                raise ValueError(f"thresholds must be 1-D or 2-D, got {t.ndim}-D")
            T = t.shape[1]
            curves = _mink2d_curves(band, t, temperature=temperature)
            return {k: v.view(Nb, Nc, T) for k, v in curves.items()}

        if self.J == 1:
            return _compute_one(flat)

        scale_results = [
            _compute_one(
                _extract_scale_flat(
                    flat, j, pbc=pbc,
                    wavelet_op=self.wavelet_op, scale_mode=self.scale_mode,
                )
            )
            for j in range(self.J)
        ]
        return {
            k: torch.stack([sr[k] for sr in scale_results], dim=2)
            for k in scale_results[0]
        }


###############################################################################


class PeakCountOperator2D:
    """
    Soft local-extrema count operator for 2D planar STL data.

    Parameters
    ----------
    shape : tuple of int
    thresholds : Tensor [T] | Tensor [B, T] | None
    temperature : float
    connectivity : int  — 4 or 8
    J : int
    scale_mode : str
    wavelet_op : callable or None
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
        self.shape = shape
        self.thresholds = thresholds
        self.temperature = temperature
        self.connectivity = connectivity
        self.J = J
        self.scale_mode = scale_mode
        self.wavelet_op = wavelet_op
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

    def _prepare(self, data):
        _check_data(data, self.shape)
        arr = data.array.abs() if torch.is_complex(data.array) else data.array
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        pbc = getattr(data, "pbc", True)
        pmode = "circular" if pbc else "replicate"
        return arr.reshape(Nb * Nc, Nx, Ny), pmode, Nb, Nc

    def _count_one(self, band, mode, thresholds, temperature, pmode, Nb, Nc):
        B, N, M = band.shape
        key = "peaks" if mode == "peaks" else "valleys"
        if mode == "peaks":
            indicator = _soft_peaks(band, temperature, self.connectivity, pmode)
            above = True
        else:
            indicator = _soft_valleys(band, temperature, self.connectivity, pmode)
            above = False

        if thresholds is None:
            count = indicator.mean(dim=(-2, -1))
            return {key: count.view(Nb, Nc)}

        t = _normalise_thresholds(thresholds, B, band.device, band.dtype)
        counts = _threshold_weighted_sum(band, indicator, t, temperature, above=above)
        return {key: counts.view(Nb, Nc, t.shape[1])}

    def peaks(self, data, thresholds=None, temperature=None):
        """
        Soft peak (local-maximum) count, optionally conditioned on threshold.

        Returns dict['peaks'] of shape:
        - J=1, no thresholds : ``[Nb, Nc]``
        - J=1, thresholds T  : ``[Nb, Nc, T]``
        - J>1               : ``[Nb, Nc, J]`` or ``[Nb, Nc, J, T]``
        """
        flat, pmode, Nb, Nc = self._prepare(data)
        thresholds = thresholds if thresholds is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature
        pbc = pmode == "circular"

        if self.J == 1:
            return self._count_one(flat, "peaks", thresholds, temperature, pmode, Nb, Nc)

        scales = [
            self._count_one(
                _extract_scale_flat(flat, j, pbc=pbc,
                                    wavelet_op=self.wavelet_op,
                                    scale_mode=self.scale_mode),
                "peaks", thresholds, temperature, pmode, Nb, Nc,
            )
            for j in range(self.J)
        ]
        return {"peaks": torch.stack([s["peaks"] for s in scales], dim=2)}

    def valleys(self, data, thresholds=None, temperature=None):
        """
        Soft valley (local-minimum) count.  Same return-shape as :meth:`peaks`.
        """
        flat, pmode, Nb, Nc = self._prepare(data)
        thresholds = thresholds if thresholds is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature
        pbc = pmode == "circular"

        if self.J == 1:
            return self._count_one(flat, "valleys", thresholds, temperature, pmode, Nb, Nc)

        scales = [
            self._count_one(
                _extract_scale_flat(flat, j, pbc=pbc,
                                    wavelet_op=self.wavelet_op,
                                    scale_mode=self.scale_mode),
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

    - **β₀(t)** ≈ soft peak count above t
    - **χ(t)**  = Minkowski W2(t)
    - **β₁(t)** = β₀(t) − χ(t)

    Parameters
    ----------
    shape : tuple of int
    thresholds : Tensor [T] | Tensor [B, T]   (required)
    temperature : float
    connectivity : int  — 4 or 8
    J : int
    scale_mode : str
    wavelet_op : callable or None
    device, dtype

    Examples
    --------
    >>> t  = torch.linspace(0.1, 0.9, 16)
    >>> op = BettiCurveOperator2D(shape=(128, 128), thresholds=t)
    >>> out = op.betti(data)
    >>> # out['beta0'].shape == [Nb, Nc, 16]

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
        self.shape = shape
        self.thresholds = thresholds
        self.temperature = temperature
        self.connectivity = connectivity
        self.J = J
        self.scale_mode = scale_mode
        self.wavelet_op = wavelet_op
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

    def _betti_one(self, band, thresholds, temperature, pmode, Nb, Nc):
        B, N, M = band.shape
        t = _normalise_thresholds(thresholds, B, band.device, band.dtype)
        T = t.shape[1]

        indicator = _soft_peaks(band, temperature, self.connectivity, pmode)
        beta0 = _threshold_weighted_sum(band, indicator, t, temperature, above=True)
        chi   = _mink2d_curves(band, t, temperature=temperature)["W2"]
        beta1 = beta0 - chi

        return {
            "beta0": beta0.view(Nb, Nc, T),
            "beta1": beta1.view(Nb, Nc, T),
            "chi":   chi.view(Nb, Nc, T),
        }

    def betti(self, data, thresholds=None, temperature=None):
        """
        Compute Betti curves β₀(t), β₁(t) and Euler characteristic χ(t).

        Returns
        -------
        dict with keys ``'beta0'``, ``'beta1'``, ``'chi'``:

        - J=1 : each ``[Nb, Nc, T]``
        - J>1 : each ``[Nb, Nc, J, T]``
        """
        _check_data(data, self.shape)

        arr = data.array.abs() if torch.is_complex(data.array) else data.array
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        flat = arr.reshape(Nb * Nc, Nx, Ny)
        pbc  = getattr(data, "pbc", True)
        pmode = "circular" if pbc else "replicate"

        thresholds = thresholds if thresholds is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature

        if thresholds is None:
            raise ValueError("thresholds must be provided for Betti curves.")

        if self.J == 1:
            return self._betti_one(flat, thresholds, temperature, pmode, Nb, Nc)

        scale_results = [
            self._betti_one(
                _extract_scale_flat(flat, j, pbc=pbc,
                                    wavelet_op=self.wavelet_op,
                                    scale_mode=self.scale_mode),
                thresholds, temperature, pmode, Nb, Nc,
            )
            for j in range(self.J)
        ]
        return {
            k: torch.stack([sr[k] for sr in scale_results], dim=2)
            for k in scale_results[0]
        }
