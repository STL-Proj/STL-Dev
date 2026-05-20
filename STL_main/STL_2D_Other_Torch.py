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
LacunarityOperator2D
    Differentiable lacunarity Λ(ε) = E[Q²]/E[Q]² at multiple window sizes.
    Optional circular-disk window for rotation invariance.
    Optional wavelet extension → 'lac_wav' [Nb, Nc, E, J, L].
GLCMOperator2D
    Differentiable GLCM statistics (contrast, correlation, energy,
    homogeneity, entropy) via soft sigmoid binning. Optional 4-direction
    averaging for rotation invariance. Wavelet ext → '*_wav' [Nb, Nc, D, J, L].

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
    Three 2D Minkowski functionals for a batch [B, H, W].

    Returns {'W0': [B], 'W1': [B], 'W2': [B]}, fully differentiable.
    Works for non-square images (H ≠ W).
    """
    B, H, W = img.shape
    if threshold is not None:
        img = torch.sigmoid(temperature * (img - _mink2d_as_threshold(threshold, img)))

    W0 = img.mean(dim=(-2, -1))

    dh = (img[:, :, 1:] - img[:, :, :-1]).abs()
    dv = (img[:, 1:, :] - img[:, :-1, :]).abs()
    W1 = (dh.sum(dim=(-2, -1)) + dv.sum(dim=(-2, -1))) / (H * W)

    Q1 = img.sum(dim=(-2, -1))
    Qh = (img[:, :, :-1] * img[:, :, 1:]).sum(dim=(-2, -1))
    Qv = (img[:, :-1, :] * img[:, 1:, :]).sum(dim=(-2, -1))
    Qf = (
        img[:, :-1, :-1] * img[:, :-1, 1:] * img[:, 1:, :-1] * img[:, 1:, 1:]
    ).sum(dim=(-2, -1))
    W2 = (Q1 - Qh - Qv + Qf) / (H * W)

    return {"W0": W0, "W1": W1, "W2": W2}


def _mink2d_curves(
    img: torch.Tensor,
    thresholds: torch.Tensor,
    temperature: float = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Minkowski functionals at multiple thresholds for [B, H, W].

    thresholds : [T] or [B, T].  Returns {'W0','W1','W2'} each [B, T].
    Works for non-square images (H ≠ W).
    """
    B, H, W = img.shape
    t = thresholds
    if t.ndim == 1:
        t = t.unsqueeze(0).expand(B, t.shape[0])
    T = t.shape[1]
    soft = torch.sigmoid(temperature * (img.unsqueeze(1) - t.view(B, T, 1, 1)))
    mf = _mink2d_functionals(soft.view(B * T, H, W))
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
# ── Wavelet-Minkowski helpers ─────────────────────────────────────────────────


def _downsample_to_scale(
    arr: torch.Tensor,
    n_steps: int,
    smooth_kernel: torch.Tensor,
    padding_mode: str,
) -> torch.Tensor:
    """
    Downsample a [B, N, M] array by ``n_steps`` successive stride-2 operations.

    Each step applies ``smooth_kernel`` then strides by 2 in both directions.

    Parameters
    ----------
    arr          : [B, N, M] float
    n_steps      : int  — number of halving steps (= j_to_dg[j])
    smooth_kernel: [1, 1, Ks, Ks] float — pre-computed Gaussian kernel
    padding_mode : str  — 'circular' or 'replicate'

    Returns
    -------
    [B, N//2^n_steps, M//2^n_steps] float
    """
    if n_steps == 0:
        return arr
    x = arr.unsqueeze(1)  # [B, 1, N, M]
    sk = smooth_kernel.to(device=x.device, dtype=x.dtype)
    ks = sk.shape[-1]
    pad = ks // 2
    for _ in range(n_steps):
        x = F.pad(x, [pad, pad, pad, pad], mode=padding_mode)
        x = F.conv2d(x, sk)[:, :, ::2, ::2]
    return x.squeeze(1)  # [B, N//2^n, M//2^n]


def _apply_wavelet_modulus(
    arr: torch.Tensor,
    kernel_j: torch.Tensor,
    pbc: bool,
) -> torch.Tensor:
    """
    Apply complex wavelet kernel at scale j to a [B, N, M] array and return
    the modulus |I * ψ_{j,θ}| for each orientation θ.

    Parameters
    ----------
    arr      : [B, N, M] real float
    kernel_j : [1, L, K, K] complex — wavelet kernels at scale j,
               L orientations stored along dim-1, leading dim-0 = 1.
    pbc      : bool — periodic boundary → circular padding.

    Returns
    -------
    [B, L, N, M] real float — modulus per orientation
    """
    # kernel_j is [1, L, K, K] complex
    k_flat = kernel_j[0]                  # [L, K, K] complex
    L, K, _ = k_flat.shape
    pad = K // 2
    padding_mode = "circular" if pbc else "replicate"

    # F.conv2d weight: [out_channels, in_channels, kH, kW] = [L, 1, K, K]
    kr = k_flat.real.unsqueeze(1).to(device=arr.device, dtype=arr.dtype)
    ki = k_flat.imag.unsqueeze(1).to(device=arr.device, dtype=arr.dtype)

    x = arr.unsqueeze(1)                                              # [B, 1, N, M]
    x_pad = F.pad(x, [pad, pad, pad, pad], mode=padding_mode)        # [B, 1, N+2p, M+2p]

    out_r = F.conv2d(x_pad, kr)  # [B, L, N, M]
    out_i = F.conv2d(x_pad, ki)  # [B, L, N, M]
    return (out_r ** 2 + out_i ** 2).sqrt()  # [B, L, N, M] — modulus


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
        Number of wavelet scales for the wavelet Minkowski outputs.
        ``J = 0`` (default) — return only the raw-image functionals.
        ``J > 0`` and ``wavelet_op`` provided — additionally return
        ``'W0_wav'``, ``'W1_wav'``, ``'W2_wav'`` of shape
        ``[Nb, Nc, J, L, T]``.
        ``J > 0`` and no ``wavelet_op`` — Gaussian DoG fallback (legacy).
    scale_mode : str
        ``'smooth'`` or ``'dog'`` — used only by the Gaussian fallback.
    wavelet_op : WaveletOperator2Dkernel_torch or ST_Operator or None
        Source of pre-computed wavelet kernels.  At initialisation the
        operator extracts and caches ``_wav_kernels``, ``_wav_j_to_dg``, and
        the smooth down-sampling kernels.  No wavelet object is needed at
        call time.
    device, dtype

    Examples
    --------
    >>> op = MinkowskiOperator2D(shape=(64, 64))
    >>> mf = op.minkowski(data)           # {'W0','W1','W2'} each [Nb, Nc]

    >>> t = torch.linspace(0.1, 0.9, 16)
    >>> op = MinkowskiOperator2D(shape=(64, 64), thresholds=t, J=3,
    ...                          wavelet_op=wav_op)
    >>> out = op.minkowski(data)
    >>> # out['W0'].shape       == [Nb, Nc, 16]           (raw image)
    >>> # out['W0_wav'].shape   == [Nb, Nc, 16, 3, L]    (wavelet norm)
    """

    def __init__(
        self,
        shape,
        thresholds=None,
        temperature: float = 20.0,
        J: int = 0,
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
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        # ── Extract wavelet kernels once at initialisation ────────────────────
        self.wavelet_op = wavelet_op
        self._wav_kernels: "list | None" = None
        self._wav_j_to_dg: "list | None" = None
        self._wav_smooth_pbc: "torch.Tensor | None" = None
        self._wav_smooth_nopbc: "torch.Tensor | None" = None
        self._wav_L: int = 0

        if wavelet_op is not None:
            # ── STL-Dev: pre-computed decimated kernels, one per scale ────────
            if hasattr(wavelet_op, "_decimated_kernels"):
                self._wav_kernels = list(wavelet_op._decimated_kernels)
            # ── pablo_one: single kernel applied after downsampling ───────────
            #    _wav_kernel_0 is used at j=0, _wav_kernel at j>0.
            elif hasattr(wavelet_op, "_wav_kernel"):
                _J = getattr(wavelet_op, "J", 1)
                self._wav_kernels = [
                    (
                        wavelet_op._wav_kernel_0
                        if j == 0 and hasattr(wavelet_op, "_wav_kernel_0")
                        else wavelet_op._wav_kernel
                    )
                    for j in range(_J)
                ]
            # ── ST_Operator: .kernels list ────────────────────────────────────
            elif hasattr(wavelet_op, "kernels") and wavelet_op.kernels is not None:
                self._wav_kernels = list(wavelet_op.kernels)

            if hasattr(wavelet_op, "j_to_dg"):
                self._wav_j_to_dg = list(wavelet_op.j_to_dg)

            if hasattr(wavelet_op, "smooth_kernel_pbc"):
                # STL-Dev: already stored as [1, 1, Ks, Ks]
                self._wav_smooth_pbc   = wavelet_op.smooth_kernel_pbc
                self._wav_smooth_nopbc = wavelet_op.smooth_kernel_nopbc
            elif hasattr(wavelet_op, "_gaussian_kernel_5x5"):
                # pablo_one: build [5,5] and unsqueeze to [1,1,5,5]
                _dev  = getattr(wavelet_op, "device", torch.device("cpu"))
                _dtyp = getattr(wavelet_op, "dtype", torch.float32)
                _sk   = wavelet_op._gaussian_kernel_5x5(device=_dev, dtype=_dtyp)
                _sk4  = _sk[None, None]   # [1, 1, 5, 5]
                self._wav_smooth_pbc   = _sk4
                self._wav_smooth_nopbc = _sk4  # same kernel; padding_mode differs

            if self._wav_kernels is not None:
                self._wav_L = self._wav_kernels[0].shape[1]

    # ── Public method ─────────────────────────────────────────────────────────

    def minkowski(self, data, thresholds=None, temperature=None):
        """
        Compute the three Minkowski functionals of 2D planar data.

        Parameters
        ----------
        data        : any object exposing .array, .N0, .pbc
        thresholds  : Tensor [T] | Tensor [Nb*Nc, T] | None
        temperature : float or None

        Returns
        -------
        dict[str, Tensor]
            Always present: ``'W0'``, ``'W1'``, ``'W2'``.

            Shape without thresholds : ``[Nb, Nc]``.
            Shape with thresholds T  : ``[Nb, Nc, T]``.

            When ``J > 0`` and ``wavelet_op`` was provided at init:
            ``'W0_wav'``, ``'W1_wav'``, ``'W2_wav'`` of shape
            ``[Nb, Nc, J, L, T]`` (or ``[Nb, Nc, J, L]`` without thresholds).
            Here L is the number of wavelet orientations and J is
            ``min(self.J, len(kernels))``.

            When ``J > 0`` but no ``wavelet_op`` (Gaussian fallback):
            raw keys are shaped ``[Nb, Nc, J]`` or ``[Nb, Nc, J, T]``.
        """
        _check_data(data, self.shape)

        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature

        arr = data.array
        if torch.is_complex(arr):
            arr = arr.abs()
        arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[None, :, :, :]
        Nb, Nc, Nx, Ny = arr.shape
        flat = arr.reshape(Nb * Nc, Nx, Ny).to(self.device, self.dtype)
        pbc  = getattr(data, "pbc", True)

        # ── Helper: Minkowski of one [B, N, M] band ───────────────────────────
        def _compute_one(band):
            if thresholds is None:
                mf = _mink2d_functionals(band, temperature=temperature)
                return {k: v.view(Nb, Nc) for k, v in mf.items()}
            t = torch.as_tensor(thresholds, dtype=band.dtype, device=band.device)
            if t.ndim == 1:
                t = t.unsqueeze(0).expand(Nb * Nc, t.shape[0])
            elif t.ndim == 2:
                if t.shape[0] != Nb * Nc:
                    raise ValueError(
                        f"thresholds batch dim {t.shape[0]} != Nb*Nc={Nb * Nc}"
                    )
            else:
                raise ValueError(f"thresholds must be 1-D or 2-D, got {t.ndim}-D")
            T_val = t.shape[1]
            curves = _mink2d_curves(band, t, temperature=temperature)
            return {k: v.view(Nb, Nc, T_val) for k, v in curves.items()}

        # ── J=0: raw image only ───────────────────────────────────────────────
        if self.J == 0:
            return _compute_one(flat)

        # ── J>0, no wavelet_op: Gaussian DoG fallback (legacy) ────────────────
        if self._wav_kernels is None:
            scale_results = [
                _compute_one(
                    _extract_scale_flat(
                        flat, j, pbc=pbc, scale_mode=self.scale_mode
                    )
                )
                for j in range(self.J)
            ]
            return {
                k: torch.stack([sr[k] for sr in scale_results], dim=2)
                for k in scale_results[0]
            }

        # ── J>0, wavelet_op provided: raw + wavelet Minkowski ─────────────────
        raw = _compute_one(flat)   # {'W0','W1','W2'} each [Nb,Nc] or [Nb,Nc,T]

        J_eff        = min(self.J, len(self._wav_kernels))
        padding_mode = "circular" if pbc else "replicate"
        smooth_k     = self._wav_smooth_pbc if pbc else self._wav_smooth_nopbc

        wav_per_j = []  # list[J] of {'W0','W1','W2'} each [Nb,Nc,L] or [Nb,Nc,L,T]
        for j in range(J_eff):
            dg_j = self._wav_j_to_dg[j] if self._wav_j_to_dg is not None else j

            # Downsample to the spatial scale of wavelet j
            ds = (
                _downsample_to_scale(flat, dg_j, smooth_k, padding_mode)
                if smooth_k is not None
                else flat
            )

            # |I * ψ_{j,θ}| for all orientations — shape [Nb*Nc, L, Nx_j, Ny_j]
            moduli = _apply_wavelet_modulus(ds, self._wav_kernels[j], pbc)

            L_j   = moduli.shape[1]
            per_l = [
                _compute_one(moduli[:, l_idx, :, :])
                for l_idx in range(L_j)
            ]

            # Stack over L → [Nb,Nc,L] or [Nb,Nc,L,T]
            wav_per_j.append(
                {k: torch.stack([r[k] for r in per_l], dim=2) for k in per_l[0]}
            )

        # Stack over J → [Nb,Nc,J,L] or [Nb,Nc,J,L,T]
        wav = {
            k: torch.stack([wr[k] for wr in wav_per_j], dim=2)
            for k in wav_per_j[0]
        }

        # Reorder T to front: [Nb,Nc,J,L,T] → [Nb,Nc,T,J,L]
        # (no-threshold case stays [Nb,Nc,J,L] — no T dim to move)
        if thresholds is not None:
            wav = {k: v.permute(0, 1, 4, 2, 3).contiguous() for k, v in wav.items()}

        return {
            "W0":     raw["W0"],
            "W1":     raw["W1"],
            "W2":     raw["W2"],
            "W0_wav": wav["W0"],
            "W1_wav": wav["W1"],
            "W2_wav": wav["W2"],
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
        Number of wavelet scales.  ``J = 0`` (default) — raw image only.
        ``J > 0`` and ``wavelet_op`` provided — also return ``'peaks_wav'``
        / ``'valleys_wav'`` of shape ``[Nb, Nc, T, J, L]`` (or
        ``[Nb, Nc, J, L]`` without thresholds).
    scale_mode : str
    wavelet_op : WaveletOperator2Dkernel_torch or ST_Operator or None
    device, dtype

    Examples
    --------
    >>> t  = torch.linspace(0.1, 0.9, 16)
    >>> op = PeakCountOperator2D(shape=(128, 128), thresholds=t)
    >>> out = op.peaks(data)     # {'peaks': [Nb, Nc, 16]}

    >>> op3 = PeakCountOperator2D(shape=(128, 128), thresholds=t,
    ...                           J=3, wavelet_op=wav_op)
    >>> out3 = op3.peaks(data)
    >>> # out3['peaks'].shape       == [Nb, Nc, 16]          (raw)
    >>> # out3['peaks_wav'].shape   == [Nb, Nc, 16, 3, L]   (wavelet)
    """

    def __init__(
        self,
        shape,
        thresholds=None,
        temperature: float = 20.0,
        connectivity: int = 8,
        J: int = 0,
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
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        # ── Extract wavelet kernels once at initialisation ────────────────────
        self.wavelet_op = wavelet_op
        self._wav_kernels: "list | None" = None
        self._wav_j_to_dg: "list | None" = None
        self._wav_smooth_pbc: "torch.Tensor | None" = None
        self._wav_smooth_nopbc: "torch.Tensor | None" = None
        self._wav_L: int = 0

        if wavelet_op is not None:
            # ── STL-Dev: pre-computed decimated kernels, one per scale ────────
            if hasattr(wavelet_op, "_decimated_kernels"):
                self._wav_kernels = list(wavelet_op._decimated_kernels)
            # ── pablo_one: single kernel applied after downsampling ───────────
            #    _wav_kernel_0 is used at j=0, _wav_kernel at j>0.
            elif hasattr(wavelet_op, "_wav_kernel"):
                _J = getattr(wavelet_op, "J", 1)
                self._wav_kernels = [
                    (
                        wavelet_op._wav_kernel_0
                        if j == 0 and hasattr(wavelet_op, "_wav_kernel_0")
                        else wavelet_op._wav_kernel
                    )
                    for j in range(_J)
                ]
            # ── ST_Operator: .kernels list ────────────────────────────────────
            elif hasattr(wavelet_op, "kernels") and wavelet_op.kernels is not None:
                self._wav_kernels = list(wavelet_op.kernels)

            if hasattr(wavelet_op, "j_to_dg"):
                self._wav_j_to_dg = list(wavelet_op.j_to_dg)

            if hasattr(wavelet_op, "smooth_kernel_pbc"):
                # STL-Dev: already stored as [1, 1, Ks, Ks]
                self._wav_smooth_pbc   = wavelet_op.smooth_kernel_pbc
                self._wav_smooth_nopbc = wavelet_op.smooth_kernel_nopbc
            elif hasattr(wavelet_op, "_gaussian_kernel_5x5"):
                # pablo_one: build [5,5] and unsqueeze to [1,1,5,5]
                _dev  = getattr(wavelet_op, "device", torch.device("cpu"))
                _dtyp = getattr(wavelet_op, "dtype", torch.float32)
                _sk   = wavelet_op._gaussian_kernel_5x5(device=_dev, dtype=_dtyp)
                _sk4  = _sk[None, None]   # [1, 1, 5, 5]
                self._wav_smooth_pbc   = _sk4
                self._wav_smooth_nopbc = _sk4  # same kernel; padding_mode differs

            if self._wav_kernels is not None:
                self._wav_L = self._wav_kernels[0].shape[1]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare(self, data):
        _check_data(data, self.shape)
        arr = data.array.abs() if torch.is_complex(data.array) else data.array
        arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        pbc = getattr(data, "pbc", True)
        pmode = "circular" if pbc else "replicate"
        return arr.reshape(Nb * Nc, Nx, Ny).to(self.device, self.dtype), pmode, Nb, Nc

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

    def _wav_pass(self, flat, mode, thresholds, temperature, pbc, Nb, Nc):
        """
        Wavelet-domain counts for all j and orientations.
        Returns {'peaks'/'valleys': [Nb,Nc,T,J,L]} or [Nb,Nc,J,L].
        """
        key        = "peaks" if mode == "peaks" else "valleys"
        J_eff      = min(self.J, len(self._wav_kernels))
        pad_mode   = "circular" if pbc else "replicate"
        smooth_k   = self._wav_smooth_pbc if pbc else self._wav_smooth_nopbc

        wav_per_j = []
        for j in range(J_eff):
            dg_j = self._wav_j_to_dg[j] if self._wav_j_to_dg is not None else j
            ds   = (
                _downsample_to_scale(flat, dg_j, smooth_k, pad_mode)
                if smooth_k is not None else flat
            )
            moduli = _apply_wavelet_modulus(ds, self._wav_kernels[j], pbc)
            # moduli: [Nb*Nc, L, Nx_j, Ny_j]
            L_j   = moduli.shape[1]
            per_l = [
                self._count_one(
                    moduli[:, l, :, :], mode, thresholds, temperature, pad_mode, Nb, Nc
                )
                for l in range(L_j)
            ]
            # stack L → [Nb,Nc,L] or [Nb,Nc,L,T]
            wav_per_j.append(
                {key: torch.stack([r[key] for r in per_l], dim=2)}
            )

        # stack J → [Nb,Nc,J,L] or [Nb,Nc,J,L,T]
        stacked = torch.stack([wr[key] for wr in wav_per_j], dim=2)

        # reorder T to front: [Nb,Nc,J,L,T] → [Nb,Nc,T,J,L]
        if thresholds is not None:
            stacked = stacked.permute(0, 1, 4, 2, 3).contiguous()

        return {key + "_wav": stacked}

    # ── Public methods ────────────────────────────────────────────────────────

    def peaks(self, data, thresholds=None, temperature=None):
        """
        Soft peak (local-maximum) count, optionally conditioned on threshold.

        Returns
        -------
        dict
            ``'peaks'``: raw image — ``[Nb, Nc]`` or ``[Nb, Nc, T]``.

            When ``J > 0`` and ``wavelet_op`` given:
            ``'peaks_wav'``: ``[Nb, Nc, T, J, L]`` or ``[Nb, Nc, J, L]``.

            Legacy (``J > 0``, no ``wavelet_op``):
            ``'peaks'``: ``[Nb, Nc, J]`` or ``[Nb, Nc, J, T]``.
        """
        flat, pmode, Nb, Nc = self._prepare(data)
        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature
        pbc         = pmode == "circular"

        if self.J == 0:
            return self._count_one(flat, "peaks", thresholds, temperature, pmode, Nb, Nc)

        if self._wav_kernels is None:
            # Gaussian DoG fallback (legacy)
            scales = [
                self._count_one(
                    _extract_scale_flat(flat, j, pbc=pbc, scale_mode=self.scale_mode),
                    "peaks", thresholds, temperature, pmode, Nb, Nc,
                )
                for j in range(self.J)
            ]
            return {"peaks": torch.stack([s["peaks"] for s in scales], dim=2)}

        raw = self._count_one(flat, "peaks", thresholds, temperature, pmode, Nb, Nc)
        wav = self._wav_pass(flat, "peaks", thresholds, temperature, pbc, Nb, Nc)
        return {**raw, **wav}

    def valleys(self, data, thresholds=None, temperature=None):
        """
        Soft valley (local-minimum) count.

        Returns
        -------
        dict
            ``'valleys'``: raw image — ``[Nb, Nc]`` or ``[Nb, Nc, T]``.

            When ``J > 0`` and ``wavelet_op`` given:
            ``'valleys_wav'``: ``[Nb, Nc, T, J, L]`` or ``[Nb, Nc, J, L]``.

            Legacy (``J > 0``, no ``wavelet_op``):
            ``'valleys'``: ``[Nb, Nc, J]`` or ``[Nb, Nc, J, T]``.
        """
        flat, pmode, Nb, Nc = self._prepare(data)
        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature
        pbc         = pmode == "circular"

        if self.J == 0:
            return self._count_one(flat, "valleys", thresholds, temperature, pmode, Nb, Nc)

        if self._wav_kernels is None:
            scales = [
                self._count_one(
                    _extract_scale_flat(flat, j, pbc=pbc, scale_mode=self.scale_mode),
                    "valleys", thresholds, temperature, pmode, Nb, Nc,
                )
                for j in range(self.J)
            ]
            return {"valleys": torch.stack([s["valleys"] for s in scales], dim=2)}

        raw = self._count_one(flat, "valleys", thresholds, temperature, pmode, Nb, Nc)
        wav = self._wav_pass(flat, "valleys", thresholds, temperature, pbc, Nb, Nc)
        return {**raw, **wav}


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
        Number of wavelet scales.  ``J = 0`` (default) — raw image only.
        ``J > 0`` and ``wavelet_op`` provided — also return
        ``'beta0_wav'``, ``'beta1_wav'``, ``'chi_wav'`` of shape
        ``[Nb, Nc, T, J, L]``.
    scale_mode : str
    wavelet_op : WaveletOperator2Dkernel_torch or ST_Operator or None
    device, dtype

    Examples
    --------
    >>> t  = torch.linspace(0.1, 0.9, 16)
    >>> op = BettiCurveOperator2D(shape=(128, 128), thresholds=t)
    >>> out = op.betti(data)
    >>> # out['beta0'].shape == [Nb, Nc, 16]

    >>> op3 = BettiCurveOperator2D(shape=(128, 128), thresholds=t,
    ...                             J=3, wavelet_op=wav_op)
    >>> out3 = op3.betti(data)
    >>> # out3['beta0'].shape       == [Nb, Nc, 16]          (raw)
    >>> # out3['beta0_wav'].shape   == [Nb, Nc, 16, 3, L]   (wavelet)
    """

    def __init__(
        self,
        shape,
        thresholds,
        temperature: float = 20.0,
        connectivity: int = 8,
        J: int = 0,
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
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        # ── Extract wavelet kernels once at initialisation ────────────────────
        self.wavelet_op = wavelet_op
        self._wav_kernels: "list | None" = None
        self._wav_j_to_dg: "list | None" = None
        self._wav_smooth_pbc: "torch.Tensor | None" = None
        self._wav_smooth_nopbc: "torch.Tensor | None" = None
        self._wav_L: int = 0

        if wavelet_op is not None:
            # ── STL-Dev: pre-computed decimated kernels, one per scale ────────
            if hasattr(wavelet_op, "_decimated_kernels"):
                self._wav_kernels = list(wavelet_op._decimated_kernels)
            # ── pablo_one: single kernel applied after downsampling ───────────
            #    _wav_kernel_0 is used at j=0, _wav_kernel at j>0.
            elif hasattr(wavelet_op, "_wav_kernel"):
                _J = getattr(wavelet_op, "J", 1)
                self._wav_kernels = [
                    (
                        wavelet_op._wav_kernel_0
                        if j == 0 and hasattr(wavelet_op, "_wav_kernel_0")
                        else wavelet_op._wav_kernel
                    )
                    for j in range(_J)
                ]
            # ── ST_Operator: .kernels list ────────────────────────────────────
            elif hasattr(wavelet_op, "kernels") and wavelet_op.kernels is not None:
                self._wav_kernels = list(wavelet_op.kernels)

            if hasattr(wavelet_op, "j_to_dg"):
                self._wav_j_to_dg = list(wavelet_op.j_to_dg)

            if hasattr(wavelet_op, "smooth_kernel_pbc"):
                # STL-Dev: already stored as [1, 1, Ks, Ks]
                self._wav_smooth_pbc   = wavelet_op.smooth_kernel_pbc
                self._wav_smooth_nopbc = wavelet_op.smooth_kernel_nopbc
            elif hasattr(wavelet_op, "_gaussian_kernel_5x5"):
                # pablo_one: build [5,5] and unsqueeze to [1,1,5,5]
                _dev  = getattr(wavelet_op, "device", torch.device("cpu"))
                _dtyp = getattr(wavelet_op, "dtype", torch.float32)
                _sk   = wavelet_op._gaussian_kernel_5x5(device=_dev, dtype=_dtyp)
                _sk4  = _sk[None, None]   # [1, 1, 5, 5]
                self._wav_smooth_pbc   = _sk4
                self._wav_smooth_nopbc = _sk4  # same kernel; padding_mode differs

            if self._wav_kernels is not None:
                self._wav_L = self._wav_kernels[0].shape[1]

    # ── Internal helpers ──────────────────────────────────────────────────────

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

    # ── Public method ─────────────────────────────────────────────────────────

    def betti(self, data, thresholds=None, temperature=None):
        """
        Compute Betti curves β₀(t), β₁(t) and Euler characteristic χ(t).

        Parameters
        ----------
        data        : any object exposing .array, .N0, .pbc
        thresholds  : Tensor [T] | Tensor [Nb*Nc, T]
        temperature : float or None

        Returns
        -------
        dict
            Always present: ``'beta0'``, ``'beta1'``, ``'chi'`` — each
            ``[Nb, Nc, T]``.

            When ``J > 0`` and ``wavelet_op`` given:
            ``'beta0_wav'``, ``'beta1_wav'``, ``'chi_wav'`` — each
            ``[Nb, Nc, T, J, L]``.

            Legacy (``J > 0``, no ``wavelet_op``):
            ``'beta0'``, ``'beta1'``, ``'chi'`` — each ``[Nb, Nc, J, T]``.
        """
        _check_data(data, self.shape)

        arr = data.array.abs() if torch.is_complex(data.array) else data.array
        arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        flat  = arr.reshape(Nb * Nc, Nx, Ny).to(self.device, self.dtype)
        pbc   = getattr(data, "pbc", True)
        pmode = "circular" if pbc else "replicate"

        thresholds  = thresholds  if thresholds  is not None else self.thresholds
        temperature = temperature if temperature is not None else self.temperature

        if thresholds is None:
            raise ValueError("thresholds must be provided for Betti curves.")

        if self.J == 0:
            return self._betti_one(flat, thresholds, temperature, pmode, Nb, Nc)

        if self._wav_kernels is None:
            # Gaussian DoG fallback (legacy)
            scale_results = [
                self._betti_one(
                    _extract_scale_flat(flat, j, pbc=pbc, scale_mode=self.scale_mode),
                    thresholds, temperature, pmode, Nb, Nc,
                )
                for j in range(self.J)
            ]
            return {
                k: torch.stack([sr[k] for sr in scale_results], dim=2)
                for k in scale_results[0]
            }

        # ── Wavelet Betti curves ───────────────────────────────────────────────
        raw   = self._betti_one(flat, thresholds, temperature, pmode, Nb, Nc)

        J_eff    = min(self.J, len(self._wav_kernels))
        pad_mode = "circular" if pbc else "replicate"
        smooth_k = self._wav_smooth_pbc if pbc else self._wav_smooth_nopbc

        wav_per_j = []
        for j in range(J_eff):
            dg_j = self._wav_j_to_dg[j] if self._wav_j_to_dg is not None else j
            ds   = (
                _downsample_to_scale(flat, dg_j, smooth_k, pad_mode)
                if smooth_k is not None else flat
            )
            moduli = _apply_wavelet_modulus(ds, self._wav_kernels[j], pbc)
            # moduli: [Nb*Nc, L, Nx_j, Ny_j]
            L_j   = moduli.shape[1]
            per_l = [
                self._betti_one(
                    moduli[:, l, :, :], thresholds, temperature, pad_mode, Nb, Nc
                )
                for l in range(L_j)
            ]
            # stack over L → [Nb,Nc,L,T]
            wav_per_j.append(
                {k: torch.stack([r[k] for r in per_l], dim=2) for k in per_l[0]}
            )

        # stack over J → [Nb,Nc,J,L,T] then permute → [Nb,Nc,T,J,L]
        wav = {
            k: torch.stack([wr[k] for wr in wav_per_j], dim=2).permute(0, 1, 4, 2, 3).contiguous()
            for k in wav_per_j[0]
        }

        return {
            "beta0":     raw["beta0"],
            "beta1":     raw["beta1"],
            "chi":       raw["chi"],
            "beta0_wav": wav["beta0"],
            "beta1_wav": wav["beta1"],
            "chi_wav":   wav["chi"],
        }


###############################################################################
# ── Lacunarity & GLCM helpers ────────────────────────────────────────────────
###############################################################################


def _circular_disk_kernel(
    eps: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Return a [1, 1, eps, eps] circular disk kernel (unnormalized sum kernel).

    Pixels within radius (eps-1)/2 of the centre are 1; others are 0.
    """
    ys = torch.arange(eps, dtype=dtype, device=device) - (eps - 1) / 2.0
    xs = torch.arange(eps, dtype=dtype, device=device) - (eps - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r    = (eps - 1) / 2.0
    mask = (xx ** 2 + yy ** 2 <= r ** 2).to(dtype)
    return mask[None, None]   # [1, 1, eps, eps]


def _lac_one_scale(
    flat: torch.Tensor,
    kernel: torch.Tensor,
    pad_mode: str,
) -> torch.Tensor:
    """
    Differentiable lacunarity at one window size.

    Parameters
    ----------
    flat   : [B, H, W]
    kernel : [1, 1, k, k]  unnormalized sum window
    pad_mode : 'circular' | 'replicate'

    Returns
    -------
    [B]  Λ(ε) = E[Q²] / E[Q]²   (≥ 1 always)
    """
    B, H, W = flat.shape
    p = kernel.shape[-1] // 2
    f = flat[:, None]                                              # [B, 1, H, W]

    # F.conv2d does not accept padding_mode — pad manually then convolve
    f_pad = F.pad(f, (p, p, p, p), mode=pad_mode)                 # [B, 1, H+2p, W+2p]
    Q = F.conv2d(f_pad, kernel, padding=0).squeeze(1)             # [B, H, W]

    mu1 = Q.mean(dim=(-2, -1))                                    # [B]
    mu2 = (Q ** 2).mean(dim=(-2, -1))                             # [B]
    return mu2 / mu1.clamp(min=1e-8) ** 2                         # [B]


def _soft_glcm(
    flat: torch.Tensor,
    offset: tuple,
    thresholds: torch.Tensor,
    temperature: float,
    pad_mode: str,
) -> torch.Tensor:
    """
    Differentiable soft GLCM for one spatial offset (dy, dx).

    Uses sigmoid-based soft histogram binning so the result is fully
    differentiable w.r.t. ``flat``.

    Parameters
    ----------
    flat        : [B, H, W]
    offset      : (dy, dx)
    thresholds  : [N+1]  bin edges
    temperature : sigmoid sharpness
    pad_mode    : 'circular' | 'replicate'

    Returns
    -------
    glcm : [B, N, N]  normalised co-occurrence matrix
    """
    B, H, W = flat.shape
    dy, dx  = offset
    N = len(thresholds) - 1

    # ── Shifted image: shifted[y, x] = img[y + dy, x + dx] (with boundary) ──
    if pad_mode == "circular":
        shifted = torch.roll(flat, shifts=(-dy, -dx), dims=(-2, -1))
    else:
        pad_t   = max(-dy, 0)
        pad_b   = max( dy, 0)
        pad_l   = max(-dx, 0)
        pad_r   = max( dx, 0)
        padded  = F.pad(flat, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
        off_h   = pad_t + dy
        off_w   = pad_l + dx
        shifted = padded[:, off_h: off_h + H, off_w: off_w + W]

    # ── Soft bin membership ───────────────────────────────────────────────────
    t    = thresholds.to(flat.device, flat.dtype)     # [N+1]
    t_lo = t[:-1]                                     # [N]
    t_hi = t[1:]                                      # [N]

    x1 = flat.unsqueeze(-1)                           # [B, H, W, 1]
    x2 = shifted.unsqueeze(-1)                        # [B, H, W, 1]

    # p_k(x) = σ(T*(x−t_k)) − σ(T*(x−t_{k+1}))
    p1 = (torch.sigmoid(temperature * (x1 - t_lo))
          - torch.sigmoid(temperature * (x1 - t_hi)))    # [B, H, W, N]
    p2 = (torch.sigmoid(temperature * (x2 - t_lo))
          - torch.sigmoid(temperature * (x2 - t_hi)))    # [B, H, W, N]

    # Outer product summed over spatial dims → [B, N, N]
    glcm = (p1.unsqueeze(-1) * p2.unsqueeze(-2)).sum(dim=(1, 2))

    # Normalise
    glcm = glcm / glcm.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
    return glcm


def _glcm_statistics(glcm: torch.Tensor) -> dict:
    """
    Extract five standard GLCM statistics.

    Parameters
    ----------
    glcm : [B, N, N]  normalised co-occurrence matrix

    Returns
    -------
    dict with keys 'contrast', 'correlation', 'energy', 'homogeneity',
    'entropy', each [B].
    """
    B, N, _ = glcm.shape
    device  = glcm.device
    dtype   = glcm.dtype

    idx = torch.arange(N, dtype=dtype, device=device)
    ii, jj = torch.meshgrid(idx, idx, indexing="ij")   # [N, N]

    # Marginal means and standard deviations
    mu_i  = (ii[None] * glcm).sum(dim=(-2, -1))        # [B]
    mu_j  = (jj[None] * glcm).sum(dim=(-2, -1))        # [B]
    sig_i = ((ii[None] - mu_i[:, None, None]) ** 2 * glcm).sum(dim=(-2, -1)).sqrt().clamp(min=1e-8)
    sig_j = ((jj[None] - mu_j[:, None, None]) ** 2 * glcm).sum(dim=(-2, -1)).sqrt().clamp(min=1e-8)

    diff      = ii - jj                                 # [N, N]
    contrast  = (diff[None] ** 2 * glcm).sum(dim=(-2, -1))                          # [B]
    homo      = (glcm / (1.0 + diff[None].abs())).sum(dim=(-2, -1))                 # [B]
    energy    = (glcm ** 2).sum(dim=(-2, -1))                                       # [B]
    entropy   = -(glcm * glcm.clamp(min=1e-10).log()).sum(dim=(-2, -1))             # [B]
    corr_num  = ((ii[None] - mu_i[:, None, None]) *
                 (jj[None] - mu_j[:, None, None]) * glcm).sum(dim=(-2, -1))
    correlation = corr_num / (sig_i * sig_j)                                        # [B]

    return {
        "contrast":    contrast,
        "correlation": correlation,
        "energy":      energy,
        "homogeneity": homo,
        "entropy":     entropy,
    }


###############################################################################


class LacunarityOperator2D:
    """
    Differentiable lacunarity descriptor for 2D planar STL data.

    For each window size ε in ``window_sizes`` computes:

        Λ(ε) = E[Q(r,ε)²] / E[Q(r,ε)]²

    where Q(r, ε) is the local pixel-sum inside a window of size ε centred
    at r (computed via ``F.conv2d``).  Λ(ε) ≥ 1 (= 1 for a uniform image,
    large for strongly clustered textures).

    Parameters
    ----------
    shape : tuple of int
        Spatial shape ``(Nx, Ny)`` matching ``data.N0``.
    window_sizes : sequence of int
        Window sizes ε₁, …, εE (pixels).
    rotation_invariant : bool
        If True, replaces the square window with a circular disk kernel,
        making Λ(ε) invariant under image rotation.
    J : int
        Wavelet scales.  ``J = 0`` — raw image only.
        ``J > 0`` + ``wavelet_op`` — also return ``'lac_wav'``
        of shape ``[Nb, Nc, E, J, L]``.
    wavelet_op : WaveletOperator2Dkernel_torch / ST_Operator / None
    device, dtype

    Examples
    --------
    >>> lac_op = LacunarityOperator2D(shape=(128, 128), window_sizes=[4,8,16,32])
    >>> out = lac_op.lacunarity(data)
    >>> # out['lac'].shape == [Nb, Nc, 4]
    """

    def __init__(
        self,
        shape,
        window_sizes=(4, 8, 16, 32),
        rotation_invariant: bool = False,
        J: int = 0,
        wavelet_op=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
    ):
        self.shape              = tuple(shape)
        self.window_sizes       = list(window_sizes)
        self.rotation_invariant = rotation_invariant
        self.J                  = J
        self.device             = _get_device(torch.device(device))
        self.dtype              = _get_dtype(dtype=dtype, device=self.device)

        # Pre-build one sum kernel per window size
        self._kernels = []
        for eps in self.window_sizes:
            if self.rotation_invariant:
                k = _circular_disk_kernel(eps, self.dtype, self.device)
            else:
                k = torch.ones(1, 1, eps, eps, dtype=self.dtype, device=self.device)
            self._kernels.append(k)

        # ── Wavelet infrastructure (identical to MinkowskiOperator2D) ─────────
        self.wavelet_op        = wavelet_op
        self._wav_kernels      = None
        self._wav_j_to_dg      = None
        self._wav_smooth_pbc   = None
        self._wav_smooth_nopbc = None
        self._wav_L: int       = 0

        if wavelet_op is not None:
            if hasattr(wavelet_op, "_decimated_kernels"):
                self._wav_kernels = list(wavelet_op._decimated_kernels)
            elif hasattr(wavelet_op, "_wav_kernel"):
                _J = getattr(wavelet_op, "J", 1)
                self._wav_kernels = [
                    (wavelet_op._wav_kernel_0
                     if j == 0 and hasattr(wavelet_op, "_wav_kernel_0")
                     else wavelet_op._wav_kernel)
                    for j in range(_J)
                ]
            elif hasattr(wavelet_op, "kernels") and wavelet_op.kernels is not None:
                self._wav_kernels = list(wavelet_op.kernels)

            if hasattr(wavelet_op, "j_to_dg"):
                self._wav_j_to_dg = list(wavelet_op.j_to_dg)

            if hasattr(wavelet_op, "smooth_kernel_pbc"):
                self._wav_smooth_pbc   = wavelet_op.smooth_kernel_pbc
                self._wav_smooth_nopbc = wavelet_op.smooth_kernel_nopbc
            elif hasattr(wavelet_op, "_gaussian_kernel_5x5"):
                _dev  = getattr(wavelet_op, "device", torch.device("cpu"))
                _dtyp = getattr(wavelet_op, "dtype", torch.float32)
                _sk4  = wavelet_op._gaussian_kernel_5x5(device=_dev, dtype=_dtyp)[None, None]
                self._wav_smooth_pbc   = _sk4
                self._wav_smooth_nopbc = _sk4

            if self._wav_kernels is not None:
                self._wav_L = self._wav_kernels[0].shape[1]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _lac_band(self, flat: torch.Tensor, pad_mode: str) -> torch.Tensor:
        """Returns [B, E] lacunarity at each window size."""
        return torch.stack(
            [_lac_one_scale(flat, k, pad_mode) for k in self._kernels],
            dim=-1,
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def lacunarity(self, data):
        """
        Compute differentiable lacunarity at all window sizes.

        Parameters
        ----------
        data : object with .array, .N0, .pbc

        Returns
        -------
        dict
            ``'lac'``     — ``[Nb, Nc, E]``
            ``'lac_wav'`` — ``[Nb, Nc, E, J, L]``  (only when J>0 + wavelet_op)
        """
        _check_data(data, self.shape)

        arr = data.array
        if torch.is_complex(arr):
            arr = arr.abs()
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        flat     = arr.reshape(Nb * Nc, Nx, Ny).to(self.device, self.dtype)
        pbc      = getattr(data, "pbc", True)
        pad_mode = "circular" if pbc else "replicate"
        E        = len(self.window_sizes)

        # ── J = 0 or no wavelet kernels ───────────────────────────────────────
        if self.J == 0 or self._wav_kernels is None:
            return {"lac": self._lac_band(flat, pad_mode).view(Nb, Nc, E)}

        # ── J > 0 : raw + wavelet lacunarity ──────────────────────────────────
        lac_raw  = self._lac_band(flat, pad_mode)              # [Nb*Nc, E]
        J_eff    = min(self.J, len(self._wav_kernels))
        smooth_k = self._wav_smooth_pbc if pbc else self._wav_smooth_nopbc

        wav_per_j = []
        for j in range(J_eff):
            dg_j = self._wav_j_to_dg[j] if self._wav_j_to_dg is not None else j
            ds   = (
                _downsample_to_scale(flat, dg_j, smooth_k, pad_mode)
                if smooth_k is not None else flat
            )
            moduli = _apply_wavelet_modulus(ds, self._wav_kernels[j], pbc)
            L_j    = moduli.shape[1]
            # [Nb*Nc, E, L_j]
            wav_per_j.append(
                torch.stack(
                    [self._lac_band(moduli[:, l, :, :], pad_mode) for l in range(L_j)],
                    dim=-1,
                )
            )

        # [Nb*Nc, E, J, L] → [Nb, Nc, E, J, L]
        lac_wav = torch.stack(wav_per_j, dim=2).view(Nb, Nc, E, J_eff, self._wav_L)

        return {
            "lac":     lac_raw.view(Nb, Nc, E),
            "lac_wav": lac_wav,
        }


###############################################################################


class GLCMOperator2D:
    """
    Differentiable Gray-Level Co-occurrence Matrix (GLCM) descriptor.

    Replaces hard histogram binning with a soft sigmoid approximation so that
    all five GLCM statistics are fully differentiable w.r.t. the input image.

    Five statistics per (distance, direction) configuration:
    **contrast**, **correlation**, **energy**, **homogeneity**, **entropy**.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape ``(Nx, Ny)`` matching ``data.N0``.
    n_bins : int
        Number of gray-level bins N.  Bin edges are uniformly spaced in
        ``val_range``.
    distances : sequence of int
        Co-occurrence pixel distances d.  One set of statistics is returned
        per distance.
    val_range : (float, float)
        Intensity range ``(lo, hi)`` used to build the bin edges.
        Defaults to ``(0.0, 1.0)``.
    temperature : float
        Sigmoid sharpness for soft histogram binning.  Higher → harder bins.
    rotation_invariant : bool
        If True, averages the GLCM over the four canonical directions
        (0°, 45°, 90°, 135°), making the statistics invariant to in-plane
        rotation.  If False, uses only 0° (dx=d) and 90° (dy=d).
    J : int
        Wavelet scales.  ``J = 0`` — raw image only.
        ``J > 0`` + ``wavelet_op`` — also return ``'*_wav'`` keys of
        shape ``[Nb, Nc, D, J, L]``.
    wavelet_op : WaveletOperator2Dkernel_torch / ST_Operator / None
    device, dtype

    Examples
    --------
    >>> glcm_op = GLCMOperator2D(shape=(128, 128), n_bins=8, distances=[1, 2],
    ...                          rotation_invariant=True)
    >>> out = glcm_op.glcm(data)
    >>> # out['contrast'].shape == [Nb, Nc, 2]   (2 distances)
    """

    # Canonical co-occurrence directions: (dy, dx) at distance 1
    _DIRECTIONS_4 = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]   # 0°,45°,90°,135°
    _DIRECTIONS_2 = [(0, 1), (-1, 0)]                        # 0°,90° only

    _STAT_KEYS = ("contrast", "correlation", "energy", "homogeneity", "entropy")

    def __init__(
        self,
        shape,
        n_bins: int = 8,
        distances=(1,),
        val_range=(0.0, 1.0),
        temperature: float = 10.0,
        rotation_invariant: bool = True,
        J: int = 0,
        wavelet_op=None,
        device=_DEFAULT_DEVICE,
        dtype=_DEFAULT_DTYPE,
    ):
        self.shape              = tuple(shape)
        self.n_bins             = n_bins
        self.distances          = list(distances)
        self.val_range          = val_range
        self.temperature        = temperature
        self.rotation_invariant = rotation_invariant
        self.J                  = J
        self.device             = _get_device(torch.device(device))
        self.dtype              = _get_dtype(dtype=dtype, device=self.device)

        # Bin edges [N+1]
        self.thresholds = torch.linspace(
            val_range[0], val_range[1], n_bins + 1,
            dtype=self.dtype, device=self.device,
        )

        # Scaled offsets per distance
        dirs = self._DIRECTIONS_4 if rotation_invariant else self._DIRECTIONS_2
        self._offsets_per_dist = [
            [(d * dy, d * dx) for (dy, dx) in dirs]
            for d in self.distances
        ]

        # ── Wavelet infrastructure (identical to MinkowskiOperator2D) ─────────
        self.wavelet_op        = wavelet_op
        self._wav_kernels      = None
        self._wav_j_to_dg      = None
        self._wav_smooth_pbc   = None
        self._wav_smooth_nopbc = None
        self._wav_L: int       = 0

        if wavelet_op is not None:
            if hasattr(wavelet_op, "_decimated_kernels"):
                self._wav_kernels = list(wavelet_op._decimated_kernels)
            elif hasattr(wavelet_op, "_wav_kernel"):
                _J = getattr(wavelet_op, "J", 1)
                self._wav_kernels = [
                    (wavelet_op._wav_kernel_0
                     if j == 0 and hasattr(wavelet_op, "_wav_kernel_0")
                     else wavelet_op._wav_kernel)
                    for j in range(_J)
                ]
            elif hasattr(wavelet_op, "kernels") and wavelet_op.kernels is not None:
                self._wav_kernels = list(wavelet_op.kernels)

            if hasattr(wavelet_op, "j_to_dg"):
                self._wav_j_to_dg = list(wavelet_op.j_to_dg)

            if hasattr(wavelet_op, "smooth_kernel_pbc"):
                self._wav_smooth_pbc   = wavelet_op.smooth_kernel_pbc
                self._wav_smooth_nopbc = wavelet_op.smooth_kernel_nopbc
            elif hasattr(wavelet_op, "_gaussian_kernel_5x5"):
                _dev  = getattr(wavelet_op, "device", torch.device("cpu"))
                _dtyp = getattr(wavelet_op, "dtype", torch.float32)
                _sk4  = wavelet_op._gaussian_kernel_5x5(device=_dev, dtype=_dtyp)[None, None]
                self._wav_smooth_pbc   = _sk4
                self._wav_smooth_nopbc = _sk4

            if self._wav_kernels is not None:
                self._wav_L = self._wav_kernels[0].shape[1]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _glcm_band(self, flat: torch.Tensor, pad_mode: str) -> dict:
        """
        Returns dict of 5 statistics, each [B, D] where D = len(distances).
        """
        per_dist = []
        for offsets in self._offsets_per_dist:
            # Average soft GLCM over all directions for this distance
            glcms   = [_soft_glcm(flat, off, self.thresholds,
                                  self.temperature, pad_mode) for off in offsets]
            glcm_avg = torch.stack(glcms, dim=0).mean(dim=0)   # [B, N, N]
            per_dist.append(_glcm_statistics(glcm_avg))

        # Stack over distances: each stat [B, D]
        return {
            k: torch.stack([pd[k] for pd in per_dist], dim=-1)
            for k in self._STAT_KEYS
        }

    # ── Public method ─────────────────────────────────────────────────────────

    def glcm(self, data):
        """
        Compute differentiable GLCM statistics.

        Parameters
        ----------
        data : object with .array, .N0, .pbc

        Returns
        -------
        dict
            ``'contrast'``, ``'correlation'``, ``'energy'``,
            ``'homogeneity'``, ``'entropy'`` — each ``[Nb, Nc, D]``.

            When J>0 + wavelet_op: ``'contrast_wav'``, … each
            ``[Nb, Nc, D, J, L]``.
        """
        _check_data(data, self.shape)

        arr = data.array
        if torch.is_complex(arr):
            arr = arr.abs()
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[None]
        Nb, Nc, Nx, Ny = arr.shape
        flat     = arr.reshape(Nb * Nc, Nx, Ny).to(self.device, self.dtype)
        pbc      = getattr(data, "pbc", True)
        pad_mode = "circular" if pbc else "replicate"
        D        = len(self.distances)

        def _reshape(stats):
            return {k: stats[k].view(Nb, Nc, D) for k in self._STAT_KEYS}

        # ── J = 0 or no wavelet kernels ───────────────────────────────────────
        if self.J == 0 or self._wav_kernels is None:
            return _reshape(self._glcm_band(flat, pad_mode))

        # ── J > 0 : raw + wavelet GLCM ───────────────────────────────────────
        raw      = _reshape(self._glcm_band(flat, pad_mode))
        J_eff    = min(self.J, len(self._wav_kernels))
        smooth_k = self._wav_smooth_pbc if pbc else self._wav_smooth_nopbc

        wav_per_j = []
        for j in range(J_eff):
            dg_j = self._wav_j_to_dg[j] if self._wav_j_to_dg is not None else j
            ds   = (
                _downsample_to_scale(flat, dg_j, smooth_k, pad_mode)
                if smooth_k is not None else flat
            )
            moduli = _apply_wavelet_modulus(ds, self._wav_kernels[j], pbc)
            L_j    = moduli.shape[1]
            per_l  = [
                self._glcm_band(moduli[:, l, :, :], pad_mode)
                for l in range(L_j)
            ]
            # each stat: [Nb*Nc, D, L_j]
            wav_per_j.append(
                {k: torch.stack([pl[k] for pl in per_l], dim=-1)
                 for k in self._STAT_KEYS}
            )

        # Stack over J → [Nb*Nc, D, J, L] → [Nb, Nc, D, J, L]
        wav = {
            k: torch.stack([wj[k] for wj in wav_per_j], dim=2).view(
                Nb, Nc, D, J_eff, self._wav_L
            )
            for k in self._STAT_KEYS
        }

        result = {k: raw[k] for k in self._STAT_KEYS}
        result.update({f"{k}_wav": wav[k] for k in self._STAT_KEYS})
        return result
