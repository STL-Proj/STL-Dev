# -*- coding: utf-8 -*-
"""
Learnable Morlet wavelet kernels for WaveletOperator2Dkernel_torch.

Two modes, selected by ``weight_free``:

1. PARAMETRIC (``weight_free=False``, default) -- learn a few interpretable
   Morlet parameters per family p:
       carrier   : mu_vec[l,p] = R_l @ [mu0 + delta_kx[p], delta_ky[p]]
       bandwidth : sigma_p     = sigma0 * exp(log_sigma[p])
       anisotropy: gamma_p     = exp(log_aniso[p])
       phase     : phase[p]
   R_l (rotation) and the dyadic carrier mu0 stay FIXED.

2. WEIGHT-FREE (``weight_free=True``) -- learn a full complex weight on each
   of the K*K pixels, independently for every (family, orientation):
       kernel[p,l] = complex(free_w_real[p,l], free_w_imag[p,l])   # [K,K]
   This is a CEILING experiment: the first-layer filter is unconstrained
   (imposed rotations R_l are dropped), so it measures how much any 5x5
   first-layer filter could improve the embedding. Zero-mean is enforced
   (wavelet admissibility). A proximity regularizer (see ``reg_proximity``)
   pulls the free weights toward the canonical Morlet, letting you
   interpolate continuously between Morlet (large lambda) and fully free
   (lambda = 0).

In BOTH modes, at initialization the generated bank reproduces EXACTLY
STL-Dev's ``_build_morlet_wavelet_kernel()`` (family 0; families p>0 get a
small symmetry-breaking jitter), so the fixed baseline and the learnable
model remain strictly comparable.

Notes
-----
- Kernel backend shares a single kernel across scales j (scale = successive
  downsampling), so no parameter can depend on j here. Per-scale variants
  need the FFT backend.
- Global per-family phase is analytically inactive on intra-family
  modulus/covariance coefficients (kept only for API completeness). This
  does NOT apply to weight-free mode, where each pixel weight is active.
- Units: delta_k in frequency pixels of the N-grid; rad/pixel = 2*pi*dk/N.
"""

import numpy as np
import torch
import torch.nn as nn


class LearnableMorletKernel2D(nn.Module):
    """Bank of P families of learnable wavelet kernels [P, L, K, K] (complex).

    Parameters
    ----------
    L : int
        Number of orientations (FIXED rotations pi*l/L in parametric mode).
    kernel_size : int
        Spatial kernel size K (odd). Must match the wavelet operator KERNELSZ.
    P : int
        Number of wavelet families. Family 0 = canonical STL Morlet;
        families p>0 get a small symmetry-breaking jitter.
    N : int
        Fourier grid size used to build the canonical kernel (STL uses 256).
    init_jitter : float
        Relative amplitude of the symmetry-breaking jitter for families p>0.
    learnable : bool
        If False, parameters are frozen (fixed baseline).
    weight_free : bool
        If True, learn a full complex weight per pixel per (p, l) instead of
        the parametric Morlet (ceiling experiment; rotations not imposed).
    dtype : torch.dtype
        Real dtype of the construction grids (float32 matches STL Morlet).
    """

    def __init__(
        self,
        L,
        kernel_size=5,
        P=1,
        N=256,
        init_jitter=0.03,
        learnable=True,
        weight_free=False,
        dtype=torch.float32,
        device=None,
        seed=1234,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.L = L
        self.K = kernel_size
        self.P = P
        self.N = N
        self.weight_free = weight_free

        # FIXED geometry: reference carrier and Fourier-envelope width.
        self.mu0 = float(N) / (2.0 * np.sqrt(2.0))
        self.sigma0 = self.mu0 / (2.0 * np.sqrt(2.0))

        angles = np.pi * torch.arange(L, dtype=dtype) / L
        self.register_buffer("cos_a", torch.cos(angles))
        self.register_buffer("sin_a", torch.sin(angles))

        x = torch.arange(N, dtype=dtype)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)
        dc_mask = torch.ones(N, N, dtype=dtype)
        dc_mask[0, 0] = 0.0
        self.register_buffer("dc_mask", dc_mask)

        gen = torch.Generator().manual_seed(seed)

        def _init(scale):
            v = torch.randn(P, generator=gen, dtype=dtype) * scale
            v[0] = 0.0
            return v

        # Parametric parameters (always defined; frozen/unused in weight-free).
        self.delta_kx = nn.Parameter(_init(init_jitter * self.mu0),
                                     requires_grad=learnable and not weight_free)
        self.delta_ky = nn.Parameter(_init(init_jitter * self.mu0),
                                     requires_grad=learnable and not weight_free)
        self.log_sigma = nn.Parameter(_init(init_jitter),
                                      requires_grad=learnable and not weight_free)
        self.log_aniso = nn.Parameter(_init(init_jitter),
                                      requires_grad=learnable and not weight_free)
        self.phase = nn.Parameter(_init(init_jitter),
                                  requires_grad=learnable and not weight_free)

        # --- Canonical bank [P, L, K, K] at zero parameters (parity anchor) ---
        with torch.no_grad():
            zeros = torch.zeros(P, dtype=dtype)
            canon = self._build_parametric(zeros, zeros, zeros, zeros, zeros)  # [P,L,K,K]

        if weight_free:
            # Free complex pixel weights, initialized on the canonical Morlet.
            wr = canon.real.clone()
            wi = canon.imag.clone()
            if init_jitter > 0 and P > 1:
                # break symmetry for families p>0 (family 0 stays canonical)
                jr = torch.randn(P, L, self.K, self.K, generator=gen, dtype=dtype)
                ji = torch.randn(P, L, self.K, self.K, generator=gen, dtype=dtype)
                scale = init_jitter * canon.abs().mean()
                jr[0] = 0.0; ji[0] = 0.0
                wr = wr + scale * jr
                wi = wi + scale * ji
            self.free_w_real = nn.Parameter(wr, requires_grad=learnable)
            self.free_w_imag = nn.Parameter(wi, requires_grad=learnable)
            # proximity target = canonical Morlet (zero-meaned, see forward)
            cz = canon - canon.mean(dim=(-2, -1), keepdim=True)
            self.register_buffer("free_w_init", cz)

        if device is not None:
            self.to(device)

    ###########################################################################
    def _build_parametric(self, dkx, dky, log_sigma, log_aniso, phase):
        """Parametric Morlet bank [P, L, K, K] from per-family [P] parameters."""
        N, K = self.N, self.K
        c = N / 2.0
        radial = self.mu0 + dkx  # [P]
        cx = c - radial[None, :] * self.sin_a[:, None] \
             + dky[None, :] * self.cos_a[:, None]
        cy = c + radial[None, :] * self.cos_a[:, None] \
             + dky[None, :] * self.sin_a[:, None]
        dX = self.X[None, None] - cx[..., None, None]  # [L, P, N, N]
        dY = self.Y[None, None] - cy[..., None, None]
        d_r = -self.sin_a[:, None, None, None] * dX + self.cos_a[:, None, None, None] * dY
        d_t = self.cos_a[:, None, None, None] * dX + self.sin_a[:, None, None, None] * dY
        sigma_p = self.sigma0 * torch.exp(log_sigma)
        gam_sq = torch.exp(0.5 * log_aniso)
        sig_r = (sigma_p / gam_sq)[None, :, None, None]
        sig_t = (sigma_p * gam_sq)[None, :, None, None]
        G = torch.exp(-(d_r**2 / (2.0 * sig_r**2) + d_t**2 / (2.0 * sig_t**2)))
        G = G * (G.detach() >= 0.1)
        bank = torch.fft.fftshift(G, dim=(-2, -1)) * self.dc_mask / 0.8
        kernel = torch.fft.fftshift(torch.fft.ifft2(bank, dim=(-2, -1)), dim=(-2, -1))
        w = K // 2
        kernel = kernel[..., N // 2 - w : N // 2 + w + 1, N // 2 - w : N // 2 + w + 1]
        kernel = kernel - kernel.mean(dim=(-2, -1), keepdim=True)
        kernel = kernel * torch.exp(1j * phase.to(kernel.real.dtype))[None, :, None, None]
        return kernel.permute(1, 0, 2, 3).contiguous()  # [P, L, K, K]

    ###########################################################################
    def forward(self):
        """Build the kernel bank [P, L, K, K] (complex), differentiable."""
        if self.weight_free:
            kernel = torch.complex(self.free_w_real, self.free_w_imag)
            # enforce zero mean (wavelet admissibility, matches STL)
            return kernel - kernel.mean(dim=(-2, -1), keepdim=True)
        return self._build_parametric(self.delta_kx, self.delta_ky,
                                      self.log_sigma, self.log_aniso, self.phase)

    ###########################################################################
    def reg_proximity(self):
        """Proximity-to-canonical-Morlet regularizer (scalar, both modes).

        - weight-free: mean squared deviation of the pixel weights from the
          canonical Morlet (interpolates Morlet <-> free as lambda varies);
        - parametric: normalized carrier + shape deviation
          sum_p |dk|^2/mu0^2 + log_sigma^2 + log_aniso^2.
        Phase is handled separately by the caller if desired.
        """
        if self.weight_free:
            k = torch.complex(self.free_w_real, self.free_w_imag)
            k = k - k.mean(dim=(-2, -1), keepdim=True)
            return (k - self.free_w_init).abs().pow(2).mean()
        r_k = ((self.delta_kx**2 + self.delta_ky**2) / self.mu0**2).sum()
        r_shape = (self.log_sigma**2 + self.log_aniso**2).sum()
        return r_k + r_shape

    ###########################################################################
    def extra_repr(self):
        mode = "weight_free" if self.weight_free else "parametric"
        return (f"L={self.L}, K={self.K}, P={self.P}, N={self.N}, mode={mode}, "
                f"mu0={self.mu0:.2f}, sigma0={self.sigma0:.2f}")

    ###########################################################################
    @torch.no_grad()
    def summary(self):
        """Human-readable dump of the learned parameters."""
        if self.weight_free:
            k = self.forward().cpu()
            init = self.free_w_init.cpu()
            lines = [f"weight-free mode: {self.P}x{self.L} free {self.K}x{self.K} "
                     f"complex kernels ({self.P * self.L * self.K * self.K * 2} params)"]
            for p in range(self.P):
                drift = (k[p] - init[p]).abs().mean().item()
                nrm = k[p].abs().pow(2).mean().sqrt().item()
                lines.append(f"  family {p}: mean|kernel-Morlet|={drift:.4f}  "
                             f"rms|kernel|={nrm:.4f}")
            return "\n".join(lines)
        dkx = self.delta_kx.cpu().numpy(); dky = self.delta_ky.cpu().numpy()
        ls = self.log_sigma.cpu().numpy(); la = self.log_aniso.cpu().numpy()
        ph = self.phase.cpu().numpy()
        lines = [f"parametric: mu0={self.mu0:.2f}, sigma0={self.sigma0:.2f} (N={self.N})"]
        for p in range(self.P):
            lines.append(
                f"  family {p}: dkx={dkx[p]:+.3f} dky={dky[p]:+.3f} "
                f"|dk|/mu0={np.hypot(dkx[p], dky[p]) / self.mu0:.4f} "
                f"sigma x{np.exp(ls[p]):.3f} gamma={np.exp(la[p]):.3f} "
                f"phase={ph[p]:+.3f}")
        return "\n".join(lines)
