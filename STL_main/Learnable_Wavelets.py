# -*- coding: utf-8 -*-
"""
Learnable Morlet wavelet kernels for WaveletOperator2Dkernel_torch.

Research prototype: keep the scattering-covariance structure (dyadic scales
via successive downsampling, imposed rotations, intra-family covariances
only) but learn a small number of wavelet parameters per family p:

    carrier   : mu_vec[l, p] = R_l @ [ mu0 + delta_kx[p],  delta_ky[p] ]
    bandwidth : sigma_p      = sigma0 * exp(log_sigma[p])
    anisotropy: gamma_p      = exp(log_aniso[p])
                (Fourier envelope elongated along/across the carrier axis:
                 sigma_radial = sigma_p / sqrt(gamma_p),
                 sigma_tangent = sigma_p * sqrt(gamma_p);
                 gamma > 1 -> curvelet-like angular elongation)
    phase     : phase[p]     (global phase)

where
    - mu0 = min(N, N) / (2*sqrt(2)) is the FIXED reference dyadic carrier
      (frequency-pixel units on the N x N Fourier grid used by
      WaveletOperator2Dkernel_torch._build_morlet_wavelet_kernel, N=256),
    - sigma0 = mu0 / (2*sqrt(2)) is the FIXED reference envelope width,
    - R_l is the FIXED rotation of orientation l (angle pi*l/L),
    - delta_kx, delta_ky, log_sigma, log_aniso, phase are LEARNED,
      one value per family p.

At delta_kx = delta_ky = log_sigma = log_aniso = phase = 0 the generated
kernel reproduces EXACTLY STL-Dev's `_build_morlet_wavelet_kernel()` chain:
    gaussian_bank(J, L, (N, N))[0] -> fftshift(ifft2(.)) -> crop KxK
    -> subtract mean
so that the fixed baseline and the learnable model are strictly comparable.

Notes
-----
- In the Kernel backend a SINGLE kernel is shared by all scales j (scale is
  obtained by successive downsampling), hence parameters cannot depend on j
  here. A per-scale variant requires the FFT backend
  (WaveletOperator2D_FFT_torch, wavelet_array [J, L, N, N]).
- A global per-family phase is analytically INACTIVE on all intra-family
  modulus/covariance coefficients (it cancels in |W|, in Cov(., .) between
  filters of the same family, including relative-phase angular pooling).
  Its gradient is therefore expected to be exactly zero with the default
  statistics; the parameter is kept for API completeness and for future
  inter-family / per-scale extensions.
- Units: delta_k are in frequency pixels of the N-grid; in rad/pixel:
  delta_k_rad = 2 * pi * delta_k / N.

Usage
-----
    from STL_main.Learnable_Wavelets import LearnableMorletKernel2D

    kernels = LearnableMorletKernel2D(L=4, kernel_size=5, P=2)
    st_op = data_example.get_ST_op(J=4, L=4, WType="Morlet", compute_PS=False)

    bank = kernels()                                   # [P, L, K, K] complex
    for p in range(kernels.P):
        st_op.wavelet_op.set_wavelet_kernel(bank[p][None])   # [1, L, K, K]
        Sx = st_op.apply(data, norm="vanilla")         # differentiable w.r.t.
                                                       # kernels.parameters()
"""

import numpy as np
import torch
import torch.nn as nn


class LearnableMorletKernel2D(nn.Module):
    """Bank of P families of learnable Morlet kernels [P, L, K, K] (complex).

    Parameters
    ----------
    L : int
        Number of orientations (FIXED rotations, angle = pi * l / L).
    kernel_size : int
        Spatial kernel size K (odd). Must match the wavelet operator KERNELSZ.
    P : int
        Number of wavelet families. Family 0 is initialized exactly on the
        canonical STL Morlet; families p > 0 receive a small random jitter to
        break the gradient symmetry (identical families receive identical
        gradients and would stay cloned forever).
    N : int
        Fourier grid size used to build the kernel (STL-Dev uses N=256 in
        `_build_wavelet_kernel_from_ifft_crop`).
    init_jitter : float
        Relative amplitude of the symmetry-breaking jitter:
        delta_k ~ N(0, (init_jitter * mu0)^2),
        log_sigma, log_aniso, phase ~ N(0, init_jitter^2).
    learnable : bool
        If False, parameters are frozen (useful for the fixed baseline).
    dtype : torch.dtype
        Real dtype of the construction grids. float32 matches the effective
        dtype of `_build_morlet_wavelet_kernel` (gaussian_bank builds float32
        tensors, giving a complex64 kernel).
    """

    def __init__(
        self,
        L,
        kernel_size=5,
        P=1,
        N=256,
        init_jitter=0.03,
        learnable=True,
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

        # FIXED geometry: reference carrier and Fourier-envelope width,
        # identical to gaussian_bank (j=0): base_mu, base_sigma.
        self.mu0 = float(N) / (2.0 * np.sqrt(2.0))
        self.sigma0 = self.mu0 / (2.0 * np.sqrt(2.0))

        angles = np.pi * torch.arange(L, dtype=dtype) / L
        self.register_buffer("cos_a", torch.cos(angles))
        self.register_buffer("sin_a", torch.sin(angles))

        # Construction grids (same convention as gaussian_2d_rotated):
        # X, Y = meshgrid(arange(N), arange(N), 'ij'), center at N/2.
        x = torch.arange(N, dtype=dtype)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)
        dc_mask = torch.ones(N, N, dtype=dtype)
        dc_mask[0, 0] = 0.0  # zero the DC bin (differentiable, via multiply)
        self.register_buffer("dc_mask", dc_mask)

        # LEARNED parameters [P], symmetry-breaking init, family 0 canonical.
        gen = torch.Generator().manual_seed(seed)

        def _init(scale):
            v = torch.randn(P, generator=gen, dtype=dtype) * scale
            v[0] = 0.0
            return v

        self.delta_kx = nn.Parameter(_init(init_jitter * self.mu0),
                                     requires_grad=learnable)
        self.delta_ky = nn.Parameter(_init(init_jitter * self.mu0),
                                     requires_grad=learnable)
        self.log_sigma = nn.Parameter(_init(init_jitter), requires_grad=learnable)
        self.log_aniso = nn.Parameter(_init(init_jitter), requires_grad=learnable)
        self.phase = nn.Parameter(_init(init_jitter), requires_grad=learnable)

        if device is not None:
            self.to(device)

    ###########################################################################
    def forward(self):
        """Build the kernel bank, differentiable w.r.t. all parameters.

        Returns
        -------
        torch.Tensor
            Complex kernel bank of shape [P, L, K, K]. Use
            `bank[p][None]` ([1, L, K, K]) with
            `WaveletOperator2Dkernel_torch.set_wavelet_kernel`.
        """
        N, K = self.N, self.K
        c = N / 2.0  # image center, as in gaussian_2d_rotated

        # Rotated carrier centers, [L, P]:
        # gaussian_2d_rotated convention: center = c + mu * (-sin a, +cos a),
        # generalized with an in-carrier offset delta_kx and an orthogonal
        # offset delta_ky:  center = c + (mu0+dkx) * (-sin a, cos a)
        #                              + dky * (cos a, sin a)
        radial = self.mu0 + self.delta_kx  # [P]
        cx = c - radial[None, :] * self.sin_a[:, None] \
             + self.delta_ky[None, :] * self.cos_a[:, None]
        cy = c + radial[None, :] * self.cos_a[:, None] \
             + self.delta_ky[None, :] * self.sin_a[:, None]

        # Anisotropic Gaussian envelope in Fourier space, axes aligned with
        # the carrier direction e_r = (-sin a, cos a) and its orthogonal
        # e_t = (cos a, sin a). gamma = 1 -> isotropic (canonical Morlet).
        dX = self.X[None, None] - cx[..., None, None]  # [L, P, N, N]
        dY = self.Y[None, None] - cy[..., None, None]
        d_r = -self.sin_a[:, None, None, None] * dX \
              + self.cos_a[:, None, None, None] * dY
        d_t = self.cos_a[:, None, None, None] * dX \
              + self.sin_a[:, None, None, None] * dY
        sigma_p = self.sigma0 * torch.exp(self.log_sigma)          # [P]
        gam_sq = torch.exp(0.5 * self.log_aniso)                   # sqrt(gamma)
        sig_r = (sigma_p / gam_sq)[None, :, None, None]
        sig_t = (sigma_p * gam_sq)[None, :, None, None]
        G = torch.exp(-(d_r**2 / (2.0 * sig_r**2) + d_t**2 / (2.0 * sig_t**2)))

        # Same hard threshold as gaussian_2d_rotated (eps=0.1). The mask is
        # detached: exact parity with STL at init, subgradient elsewhere.
        G = G * (G.detach() >= 0.1)

        # gaussian_bank: fftshift -> zero DC -> /0.8
        bank = torch.fft.fftshift(G, dim=(-2, -1)) * self.dc_mask / 0.8

        # _build_wavelet_kernel_from_ifft_crop: ifft2 -> fftshift -> crop
        kernel = torch.fft.fftshift(torch.fft.ifft2(bank, dim=(-2, -1)),
                                    dim=(-2, -1))
        w = K // 2
        kernel = kernel[..., N // 2 - w : N // 2 + w + 1,
                        N // 2 - w : N // 2 + w + 1]

        # zero mean (as in STL), then global per-family phase
        kernel = kernel - kernel.mean(dim=(-2, -1), keepdim=True)
        kernel = kernel * torch.exp(
            1j * self.phase.to(kernel.real.dtype)
        )[None, :, None, None]

        return kernel.permute(1, 0, 2, 3).contiguous()  # [P, L, K, K]

    ###########################################################################
    def extra_repr(self):
        return (
            f"L={self.L}, K={self.K}, P={self.P}, N={self.N}, "
            f"mu0={self.mu0:.2f}, sigma0={self.sigma0:.2f}"
        )

    ###########################################################################
    @torch.no_grad()
    def summary(self):
        """Human-readable dump of the learned parameters."""
        dkx = self.delta_kx.cpu().numpy()
        dky = self.delta_ky.cpu().numpy()
        ls = self.log_sigma.cpu().numpy()
        la = self.log_aniso.cpu().numpy()
        ph = self.phase.cpu().numpy()
        lines = [f"mu0={self.mu0:.2f} freq-px, sigma0={self.sigma0:.2f} (grid N={self.N})"]
        for p in range(self.P):
            lines.append(
                f"  family {p}: dkx={dkx[p]:+.3f} dky={dky[p]:+.3f} "
                f"|dk|/mu0={np.hypot(dkx[p], dky[p]) / self.mu0:.4f} "
                f"sigma x{np.exp(ls[p]):.3f} gamma={np.exp(la[p]):.3f} "
                f"phase={ph[p]:+.3f}"
            )
        return "\n".join(lines)
