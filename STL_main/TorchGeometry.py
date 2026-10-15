"""HEALPix convolution geometry computed on the device, in torch.

healpix-analyse builds the gather indices and bilinear weights of
``HealPixConv`` on the CPU (healpix-geo for pixel centres and the
interpolation, then a torch binding); at 16 us per pixel this is a hundred
times the cost of the convolution itself and the bottleneck of the
patch-local engine, which rebuilds the geometry of every block at every
evaluation.  This module reproduces the same pipeline in vectorised torch:

1. NESTED pixel centres (``nest2xyf`` + ``xyf2ang``, HEALPix reference
   formulas);
2. rotation of the North-Pole tangent stencil to each pixel and gauge
   (same ``_build_rotation_matrices`` as the library);
3. bilinear interpolation of every rotated direction (``get_interpol`` of
   the HEALPix C++ library in the RING scheme, then ``ring2nest``), the
   algorithm healpix-analyse's ``get_interp_weights`` ports for the sphere;
4. binding on the (partial) grid with renormalised weights
   (``_bind_support_batched``).

``torch_conv`` returns an object with the attributes the explicit engine
reads (``_pos_safe``, ``_w_norm``, ``_sort_order``, ``_inv_order``,
``weight``, ``bias``, ``G``, ``K``, ``P``, ``in_channels``,
``out_channels``, ``dtype``, ``use_norm``) and a ``__call__`` through
``AdjointOps.convolution_forward``.  Only ``ellipsoid="sphere"`` is
supported; ``LocalGeometry`` falls back to healpix-analyse otherwise.
The kernel is cast through float32 like ``HealPixConv.set_kernel`` so that
results match the library bit for bit.
"""

import math

import numpy as np
import torch

JRLL = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)


# ---------------------------------------------------------------------------
# NESTED <-> (x, y, face) <-> angles, RING <-> NESTED
# ---------------------------------------------------------------------------


def _compress_bits(v):
    """Keep the even bits of ``v`` (int64), packed: inverse of ``_spread_bits``."""
    v = v & 0x5555555555555555
    v = (v | (v >> 1)) & 0x3333333333333333
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFF
    v = (v | (v >> 16)) & 0x00000000FFFFFFFF
    return v


def _spread_bits(v):
    """Interleave zeros between the bits of ``v`` (v < 2**32)."""
    v = v & 0x00000000FFFFFFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def nest2xyf(pix, order):
    face = pix >> (2 * order)
    local = pix & ((1 << (2 * order)) - 1)
    return _compress_bits(local), _compress_bits(local >> 1), face


def xyf2nest(ix, iy, face, order):
    return (face << (2 * order)) + _spread_bits(ix) + (_spread_bits(iy) << 1)


def xyf2ang(ix, iy, face, nside):
    """Colatitude and longitude (radians) of pixel centres (healpix_base ``pix2loc``)."""
    nside_t = torch.as_tensor(nside, dtype=torch.long, device=ix.device)
    jrll = torch.tensor(JRLL, dtype=torch.long, device=ix.device)[face]
    jpll = torch.tensor(JPLL, dtype=torch.long, device=ix.device)[face]
    jr = jrll * nside - ix - iy - 1
    fact2 = 4.0 / (12.0 * nside * nside)
    fact1 = 2.0 / (3.0 * nside)
    north = jr < nside
    south = jr > 3 * nside
    nr = torch.where(north, jr, torch.where(south, 4 * nside - jr, nside_t))
    kshift = torch.where(north | south, torch.zeros_like(jr), (jr - nside) & 1)
    nrf = nr.to(torch.float64)
    tmp = nrf * nrf * fact2
    z_cap = 1.0 - tmp
    sth_cap = torch.sqrt(tmp * (2.0 - tmp))
    z_eq = (2.0 * nside - jr.to(torch.float64)) * fact1
    z = torch.where(north, z_cap, torch.where(south, -z_cap, z_eq))
    theta_cap = torch.atan2(sth_cap, z_cap)
    theta = torch.where(
        north, theta_cap, torch.where(south, math.pi - theta_cap, torch.acos(z_eq))
    )
    jp = (jpll * nr + ix - iy + 1 + kshift) // 2
    jp = torch.where(jp > 4 * nr, jp - 4 * nr, jp)
    jp = torch.where(jp < 1, jp + 4 * nr, jp)
    phi = (jp.to(torch.float64) - (kshift.to(torch.float64) + 1.0) * 0.5) * (
        0.5 * math.pi / nrf
    )
    return theta, phi


def _isqrt(v):
    r = torch.sqrt(v.to(torch.float64)).to(torch.long)
    r = torch.where(r * r > v, r - 1, r)
    r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


def ring2nest(pix, order):
    """RING -> NESTED pixel numbers (healpix_base ``ring2xyf`` then ``xyf2nest``)."""
    nside = 1 << order
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    nl2 = 2 * nside
    device = pix.device
    jrll = torch.tensor(JRLL, dtype=torch.long, device=device)
    jpll = torch.tensor(JPLL, dtype=torch.long, device=device)

    north = pix < ncap
    south = pix >= npix - ncap
    equator = ~(north | south)

    iring = torch.zeros_like(pix)
    iphi = torch.zeros_like(pix)
    kshift = torch.zeros_like(pix)
    nr = torch.zeros_like(pix)
    face = torch.zeros_like(pix)

    # North polar cap.
    ir_n = (1 + _isqrt(1 + 2 * pix)) >> 1
    ip_n = pix + 1 - 2 * ir_n * (ir_n - 1)
    iring = torch.where(north, ir_n, iring)
    iphi = torch.where(north, ip_n, iphi)
    nr = torch.where(north, ir_n, nr)
    face = torch.where(north, (ip_n - 1) // ir_n.clamp_min(1), face)

    # Equatorial belt.
    ip_e = pix - ncap
    tmp = ip_e >> (order + 2)
    ir_e = tmp + nside
    ip_e2 = ip_e - tmp * 4 * nside + 1
    ks_e = (ir_e + nside) & 1
    ire = ir_e - nside + 1
    irm = nl2 + 2 - ire
    ifm = (ip_e2 - ire // 2 + nside - 1) >> order
    ifp = (ip_e2 - irm // 2 + nside - 1) >> order
    face_e = torch.where(ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8))
    iring = torch.where(equator, ir_e, iring)
    iphi = torch.where(equator, ip_e2, iphi)
    kshift = torch.where(equator, ks_e, kshift)
    nr = torch.where(equator, torch.full_like(pix, nside), nr)
    face = torch.where(equator, face_e, face)

    # South polar cap.
    ip_s = npix - pix
    ir_s = (1 + _isqrt(2 * ip_s - 1)) >> 1
    ip_s2 = 4 * ir_s + 1 - (ip_s - 2 * ir_s * (ir_s - 1))
    iring = torch.where(south, 2 * nl2 - ir_s, iring)
    iphi = torch.where(south, ip_s2, iphi)
    nr = torch.where(south, ir_s, nr)
    face = torch.where(south, 8 + (ip_s2 - 1) // ir_s.clamp_min(1), face)

    irt = iring - jrll[face] * nside + 1
    ipt = 2 * iphi - jpll[face] * nr - kshift - 1
    ipt = torch.where(ipt >= nl2, ipt - 8 * nside, ipt)
    ix = (ipt - irt) >> 1
    iy = (-ipt - irt) >> 1
    return xyf2nest(ix, iy, face, order)


# ---------------------------------------------------------------------------
# Bilinear interpolation (RING scheme), port of T_Healpix_Base::get_interpol
# ---------------------------------------------------------------------------


def _ring_above(z, nside):
    az = z.abs()
    eq = az <= 2.0 / 3.0
    out_eq = torch.trunc(nside * (2.0 - 1.5 * z)).to(torch.long)
    iring = torch.trunc(nside * torch.sqrt(3.0 * (1.0 - az))).to(torch.long)
    out_pol = torch.where(z > 0, iring, 4 * nside - iring - 1)
    return torch.where(eq, out_eq, out_pol)


def _ring_info(ring, nside):
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    fact1 = 2.0 / (3.0 * nside)
    fact2 = 1.0 / (3.0 * nside * nside)
    northring = torch.where(ring > 2 * nside, 4 * nside - ring, ring)
    cap = northring < nside
    nrf = northring.to(torch.float64)
    tmp = nrf * nrf * fact2
    theta_cap = torch.atan2(torch.sqrt(tmp * (2.0 - tmp)), 1.0 - tmp)
    theta_eq = torch.acos((2 * nside - northring).to(torch.float64) * fact1)
    theta = torch.where(cap, theta_cap, theta_eq)
    ringpix = torch.where(cap, 4 * northring, torch.full_like(ring, 4 * nside))
    shifted = torch.where(cap, torch.ones_like(cap), ((northring - nside) & 1) == 0)
    startpix = torch.where(
        cap, 2 * northring * (northring - 1), ncap + (northring - nside) * 4 * nside
    )
    south = northring != ring
    theta = torch.where(south, math.pi - theta, theta)
    startpix = torch.where(south, npix - startpix - ringpix, startpix)
    return startpix, ringpix, theta, shifted


def _ring_pair(phi, ring, nside):
    sp, nr, theta_ring, shifted = _ring_info(ring, nside)
    dphi = (2.0 * math.pi) / nr.to(torch.float64)
    half = torch.where(shifted, 0.5, 0.0).to(torch.float64)
    tmp = phi / dphi - half
    i1 = torch.where(tmp < 0, torch.trunc(tmp).to(torch.long) - 1, torch.trunc(tmp).to(torch.long))
    w1 = (phi - (i1.to(torch.float64) + half) * dphi) / dphi
    i2 = i1 + 1
    i1 = torch.where(i1 < 0, i1 + nr, i1)
    i2 = torch.where(i2 >= nr, i2 - nr, i2)
    return sp + i1, sp + i2, 1.0 - w1, w1, theta_ring


def get_interpol_ring(theta, phi, nside):
    """Four RING pixels and weights for directions ``(theta, phi)`` (float64)."""
    npix = 12 * nside * nside
    z = torch.cos(theta)
    ir1 = _ring_above(z, nside)
    ir2 = ir1 + 1
    pix = torch.zeros(theta.shape[0], 4, dtype=torch.long, device=theta.device)
    wgt = torch.zeros(theta.shape[0], 4, dtype=torch.float64, device=theta.device)
    m1 = ir1 > 0
    p0, p1, w0, w1, theta1 = _ring_pair(phi, ir1.clamp_min(1), nside)
    pix[:, 0] = torch.where(m1, p0, 0)
    pix[:, 1] = torch.where(m1, p1, 0)
    wgt[:, 0] = torch.where(m1, w0, 0.0)
    wgt[:, 1] = torch.where(m1, w1, 0.0)
    m2 = ir2 < 4 * nside
    p2, p3, w2, w3, theta2 = _ring_pair(phi, ir2.clamp_max(4 * nside - 1), nside)
    pix[:, 2] = torch.where(m2, p2, 0)
    pix[:, 3] = torch.where(m2, p3, 0)
    wgt[:, 2] = torch.where(m2, w2, 0.0)
    wgt[:, 3] = torch.where(m2, w3, 0.0)

    north = ir1 == 0
    south = (ir2 == 4 * nside) & ~north
    mid = ~north & ~south
    wn = theta / theta2
    facn = (1.0 - wn) * 0.25
    ws = (theta - theta1) / (math.pi - theta1)
    facs = ws * 0.25
    wm = (theta - theta1) / (theta2 - theta1)

    w0n, w1n = facn, facn
    w2n, w3n = wgt[:, 2] * wn + facn, wgt[:, 3] * wn + facn
    w0s, w1s = wgt[:, 0] * (1.0 - ws) + facs, wgt[:, 1] * (1.0 - ws) + facs
    w2s, w3s = facs, facs
    w0m, w1m = wgt[:, 0] * (1.0 - wm), wgt[:, 1] * (1.0 - wm)
    w2m, w3m = wgt[:, 2] * wm, wgt[:, 3] * wm
    out_w = torch.stack(
        [
            torch.where(north, w0n, torch.where(south, w0s, w0m)),
            torch.where(north, w1n, torch.where(south, w1s, w1m)),
            torch.where(north, w2n, torch.where(south, w2s, w2m)),
            torch.where(north, w3n, torch.where(south, w3s, w3m)),
        ],
        dim=1,
    )
    p0n, p1n = (pix[:, 2] + 2) & 3, (pix[:, 3] + 2) & 3
    p2s, p3s = ((pix[:, 0] + 2) & 3) + npix - 4, ((pix[:, 1] + 2) & 3) + npix - 4
    out_p = torch.stack(
        [
            torch.where(north, p0n, pix[:, 0]),
            torch.where(north, p1n, pix[:, 1]),
            torch.where(south, p2s, pix[:, 2]),
            torch.where(south, p3s, pix[:, 3]),
        ],
        dim=1,
    )
    return out_p, out_w


def interp_weights_nested(vecs, order):
    """Four NESTED pixels and weights of unit vectors ``[M, 3]`` (sphere).

    Follows ``healpix_analyse.convol._get_interp_weights``: direction ->
    (theta, phi) -> degrees -> ``get_interp_weights`` (radians again) so that
    the rounding path is the same as the library's.
    """
    vn = vecs / vecs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    theta = torch.acos(vn[:, 2].clamp(-1.0, 1.0))
    phi = torch.atan2(vn[:, 1], vn[:, 0]) % (2.0 * math.pi)
    lon_deg = torch.rad2deg(phi)
    lat_deg = 90.0 - torch.rad2deg(theta)
    theta2 = torch.deg2rad(90.0 - lat_deg)
    phi2 = torch.deg2rad(lon_deg) % (2.0 * math.pi)
    pix_ring, weights = get_interpol_ring(theta2, phi2, 1 << order)
    return ring2nest(pix_ring.reshape(-1), order).reshape(-1, 4), weights


# ---------------------------------------------------------------------------
# Stencil rotation and binding (ports of the library helpers)
# ---------------------------------------------------------------------------


def rotation_matrices(theta, phi, G, gauge_type, dtype):
    """``_build_rotation_matrices`` of healpix-analyse (``phi``/``cosmo`` gauges)."""
    device = theta.device
    th_t, ph_t = theta.to(dtype), phi.to(dtype)
    K = th_t.shape[0]
    ct, st = torch.cos(th_t), torch.sin(th_t)
    cp, sp = torch.cos(ph_t), torch.sin(ph_t)
    R_base = torch.zeros(K, 3, 3, device=device, dtype=dtype)
    R_base[:, 0, 0] = cp * ct
    R_base[:, 0, 1] = -sp
    R_base[:, 0, 2] = cp * st
    R_base[:, 1, 0] = sp * ct
    R_base[:, 1, 1] = cp
    R_base[:, 1, 2] = sp * st
    R_base[:, 2, 0] = -st
    R_base[:, 2, 1] = 0.0
    R_base[:, 2, 2] = ct
    n = R_base[:, :, 2]
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if gauge_type == "cosmo":
        is_south = th_t > math.pi / 2
        alpha_base = torch.where(is_south, ph_t, -ph_t)
        sign_g = torch.where(is_south, -torch.ones_like(th_t), torch.ones_like(th_t))
    elif gauge_type == "phi":
        alpha_base = torch.zeros_like(th_t)
        sign_g = torch.ones_like(th_t)
    else:
        raise NotImplementedError("torch geometry supports gauge_type 'phi' and 'cosmo'")
    g_shifts = torch.arange(G, device=device, dtype=dtype) * (math.pi / G)
    alpha_g = alpha_base[:, None] + sign_g[:, None] * g_shifts[None, :]
    ca, sa = torch.cos(alpha_g), torch.sin(alpha_g)
    n_g = n[:, None, :].expand(K, G, 3)
    nxg, nyg, nzg = n_g[..., 0], n_g[..., 1], n_g[..., 2]
    K_skew = torch.zeros(K, G, 3, 3, device=device, dtype=dtype)
    K_skew[..., 0, 1] = -nzg
    K_skew[..., 0, 2] = nyg
    K_skew[..., 1, 0] = nzg
    K_skew[..., 1, 2] = -nxg
    K_skew[..., 2, 0] = -nyg
    K_skew[..., 2, 1] = nxg
    outer = n_g.unsqueeze(-1) * n_g.unsqueeze(-2)
    I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    R_gauge = (
        I * ca.view(K, G, 1, 1)
        + K_skew * sa.view(K, G, 1, 1)
        + outer * (1.0 - ca).view(K, G, 1, 1)
    )
    return torch.matmul(R_gauge, R_base[:, None, :, :].expand(K, G, 3, 3))


def local_kernel_grid(kernel_sz, nside, device, dtype):
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    alpha_pix = np.sqrt(4 * np.pi / (12 * nside**2))
    dtheta = np.sqrt(xx**2 + yy**2).ravel() * alpha_pix
    dphi = np.arctan2(yy, xx).ravel()
    vec = np.stack(
        [np.sin(dtheta) * np.cos(dphi), np.sin(dtheta) * np.sin(dphi), np.cos(dtheta)],
        axis=-1,
    ).astype(np.float64)
    return torch.as_tensor(vec, device=device, dtype=dtype)


def bind_support(idx_t, w_t, ids_sorted, kernel_sz, K, P):
    """``_bind_support_batched`` of healpix-analyse: ``[G, 4, K*P]`` -> (pos, w_norm)."""
    G = idx_t.shape[0]
    M = K * P
    N_sorted = ids_sorted.numel()
    device = idx_t.device
    pos = torch.searchsorted(ids_sorted, idx_t.reshape(-1)).view(G, 4, M)
    in_range = pos < N_sorted
    cmp_vals = torch.full_like(idx_t, -1)
    cmp_vals[in_range] = ids_sorted[pos[in_range]]
    present = cmp_vals == idx_t
    p_ref = (kernel_sz // 2) * (kernel_sz + 1)
    empty = ~present.any(dim=1)
    if bool(empty.any()):
        k_id = torch.div(torch.arange(M, device=device), P, rounding_mode="floor")
        ref_cols = k_id * P + p_ref
        for g in range(G):
            empty_g = empty[g]
            if not bool(empty_g.any()):
                continue
            ref_g = ref_cols[empty_g]
            idx_t[g, :, empty_g] = idx_t[g, :, ref_g]
            w_t[g, :, empty_g] = w_t[g, :, ref_g]
            idx_e = idx_t[g, :, empty_g].reshape(-1)
            pos_e = torch.searchsorted(ids_sorted, idx_e)
            valid_e = pos_e < N_sorted
            pos_e_c = pos_e.clamp(0, max(N_sorted - 1, 0))
            pres_e = valid_e & (ids_sorted[pos_e_c] == idx_e)
            present[g, :, empty_g] = pres_e.view(4, -1)
            pos[g, :, empty_g] = pos_e_c.view(4, -1)
    w = w_t * present
    colsum = w.sum(dim=1, keepdim=True)
    zero_c = colsum == 0
    if bool(zero_c.any()):
        zero_mask = zero_c.squeeze(1)
        w[:, 0, :] = torch.where(zero_mask, present[:, 0, :].to(w.dtype), w[:, 0, :])
        colsum = w.sum(dim=1, keepdim=True)
    w_norm = w / colsum.clamp_min(1e-12)
    pos_safe = torch.where(present, pos, torch.zeros_like(pos))
    # Rows whose every read is present: their result equals the full-sky one.
    complete = present.all(dim=1).view(G, K, P).all(dim=2).all(dim=0)
    return pos_safe, w_norm, complete


# ---------------------------------------------------------------------------
# Anti-aliased decimation (port of HealPixDown("smooth"), sphere)
# ---------------------------------------------------------------------------


def _haversine(lat1, lon1, lat2, lon2):
    dlat = 0.5 * (lat2 - lat1)
    dlon = 0.5 * (lon2 - lon1)
    a = (torch.sin(dlat) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon) ** 2).clamp(
        0.0, 1.0
    )
    return 2.0 * torch.asin(torch.sqrt(a))


def centres_lonlat(ids, nside):
    """Longitude and latitude (radians, float64) of NESTED pixel centres,
    rounded through degrees like the healpix-geo output used by the library."""
    order = int(round(math.log2(nside)))
    theta, phi = xyf2ang(*nest2xyf(ids.to(torch.long), order), nside)
    lon_deg = torch.rad2deg(phi)
    lat_deg = 90.0 - torch.rad2deg(theta)
    return torch.deg2rad(lon_deg), torch.deg2rad(lat_deg)


def smooth_decimation_csr(ids, nside, candidates, dtype, radius_deg=None, sigma_deg=None, weight_norm="l1"):
    """CSR matrices (forward ``[N_out, N_in]`` and transpose) of the smooth
    decimation of ``HealPixDown`` restricted to the fine pixels ``ids``
    (sorted, NESTED, resolution ``nside``).

    ``candidates`` is ``[N_out, M]`` of fine NESTED ids (or -1) that may lie
    within the cone of each coarse pixel (parents of ``ids``, sorted); fine
    pixels outside ``ids`` are unavailable, as in the partial operator.  The
    weights follow ``_build_smooth_matrix``: Gaussian of the haversine
    distance to the coarse centre, zero beyond ``radius``, rows normalised
    over the available pixels, zero entries dropped.  Returns
    ``(coarse_ids, forward, transposed)``.
    """
    device = ids.device
    npix_fine = 12 * nside * nside
    pix_deg = math.degrees(math.sqrt(4.0 * math.pi / npix_fine))
    radius = math.radians(3.0 * pix_deg if radius_deg is None else float(radius_deg))
    sigma = radius / 2.0 if sigma_deg is None else math.radians(float(sigma_deg))
    coarse = torch.unique(torch.div(ids, 4, rounding_mode="floor"))
    n_out, n_in = coarse.numel(), ids.numel()
    lon_c, lat_c = centres_lonlat(coarse, nside // 2)
    # Keep candidate fine pixels that are available (inside ``ids``).
    cand = candidates.reshape(-1)
    valid = cand >= 0
    pos = torch.searchsorted(ids, cand.clamp_min(0))
    pos = pos.clamp_max(max(n_in - 1, 0))
    present = valid & (ids[pos] == cand)
    rows = torch.arange(n_out, device=device).repeat_interleave(candidates.shape[1])
    rows, cols = rows[present], pos[present]
    # Candidate lists may repeat a pair; keep each (row, col) once.
    key = torch.unique(rows * n_in + cols)
    rows, cols = torch.div(key, n_in, rounding_mode="floor"), key % n_in
    lon_f, lat_f = centres_lonlat(ids[cols], nside)
    gamma = _haversine(lat_c[rows], lon_c[rows], lat_f, lon_f)
    w = torch.exp(-0.5 * (gamma / sigma) ** 2)
    w = torch.where(gamma > radius, torch.zeros_like(w), w)
    row_sum = torch.zeros(n_out, dtype=w.dtype, device=device).index_add_(0, rows, w)
    # Rows whose every candidate is cut: uniform over available (library fallback).
    empty = row_sum[rows] <= 0.0
    if bool(empty.any()):
        count = torch.zeros(n_out, dtype=w.dtype, device=device).index_add_(0, rows, torch.ones_like(w))
        w = torch.where(empty, 1.0 / count[rows], w)
        row_sum = torch.zeros(n_out, dtype=w.dtype, device=device).index_add_(0, rows, w)
    if weight_norm == "l1":
        w = w / row_sum[rows].clamp_min(1e-300)
    elif weight_norm == "l2":
        norm = torch.sqrt(torch.zeros(n_out, dtype=w.dtype, device=device).index_add_(0, rows, w * w))
        w = w / norm[rows].clamp_min(1e-300)
    keep = w != 0.0
    rows, cols, w = rows[keep], cols[keep], w[keep].to(dtype)
    forward = torch.sparse_coo_tensor(torch.stack([rows, cols]), w, (n_out, n_in)).coalesce().to_sparse_csr()
    transposed = torch.sparse_coo_tensor(torch.stack([cols, rows]), w, (n_in, n_out)).coalesce().to_sparse_csr()
    return coarse, forward, transposed


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------


class TorchConv:
    """Geometry-compatible stand-in for a linear ``HealPixConv`` (sphere)."""

    use_norm = False

    def __init__(self, ids, nside, kernel_sz, G, gauge_type, weights, dtype, pixel_batch_size=16384):
        """``ids``: NESTED cell ids (long tensor, any order) of the partial grid;
        ``weights``: ``[C_in, C_out, P]`` kernel (cast through float32 as the
        library does)."""
        device = ids.device
        order = int(round(math.log2(nside)))
        if 1 << order != nside:
            raise ValueError("nside must be a power of two")
        self.nside, self.level, self.kernel_sz = nside, order, int(kernel_sz)
        self.G, self.P, self.K = int(G), int(kernel_sz) ** 2, int(ids.numel())
        self.dtype, self.device, self.gauge_type = dtype, device, gauge_type
        self.pixel_batch_size = pixel_batch_size
        ix, iy, face = nest2xyf(ids.to(torch.long), order)
        theta, phi = xyf2ang(ix, iy, face, nside)
        # The library rounds pixel centres through degrees (healpix_geo output).
        theta = torch.deg2rad(90.0 - torch.rad2deg(math.pi / 2 - theta))
        phi = torch.deg2rad(torch.rad2deg(phi))
        R = rotation_matrices(theta, phi, self.G, gauge_type, dtype)  # [K, G, 3, 3]
        vec = local_kernel_grid(kernel_sz, nside, device, dtype)  # [P, 3]
        rotated = torch.einsum("kgij,pj->kgpi", R, vec).reshape(-1, 3)
        idx_flat, w_flat = interp_weights_nested(rotated.to(torch.float64), order)
        idx_all = idx_flat.view(self.K, self.G, self.P, 4).permute(1, 3, 0, 2).reshape(self.G, 4, -1)
        w_all = w_flat.to(dtype).view(self.K, self.G, self.P, 4).permute(1, 3, 0, 2).reshape(self.G, 4, -1)
        ids_sorted, sort_order = torch.sort(ids.to(torch.long))
        inv_order = torch.empty_like(sort_order)
        inv_order[sort_order] = torch.arange(self.K, device=device)
        pos_all, w_norm, complete = bind_support(
            idx_all.clone(), w_all.clone(), ids_sorted, kernel_sz, self.K, self.P
        )
        self._pos_safe = pos_all.permute(1, 0, 2).contiguous()
        self._w_norm = w_norm.permute(1, 0, 2).contiguous()
        self._sort_order, self._inv_order = sort_order, inv_order
        # complete[i] refers to output position i (row k = inv_order[i]).
        self._complete = complete[inv_order]
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 3:
            w = np.broadcast_to(w[None], (self.G,) + w.shape).copy()
        self.in_channels, self.out_channels = int(w.shape[1]), int(w.shape[2])
        self.weight = torch.as_tensor(w, dtype=dtype, device=device)
        self.bias = torch.zeros(self.G * self.out_channels, dtype=dtype, device=device)

    def __call__(self, x):
        from STL_main.AdjointOps import convolution_forward

        return convolution_forward(self, x, self.pixel_batch_size)
