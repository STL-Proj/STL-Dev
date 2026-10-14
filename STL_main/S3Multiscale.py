"""One S3 scale pair at a time, with the fine scale streamed by sub-patches.

S3(j1, j3) with ``j1 < j3`` couples a wavelet of the finest resolution with
one of the coarsest: ``mean( psi_j3 * D^(n-d1) |psi_j1 * x|  .  conj(psi_j3 * x) )``.
Its gradient reads every fine pixel.  The chain is linear from the modulus
upwards, so it decomposes over fine sub-patches ``F_k``:

    m = D^(n-d1) |psi_j1 * x| = sum_k D^(n-d1) ( |psi_j1 * x| . 1_{F_k} )

Each piece is computed with the fine wavelet on ``F_k`` only, pushed through
the decimation chain restricted to its support (``DownPiece``), and summed at
the coarse level, which is small and handled as in the full-sky engine.  The
reverse pass redistributes the coarse cotangent through the transposed
pieces, sub-patch by sub-patch, so no field with an orientation axis ever
exceeds one fine sub-patch (plus its stencil halo).  Memory therefore scales
with ``fine_patch_pixels`` and not with ``4**(j3 - j1)``.

The coarse level can itself be split into cores (``coarse_patch_pixels``),
but by default one coarse core covers the whole level ("old mode" for the
large scales, as suggested by the structure of the problem).
"""

import torch

from STL_main.AdjointOps import modulus_vjp
from STL_main.PatchOps import (
    ConvRows,
    DownPiece,
    conv_adjoint_rows,
    conv_forward_rows,
    down_input_domain,
)
from STL_main.PatchScattering import PatchGeometry
from STL_main.PyramidScattering import PyramidStatistics, build_plan


def s3_pair_plan(operator, channels, j1, j3):
    return tuple(
        c
        for c in build_plan(operator, channels, families=("S3",))
        if c.j1 == j1 and c.j3 == j3
    )


class _Chain:
    """Decimation pieces carrying one fine sub-patch up to level ``n``."""

    def __init__(self, geometry, dg_fine, n, fine_positions):
        self.pieces, support = [], fine_positions
        for level in range(dg_fine, n):
            # No index table here: the coarse table may be filled by the
            # caller, and the pieces' supports are small.
            piece = DownPiece(geometry.down(level), support)
            self.pieces.append(piece)
            support = piece.out_positions
        self.top = support

    def forward(self, x):
        for piece in self.pieces:
            x = piece.forward(x)
        return x

    def adjoint(self, g):
        for piece in reversed(self.pieces):
            g = piece.adjoint(g)
        return g


def _split(x):
    """Complex ``[..., L, m] -> [..., 2L, m]`` in the (real, imag) tap layout."""
    return torch.stack((x.real, x.imag), dim=-2).reshape(*x.shape[:-2], -1, x.shape[-1])


def _join(y, L):
    """Inverse of the conv output layout ``[B, L, 2, m] -> complex [B, L, m]``."""
    y = y.reshape(y.shape[0], L, 2, -1)
    return torch.complex(y[..., 0, :], y[..., 1, :])


@torch.no_grad()
def s3_scale_pair(
    operator,
    pyramid,
    j1,
    j3,
    gradient=None,
    plan=None,
    fine_patch_pixels=4096,
    coarse_patch_pixels=None,
    tracker=None,
    maps=None,
    maps_gradient=None,
):
    """Statistics and map cotangents of the S3 coefficients with scales (j1, j3).

    Returns ``(PyramidStatistics, maps_gradient)``; ``maps_gradient`` is
    ``None`` without ``gradient`` and is otherwise a list of per-level map
    cotangents to pass to ``pyramid.analysis_vjp`` (contributions are added
    into ``maps_gradient`` when one is supplied, so several scale pairs can
    be accumulated).  ``gradient`` has shape ``[batch, len(plan)]``.
    """
    w = operator.wavelet_op
    if not 0 <= j1 <= j3 < w.J:
        raise ValueError("Expected 0 <= j1 <= j3 < J")
    n, d1 = w.j_to_dg[j3], w.j_to_dg[j1]
    channels = pyramid.levels[0].shape[1]
    plan = s3_pair_plan(operator, channels, j1, j3) if plan is None else tuple(plan)
    if any(c.family != "S3" or c.j1 != j1 or c.j3 != j3 for c in plan):
        raise ValueError("Plan must contain S3 coefficients of the requested pair")
    if maps is None:
        maps = pyramid.analysis_levels()
    batch = maps[0].shape[0]
    if gradient is not None:
        if gradient.shape != (batch, len(plan)):
            raise ValueError("Gradient must have shape [batch, len(plan)]")
        if maps_gradient is None:
            maps_gradient = [torch.zeros_like(x) for x in maps]
    L, fewer = w.L, operator.has_fewer_convolutions
    n_total = maps[n].shape[-1]
    geometry = PatchGeometry(
        operator, pyramid, coarse_patch_pixels or n_total
    )
    conv_n, conv_1 = geometry.conv(n), geometry.conv(d1)
    table_n, table_1 = geometry.table(n), geometry.table(d1)
    pixel_batch = w.pyramid_pixel_batch_size
    values = torch.zeros(
        batch, len(plan),
        dtype=torch.promote_types(maps[0].dtype, torch.complex64),
        device=maps[0].device,
    )

    def trace(name, dg, x):
        if tracker is not None:
            tracker(name, dg, x)

    def fine_blocks(region):
        if d1 == n:
            return [region]
        size = max(1, int(fine_patch_pixels))
        return [region[s : s + size] for s in range(0, region.numel(), size)]

    def fine_wave(block):
        """Fine wavelet j1 on ``block``: rows, complex [B, C, L, |block|]."""
        rows = ConvRows(conv_1, block, table=table_1)
        x = maps[d1].index_select(-1, rows.in_domain)
        y = conv_forward_rows(rows, x.reshape(-1, 1, x.shape[-1]), pixel_batch)
        z = _join(y, L).reshape(batch, channels, L, -1)
        trace("first_wavelet", d1, z)
        return rows, z

    for core in geometry.cores(n, n):
        # ---- coarse geometry: core, its stencil domain a1 and the fine region
        a1 = table_n.collect(
            torch.cat([ConvRows(conv_n, core, table=table_n).in_domain, core])
        )
        second = ConvRows(conv_n, core, a1, table_n)
        region = a1
        for level in range(n - 1, d1 - 1, -1):
            region = down_input_domain(geometry.down(level), region, geometry.table(level))
        blocks = fine_blocks(region)

        # ---- forward: decimated fine modulus on a1, accumulated by sub-patch
        m = torch.zeros(batch, channels, L, a1.numel(), dtype=maps[0].dtype, device=maps[0].device)
        for block in blocks:
            rows, z = fine_wave(block)
            a = z.abs()
            if d1 < n:
                chain = _Chain(geometry, d1, n, block)
                a = chain.forward(a)
                idx = table_n.map(a1, chain.top, optional=True)
                keep = idx >= 0
                m.index_add_(-1, idx[keep], a[..., keep])
            else:
                m.index_add_(-1, table_n.map(a1, block), a)
            del rows, z, a

        # ---- coarse level: second wavelet on the core, direct operand, sums
        y = conv_forward_rows(second, m.reshape(-1, 1, a1.numel()), pixel_batch)
        branch = _join(y, L).reshape(batch, channels, L, L, -1)  # [B, C, l1, l3, |core|]
        trace("second_wavelet", n, branch)
        rows3 = ConvRows(conv_n, core, table=table_n)
        x3 = maps[n].index_select(-1, rows3.in_domain)
        z3 = _join(
            conv_forward_rows(rows3, x3.reshape(-1, 1, x3.shape[-1]), pixel_batch), L
        ).reshape(batch, channels, L, -1)
        direct = maps[n].index_select(-1, core)  # fewer-convolution operand
        gbranch = torch.zeros_like(branch) if gradient is not None else None
        gz3 = torch.zeros_like(z3) if gradient is not None else None
        for i, c in enumerate(plan):
            if fewer:
                a, b = direct[:, c.c1], branch[:, c.c2, c.l1, c.l3]
            else:
                a, b = branch[:, c.c1, c.l1, c.l3], z3[:, c.c2, c.l3]
            values[:, i] += (a * b.conj()).sum(-1) / n_total
            if gradient is None:
                continue
            g = gradient[:, i]
            ga, gb = g[..., None] * b / n_total, g.conj()[..., None] * a / n_total
            if fewer:
                maps_gradient[n][:, c.c1].index_add_(-1, core, ga.real)
                gbranch[:, c.c2, c.l1, c.l3].add_(gb)
            else:
                gbranch[:, c.c1, c.l1, c.l3].add_(ga)
                gz3[:, c.c2, c.l3].add_(gb)
        if gradient is None:
            continue

        # ---- reverse, coarse part
        gx3 = conv_adjoint_rows(rows3, _split(gz3).reshape(batch * channels, 2 * L, -1), pixel_batch)
        maps_gradient[n].index_add_(-1, rows3.in_domain, gx3.reshape(batch, channels, -1))
        gm = conv_adjoint_rows(
            second, _split(gbranch).reshape(batch * channels * L, 2 * L, -1), pixel_batch
        ).reshape(batch, channels, L, -1)  # real cotangent of m on a1
        del gbranch, gz3, branch, z3, y

        # ---- reverse, fine part streamed by sub-patch
        for block in blocks:
            rows, z = fine_wave(block)
            if d1 < n:
                chain = _Chain(geometry, d1, n, block)
                idx = table_n.map(a1, chain.top, optional=True)
                keep = idx >= 0
                gtop = gm.new_zeros(*gm.shape[:-1], chain.top.numel())
                gtop[..., keep] = gm[..., idx[keep]]
                gk = chain.adjoint(gtop)
            else:
                gk = gm[..., table_n.map(a1, block)]
            gz = modulus_vjp(z, gk)
            trace("native_gradient", d1, gz)
            gx = conv_adjoint_rows(rows, _split(gz).reshape(batch * channels, 2 * L, -1), pixel_batch)
            maps_gradient[d1].index_add_(-1, rows.in_domain, gx.reshape(batch, channels, -1))
            del rows, z, gk, gz, gx
    return PyramidStatistics(plan, values), maps_gradient
