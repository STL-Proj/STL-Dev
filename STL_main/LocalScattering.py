"""Complete HEALPix ScatCov (mean, var, S1, S2, S3, S4) with patch-local geometry.

Structure.  The wavelet kernel is the same at every level, so the first
wavelet field depends only on the resolution ``d``: ``z_d = psi * maps[d]``.
Every coefficient of native level ``n`` is a pixel mean at level ``n`` of
products of ``z_n`` (on the core), of the second wavelet applied to the
decimated moduli ``m[d][n] = D**(n-d) |z_d|`` (``d < n``) or to ``|z_n|``
(``d == n``), and of ``maps[n]``.

Forward sweep, levels ascending: each level ``n`` is cut into blocks ``F``
(``block_pixels`` at every level, so a level smaller than one block is
handled whole, the "old mode" for the large scales).  For a block,
``z_n`` is computed on ``a1 = F + conv halo`` through one partial
convolution on ``D = a1 + conv halo``; the moduli ``m[d][n]`` of finer
levels are read on ``a1`` from stored maps; the second wavelets are
evaluated on ``F`` through the same partial convolution; partial sums are
accumulated.  Then ``|z_n|`` restricted to ``F`` is pushed through the
decimation chain pieces (``LocalGeometry.piece_forward``) into the stored
maps ``m[n][n']`` of every coarser level ``n'``.

Reverse sweep, levels descending: the same blocks are revisited, the fields
recomputed, the coefficient cotangents applied; the cotangents of the
stored moduli ``gm[d][n]`` are accumulated (complete once every coarser
level is done); the block's own chain is reversed (``piece_adjoint``) from
``gm[n][n']``, the modulus VJP and the wavelet adjoint are applied, and the
map cotangent of level ``n`` is accumulated.  ``LocalGeometry.analysis_vjp``
then returns the band gradients.

Memory.  Per block: a few fields of size ``|a1|`` and ``|F|`` with
orientation axes.  Globally: the maps (``4N/3``), their cotangents, and the
stored decimated moduli and their cotangents, real ``[B, C, L, N_n]`` for
every pair ``d < n`` (about ``4N/9`` values times ``B*C*L`` in total).  No
full-sky geometry beyond the levels that fit in one block.
"""

import torch

from STL_main.AdjointOps import convolution_adjoint, modulus_vjp, root_modulus_vjp
from STL_main.LocalGeometry import LocalGeometry, dilate_fine, parents, positions
from STL_main.PyramidScattering import PyramidStatistics, _validate, build_plan
from STL_main.S3Multiscale import _join, _split


def _group_by_level(plan):
    groups = {}
    for i, c in enumerate(plan):
        groups.setdefault(c.native_dg, []).append((i, c))
    return groups


class _Block:
    """Fields of one block ``F`` of level ``n``: ``z_n`` on ``a1``, branches on ``F``."""

    def __init__(self, engine, n, block):
        geo, w = engine.geo, engine.w
        self.engine, self.n, self.block = engine, n, block
        nside = geo.nside(n)
        self.a1 = dilate_fine(block, nside, geo.conv_rings)
        self.domain = dilate_fine(self.a1, nside, geo.conv_rings)
        self.conv = geo.conv(n, self.domain)
        self.rows_a1 = positions(self.domain, self.a1)
        self.rows_f = positions(self.domain, block)
        self.f_in_a1 = positions(self.a1, block)
        maps = engine.maps
        x = maps[n][..., self.domain]
        batch, channels = x.shape[:2]
        self.batch, self.channels = batch, channels
        y = self.conv(x.reshape(-1, 1, self.domain.numel()).to(self.conv.dtype))
        self.z = _join(y[..., self.rows_a1], w.L).reshape(batch, channels, w.L, -1)
        engine.trace("first_wavelet", n, self.z)
        self._modulus, self._branch = {}, {}
        self.gz = None
        self.gbranch, self.gmod = {}, {}

    # -- forward fields -----------------------------------------------------
    def modulus(self, d):
        """Decimated modulus of level ``d <= n`` on ``a1``: ``[B, C, L, |a1|]``."""
        if d not in self._modulus:
            if d == self.n:
                self._modulus[d] = self.z.abs()
            else:
                self._modulus[d] = self.engine.m[d][self.n][..., self.a1]
        return self._modulus[d]

    def branch(self, d):
        """Second wavelet of ``modulus(d)`` on ``F``: ``[B, C, L(l1), L(l3), |F|]``."""
        if d not in self._branch:
            w = self.engine.w
            m = self.modulus(d)
            padded = m.new_zeros(*m.shape[:-1], self.domain.numel())
            padded[..., self.rows_a1] = m
            y = self.conv(padded.reshape(-1, 1, self.domain.numel()).to(self.conv.dtype))
            y = _join(y[..., self.rows_f], w.L)
            self._branch[d] = y.reshape(self.batch, self.channels, w.L, w.L, -1)
            self.engine.trace("second_wavelet", self.n, self._branch[d])
        return self._branch[d]

    def z_core(self, channel, angle):
        return self.z[:, channel, angle, self.f_in_a1]

    def operands(self, c):
        engine, maps = self.engine, self.engine.maps
        dg = engine.w.j_to_dg
        if c.family in ("mean", "var"):
            return maps[0][:, c.c1, self.block], None
        if c.family in ("S1", "S2"):
            return self.z_core(c.c1, c.l1), self.z_core(c.c2, c.l1)
        if c.family == "S3":
            if not engine.fewer:
                return self.branch(dg[c.j1])[:, c.c1, c.l1, c.l3], self.z_core(c.c2, c.l3)
            return maps[self.n][:, c.c1, self.block], self.branch(dg[c.j1])[:, c.c2, c.l1, c.l3]
        a = (
            self.modulus(dg[c.j1])[:, c.c1, c.l1, self.f_in_a1]
            if engine.fewer
            else self.branch(dg[c.j1])[:, c.c1, c.l1, c.l3]
        )
        return a, self.branch(dg[c.j2])[:, c.c2, c.l2, c.l3]

    def partial(self, c, a, b):
        n_total = self.engine.n_pixels[self.n]
        if c.family == "mean":
            return a.sum(-1) / n_total
        if c.family == "var" or (c.family == "S2" and c.c1 == c.c2):
            return a.abs().square().sum(-1) / n_total
        if c.family == "S1":
            if c.c1 == c.c2:
                return a.abs().sum(-1) / n_total
            a, b = a / (a.abs() + 1e-8).sqrt(), b / (b.abs() + 1e-8).sqrt()
        return (a * b.conj()).sum(-1) / n_total

    # -- reverse --------------------------------------------------------------
    def add_z(self, channel, angle, g):
        if self.gz is None:
            self.gz = torch.zeros_like(self.z)
        self.gz[:, channel, angle].index_add_(-1, self.f_in_a1, g)

    def add_branch(self, d, channel, l1, l3, g):
        if d not in self.gbranch:
            self.gbranch[d] = torch.zeros_like(self.branch(d))
        self.gbranch[d][:, channel, l1, l3].add_(g)

    def add_modulus_core(self, d, channel, l1, g):
        if d not in self.gmod:
            self.gmod[d] = torch.zeros_like(self.modulus(d))
        self.gmod[d][:, channel, l1].index_add_(-1, self.f_in_a1, g.real)

    def reverse_coefficient(self, c, a, b, g, maps_gradient):
        engine = self.engine
        n, dg, n_total = self.n, engine.w.j_to_dg, engine.n_pixels[self.n]
        if c.family == "mean":
            ga = (g.real / n_total)[..., None].expand_as(a)
            maps_gradient[0][:, c.c1].index_add_(-1, self.block, ga)
        elif c.family == "var":
            ga = 2 * g.real[..., None] * a / n_total
            maps_gradient[0][:, c.c1].index_add_(-1, self.block, ga)
        elif c.family in ("S1", "S2"):
            if c.c1 == c.c2:
                if c.family == "S1":
                    ga = modulus_vjp(a, (g.real / n_total)[..., None].expand_as(a))
                else:
                    ga = 2 * g.real[..., None] * a / n_total
                self.add_z(c.c1, c.l1, ga)
            else:
                if c.family == "S1":
                    ar, br = a / (a.abs() + 1e-8).sqrt(), b / (b.abs() + 1e-8).sqrt()
                    ga, gb = g[..., None] * br / n_total, g.conj()[..., None] * ar / n_total
                    ga, gb = root_modulus_vjp(a, ga), root_modulus_vjp(b, gb)
                else:
                    ga, gb = g[..., None] * b / n_total, g.conj()[..., None] * a / n_total
                self.add_z(c.c1, c.l1, ga)
                self.add_z(c.c2, c.l1, gb)
        else:
            ga, gb = g[..., None] * b / n_total, g.conj()[..., None] * a / n_total
            if c.family == "S3":
                if engine.fewer:
                    maps_gradient[n][:, c.c1].index_add_(-1, self.block, ga.real)
                    self.add_branch(dg[c.j1], c.c2, c.l1, c.l3, gb)
                else:
                    self.add_branch(dg[c.j1], c.c1, c.l1, c.l3, ga)
                    self.add_z(c.c2, c.l3, gb)
            else:
                if engine.fewer:
                    self.add_modulus_core(dg[c.j1], c.c1, c.l1, ga)
                else:
                    self.add_branch(dg[c.j1], c.c1, c.l1, c.l3, ga)
                self.add_branch(dg[c.j2], c.c2, c.l2, c.l3, gb)
        engine.trace("native_gradient", n, ga)

    def finish_reverse(self, maps_gradient):
        engine, geo, w = self.engine, self.engine.geo, self.engine.w
        n, L = self.n, self.engine.w.L
        pixel_batch = w.pyramid_pixel_batch_size
        # Second-wavelet cotangents -> modulus cotangents on a1.
        for d, g in self.gbranch.items():
            padded = g.new_zeros(self.batch * self.channels * L, 2 * L, self.domain.numel(), dtype=g.real.dtype)
            padded[..., self.rows_f] = _split(g).reshape(self.batch * self.channels * L, 2 * L, -1)
            gm = convolution_adjoint(self.conv, padded, pixel_batch)[..., self.rows_a1]
            gm = gm.reshape(self.batch, self.channels, L, -1)
            if d not in self.gmod:
                self.gmod[d] = gm
            else:
                self.gmod[d].add_(gm)
        self.gbranch.clear()
        # Modulus cotangents: own level -> modulus VJP; finer levels -> stored.
        for d, gm in self.gmod.items():
            if d == n:
                if self.gz is None:
                    self.gz = torch.zeros_like(self.z)
                self.gz.add_(modulus_vjp(self.z, gm))
            else:
                engine.gm[d][n].index_add_(-1, self.a1, gm)
        self.gmod.clear()
        # The block's own chain to coarser levels.
        chain = engine.supports(n, self.block)
        g_fine = None
        for n2 in range(n + 1, engine.levels):
            g = engine.gm[n][n2][..., chain[n2 - n]]
            for level in range(n2 - 1, n - 1, -1):
                g = geo.piece_adjoint(g, chain[level - n], chain[level - n + 1], level)
            g_fine = g if g_fine is None else g_fine + g
        if g_fine is not None:
            if self.gz is None:
                self.gz = torch.zeros_like(self.z)
            self.gz[..., self.f_in_a1] += modulus_vjp(self.z[..., self.f_in_a1], g_fine)
        if self.gz is None:
            return
        # Wavelet adjoint on a1 rows -> map cotangent on the domain.
        padded = self.gz.new_zeros(self.batch * self.channels, 2 * L, self.domain.numel(), dtype=self.gz.real.dtype)
        padded[..., self.rows_a1] = _split(self.gz).reshape(self.batch * self.channels, 2 * L, -1)
        gx = convolution_adjoint(self.conv, padded, pixel_batch)
        maps_gradient[n].index_add_(-1, self.domain, gx.reshape(self.batch, self.channels, -1))


class LocalEngine:
    """One forward (and optionally reverse) pass of a plan with local geometry."""

    def __init__(self, operator, pyramid, geometry, tracker=None, maps=None):
        self.op, self.p, self.geo, self.tracker = operator, pyramid, geometry, tracker
        self.w = operator.wavelet_op
        self.fewer = operator.has_fewer_convolutions
        self.maps = geometry.analysis_levels(pyramid) if maps is None else maps
        self.levels = len(self.maps)
        self.n_pixels = [x.shape[-1] for x in self.maps]
        batch, channels = self.maps[0].shape[:2]
        shape = (batch, channels, self.w.L)
        self.m = {
            d: {
                n: torch.zeros(*shape, self.n_pixels[n], dtype=self.maps[0].dtype, device=self.maps[0].device)
                for n in range(d + 1, self.levels)
            }
            for d in range(self.levels)
        }
        self.gm = None
        self._supports = {}

    def trace(self, name, dg, x):
        if self.tracker is not None:
            self.tracker(name, dg, x)

    def supports(self, n, block):
        key = (n, int(block[0]), block.numel())
        if key not in self._supports:
            chain, support = [block], block
            for level in range(n, self.levels - 1):
                support = parents(dilate_fine(support, self.geo.nside(level), self.geo.down_rings(level)))
                chain.append(support)
            self._supports[key] = chain
        return self._supports[key]

    def forward(self, plan, values):
        groups = _group_by_level(plan)
        for n in range(self.levels):
            for block in self.geo.blocks(n):
                fields = _Block(self, n, block)
                for i, c in groups.get(n, ()):
                    a, b = fields.operands(c)
                    values[:, i] += fields.partial(c, a, b)
                # Push |z_n| on F to every coarser level.
                a, chain = fields.z.abs()[..., fields.f_in_a1], self.supports(n, block)
                for n2 in range(n + 1, self.levels):
                    a, _ = self.geo.piece_forward(a, chain[n2 - n - 1], n2 - 1)
                    self.m[n][n2].index_add_(-1, chain[n2 - n], a)
                del fields
        return values

    def reverse(self, plan, cotangent, active, maps_gradient):
        groups = _group_by_level(plan)
        self.gm = {d: {n: torch.zeros_like(x) for n, x in maps.items()} for d, maps in self.m.items()}
        for n in range(self.levels - 1, -1, -1):
            for block in self.geo.blocks(n):
                fields = _Block(self, n, block)
                for i, c in groups.get(n, ()):
                    if active is not None and not active[i]:
                        continue
                    a, b = fields.operands(c)
                    fields.reverse_coefficient(c, a, b, cotangent[:, i], maps_gradient)
                fields.finish_reverse(maps_gradient)
                del fields
        self.gm = None
        return maps_gradient


def _prepare(operator, pyramid, plan, geometry):
    plan = tuple(build_plan(operator, pyramid.levels[0].shape[1]) if plan is None else plan)
    _validate(operator, pyramid, plan, 1, "recompute")
    if geometry is None:
        geometry = LocalGeometry(operator.wavelet_op)
    elif geometry.w is not operator.wavelet_op:
        raise ValueError("LocalGeometry was built for another wavelet operator")
    return plan, geometry


def _values(maps, plan):
    return torch.zeros(
        maps[0].shape[0], len(plan),
        dtype=torch.promote_types(maps[0].dtype, torch.complex64), device=maps[0].device,
    )


@torch.no_grad()
def apply_local(operator, pyramid, plan=None, geometry=None, tracker=None):
    """Same coefficients as ``apply_pyramid`` with patch-local geometry."""
    plan, geometry = _prepare(operator, pyramid, plan, geometry)
    engine = LocalEngine(operator, pyramid, geometry, tracker)
    return PyramidStatistics(plan, engine.forward(plan, _values(engine.maps, plan)))


@torch.no_grad()
def vjp_local(operator, pyramid, gradient, plan=None, geometry=None, tracker=None, return_maps_gradient=False):
    """Band gradients of ``Re sum(conj(gradient) * statistics)`` with local geometry."""
    if isinstance(gradient, PyramidStatistics):
        if plan is not None and tuple(plan) != gradient.plan:
            raise ValueError("Gradient and requested plan differ")
        plan, gradient = gradient.plan, gradient.values
    plan, geometry = _prepare(operator, pyramid, plan, geometry)
    engine = LocalEngine(operator, pyramid, geometry, tracker)
    if gradient.shape != (engine.maps[0].shape[0], len(plan)):
        raise ValueError("Coefficient cotangent must have shape [batch, len(plan)]")
    gradient = gradient.to(engine.maps[0].device)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("Coefficient cotangent must be finite")
    active = (gradient != 0).any(dim=0).cpu().tolist()
    engine.forward(plan, _values(engine.maps, plan))  # fills the decimated moduli
    maps_gradient = engine.reverse(plan, gradient, active, [torch.zeros_like(x) for x in engine.maps])
    if return_maps_gradient:
        return maps_gradient
    return geometry.analysis_vjp(pyramid, maps_gradient)


@torch.no_grad()
def loss_and_grad_local(operator, pyramid, target, weights=1.0, geometry=None, tracker=None):
    """Fused weighted squared loss and band gradients with local geometry."""
    from STL_main.LossOptim import SquaredStatisticsLoss

    plan, geometry = _prepare(operator, pyramid, target.plan, geometry)
    engine = LocalEngine(operator, pyramid, geometry, tracker)
    expected = (engine.maps[0].shape[0], len(plan))
    if target.values.shape != expected or target.values.device != pyramid.device:
        raise ValueError("Target shape/device does not match the pyramid")
    _, weight = SquaredStatisticsLoss(weights)._terms(target, target)
    weight = torch.broadcast_to(weight, expected)
    active = (weight != 0).any(dim=0).cpu().tolist()
    values = engine.forward(plan, _values(engine.maps, plan))
    residual = values - target.values
    loss = (weight * residual.abs().square()).sum()
    maps_gradient = engine.reverse(
        plan, 2 * weight * residual, active, [torch.zeros_like(x) for x in engine.maps]
    )
    return loss, geometry.analysis_vjp(pyramid, maps_gradient)
