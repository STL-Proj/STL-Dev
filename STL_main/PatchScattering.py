"""Patch-localised HEALPix ScatCov: exact statistics and VJP per spatial patch.

This engine computes the same coefficients and the same band gradients as
``PyramidScattering`` but never materialises a wavelet field on the whole
sphere.  The sphere at the native resolution of a coefficient batch is split
into disjoint *cores*; every coefficient is a pixel mean, so its value is the
sum of partial sums over cores, and the gradient of the loss is the sum over
cores of the adjoint chain applied to the cotangent restricted to that core.

For each core the chain only reads a bounded neighbourhood, derived exactly
from the operator index structure (``PatchOps``):

* ``a1``       - native positions read by the native wavelet for the core
                 (plus the core itself);
* ``chain[j]`` - for every inner scale ``j``, the positions needed at each
                 level between ``dg_j`` and the native level so that the
                 decimated modulus is exact on ``a1``;
* ``inner[j]`` - ``chain[j][dg_j]``, where the first wavelet ``j`` is evaluated;
* ``outer[j]`` - positions of ``maps[dg_j]`` read by that wavelet.

The receptive field of a cross-scale coefficient (``j1 < j3``) is inherent:
one native pixel at level ``n`` reads about ``4**(n - dg_j1)`` fine pixels
times the stencil halo.  ``patch_pixels`` is therefore a budget at the finest
level involved in a batch, and the core at the native level is shrunk by
``4**(n - dg_min)`` so that the fine domain stays near the budget.

Batches sharing a native and a finest level are processed together: the
patch loop is outside the batch loop so that the index structures of a patch
are built once per level, while fields still live for one batch only.

Contributions of different patches are additive, so patches can be processed
on different workers and their ``maps_gradient`` accumulators summed before
``HealpixPyramid.analysis_vjp``.  The driver here is sequential.
"""

import torch

from STL_main.AdjointOps import modulus_vjp, root_modulus_vjp
from STL_main.PatchOps import (
    ConvRows,
    DownRows,
    IndexTable,
    conv_adjoint_rows,
    conv_forward_rows,
    conv_input_domain,
    down_adjoint_rows,
    down_forward_rows,
    down_input_domain,
)
from STL_main.PyramidScattering import (
    PyramidStatistics,
    _batches,
    _validate,
    build_plan,
)


def _scales(c):
    return [j for j in (c.j1, c.j2, c.j3) if j is not None]


def _finest(batch):
    return min(c.native_dg if not _scales(c) else min(_scales(c)) for c in batch)


def _levels(plan, size):
    """Group coefficient batches by (native level, finest level), in plan order."""
    groups = {}
    for start, batch in _batches(plan, size):
        groups.setdefault((batch[0].native_dg, _finest(batch)), []).append(
            (start, batch)
        )
    return groups


class _PatchContext:
    """Exact domains and restricted operators of one patch; data independent."""

    def __init__(self, geometry, native_dg, core):
        self.geo, self.n, self.core = geometry, native_dg, core
        w = geometry.w
        table = geometry.table(native_dg)
        # On unsorted partial grids healpix-analyse does not co-locate output
        # position i with input position i, so the core is added explicitly:
        # native fields (direct operands, S4 moduli) are also read on the core.
        self.a1 = table.collect(
            torch.cat([conv_input_domain(geometry.conv(native_dg), core, table), core])
        )
        self.core_in_a1 = table.map(self.a1, core)
        self.chain, self.inner, self.outer, self.core_in_inner = {}, {}, {}, {}
        for j in range(w.J):
            dg = w.j_to_dg[j]
            if dg > native_dg:
                continue
            levels = {native_dg: self.a1}
            for level in range(native_dg - 1, dg - 1, -1):
                levels[level] = down_input_domain(
                    geometry.down(level), levels[level + 1], geometry.table(level)
                )
            self.chain[j] = levels
            self.inner[j] = levels[dg]
            self.outer[j] = conv_input_domain(
                geometry.conv(dg), self.inner[j], geometry.table(dg)
            )
            if dg == native_dg:
                self.core_in_inner[j] = table.map(self.inner[j], core)
        self._rows = {}

    def first_rows(self, j):
        if ("first", j) not in self._rows:
            dg = self.geo.w.j_to_dg[j]
            self._rows[("first", j)] = ConvRows(
                self.geo.conv(dg), self.inner[j], self.outer[j], self.geo.table(dg)
            )
        return self._rows[("first", j)]

    def second_rows(self):
        if "second" not in self._rows:
            self._rows["second"] = ConvRows(
                self.geo.conv(self.n), self.core, self.a1, self.geo.table(self.n)
            )
        return self._rows["second"]

    def down_rows(self, j, level):
        """Decimation ``level -> level + 1`` inside the chain of scale ``j``."""
        key = ("down", j, level)
        if key not in self._rows:
            chain = self.chain[j]
            self._rows[key] = DownRows(
                self.geo.down(level),
                chain[level + 1],
                chain[level],
                self.geo.table(level),
            )
        return self._rows[key]

    def largest_field(self):
        """Largest number of positions of any first-wavelet field."""
        return max(v.numel() for v in self.inner.values())


class PatchGeometry:
    """Operators, index tables and per-patch contexts for one (operator, pyramid).

    Contexts depend only on the geometry, never on the data.  With
    ``cache=True`` they are kept across calls (memory grows with the number of
    patches); otherwise a context lives while its patch is processed.
    """

    def __init__(self, operator, pyramid, patch_pixels=4096, cache=False):
        if not isinstance(patch_pixels, int) or patch_pixels < 1:
            raise ValueError("patch_pixels must be a positive integer")
        self.op, self.p, self.w = operator, pyramid, operator.wavelet_op
        self.patch_pixels, self.cache = patch_pixels, cache
        self._convs, self._downs, self._tables, self._contexts = {}, {}, {}, {}

    def conv(self, dg):
        if dg not in self._convs:
            template = self.p.templates[dg]
            self._convs[dg] = self.w._get_conv(
                dg, template.cell_ids, self.w.L, self.w._wav_weights
            )
        return self._convs[dg]

    def down(self, dg):
        """Decimation from level ``dg`` to ``dg + 1``."""
        if dg not in self._downs:
            template = self.p.templates[dg]
            self._downs[dg] = self.w._get_down(dg, template.cell_ids)
        return self._downs[dg]

    def table(self, dg):
        if dg not in self._tables:
            self._tables[dg] = IndexTable(self.pixels(dg), self.p.levels[dg].device)
        return self._tables[dg]

    def pixels(self, dg):
        return self.p.levels[dg].shape[-1]

    def core_pixels(self, native_dg, finest_dg):
        """Core size at the native level for a budget at the finest level."""
        return max(1, self.patch_pixels // 4 ** (native_dg - finest_dg))

    def cores(self, native_dg, finest_dg):
        """Disjoint contiguous position blocks; NESTED-aligned when possible."""
        n, size = self.pixels(native_dg), self.core_pixels(native_dg, finest_dg)
        device = self.p.levels[native_dg].device
        return [
            torch.arange(start, min(start + size, n), device=device)
            for start in range(0, n, size)
        ]

    def context(self, native_dg, finest_dg, index, core):
        key = (native_dg, finest_dg, index)
        if key in self._contexts:
            return self._contexts[key]
        context = _PatchContext(self, native_dg, core)
        if self.cache:
            self._contexts[key] = context
        return context

    def patches(self, native_dg, finest_dg, patches=None):
        for index, core in enumerate(self.cores(native_dg, finest_dg)):
            if patches is None or index in patches:
                yield self.context(native_dg, finest_dg, index, core)


class _PatchGroup:
    """Fields of one coefficient batch on one patch; mirrors ``_Group``.

    Pixel means use the native pixel count ``n_total`` of the whole level, so
    ``partial`` returns the core's contribution to the global statistic and
    the cotangents are already scaled for the global mean.
    """

    def __init__(self, operator, context, maps, tracker):
        self.operator, self.ctx, self.maps = operator, context, maps
        self.w = operator.wavelet_op
        self.n, self.tracker = context.n, tracker
        self.n_total = maps[self.n].shape[-1]
        self.core, self.a1 = context.core, context.a1
        self.pixel_batch_size = self.w.pyramid_pixel_batch_size
        self.first, self.second, self.moduli = {}, {}, {}
        self.gfirst, self.gsecond, self.gmoduli = {}, {}, {}

    def trace(self, name, dg, x):
        if self.tracker is not None:
            self.tracker(name, dg, x)

    # -- forward fields -----------------------------------------------------
    def wave(self, j):
        """First wavelet ``j`` on ``inner[j]``: ``[B, C, L, |inner|]`` complex."""
        if j not in self.first:
            dg = self.w.j_to_dg[j]
            x = self.maps[dg].index_select(-1, self.ctx.outer[j])
            batch, channels, n_in = x.shape
            y = conv_forward_rows(
                self.ctx.first_rows(j), x.reshape(-1, 1, n_in), self.pixel_batch_size
            )
            y = y.reshape(batch, channels, self.w.L, 2, -1)
            self.first[j] = torch.complex(y[..., 0, :], y[..., 1, :])
            self.trace("first_wavelet", dg, self.first[j])
        return self.first[j]

    def mod(self, key):
        """Decimated modulus on ``a1`` (native level)."""
        j, channel, angle, _ = key
        if key not in self.moduli:
            m = self.wave(j)[:, channel, angle].abs()
            for level in range(self.w.j_to_dg[j], self.n):
                m = down_forward_rows(self.ctx.down_rows(j, level), m)
            self.moduli[key] = m
        return self.moduli[key]

    def branch(self, key):
        """Second wavelet on the core: ``[B, L, |core|]`` complex."""
        if key not in self.second:
            m = self.mod(key)
            y = conv_forward_rows(
                self.ctx.second_rows(), m[:, None, :], self.pixel_batch_size
            )
            y = y.reshape(m.shape[0], self.w.L, 2, -1)
            self.second[key] = torch.complex(y[..., 0, :], y[..., 1, :])
            self.trace("second_wavelet", self.n, self.second[key])
            self.moduli.pop(key, None)
        return self.second[key]

    def wave_core(self, j, channel, angle):
        return self.wave(j)[:, channel, angle].index_select(
            -1, self.ctx.core_in_inner[j]
        )

    def operands(self, c):
        if c.family in ("mean", "var"):
            return self.maps[0][:, c.c1].index_select(-1, self.core), None
        if c.family in ("S1", "S2"):
            return (
                self.wave_core(c.j1, c.c1, c.l1),
                self.wave_core(c.j1, c.c2, c.l1),
            )
        if c.family == "S3":
            if not self.operator.has_fewer_convolutions:
                return (
                    self.branch((c.j1, c.c1, c.l1, c.j3))[:, c.l3],
                    self.wave_core(c.j3, c.c2, c.l3),
                )
            return (
                self.maps[c.native_dg][:, c.c1].index_select(-1, self.core),
                self.branch((c.j1, c.c2, c.l1, c.j3))[:, c.l3],
            )
        akey, bkey = (c.j1, c.c1, c.l1, c.j3), (c.j2, c.c2, c.l2, c.j3)
        a = (
            self.mod(akey).index_select(-1, self.ctx.core_in_a1)
            if self.operator.has_fewer_convolutions
            else self.branch(akey)[:, c.l3]
        )
        return a, self.branch(bkey)[:, c.l3]

    def partial(self, c, a, b):
        """Core contribution to the coefficient (sum over core / n_total)."""
        if c.family == "mean":
            return a.sum(-1) / self.n_total
        if c.family == "var" or (c.family == "S2" and c.c1 == c.c2):
            return a.abs().square().sum(-1) / self.n_total
        if c.family == "S1":
            if c.c1 == c.c2:
                return a.abs().sum(-1) / self.n_total
            a, b = a / (a.abs() + 1e-8).sqrt(), b / (b.abs() + 1e-8).sqrt()
        return (a * b.conj()).sum(-1) / self.n_total

    # -- reverse --------------------------------------------------------------
    def add_wave(self, j, channel, angle, g):
        """Cotangent ``g`` on the core of ``wave(j)[:, channel, angle]``."""
        if j not in self.gfirst:
            self.gfirst[j] = torch.zeros_like(self.wave(j))
        self.gfirst[j][:, channel, angle].index_add_(
            -1, self.ctx.core_in_inner[j], g
        )

    def add_branch(self, key, angle, g):
        if key not in self.gsecond:
            self.gsecond[key] = torch.zeros_like(self.branch(key))
        self.gsecond[key][:, angle].add_(g)

    def add_mod(self, key, g, on_core=False):
        """Real cotangent of the decimated modulus, on ``a1`` or on the core."""
        if key not in self.gmoduli:
            self.gmoduli[key] = torch.zeros(
                g.shape[0], self.a1.numel(), dtype=g.real.dtype, device=g.device
            )
        if on_core:
            self.gmoduli[key].index_add_(-1, self.ctx.core_in_a1, g.real)
        else:
            self.gmoduli[key].add_(g.real)

    def reverse_coefficient(self, c, a, b, g, maps_gradient):
        n = self.n_total
        if c.family == "mean":
            ga = (g.real / n)[..., None].expand_as(a)
            maps_gradient[0][:, c.c1].index_add_(-1, self.core, ga)
        elif c.family == "var":
            ga = 2 * g.real[..., None] * a / n
            maps_gradient[0][:, c.c1].index_add_(-1, self.core, ga)
        elif c.family in ("S1", "S2"):
            if c.c1 == c.c2:
                if c.family == "S1":
                    ga = modulus_vjp(a, (g.real / n)[..., None].expand_as(a))
                else:
                    ga = 2 * g.real[..., None] * a / n
                self.add_wave(c.j1, c.c1, c.l1, ga)
            else:
                if c.family == "S1":
                    ar, br = a / (a.abs() + 1e-8).sqrt(), b / (b.abs() + 1e-8).sqrt()
                    ga, gb = g[..., None] * br / n, g.conj()[..., None] * ar / n
                    ga, gb = root_modulus_vjp(a, ga), root_modulus_vjp(b, gb)
                else:
                    ga, gb = g[..., None] * b / n, g.conj()[..., None] * a / n
                self.add_wave(c.j1, c.c1, c.l1, ga)
                self.add_wave(c.j1, c.c2, c.l1, gb)
        else:
            ga, gb = g[..., None] * b / n, g.conj()[..., None] * a / n
            if c.family == "S3":
                if self.operator.has_fewer_convolutions:
                    maps_gradient[c.native_dg][:, c.c1].index_add_(
                        -1, self.core, ga.real
                    )
                    self.add_branch((c.j1, c.c2, c.l1, c.j3), c.l3, gb)
                else:
                    self.add_branch((c.j1, c.c1, c.l1, c.j3), c.l3, ga)
                    self.add_wave(c.j3, c.c2, c.l3, gb)
            else:
                akey = (c.j1, c.c1, c.l1, c.j3)
                if self.operator.has_fewer_convolutions:
                    self.add_mod(akey, ga, on_core=True)
                else:
                    self.add_branch(akey, c.l3, ga)
                self.add_branch((c.j2, c.c2, c.l2, c.j3), c.l3, gb)
        self.trace("native_gradient", c.native_dg, ga)

    def finish_reverse(self, maps_gradient):
        self.second.clear()
        self.moduli.clear()
        while self.gsecond:
            key, g = self.gsecond.popitem()
            batch = g.shape[0]
            split = torch.stack((g.real, g.imag), dim=-2).reshape(batch, -1, g.shape[-1])
            gm = conv_adjoint_rows(self.ctx.second_rows(), split, self.pixel_batch_size)
            self.add_mod(key, gm[:, 0])
        while self.gmoduli:
            (j, channel, angle, _), g = self.gmoduli.popitem()
            for level in range(self.n - 1, self.w.j_to_dg[j] - 1, -1):
                g = down_adjoint_rows(self.ctx.down_rows(j, level), g)
            z = self.wave(j)[:, channel, angle]
            if j not in self.gfirst:
                self.gfirst[j] = torch.zeros_like(self.wave(j))
            self.gfirst[j][:, channel, angle].add_(modulus_vjp(z, g))
        self.first.clear()
        while self.gfirst:
            j, g = self.gfirst.popitem()
            dg = self.w.j_to_dg[j]
            batch, channels = g.shape[:2]
            split = torch.stack((g.real, g.imag), dim=-2)
            split = split.reshape(batch * channels, 2 * self.w.L, g.shape[-1])
            gx = conv_adjoint_rows(self.ctx.first_rows(j), split, self.pixel_batch_size)
            maps_gradient[dg].index_add_(
                -1, self.ctx.outer[j], gx.reshape(batch, channels, -1)
            )


def _prepare(operator, pyramid, plan, coefficient_batch_size, patch_pixels, geometry):
    plan = tuple(
        build_plan(operator, pyramid.levels[0].shape[1]) if plan is None else plan
    )
    _validate(operator, pyramid, plan, coefficient_batch_size, "recompute")
    if geometry is None:
        geometry = PatchGeometry(operator, pyramid, patch_pixels)
    elif geometry.op is not operator or geometry.p is not pyramid:
        raise ValueError("PatchGeometry was built for another operator or pyramid")
    return plan, geometry


def _forward_sweep(operator, geometry, maps, level, batches, values, active, tracker, patches):
    n, finest = level
    for context in geometry.patches(n, finest, patches):
        for start, batch in batches:
            group = _PatchGroup(operator, context, maps, tracker)
            for i, c in enumerate(batch, start):
                if active is None or active[i]:
                    a, b = group.operands(c)
                    values[:, i] += group.partial(c, a, b)
            del group


def _reverse_sweep(
    operator, geometry, maps, level, batches, cotangent, active, maps_gradient, tracker, patches
):
    n, finest = level
    for context in geometry.patches(n, finest, patches):
        for start, batch in batches:
            if not any(active[start : start + len(batch)]):
                continue
            group = _PatchGroup(operator, context, maps, tracker)
            for i, c in enumerate(batch, start):
                if not active[i]:
                    continue
                a, b = group.operands(c)
                group.reverse_coefficient(c, a, b, cotangent[:, i], maps_gradient)
                del a, b
            group.finish_reverse(maps_gradient)
            del group


@torch.no_grad()
def apply_patched(
    operator,
    pyramid,
    plan=None,
    patch_pixels=4096,
    coefficient_batch_size=64,
    tracker=None,
    geometry=None,
    patches=None,
):
    """Same coefficients as ``apply_pyramid``, accumulated patch by patch.

    With ``patches`` (a set of patch indices) only those cores contribute; the
    partial statistics of disjoint patch sets add up to the full statistics.
    """
    plan, geometry = _prepare(
        operator, pyramid, plan, coefficient_batch_size, patch_pixels, geometry
    )
    maps = pyramid.analysis_levels()
    values = torch.zeros(
        maps[0].shape[0],
        len(plan),
        dtype=torch.promote_types(maps[0].dtype, torch.complex64),
        device=maps[0].device,
    )
    for level, batches in _levels(plan, coefficient_batch_size).items():
        _forward_sweep(
            operator, geometry, maps, level, batches, values, None, tracker, patches
        )
    return PyramidStatistics(plan, values)


@torch.no_grad()
def vjp_patched(
    operator,
    pyramid,
    gradient,
    plan=None,
    patch_pixels=4096,
    coefficient_batch_size=64,
    tracker=None,
    geometry=None,
    patches=None,
    gradient_callback=None,
    return_maps_gradient=False,
):
    """VJP of ``Re sum(conj(gradient) * statistics)`` computed patch by patch.

    ``return_maps_gradient=True`` returns the per-level map cotangents before
    the analysis VJP; those of disjoint patch sets can be summed on one worker
    and passed to ``pyramid.analysis_vjp``.
    """
    if isinstance(gradient, PyramidStatistics):
        if plan is not None and tuple(plan) != gradient.plan:
            raise ValueError("Gradient and requested plan differ")
        plan, gradient = gradient.plan, gradient.values
    plan, geometry = _prepare(
        operator, pyramid, plan, coefficient_batch_size, patch_pixels, geometry
    )
    if gradient.shape != (pyramid.levels[0].shape[0], len(plan)):
        raise ValueError("Coefficient cotangent must have shape [batch, len(plan)]")
    if gradient.device != pyramid.device or not bool(torch.isfinite(gradient).all()):
        raise ValueError(
            "Coefficient cotangent must be finite and on the pyramid device"
        )
    maps = pyramid.analysis_levels()
    maps_gradient = [torch.zeros_like(x) for x in maps]
    active = (gradient != 0).any(dim=0).cpu().tolist()
    for level, batches in _levels(plan, coefficient_batch_size).items():
        _reverse_sweep(
            operator, geometry, maps, level, batches, gradient, active,
            maps_gradient, tracker, patches,
        )
    del maps
    if return_maps_gradient:
        return maps_gradient
    return pyramid.analysis_vjp(maps_gradient, gradient_callback)


@torch.no_grad()
def loss_and_grad_patched(
    operator,
    pyramid,
    target,
    weights=1.0,
    patch_pixels=4096,
    coefficient_batch_size=64,
    tracker=None,
    geometry=None,
    gradient_callback=None,
):
    """Fused weighted squared loss and VJP, two patch sweeps per level.

    The first sweep accumulates the statistics of every batch of a level (a
    global pixel mean), the second applies the residual cotangents; fields
    are recomputed between the two so that no field ever covers more than
    one patch and one batch.
    """
    from STL_main.LossOptim import SquaredStatisticsLoss

    plan, geometry = _prepare(
        operator, pyramid, target.plan, coefficient_batch_size, patch_pixels, geometry
    )
    expected = (pyramid.levels[0].shape[0], len(plan))
    if target.values.shape != expected or target.values.device != pyramid.device:
        raise ValueError("Target shape/device does not match the pyramid")
    if not bool(torch.isfinite(target.values).all()):
        raise ValueError("Target coefficients must be finite")
    _, weight = SquaredStatisticsLoss(weights)._terms(target, target)
    weight = torch.broadcast_to(weight, expected)
    active = (weight != 0).any(dim=0).cpu().tolist()
    maps = pyramid.analysis_levels()
    maps_gradient = [torch.zeros_like(x) for x in maps]
    loss = maps[0].new_zeros(())
    values = torch.zeros(expected, dtype=target.values.dtype, device=maps[0].device)
    for level, batches in _levels(plan, coefficient_batch_size).items():
        _forward_sweep(
            operator, geometry, maps, level, batches, values, active, tracker, None
        )
        residual = values - target.values
        cotangent = 2 * weight * residual
        _reverse_sweep(
            operator, geometry, maps, level, batches, cotangent, active,
            maps_gradient, tracker, None,
        )
    loss = (weight * (values - target.values).abs().square()).sum()
    del maps
    return loss, pyramid.analysis_vjp(maps_gradient, gradient_callback)
