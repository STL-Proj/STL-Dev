"""Streaming HEALPix ScatCov with explicit, bounded-lifetime reverse passes.

The old ST_Operator.apply remains the scientific reference. No automatic
differentiation is used here. A coefficient describes one channel/orientation
pair, with batches of maps evaluated together.
"""

from dataclasses import dataclass
from itertools import product

import torch

from STL_main.AdjointOps import (
    covariance,
    covariance_vjp,
    mean_vjp,
    modulus_vjp,
    power_mean_vjp,
    root_modulus_vjp,
)


@dataclass(frozen=True)
class ScatCoefficient:
    family: str
    native_dg: int
    c1: int = 0
    c2: int = 0
    j1: int | None = None
    j2: int | None = None
    j3: int | None = None
    l1: int | None = None
    l2: int | None = None
    l3: int | None = None

    @property
    def legacy_index(self):
        if self.family in ("mean", "var"):
            return (slice(None), self.c1)
        index = (slice(None), self.c1, self.c2)
        if self.family in ("S1", "S2"):
            return index + (self.j1, self.l1)
        if self.family == "S3":
            return index + (self.j1, self.j3, self.l1, self.l3)
        return index + (self.j1, self.j2, self.j3, self.l1, self.l2, self.l3)


@dataclass
class PyramidStatistics:
    """Compact coefficients [batch, coefficient]; no NaN-padded dense cubes."""

    plan: tuple
    values: torch.Tensor

    @classmethod
    def from_legacy(cls, statistics, plan):
        return cls(
            tuple(plan),
            torch.stack(
                [getattr(statistics, c.family)[c.legacy_index] for c in plan], dim=-1
            )
            .detach()
            .clone(),
        )

    def family(self, name):
        indices = [i for i, c in enumerate(self.plan) if c.family == name]
        return self.values[:, indices]


TargetStatistics = PyramidStatistics


class MemoryTracker:
    """Optional event sink; records shapes/bytes, never references tensors.

    Events describe individual work fields, not process or allocator peaks.
    Use the benchmark for actual allocator/process memory measurements.
    """

    def __init__(self):
        self.events = []

    def __call__(self, operation, dg, tensor):
        self.events.append(
            dict(
                operation=operation,
                dg=dg,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                bytes=tensor.numel() * tensor.element_size(),
            )
        )


def build_plan(
    operator,
    channels=1,
    compute_cross_matrix=None,
    families=("mean", "var", "S1", "S2", "S3", "S4"),
):
    if operator.DT != "HealpixKernel_torch" or operator.SC != "ScatCov":
        raise NotImplementedError(
            "Explicit pyramid engine supports HEALPix ScatCov only"
        )
    if channels < 1:
        raise ValueError("channels must be positive")
    if set(families) - {"mean", "var", "S1", "S2", "S3", "S4"}:
        raise ValueError("Unknown scattering family")
    cross = (
        torch.ones(channels, channels, dtype=torch.bool)
        if compute_cross_matrix is None
        else compute_cross_matrix.bool().cpu()
    )
    if cross.shape != (channels, channels):
        raise ValueError("compute_cross_matrix has the wrong channel dimensions")
    plan, angles = [], range(operator.L)
    for family in ("mean", "var"):
        if family in families:
            plan.extend(ScatCoefficient(family, 0, c, c) for c in range(channels))
    for j3 in range(operator.J):
        dg = operator.wavelet_op.j_to_dg[j3]
        for c1 in range(channels):
            for c2 in range(c1, channels):
                if not cross[c1, c2]:
                    continue
                for family in ("S1", "S2"):
                    if family in families:
                        plan.extend(
                            ScatCoefficient(family, dg, c1, c2, j1=j3, l1=l)
                            for l in angles
                        )
                pairs = [(c1, c2)] if c1 == c2 else [(c1, c2), (c2, c1)]
                for ca, cb in pairs:
                    for j2 in range(j3 + 1):
                        if "S3" in families:
                            plan.extend(
                                ScatCoefficient(
                                    "S3", dg, ca, cb, j1=j2, j3=j3, l1=l2, l3=l3
                                )
                                for l2, l3 in product(angles, repeat=2)
                            )
                        if "S4" in families:
                            for j1 in range(j2 + 1):
                                plan.extend(
                                    ScatCoefficient(
                                        "S4",
                                        dg,
                                        ca,
                                        cb,
                                        j1=j1,
                                        j2=j2,
                                        j3=j3,
                                        l1=l1,
                                        l2=l2,
                                        l3=l3,
                                    )
                                    for l1, l2, l3 in product(angles, repeat=3)
                                )
    return tuple(sorted(plan, key=lambda coefficient: coefficient.native_dg))


class _Group:
    """Caches only the first/second-layer fields used by one coefficient batch."""

    def __init__(self, operator, pyramid, maps, tracker, strategy):
        self.operator, self.p, self.maps = operator, pyramid, maps
        self.w = operator.wavelet_op
        self.tracker, self.strategy = tracker, strategy
        self.first, self.second, self.moduli = {}, {}, {}
        self.gfirst, self.gsecond, self.gmoduli = {}, {}, {}

    def trace(self, name, dg, x):
        if self.tracker is not None:
            self.tracker(name, dg, x)

    def wave(self, j):
        if j not in self.first:
            dg = self.w.j_to_dg[j]
            self.first[j] = self.w.apply_level(self.p.wrap(self.maps[dg], dg), j).array
            self.trace("first_wavelet", dg, self.first[j])
        return self.first[j]

    def mod(self, key):
        j, channel, angle, outer = key
        if key not in self.moduli:
            z = self.wave(j)[:, channel, angle]
            self.moduli[key] = self.p.down(
                z.abs(), self.w.j_to_dg[j], self.w.j_to_dg[outer]
            )
        return self.moduli[key]

    def branch(self, key):
        if key not in self.second:
            j, _, _, outer = key
            dg = self.w.j_to_dg[outer]
            self.second[key] = self.w.apply_level(
                self.p.wrap(self.mod(key), dg, (j,)), outer
            ).array
            self.trace("second_wavelet", dg, self.second[key])
            # This modulus is no longer needed for the branch VJP.
            self.moduli.pop(key, None)
        return self.second[key]

    def operands(self, c):
        if c.family in ("mean", "var"):
            return self.maps[0][:, c.c1], None
        if c.family in ("S1", "S2"):
            return self.wave(c.j1)[:, c.c1, c.l1], self.wave(c.j1)[:, c.c2, c.l1]
        if c.family == "S3":
            if not self.operator.has_fewer_convolutions:
                return (
                    self.branch((c.j1, c.c1, c.l1, c.j3))[:, c.l3],
                    self.wave(c.j3)[:, c.c2, c.l3],
                )
            return (
                self.maps[c.native_dg][:, c.c1],
                self.branch((c.j1, c.c2, c.l1, c.j3))[:, c.l3],
            )
        akey, bkey = (c.j1, c.c1, c.l1, c.j3), (c.j2, c.c2, c.l2, c.j3)
        a = (
            self.mod(akey)
            if self.operator.has_fewer_convolutions
            else self.branch(akey)[:, c.l3]
        )
        return a, self.branch(bkey)[:, c.l3]

    def value(self, c, a, b):
        if c.family == "mean":
            return a.mean(-1)
        if c.family == "var" or (c.family == "S2" and c.c1 == c.c2):
            return a.abs().square().mean(-1)
        if c.family == "S1":
            if c.c1 == c.c2:
                return a.abs().mean(-1)
            a, b = a / (a.abs() + 1e-8).sqrt(), b / (b.abs() + 1e-8).sqrt()
        return covariance(a, b)

    def add_wave(self, j, channel, angle, g):
        if j not in self.gfirst:
            self.gfirst[j] = torch.zeros_like(self.wave(j))
        self.gfirst[j][:, channel, angle].add_(g)

    def add_branch(self, key, angle, g):
        if key not in self.gsecond:
            self.gsecond[key] = torch.zeros_like(self.branch(key))
        self.gsecond[key][:, angle].add_(g)

    def add_mod(self, key, g):
        if key in self.gmoduli:
            self.gmoduli[key].add_(g.real)
        else:
            self.gmoduli[key] = g.real.clone()

    def reverse_coefficient(self, c, a, b, g, maps_gradient):
        if c.family == "mean":
            ga = mean_vjp(a, g.real)
            maps_gradient[0][:, c.c1].add_(ga)
        elif c.family == "var":
            ga = power_mean_vjp(a, g)
            maps_gradient[0][:, c.c1].add_(ga)
        elif c.family in ("S1", "S2"):
            if c.c1 == c.c2:
                ga = (
                    modulus_vjp(a, mean_vjp(a, g.real))
                    if c.family == "S1"
                    else power_mean_vjp(a, g)
                )
                self.add_wave(c.j1, c.c1, c.l1, ga)
            else:
                if c.family == "S1":
                    ar, br = a / (a.abs() + 1e-8).sqrt(), b / (b.abs() + 1e-8).sqrt()
                    ga, gb = covariance_vjp(ar, br, g)
                    ga, gb = root_modulus_vjp(a, ga), root_modulus_vjp(b, gb)
                else:
                    ga, gb = covariance_vjp(a, b, g)
                self.add_wave(c.j1, c.c1, c.l1, ga)
                self.add_wave(c.j1, c.c2, c.l1, gb)
        else:
            ga, gb = covariance_vjp(a, b, g)
            if c.family == "S3":
                if self.operator.has_fewer_convolutions:
                    maps_gradient[c.native_dg][:, c.c1].add_(ga.real)
                    self.add_branch((c.j1, c.c2, c.l1, c.j3), c.l3, gb)
                else:
                    self.add_branch((c.j1, c.c1, c.l1, c.j3), c.l3, ga)
                    self.add_wave(c.j3, c.c2, c.l3, gb)
            else:
                akey = (c.j1, c.c1, c.l1, c.j3)
                if self.operator.has_fewer_convolutions:
                    self.add_mod(akey, ga)
                else:
                    self.add_branch(akey, c.l3, ga)
                self.add_branch((c.j2, c.c2, c.l2, c.j3), c.l3, gb)
        self.trace("native_gradient", c.native_dg, ga)

    def finish_reverse(self, maps_gradient):
        self.second.clear()
        self.moduli.clear()
        if self.strategy == "recompute":
            self.first.clear()
        while self.gsecond:
            key, g = self.gsecond.popitem()
            outer = key[-1]
            dg = self.w.j_to_dg[outer]
            gm = self.w.adjoint_level(g, self.p.templates[dg], outer)
            self.add_mod(key, gm)
        while self.gmoduli:
            (j, channel, angle, outer), g = self.gmoduli.popitem()
            g = self.p.down_adjoint(g, self.w.j_to_dg[j], self.w.j_to_dg[outer])
            self.add_wave(
                j, channel, angle, modulus_vjp(self.wave(j)[:, channel, angle], g)
            )
        self.first.clear()
        while self.gfirst:
            j, g = self.gfirst.popitem()
            dg = self.w.j_to_dg[j]
            maps_gradient[dg].add_(self.w.adjoint_level(g, self.p.templates[dg], j))


def _validate(operator, pyramid, plan, coefficient_batch_size, strategy):
    if operator.wavelet_op is not pyramid.wavelet_op:
        raise ValueError(
            "Use the same wavelet operator to construct the pyramid and scattering"
        )
    if (
        operator.norm != "vanilla"
        or operator.compute_PS
        or operator.iso
        or operator.angular_ft
        or operator.scale_ft
        or operator.flatten
    ):
        raise NotImplementedError(
            "Construct the pyramid operator with norm='vanilla', compute_PS=False and no compression; use fixed loss weights for normalization"
        )
    if operator.SC != "ScatCov" or operator.DT != "HealpixKernel_torch":
        raise NotImplementedError("Only HEALPix ScatCov is supported")
    if not isinstance(coefficient_batch_size, int) or coefficient_batch_size < 1:
        raise ValueError("coefficient_batch_size must be a positive integer")
    if strategy not in ("store", "recompute"):
        raise ValueError("strategy must be 'store' or 'recompute'")
    valid = set(build_plan(operator, pyramid.levels[0].shape[1]))
    if not plan or any(c not in valid for c in plan) or len(set(plan)) != len(plan):
        raise ValueError(
            "Plan must contain unique valid coefficients with correct native_dg"
        )


def _batches(plan, size):
    """Never combine native resolutions, even for a caller-supplied plan."""
    start = 0
    while start < len(plan):
        stop = start + 1
        while (
            stop < min(start + size, len(plan))
            and plan[stop].native_dg == plan[start].native_dg
        ):
            stop += 1
        yield start, plan[start:stop]
        start = stop


@torch.no_grad()
def apply_pyramid(
    operator,
    pyramid,
    plan=None,
    coefficient_batch_size=64,
    strategy="recompute",
    tracker=None,
):
    plan = tuple(
        build_plan(operator, pyramid.levels[0].shape[1]) if plan is None else plan
    )
    _validate(operator, pyramid, plan, coefficient_batch_size, strategy)
    maps = pyramid.analysis_levels()
    values = []
    for start, batch in _batches(plan, coefficient_batch_size):
        group = _Group(operator, pyramid, maps, tracker, strategy)
        for c in batch:
            a, b = group.operands(c)
            values.append(group.value(c, a, b))
        del group, a, b
    return PyramidStatistics(plan, torch.stack(values, dim=-1))


@torch.no_grad()
def vjp_pyramid(
    operator,
    pyramid,
    gradient,
    plan=None,
    coefficient_batch_size=64,
    strategy="recompute",
    tracker=None,
    gradient_callback=None,
):
    """VJP for Re sum(conj(gradient) * statistics), returning one tensor/band."""
    if isinstance(gradient, PyramidStatistics):
        if plan is not None and tuple(plan) != gradient.plan:
            raise ValueError("Gradient and requested plan differ")
        plan, gradient = gradient.plan, gradient.values
    plan = tuple(
        build_plan(operator, pyramid.levels[0].shape[1]) if plan is None else plan
    )
    _validate(operator, pyramid, plan, coefficient_batch_size, strategy)
    if gradient.shape != (pyramid.levels[0].shape[0], len(plan)):
        raise ValueError("Coefficient cotangent must have shape [batch, len(plan)]")
    if gradient.device != pyramid.device or not bool(torch.isfinite(gradient).all()):
        raise ValueError(
            "Coefficient cotangent must be finite and on the pyramid device"
        )
    maps = pyramid.analysis_levels()
    maps_gradient = [torch.zeros_like(x) for x in maps]
    # Transfer one small activity mask, avoiding one CUDA synchronization per
    # coefficient when zero cotangents are skipped.
    active = (gradient != 0).any(dim=0).cpu().tolist()
    for start, batch in _batches(plan, coefficient_batch_size):
        group = _Group(operator, pyramid, maps, tracker, strategy)
        for i, c in enumerate(batch, start):
            # Selected-coefficient losses need not evaluate zero-weight branches.
            if not active[i]:
                continue
            a, b = group.operands(c)
            group.reverse_coefficient(c, a, b, gradient[:, i], maps_gradient)
            del a, b
        group.finish_reverse(maps_gradient)
        del group
    del maps
    return pyramid.analysis_vjp(maps_gradient, gradient_callback)


@torch.no_grad()
def loss_and_grad_pyramid(
    operator,
    pyramid,
    target,
    weights=1.0,
    coefficient_batch_size=64,
    strategy="recompute",
    tracker=None,
    gradient_callback=None,
):
    """Fused weighted squared loss/VJP: one analysis and one forward per batch.

    The target has fixed (detached) coefficients. Weights can include fixed
    reference normalization. No optimizer update occurs until the VJP completes.
    """
    from STL_main.LossOptim import SquaredStatisticsLoss

    plan = target.plan
    _validate(operator, pyramid, plan, coefficient_batch_size, strategy)
    expected = (pyramid.levels[0].shape[0], len(plan))
    if target.values.shape != expected or target.values.device != pyramid.device:
        raise ValueError("Target shape/device does not match the pyramid")
    if not bool(torch.isfinite(target.values).all()):
        raise ValueError("Target coefficients must be finite")
    # Validate and broadcast weights once, before computing large fields.
    _, weight = SquaredStatisticsLoss(weights)._terms(target, target)
    weight = torch.broadcast_to(weight, expected)
    active = (weight != 0).any(dim=0).cpu().tolist()
    maps = pyramid.analysis_levels()
    maps_gradient = [torch.zeros_like(x) for x in maps]
    loss = maps[0].new_zeros(())
    for start, batch in _batches(plan, coefficient_batch_size):
        group = _Group(operator, pyramid, maps, tracker, strategy)
        for i, c in enumerate(batch, start):
            if not active[i]:
                continue
            a, b = group.operands(c)
            residual = group.value(c, a, b) - target.values[:, i]
            loss.add_((weight[:, i] * residual.abs().square()).sum())
            group.reverse_coefficient(
                c, a, b, 2 * weight[:, i] * residual, maps_gradient
            )
            del a, b
        group.finish_reverse(maps_gradient)
        del group
    del maps
    return loss, pyramid.analysis_vjp(maps_gradient, gradient_callback)
