"""Small-map validation: legacy coefficients, explicit adjoints and band VJPs."""

import pytest
import torch

pytest.importorskip("healpix_analyse")

from STL_main.AdjointOps import (
    covariance,
    covariance_vjp,
    modulus_vjp,
    root_modulus_vjp,
)
from STL_main.LossOptim import PyramidOptimizer, PyramidSynthesis, SquaredStatisticsLoss
from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import MemoryTracker, PyramidStatistics
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def setup_problem(
    nside=4, channels=1, batch=1, J=2, fewer=False, partial=False, dtype=torch.float64
):
    torch.manual_seed(17)
    ids = torch.arange(12 * nside**2)
    if partial:
        ids = ids[: len(ids) // 2][torch.randperm(len(ids) // 2)]
    data = STL_Healpix_Kernel_Torch(
        torch.randn(batch, channels, len(ids), dtype=dtype), nside=nside, cell_ids=ids
    )
    op = data.get_ST_op(
        J=J,
        L=2,
        norm="vanilla",
        compute_PS=False,
        has_fewer_convolutions=fewer,
        wavelet_op_kwargs=dict(kernel_size=3, ellipsoid="sphere"),
    )
    return data, op, HealpixPyramid.from_map(data, op.wavelet_op)


def legacy_values(op, data, plan):
    stats = op.apply(data.copy())
    return torch.stack([getattr(stats, c.family)[c.legacy_index] for c in plan], -1)


@pytest.mark.parametrize("partial", [False, True])
def test_reconstruction_and_linear_adjoints(partial):
    data, op, p = setup_problem(partial=partial)
    torch.testing.assert_close(p.reconstruct(), data.array, atol=1e-14, rtol=1e-14)
    x, y = torch.randn_like(p.levels[0]), torch.randn_like(p.levels[1])
    torch.testing.assert_close(
        (p.down(x, 0, 1) * y).sum(),
        (x * p.down_adjoint(y, 0, 1)).sum(),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        (p.up(y, 1, 0) * x).sum(),
        (y * p.up_adjoint(x, 1, 0)).sum(),
        atol=1e-10,
        rtol=1e-10,
    )
    x.requires_grad_()
    z = op.wavelet_op.apply(p.wrap(x, 0), j=0).array
    g = torch.randn_like(z)
    explicit = op.wavelet_op.adjoint_level(g, p.templates[0], 0, pixel_batch_size=19)
    (reference,) = torch.autograd.grad((z.conj() * g).real.sum(), x)
    torch.testing.assert_close(explicit, reference, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        (z.conj() * g).real.sum(), (x * explicit).sum(), atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("fewer", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_forward_every_coefficient(fewer, partial):
    data, op, p = setup_problem(channels=2, batch=2, fewer=fewer, partial=partial)
    actual = op.apply_pyramid(p, coefficient_batch_size=23)
    expected = legacy_values(op, data, actual.plan)
    for family in ("mean", "var", "S1", "S2", "S3", "S4"):
        ids = [i for i, c in enumerate(actual.plan) if c.family == family]
        torch.testing.assert_close(
            actual.values[:, ids], expected[:, ids], atol=1e-11, rtol=1e-10
        )
    assert not actual.values.requires_grad


@pytest.mark.parametrize("fewer", [False, True])
@pytest.mark.parametrize("strategy", ["recompute", "store"])
def test_manual_gradient_against_legacy_and_finite_difference(fewer, strategy):
    _, op, p = setup_problem(channels=2, fewer=fewer)
    # Arbitrary (noncanonical) bands expose the D U != I trap.
    for band in p.levels:
        band.add_(0.2 * torch.randn_like(band))
    plan = op.build_plan(channels=2)
    g = torch.randn(1, len(plan), dtype=torch.complex128)
    actual = op.vjp_pyramid(
        p, g, plan=plan, coefficient_batch_size=31, strategy=strategy
    )
    leaves = [band.detach().clone().requires_grad_() for band in p.levels]
    reference_p = HealpixPyramid(leaves, p.templates, op.wavelet_op)
    vals = legacy_values(op, p.wrap(reference_p.reconstruct(), 0), plan)
    reference = torch.autograd.grad((vals * g.conj()).real.sum(), leaves)
    for d, (manual, automatic) in enumerate(zip(actual, reference)):
        torch.testing.assert_close(manual, automatic, atol=2e-10, rtol=2e-9)
        assert not manual.requires_grad
        h = torch.randn_like(p.levels[d])
        eps = 1e-5
        original = p.levels[d].clone()
        p.levels[d].copy_(original + eps * h)
        plus = op.apply_pyramid(p, plan=plan).values
        p.levels[d].copy_(original - eps * h)
        minus = op.apply_pyramid(p, plan=plan).values
        p.levels[d].copy_(original)
        fd = ((plus - minus) * g.conj()).real.sum() / (2 * eps)
        torch.testing.assert_close((manual * h).sum(), fd, atol=2e-8, rtol=2e-6)


def test_primitives_complex_and_zero():
    torch.manual_seed(42)
    a = torch.randn(3, 11, dtype=torch.complex128, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    g = torch.randn(3, dtype=a.dtype)
    ref = torch.autograd.grad((covariance(a, b) * g.conj()).real.sum(), (a, b))
    for manual, automatic in zip(covariance_vjp(a, b, g), ref):
        torch.testing.assert_close(manual, automatic)
    for transform, vjp in [
        (torch.abs, modulus_vjp),
        (lambda z: z / (z.abs() + 1e-8).sqrt(), root_modulus_vjp),
    ]:
        z = torch.tensor(
            [0, 1e-15 + 1e-15j, 1 + 2j], dtype=torch.complex128, requires_grad=True
        )
        y = transform(z)
        g = torch.randn_like(y)
        (ref,) = torch.autograd.grad((y * g.conj()).real.sum(), z)
        torch.testing.assert_close(vjp(z, g), ref, rtol=1e-10, atol=1e-10)


def test_j0_jmax_native_gradient_and_no_autograd(monkeypatch):
    data, op, p = setup_problem(nside=8, J=3)
    plan = tuple(
        c
        for c in op.build_plan()
        if c.family == "S3" and c.j1 == 0 and c.j3 == 2 and c.l1 == 0 and c.l3 == 1
    )
    reference = legacy_values(op, data, plan)

    def forbidden(*args, **kwargs):
        raise AssertionError("Production engine invoked autograd")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    tracker = MemoryTracker()
    stats = op.apply_pyramid(p, plan=plan, tracker=tracker)
    torch.testing.assert_close(stats.values, reference)
    gradients = op.vjp_pyramid(
        p, torch.ones_like(stats.values), plan=plan, tracker=tracker
    )
    events = [e for e in tracker.events if e["operation"] == "native_gradient"]
    assert len(events) == 1
    assert events[0]["dg"] == op.wavelet_op.j_to_dg[2]
    assert events[0]["shape"][-1] == 12 * (8 // 4) ** 2
    callback_result = {}
    result = op.vjp_pyramid(
        p,
        torch.ones_like(stats.values),
        plan=plan,
        gradient_callback=lambda dg, g: callback_result.update({dg: g.clone()}),
    )
    assert result is None
    for d, g in enumerate(gradients):
        torch.testing.assert_close(g, callback_result[d])


def test_synthesis_decreases_loss():
    data, op, target_p = setup_problem()
    target = op.apply_pyramid(target_p)
    solver = PyramidSynthesis(op, target, optimizer=PyramidOptimizer(lr=[0.01, 0.02]))
    p = solver.initialize_from_noise(data, seed=24)
    before = solver.loss.value(op.apply_pyramid(p), target)
    history = solver.run(p, niter=12)
    after = solver.loss.value(op.apply_pyramid(p), target)
    assert after < before
    assert len(history) == 12
    assert solver.reconstruct().shape == data.array.shape


def test_invalid_modes_and_loss_weights():
    _, op, p = setup_problem()
    with pytest.raises(ValueError, match="positive integer"):
        op.apply_pyramid(p, coefficient_batch_size=0)
    op.norm = "store_ref"
    with pytest.raises(NotImplementedError, match="vanilla"):
        op.apply_pyramid(p)
    op.norm = "vanilla"
    stats = op.apply_pyramid(p)
    with pytest.raises(ValueError, match="nonnegative"):
        SquaredStatisticsLoss(-1).value(stats, stats)


def test_float32_and_batch_invariance():
    _, op, p = setup_problem(dtype=torch.float32)
    small = op.apply_pyramid(p, coefficient_batch_size=1)
    large = op.apply_pyramid(p, coefficient_batch_size=128)
    torch.testing.assert_close(small.values, large.values)
    g = torch.randn_like(small.values)
    a = op.vjp_pyramid(p, g, coefficient_batch_size=1)
    b = op.vjp_pyramid(p, g, coefficient_batch_size=128)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)


def test_fused_loss_gradient_and_callback():
    _, op, p = setup_problem(channels=2, batch=2, partial=True)
    stats = op.apply_pyramid(p)
    target = PyramidStatistics(stats.plan, stats.values * 0.7)
    weights = torch.linspace(0, 2, len(stats.plan), dtype=torch.float64)
    loss = SquaredStatisticsLoss(weights)
    ref_value = loss.value(stats, target)
    ref_gradient = op.vjp_pyramid(p, loss.gradient(stats, target))
    value, gradient = op.loss_and_grad_pyramid(
        p, target, weights=weights, coefficient_batch_size=17
    )
    torch.testing.assert_close(value, ref_value)
    for a, b in zip(gradient, ref_gradient):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    received = {}
    _, result = op.loss_and_grad_pyramid(
        p,
        target,
        weights=weights,
        gradient_callback=lambda d, g: received.update({d: g.clone()}),
    )
    assert result is None
    for d, g in enumerate(gradient):
        torch.testing.assert_close(received[d], g)


def test_shared_native_resolutions_and_larger_map():
    data, op, _ = setup_problem(nside=16, J=3)
    op.wavelet_op.j_to_dg = [0, 0, 1]
    op.wavelet_op.pyramid_pixel_batch_size = 101
    p = HealpixPyramid.from_map(data, op.wavelet_op)
    stats = op.apply_pyramid(p)
    torch.testing.assert_close(
        stats.values, legacy_values(op, data, stats.plan), atol=1e-10, rtol=1e-9
    )
    assert {c.native_dg for c in stats.plan} == {0, 1}
    chosen = tuple(
        c for c in stats.plan if c.family == "S3" and c.j1 == 0 and c.j3 == 2
    )
    g = torch.randn(1, len(chosen), dtype=torch.complex128)
    actual = op.vjp_pyramid(p, g, plan=chosen)
    leaves = [b.clone().requires_grad_() for b in p.levels]
    pp = HealpixPyramid(leaves, p.templates, op.wavelet_op)
    v = legacy_values(op, p.wrap(pp.reconstruct(), 0), chosen)
    expected = torch.autograd.grad((v * g.conj()).real.sum(), leaves)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_adam_matches_torch_reference():
    _, _, p = setup_problem()
    manual = PyramidOptimizer(lr=0.003)
    leaves = [b.clone().requires_grad_() for b in p.levels]
    reference = torch.optim.Adam(leaves, lr=0.003)
    for _ in range(3):
        grads = [torch.randn_like(b) for b in p.levels]
        for leaf, g in zip(leaves, grads):
            leaf.grad = g.clone()
        reference.step()
        manual.step(p, grads)
    for a, b in zip(p.levels, leaves):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
