"""Patch-localised ScatCov must reproduce the full-sky explicit engine exactly."""

import pytest
import torch

pytest.importorskip("healpix_analyse")

from STL_main.LossOptim import PyramidOptimizer, PyramidSynthesis
from STL_main.PatchOps import (
    ConvRows,
    DownRows,
    conv_adjoint_rows,
    conv_forward_rows,
    down_adjoint_rows,
    down_forward_rows,
)
from STL_main.PatchScattering import (
    PatchGeometry,
    apply_patched,
    loss_and_grad_patched,
    vjp_patched,
)
from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import MemoryTracker, PyramidStatistics
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def setup_problem(
    nside=8, channels=1, batch=1, J=3, fewer=False, partial=False, dtype=torch.float64
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


@pytest.mark.parametrize("partial", [False, True])
def test_row_restricted_operators_match_full_operators(partial):
    _, op, p = setup_problem(partial=partial)
    w = op.wavelet_op
    conv = w._get_conv(0, p.templates[0].cell_ids, w.L, w._wav_weights)
    x = torch.randn(2, 1, conv.K, dtype=torch.float64)
    rows_out = torch.tensor([3, 5, 11, 40, 41, conv.K - 1])
    rows = ConvRows(conv, rows_out)
    full = w.apply_level(p.wrap(x[:, 0], 0), 0).array  # [2, L, K] complex
    local = conv_forward_rows(rows, x.index_select(-1, rows.in_domain), 5)
    local = local.reshape(2, w.L, 2, -1)
    local = torch.complex(local[..., 0, :], local[..., 1, :])
    torch.testing.assert_close(local, full.index_select(-1, rows_out))
    # Adjoint identity <conv x, g> = <x, conv^T g> on the restricted rows.
    g = torch.randn(2, 2 * w.L, rows_out.numel(), dtype=torch.float64)
    y = conv_forward_rows(rows, x.index_select(-1, rows.in_domain), 4)
    gx = conv_adjoint_rows(rows, g, 4)
    torch.testing.assert_close(
        (y * g).sum(), (x.index_select(-1, rows.in_domain) * gx).sum()
    )
    down = w._get_down(0, p.templates[0].cell_ids)
    coarse = torch.tensor([0, 7, p.levels[1].shape[-1] - 1])
    drows = DownRows(down, coarse)
    z = torch.randn(3, conv.K, dtype=torch.complex128)
    full_down = p.down(z, 0, 1)
    local_down = down_forward_rows(drows, z.index_select(-1, drows.in_domain))
    torch.testing.assert_close(local_down, full_down.index_select(-1, coarse))
    gc = torch.randn(3, coarse.numel(), dtype=torch.complex128)
    back = down_adjoint_rows(drows, gc)
    torch.testing.assert_close(
        (local_down.conj() * gc).sum().real,
        (z.index_select(-1, drows.in_domain).conj() * back).sum().real,
    )


@pytest.mark.parametrize("fewer", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("patch_pixels", [7, 64])
def test_patched_statistics_match_full_sky(fewer, partial, patch_pixels):
    _, op, p = setup_problem(channels=2, batch=2, fewer=fewer, partial=partial)
    reference = op.apply_pyramid(p, coefficient_batch_size=23)
    actual = apply_patched(
        op, p, patch_pixels=patch_pixels, coefficient_batch_size=23
    )
    assert actual.plan == reference.plan
    torch.testing.assert_close(
        actual.values, reference.values, atol=1e-11, rtol=1e-10
    )


@pytest.mark.parametrize("fewer", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_patched_vjp_matches_full_sky(fewer, partial):
    _, op, p = setup_problem(channels=2, fewer=fewer, partial=partial)
    for band in p.levels:
        band.add_(0.2 * torch.randn_like(band))
    plan = op.build_plan(channels=2)
    g = torch.randn(1, len(plan), dtype=torch.complex128)
    reference = op.vjp_pyramid(p, g, plan=plan, coefficient_batch_size=31)
    actual = vjp_patched(
        op, p, g, plan=plan, patch_pixels=16, coefficient_batch_size=31
    )
    for manual, full in zip(actual, reference):
        torch.testing.assert_close(manual, full, atol=1e-10, rtol=1e-9)
        assert not manual.requires_grad


def test_patch_contributions_are_additive_and_fields_are_patch_sized():
    _, op, p = setup_problem(nside=16, J=3)
    plan = tuple(c for c in op.build_plan() if c.family in ("S2", "S3"))
    g = torch.randn(1, len(plan), dtype=torch.complex128)
    geometry = PatchGeometry(op, p, patch_pixels=32, cache=True)
    full = vjp_patched(op, p, g, plan=plan, geometry=geometry, return_maps_gradient=True)
    n_patches = len(geometry.cores(0, 0))
    even = vjp_patched(
        op, p, g, plan=plan, geometry=geometry, patches=set(range(0, n_patches, 2)),
        return_maps_gradient=True,
    )
    odd = vjp_patched(
        op, p, g, plan=plan, geometry=geometry,
        patches=set(range(1, max(n_patches, 1), 2)), return_maps_gradient=True,
    )
    for a, b, c in zip(full, even, odd):
        torch.testing.assert_close(a, b + c, atol=1e-12, rtol=1e-12)
    stats_full = apply_patched(op, p, plan=plan, geometry=geometry)
    stats_even = apply_patched(
        op, p, plan=plan, geometry=geometry, patches=set(range(0, n_patches, 2))
    )
    stats_odd = apply_patched(
        op, p, plan=plan, geometry=geometry, patches=set(range(1, n_patches, 2))
    )
    torch.testing.assert_close(
        stats_full.values, stats_even.values + stats_odd.values, atol=1e-12, rtol=1e-12
    )
    tracker = MemoryTracker()
    vjp_patched(op, p, g, plan=plan, patch_pixels=32, tracker=tracker)
    fine = [e for e in tracker.events if e["operation"] == "first_wavelet" and e["dg"] == 0]
    # Cross-scale halos are inherent (see docs); the point is that no field
    # covers the sphere.  Measured: 898 of 3072 fine pixels for this setup.
    assert fine and max(e["shape"][-1] for e in fine) < 12 * 16**2 // 3
    tracker_small = MemoryTracker()
    vjp_patched(op, p, g, plan=plan, patch_pixels=8, tracker=tracker_small)
    small = [
        e["shape"][-1]
        for e in tracker_small.events
        if e["operation"] == "first_wavelet" and e["dg"] == 0
    ]
    assert max(small) < max(e["shape"][-1] for e in fine)


def test_fused_loss_matches_and_uses_no_autograd(monkeypatch):
    _, op, p = setup_problem(channels=2, batch=2, partial=True)
    stats = op.apply_pyramid(p)
    target = PyramidStatistics(stats.plan, stats.values * 0.7)
    weights = torch.linspace(0, 2, len(stats.plan), dtype=torch.float64)
    ref_value, ref_gradient = op.loss_and_grad_pyramid(
        p, target, weights=weights, coefficient_batch_size=17
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Patched engine invoked autograd")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    value, gradient = loss_and_grad_patched(
        op, p, target, weights=weights, patch_pixels=20, coefficient_batch_size=17
    )
    torch.testing.assert_close(value, ref_value)
    for a, b in zip(gradient, ref_gradient):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)


def test_shared_native_resolutions_against_legacy_autograd():
    data, op, _ = setup_problem(nside=16, J=3)
    op.wavelet_op.j_to_dg = [0, 0, 1]
    op.wavelet_op.pyramid_pixel_batch_size = 101
    p = HealpixPyramid.from_map(data, op.wavelet_op)
    chosen = tuple(
        c for c in op.build_plan() if c.family in ("S3", "S4") and c.j3 == 2
    )
    g = torch.randn(1, len(chosen), dtype=torch.complex128)
    actual = vjp_patched(op, p, g, plan=chosen, patch_pixels=48)
    leaves = [b.clone().requires_grad_() for b in p.levels]
    pp = HealpixPyramid(leaves, p.templates, op.wavelet_op)
    stats = op.apply(p.wrap(pp.reconstruct(), 0))
    v = torch.stack([getattr(stats, c.family)[c.legacy_index] for c in chosen], -1)
    expected = torch.autograd.grad((v * g.conj()).real.sum(), leaves)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_patched_synthesis_decreases_loss():
    data, op, target_p = setup_problem(nside=4, J=2)
    target = op.apply_pyramid(target_p)
    solver = PyramidSynthesis(
        op, target, optimizer=PyramidOptimizer(lr=[0.01, 0.02]), patch_pixels=24
    )
    p = solver.initialize_from_noise(data, seed=24)
    before = solver.loss.value(op.apply_pyramid(p), target)
    history = solver.run(p, niter=8)
    after = solver.loss.value(op.apply_pyramid(p), target)
    assert after < before and len(history) == 8
