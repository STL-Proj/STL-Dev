"""Full ScatCov with patch-local geometry must equal the full-sky explicit engine."""

import pytest
import torch

pytest.importorskip("healpix_analyse")

from STL_main.LocalGeometry import LocalGeometry
from STL_main.LocalScattering import apply_local, loss_and_grad_local, vjp_local
from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import MemoryTracker, PyramidStatistics
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def setup_problem(nside=16, channels=2, batch=2, J=3, kernel_size=3, fewer=False):
    torch.manual_seed(21)
    data = STL_Healpix_Kernel_Torch(
        torch.randn(batch, channels, 12 * nside**2, dtype=torch.float64), nside=nside
    )
    op = data.get_ST_op(
        J=J, L=2, norm="vanilla", compute_PS=False, has_fewer_convolutions=fewer,
        # The library disk cache would hand float32-rounded weights to the
        # full-sky reference; the local operators are always built exactly.
        wavelet_op_kwargs=dict(kernel_size=kernel_size, ellipsoid="sphere", conv_cache_dir=None),
    )
    p = HealpixPyramid.from_map(data, op.wavelet_op)
    for band in p.levels:
        band.add_(0.2 * torch.randn_like(band))
    return op, p


@pytest.mark.parametrize("fewer", [False, True])
@pytest.mark.parametrize("block", [64, 100000])
def test_all_families_match_full_sky(fewer, block):
    op, p = setup_problem(fewer=fewer)
    geo = LocalGeometry(op.wavelet_op, block_pixels=block)
    reference = op.apply_pyramid(p)
    actual = apply_local(op, p, geometry=geo)
    assert actual.plan == reference.plan
    for family in ("mean", "var", "S1", "S2", "S3", "S4"):
        ids = [i for i, c in enumerate(actual.plan) if c.family == family]
        torch.testing.assert_close(
            actual.values[:, ids], reference.values[:, ids], atol=1e-11, rtol=1e-10
        )
    g = torch.randn_like(reference.values)
    expected = op.vjp_pyramid(p, g)
    bands = vjp_local(op, p, g, geometry=geo)
    for a, b in zip(bands, expected):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_fused_loss_and_shared_resolutions():
    op, p = setup_problem(nside=32, J=4, channels=1, batch=1, kernel_size=5)
    op.wavelet_op.j_to_dg = [0, 1, 1, 2]
    p = HealpixPyramid.from_map(p.wrap(p.reconstruct(), 0), op.wavelet_op)
    stats = op.apply_pyramid(p)
    target = PyramidStatistics(stats.plan, stats.values * 0.8)
    weights = torch.linspace(0.5, 2, len(stats.plan), dtype=torch.float64)
    ref_loss, ref_bands = op.loss_and_grad_pyramid(p, target, weights=weights)
    geo = LocalGeometry(op.wavelet_op, block_pixels=256)
    tracker = MemoryTracker()
    loss, bands = loss_and_grad_local(op, p, target, weights=weights, geometry=geo, tracker=tracker)
    torch.testing.assert_close(loss, ref_loss)
    for a, b in zip(bands, ref_bands):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    fine = [e["shape"][-1] for e in tracker.events if e["operation"] == "first_wavelet" and e["dg"] == 0]
    assert max(fine) < 12 * 32**2 // 4


def test_synthesis_with_local_geometry_decreases_loss(monkeypatch):
    from STL_main.LossOptim import PyramidLBFGS, PyramidSynthesis

    op, target_p = setup_problem(nside=8, J=2, channels=1, batch=1)
    geo = LocalGeometry(op.wavelet_op, block_pixels=128)
    target = apply_local(op, target_p, geometry=geo)
    solver = PyramidSynthesis(op, target, optimizer=PyramidLBFGS(lr=1.0, history_size=4), geometry=geo)
    example = STL_Healpix_Kernel_Torch(target_p.reconstruct()[0, 0], nside=8)
    p = solver.initialize_from_noise(example, seed=3)
    before = solver.loss.value(apply_local(op, p, geometry=geo), target)

    def forbidden(*args, **kwargs):
        raise AssertionError("Local synthesis invoked autograd")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    history = solver.run(p, niter=4)
    after = solver.loss.value(apply_local(op, p, geometry=geo), target)
    assert after < before and history
    assert geo.built["conv"] > 0
