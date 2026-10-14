"""S3(j1, j3) streamed by fine sub-patches must match the full-sky engine."""

import pytest
import torch

pytest.importorskip("healpix_analyse")

from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import MemoryTracker
from STL_main.S3Multiscale import s3_pair_plan, s3_scale_pair
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def setup_problem(nside=8, channels=2, batch=1, J=3, fewer=False, partial=False):
    torch.manual_seed(5)
    ids = torch.arange(12 * nside**2)
    if partial:
        ids = ids[: len(ids) // 2][torch.randperm(len(ids) // 2)]
    data = STL_Healpix_Kernel_Torch(
        torch.randn(batch, channels, len(ids), dtype=torch.float64),
        nside=nside,
        cell_ids=ids,
    )
    op = data.get_ST_op(
        J=J,
        L=2,
        norm="vanilla",
        compute_PS=False,
        has_fewer_convolutions=fewer,
        wavelet_op_kwargs=dict(kernel_size=3, ellipsoid="sphere"),
    )
    p = HealpixPyramid.from_map(data, op.wavelet_op)
    for band in p.levels:
        band.add_(0.2 * torch.randn_like(band))
    return op, p


@pytest.mark.parametrize("fewer", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("pair", [(0, 2), (1, 2), (2, 2), (0, 1)])
@pytest.mark.parametrize("fine_patch_pixels", [5, 64, 100000])
def test_scale_pair_matches_full_sky(fewer, partial, pair, fine_patch_pixels):
    op, p = setup_problem(fewer=fewer, partial=partial)
    plan = s3_pair_plan(op, 2, *pair)
    assert plan
    g = torch.randn(1, len(plan), dtype=torch.complex128)
    reference_stats = op.apply_pyramid(p, plan=plan)
    reference = op.vjp_pyramid(p, g, plan=plan)
    stats, maps_gradient = s3_scale_pair(
        op, p, *pair, gradient=g, fine_patch_pixels=fine_patch_pixels
    )
    torch.testing.assert_close(stats.values, reference_stats.values, atol=1e-11, rtol=1e-10)
    bands = p.analysis_vjp(maps_gradient)
    for a, b in zip(bands, reference):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_coarse_cores_and_accumulation():
    op, p = setup_problem(nside=16, J=3)
    g = {pair: None for pair in [(0, 2), (1, 2)]}
    full_plan = sum((s3_pair_plan(op, 2, *pair) for pair in g), ())
    cot = torch.randn(1, len(full_plan), dtype=torch.complex128)
    reference = op.vjp_pyramid(p, cot, plan=full_plan)
    maps_gradient = None
    start = 0
    for pair in g:
        plan = s3_pair_plan(op, 2, *pair)
        _, maps_gradient = s3_scale_pair(
            op, p, *pair, gradient=cot[:, start : start + len(plan)],
            fine_patch_pixels=37, coarse_patch_pixels=7, maps_gradient=maps_gradient,
        )
        start += len(plan)
    for a, b in zip(p.analysis_vjp(maps_gradient), reference):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_fine_fields_scale_with_the_sub_patch():
    op, p = setup_problem(nside=16, J=3, channels=1)
    plan = s3_pair_plan(op, 1, 0, 2)
    g = torch.ones(1, len(plan), dtype=torch.complex128)
    sizes = {}
    for fine in (16, 64, 256):
        tracker = MemoryTracker()
        s3_scale_pair(op, p, 0, 2, gradient=g, fine_patch_pixels=fine, tracker=tracker)
        sizes[fine] = max(
            e["shape"][-1] for e in tracker.events if e["operation"] == "first_wavelet"
        )
    assert sizes[16] < sizes[64] < sizes[256] < 12 * 16**2
