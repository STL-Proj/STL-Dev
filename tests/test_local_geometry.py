"""Patch-local partial operators must reproduce the full-sky engine exactly."""

import pytest
import torch

pytest.importorskip("healpix_analyse")

from STL_main.LocalGeometry import LocalGeometry, s3_scale_pair_local
from STL_main.Pyramid import HealpixPyramid
from STL_main.S3Multiscale import s3_pair_plan, s3_scale_pair
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def setup_problem(nside=32, channels=2, batch=1, J=4, kernel_size=3, fewer=False):
    torch.manual_seed(9)
    data = STL_Healpix_Kernel_Torch(
        torch.randn(batch, channels, 12 * nside**2, dtype=torch.float64), nside=nside
    )
    op = data.get_ST_op(
        J=J,
        L=2,
        norm="vanilla",
        compute_PS=False,
        has_fewer_convolutions=fewer,
        wavelet_op_kwargs=dict(kernel_size=kernel_size, ellipsoid="sphere", conv_cache_dir=None),
    )
    p = HealpixPyramid.from_map(data, op.wavelet_op)
    for band in p.levels:
        band.add_(0.2 * torch.randn_like(band))
    return data, op, p


@pytest.mark.parametrize("block", [64, 1024])
def test_block_decimation_and_pyramid_match_full_sky(block):
    data, op, p = setup_problem()
    geo = LocalGeometry(op.wavelet_op, block_pixels=block)
    reference = p.analysis_levels()
    local = geo.analysis_levels(p)
    for a, b in zip(local, reference):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    g = [torch.randn_like(x) for x in reference]
    expected = p.analysis_vjp([x.clone() for x in g])
    actual = geo.analysis_vjp(p, [x.clone() for x in g])
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    q = geo.pyramid_from_map(data)
    r = HealpixPyramid.from_map(data, op.wavelet_op)
    for a, b in zip(q.levels, r.levels):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    assert geo.built["conv"] == 0


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("fewer", [False, True])
@pytest.mark.parametrize("pair", [(0, 3), (1, 3), (0, 2), (3, 3)])
@pytest.mark.parametrize("fine_patch", [256, 4096])
def test_local_s3_pair_matches_full_sky(kernel_size, fewer, pair, fine_patch):
    _, op, p = setup_problem(kernel_size=kernel_size, fewer=fewer)
    plan = s3_pair_plan(op, 2, *pair)
    g = torch.randn(1, len(plan), dtype=torch.complex128, device=p.device)
    stats_ref, maps_ref = s3_scale_pair(op, p, *pair, gradient=g, fine_patch_pixels=fine_patch)
    geo = LocalGeometry(op.wavelet_op, block_pixels=fine_patch)
    stats, maps_gradient = s3_scale_pair_local(
        op, p, *pair, gradient=g, geometry=geo, fine_patch_pixels=fine_patch
    )
    torch.testing.assert_close(stats.values, stats_ref.values, atol=1e-11, rtol=1e-10)
    for a, b in zip(maps_gradient, maps_ref):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    bands = geo.analysis_vjp(p, maps_gradient)
    reference = op.vjp_pyramid(p, g, plan=plan)
    for a, b in zip(bands, reference):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_exact_geometry_cache_roundtrip(tmp_path):
    _, op, _ = setup_problem(nside=16, J=2, kernel_size=5)
    # The disk cache belongs to the healpix-analyse fallback path.
    cached = LocalGeometry(
        op.wavelet_op, block_pixels=256, cache_dir=str(tmp_path), torch_geometry=False
    )
    fresh = LocalGeometry(op.wavelet_op, block_pixels=256, torch_geometry=False)
    ids = cached.blocks(0)[2]
    first = cached.conv(0, ids)
    assert cached.built.get("saved") == 1
    reloaded = cached.conv(0, ids)
    reference = fresh.conv(0, ids)
    assert reloaded._w_norm.dtype == reference._w_norm.dtype == torch.float64
    torch.testing.assert_close(reloaded._w_norm, reference._w_norm, atol=0, rtol=0)
    assert torch.equal(reloaded._pos_safe, reference._pos_safe)
    assert torch.equal(first._pos_safe, reference._pos_safe)


def test_local_s3_pair_larger_map_and_cached_downs():
    """Halos of several rings at intermediate levels (nside 64, J=5)."""
    _, op, p = setup_problem(nside=64, J=5, channels=1, kernel_size=5)
    plan = s3_pair_plan(op, 1, 0, 4)
    g = torch.randn(1, len(plan), dtype=torch.complex128, device=p.device)
    _, maps_ref = s3_scale_pair(op, p, 0, 4, gradient=g, fine_patch_pixels=4096)
    # Fallback path (library operators), whose decimations are cached.
    geo = LocalGeometry(op.wavelet_op, block_pixels=4096, torch_geometry=False)
    _, maps_gradient = s3_scale_pair_local(op, p, 0, 4, gradient=g, geometry=geo)
    for a, b in zip(maps_gradient, maps_ref):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    built = dict(geo.built)
    _, again = s3_scale_pair_local(op, p, 0, 4, gradient=g, geometry=geo)
    assert geo.built["down"] == built["down"]  # cached across calls
    assert geo.built["conv"] == 2 * built["conv"]
    for a, b in zip(again, maps_ref):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
