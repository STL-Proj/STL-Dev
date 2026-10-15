"""The torch geometry must reproduce healpix-analyse's HealPixConv exactly (sphere)."""

import numpy as np
import pytest
import torch

pytest.importorskip("healpix_analyse")

from healpix_analyse.convol import HealPixConv
from healpix_geo.nested import healpix_to_lonlat

from STL_main.TorchGeometry import TorchConv, nest2xyf, ring2nest, xyf2ang


def _library_conv(ids, nside, kernel_sz, G, gauge_type, weights):
    conv = HealPixConv(
        level=int(np.log2(nside)), in_channels=1, out_channels=2, kernel_sz=kernel_sz,
        n_gauges=G, gauge_type=gauge_type, cell_ids=ids.cpu().numpy(), nest=True,
        device="cpu", dtype=torch.float64, ellipsoid="sphere", cache_dir=None,
    )
    conv.set_kernel(weights)
    return conv


@pytest.mark.parametrize("nside", [8, 32, 256])
def test_pixel_centres_match_healpix_geo(nside):
    order = int(np.log2(nside))
    ids = torch.randperm(12 * nside**2, generator=torch.Generator().manual_seed(1))[:5000]
    theta, phi = xyf2ang(*nest2xyf(ids, order), nside)
    lon, lat = healpix_to_lonlat(ids.numpy().astype(np.uint64), order, ellipsoid="sphere")
    torch.testing.assert_close(torch.rad2deg(phi), torch.as_tensor(np.asarray(lon)), atol=1e-11, rtol=0)
    torch.testing.assert_close(90 - torch.rad2deg(theta), torch.as_tensor(np.asarray(lat)), atol=1e-11, rtol=0)


@pytest.mark.parametrize("order", [1, 3, 6])
def test_ring2nest_roundtrip(order):
    from healpix_geo.ring import to_nested

    nside = 1 << order
    ring = torch.arange(12 * nside**2)
    expected = torch.as_tensor(np.asarray(to_nested(ring.numpy().astype(np.uint64), order)).astype(np.int64))
    assert torch.equal(ring2nest(ring, order), expected)


@pytest.mark.parametrize("nside,count", [(8, 768), (32, 3000), (128, 6000)])
@pytest.mark.parametrize("kernel_sz,G,gauge", [(3, 2, "cosmo"), (5, 4, "cosmo"), (5, 2, "phi")])
@pytest.mark.parametrize("shuffle", [False, True])
def test_torch_conv_matches_library(nside, count, kernel_sz, G, gauge, shuffle):
    g = torch.Generator().manual_seed(nside + kernel_sz)
    npix = 12 * nside**2
    # A compact NESTED block (complete stencils inside), optionally stored in
    # a shuffled order to exercise the sort/unsort bookkeeping.
    ids = torch.arange(npix)[:count]
    if shuffle:
        ids = ids[torch.randperm(count, generator=g)]
    weights = np.random.default_rng(3).standard_normal((1, 2, kernel_sz**2))
    reference = _library_conv(ids, nside, kernel_sz, G, gauge, weights)
    mine = TorchConv(ids, nside, kernel_sz, G, gauge, weights, torch.float64)
    assert torch.equal(mine._sort_order, reference._sort_order)
    assert torch.equal(mine._inv_order, reference._inv_order)
    torch.testing.assert_close(mine.weight, reference.weight.detach(), atol=0, rtol=0)
    # Directions that fall exactly on a pixel centre (the central tap) get
    # their unit weight in a different slot depending on float rounding, so
    # the raw buffers may differ while the operators are identical: compare
    # the linear operators on random inputs and the per-tap weight sums.
    # Rows with a truncated stencil (partial grids) fall back differently in
    # exact-tie cases; only complete rows are used by the local geometry.
    complete = mine._complete
    assert complete.all() if count == 12 * nside**2 else complete.sum() > count // 2
    for _ in range(3):
        x = torch.randn(2, 1, count, dtype=torch.float64, generator=g)
        # ~1e-12 is the rounding level of the interpolation weights themselves
        # (the library documents the same level against healpy).
        torch.testing.assert_close(
            mine(x)[..., complete], reference(x)[..., complete], atol=1e-10, rtol=1e-10
        )
