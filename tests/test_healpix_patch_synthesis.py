import numpy as np
import torch

from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch
from STL_main.Synthesis import (
    _healpix_patch_domains,
    synthesize_healpix_from_patches,
)


def test_nested_patch_domains_partition_the_full_sky_and_keep_core_in_halo():
    nside = 8
    domains = _healpix_patch_domains(nside, patch_nside=2, halo_rings=1)

    cores = np.concatenate([core for core, _, _ in domains])
    np.testing.assert_array_equal(np.sort(cores), np.arange(12 * nside**2))
    assert len(domains) == 48
    for core, support, core_positions in domains:
        np.testing.assert_array_equal(support[core_positions], core)


def test_block_gradients_equal_the_conventional_global_gradient():
    nside = 4
    generator = torch.Generator().manual_seed(4)
    target = STL_Healpix_Kernel_Torch(torch.randn(12 * nside**2, generator=generator))

    result, diagnostics = synthesize_healpix_from_patches(
        target,
        patch_nside=2,
        J=2,
        L=1,
        has_fewer_convolutions=True,
        max_iter=1,
        lr=2e-2,
        wavelet_op_kwargs={"kernel_size": 3},
        verbose=False,
        track_memory=False,
        verify_gradient=True,
        return_diagnostics=True,
    )

    assert result.shape == target.array.shape
    assert torch.isfinite(result).all()
    assert diagnostics["objective"] == "single global ScatCov loss"
    assert diagnostics["block_count"] == 48
    assert diagnostics["block_pixels"] == 4
    assert diagnostics["graph_pixels"] == diagnostics["full_sky_pixels"]
    assert diagnostics["global_gradient_max_abs_error"] < 1e-6
    assert diagnostics["global_gradient_max_rel_error"] < 1e-6
