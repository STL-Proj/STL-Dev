import numpy as np
import torch

from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch
from STL_main.Synthesis import (
    _healpix_patch_domains,
    reweight,
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


def test_global_loss_and_gradient_are_bitwise_identical_with_and_without_blocks():
    """A blockwise VJP must only partition dL/dx, never change L."""
    nside = 4
    npix = 12 * nside**2
    generator = torch.Generator().manual_seed(17)
    target = STL_Healpix_Kernel_Torch(
        torch.randn(npix, generator=generator, dtype=torch.float64)
    )

    # Use one explicitly shared target normalization and one explicitly shared
    # scattering operator for both calculations.
    target_array = target.array[None, None, :]
    target_mean = target_array.mean(dim=-1, keepdim=True)
    target_std = target_array.std(dim=-1, correction=0, keepdim=True)
    target_standardized = (target_array - target_mean) / target_std
    target_data = target.new_like(target_standardized, pbc=True)
    st_op = target_data.get_ST_op(
        J=2,
        L=1,
        compute_PS=False,
        has_fewer_convolutions=True,
        replace_nan_value=None,
        wavelet_op_kwargs={"kernel_size": 3},
    )
    weights = {"S3": 3.5, "S4": 3.5**2}

    with torch.no_grad():
        target_stats = st_op.apply(
            target_data,
            has_fewer_convolutions=True,
            compute_PS=False,
            norm="store_ref",
            norm_batch_mean=True,
        )
        reweight(target_stats, weights)
        target_flat = target_stats.to_flatten(
            mean_along_batch=True, keepnans=False
        ).detach()

    def loss_fn(array):
        stats = st_op.apply(
            target.new_like(array, pbc=True),
            has_fewer_convolutions=True,
            compute_PS=False,
            norm="load_ref",
        )
        reweight(stats, weights)
        flat = stats.to_flatten(mean_along_batch=True, keepnans=False)
        return ((flat - target_flat).abs() ** 2).sum()

    initial = torch.randn((1, 1, npix), generator=generator, dtype=torch.float64)

    # Reference: one loss and one backward with the whole map as the leaf.
    full_map = initial.clone().requires_grad_(True)
    full_loss = loss_fn(full_map)
    (full_gradient,) = torch.autograd.grad(full_loss, full_map)

    # Block version: the full map values stay identical for every evaluation;
    # only the leaf with respect to which autograd differentiates changes.
    frozen_map = initial.detach()
    assembled_gradient = torch.zeros_like(frozen_map)
    block_losses = []
    blocks = _healpix_patch_domains(nside, patch_nside=2, halo_rings=0)
    for block_np, _, _ in blocks:
        block = torch.as_tensor(block_np, dtype=torch.long)
        local_leaf = frozen_map.index_select(-1, block).clone().requires_grad_(True)
        reconstructed_map = frozen_map.index_copy(-1, block, local_leaf)
        block_loss = loss_fn(reconstructed_map)
        (block_gradient,) = torch.autograd.grad(block_loss, local_leaf)

        # Every reconstructed map equals initial exactly, hence every loss must
        # be the exact same floating-point number as the reference loss.
        torch.testing.assert_close(
            block_loss.detach(), full_loss.detach(), rtol=0.0, atol=0.0
        )
        assembled_gradient.index_copy_(-1, block, block_gradient)
        block_losses.append(block_loss.detach())

    torch.testing.assert_close(
        torch.stack(block_losses),
        full_loss.detach().expand(len(block_losses)),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(assembled_gradient, full_gradient, rtol=0.0, atol=0.0)
