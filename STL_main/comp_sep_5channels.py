import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import torch

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch


# Full phase (5-channel phase)
def baseline_comp_sep(
    d_U,
    d_Q,
    d_I,
    c_U,
    c_Q,
    max_iter=25,
    batch_size=None,
    resampling_period=None,
    epoch_period=None,
    STL_DataClass=STL_2D_FFT_Torch,
):

    # Set default values for optional parameters
    N = c_U.shape[0]  # Number of contamination maps in the ensemble
    if batch_size is None:
        batch_size = N  # default to full batch if not specified

    if resampling_period is None:
        resampling_period = 1  # default value

    elif resampling_period != 1 and resampling_period != max_iter:
        warnings.warn(
            "Semi-stochastic gradient descent optimization with mini-batch resampling period different from 1 or max_iter. "
            "May be trapped in local optima.",
            UserWarning,
        )

    if epoch_period is None:
        epoch_period = max_iter  # Default to single epoch if not specified

    # Sanity checks
    assert (
        batch_size <= N
    ), "Mini batch size in SGD must be less than or equal to the number of contamination maps in the ensemble."
    assert (
        resampling_period <= max_iter
    ), "Resampling period must be less than or equal to max iterations."

    target_maps = torch.stack(
        [
            d_U.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
            c_U,  # [N, H, W]
            d_Q.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
            c_Q,  # [N, H, W]
            d_I.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
        ],
        dim=1,
    )  # [N, 5, H, W]

    compute_PS = not target_maps.isnan().any()

    running_maps = torch.stack([d_U.clone(), d_Q.clone()], dim=0)  # [2, H, W]
    running_maps.requires_grad_(True)

    compute_cross_matrix = torch.tensor(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
        ],
        dtype=torch.bool,
    )

    optimizer = torch.optim.LBFGS(
        [running_maps],
        lr=1,
        max_iter=max_iter,
        tolerance_grad=1e-17,
        tolerance_change=1e-17,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    # Utils function to compute stats and flatten them for the loss computation
    def stats_flatten(data, st_op, compute_PS, norm=None):
        return st_op.apply(
            data,
            norm=norm,
            norm_batch_mean=True,
            compute_cross_matrix=compute_cross_matrix,
            compute_PS=compute_PS,
        ).to_flatten(keep_channel_dim=True)

    # Split total iterations into epoch iterations
    n_epochs = max_iter // epoch_period
    n_iter_per_epoch = [epoch_period] * n_epochs + (
        [max_iter % epoch_period] if max_iter % epoch_period > 0 else []
    )

    loss_DC1_history = []
    loss_DC2_history = []
    loss_MC1_history = []
    loss_MC2_history = []
    loss_history = []
    print_iter = 5

    # Use a mutable objects to allow modification inside closure
    iter_counter = [0]
    indices = torch.arange(
        batch_size
    )  # Initialize randomly, will be updated in closure

    def closure():
        optimizer.zero_grad()

        if iter_counter[0] % resampling_period == 0:
            perm = torch.randperm(N)
            indices[:] = perm[:batch_size]

        iter_counter[0] += 1

        rm = torch.stack(
            [
                running_maps[0].unsqueeze(0) + c_U[indices],  # [batch_size, H, W]
                (d_U - running_maps[0])
                .unsqueeze(0)
                .repeat(batch_size, 1, 1),  # [batch_size, H, W]
                running_maps[1].unsqueeze(0) + c_Q[indices],  # [batch_size, H, W]
                (d_Q - running_maps[1])
                .unsqueeze(0)
                .repeat(batch_size, 1, 1),  # [batch_size, H, W]
                d_I.unsqueeze(0).repeat(batch_size, 1, 1),  # [batch_size, H, W]
            ],
            dim=1,
        )  # [batch_size, 5, H, W]

        stl_rm = STL_DataClass(rm, pbc=True)

        stats_rm = stats_flatten(
            stl_rm, st_op, norm="load_ref", compute_PS=compute_PS
        )  # [batch_size, Nc, Nc, n_stats]

        loss_DC1 = (
            (
                stats_rm[:, 0, 0].mean(dim=0)
                - stats_target_maps[indices][:, 0, 0].mean(dim=0)
            )
            .abs()
            .square()
            .sum()
        )
        loss_MC1 = (
            (
                stats_rm[:, 1, 1].mean(dim=0)
                - stats_target_maps[indices][:, 1, 1].mean(dim=0)
            )
            .abs()
            .square()
            .sum()
        )
        loss_DC2 = (
            (
                stats_rm[:, 2, 2].mean(dim=0)
                - stats_target_maps[indices][:, 2, 2].mean(dim=0)
            )
            .abs()
            .square()
            .sum()
        )
        loss_MC2 = (
            (
                stats_rm[:, 3, 3].mean(dim=0)
                - stats_target_maps[indices][:, 3, 3].mean(dim=0)
            )
            .abs()
            .square()
            .sum()
        )

        loss = loss_DC1 + loss_MC1 + loss_DC2 + loss_MC2

        loss.backward()
        loss_DC1_history.append(loss_DC1.item())
        loss_DC2_history.append(loss_DC2.item())
        loss_MC1_history.append(loss_MC1.item())
        loss_MC2_history.append(loss_MC2.item())
        loss_history.append(loss.item())

        if len(loss_history) % print_iter == 0:
            print(f"[LBFGS] iter {len(loss_history)}, loss = {loss.item():.6e}")

        return loss

    start = time.perf_counter()

    with torch.no_grad():
        stl_target_maps = STL_DataClass(target_maps, pbc=True)
        st_op = stl_target_maps.get_ST_op()
        stats_target_maps = stats_flatten(
            stl_target_maps, st_op, norm="store_ref", compute_PS=compute_PS
        )  # [N, Nc, Nc, n_stats]

    for epoch_index, n_iters in enumerate(n_iter_per_epoch):
        print(
            f"--- Epoch {epoch_index+1}/{len(n_iter_per_epoch)} ({n_iters} iters) ---"
        )

        optimizer.param_groups[0]["max_iter"] = n_iters

        # Update normalisation term with the current running maps stats at the beginning of each epoch
        if epoch_index > 0:
            with torch.no_grad():
                stl_current_maps = STL_DataClass(
                    torch.stack(
                        [
                            running_maps[0].unsqueeze(0) + c_U,  # [N, H, W]
                            (d_U - running_maps[0])
                            .unsqueeze(0)
                            .repeat(N, 1, 1),  # [N, H, W]
                            running_maps[1].unsqueeze(0) + c_Q,  # [N, H, W]
                            (d_Q - running_maps[1])
                            .unsqueeze(0)
                            .repeat(N, 1, 1),  # [N, H, W]
                            d_I.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
                        ],
                        dim=1,
                    ),
                    pbc=True,
                )  # [N, 5, H, W]
                st_op_current = stl_current_maps.get_ST_op()
                stats_current_maps = st_op_current.apply(
                    stl_current_maps,
                    norm="store_ref",
                    norm_batch_mean=True,
                    compute_cross_matrix=compute_cross_matrix,
                    compute_PS=compute_PS,
                )

                # ------- Transfer reference normalization from current running operator to target operator -------
                st_op.S2_ref_sqrt_chan_diag = st_op_current.S2_ref_sqrt_chan_diag
                st_op.var_ref = st_op_current.var_ref
                if compute_PS:
                    st_op.PS_ref_sqrt_chan_diag = st_op_current.PS_ref_sqrt_chan_diag

        optimizer.step(closure)

    end = time.perf_counter()

    print(f"{len(loss_history)} iterations of synthesis.")
    print(f"Execution time: {end - start:.3f} s")

    running_maps = running_maps.detach()
    s_U_opt = running_maps[0]
    s_Q_opt = running_maps[1]

    return (s_U_opt, s_Q_opt), (
        loss_DC1_history,
        loss_DC2_history,
        loss_MC1_history,
        loss_MC2_history,
        loss_history,
    )
