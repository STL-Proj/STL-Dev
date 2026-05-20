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

    compute_PS = False

    running_maps = torch.stack([d_U.clone(), d_Q.clone()], dim=0)  # [2, H, W]

    running_maps.requires_grad_(True)

    compute_cross_matrix = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, False, False, False],
            [False, False, True, False, True],
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
    def stats_flatten(data, st_op, norm=None):
        return st_op.apply(
            data,
            norm=norm,
            norm_batch_mean=True,
            compute_cross_matrix=compute_cross_matrix,
            compute_PS=compute_PS,
        ).to_flatten(keep_batch_dim=True)

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

        stats_rm = stats_flatten(stl_rm, st_op, norm="load_ref")
        loss = (
            (stats_rm.mean(dim=0) - stats_target[indices].mean(dim=0))
            .abs()
            .square()
            .sum()
        )

        loss.backward()

        loss_history.append(loss.item())

        if len(loss_history) % print_iter == 0:
            print(f"[LBFGS] iter {len(loss_history):4d} - loss: {loss.item():.6e}")
        return loss

    start = time.perf_counter()

    with torch.no_grad():
        stl_target = STL_DataClass(target_maps, pbc=True)
        st_op = stl_target.get_ST_op()

        stats_target = stats_flatten(stl_target, st_op, norm="store_ref")

    optimizer.step(closure)

    end = time.perf_counter()

    print(f"{len(loss_history)} iterations of synthesis.")
    print(f"Execution time: {end - start:.3f} s")

    running_maps = running_maps.detach()
    s_U_opt = running_maps[0]
    s_Q_opt = running_maps[1]

    return (s_U_opt, s_Q_opt), loss_history
