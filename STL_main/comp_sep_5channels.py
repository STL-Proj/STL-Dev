import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch


def save_comp_sep(
    d_U,
    d_Q,
    s_U_opt,
    s_Q_opt,
    s_U_opt_noisy,
    s_Q_opt_noisy,
    residual_U,
    residual_Q,
    output_dir_name,
    filename,
):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # common bounds for U
    vmin_U = d_U.min().item()
    vmax_U = d_U.max().item()

    # common bounds for Q
    vmin_Q = d_Q.min().item()
    vmax_Q = d_Q.max().item()

    # plotting d_U, s_U_opt, s_U_opt_noisy, and residual_U
    im_U1 = axes[0, 0].imshow(
        d_U.cpu().numpy(), vmin=vmin_U, vmax=vmax_U, cmap="viridis"
    )
    axes[0, 0].set_title("$d_U$")

    im_U2 = axes[0, 1].imshow(
        s_U_opt.cpu().numpy(), vmin=vmin_U, vmax=vmax_U, cmap="viridis"
    )
    axes[0, 1].set_title("$\\tilde{s}_U$")

    im_U3 = axes[0, 2].imshow(
        s_U_opt_noisy.cpu().numpy(), vmin=vmin_U, vmax=vmax_U, cmap="viridis"
    )
    axes[0, 2].set_title("$\\tilde{s}_U + c_U$")

    im_U4 = axes[0, 3].imshow(
        residual_U.cpu().numpy(),
        vmin=residual_U.min().item(),
        vmax=residual_U.max().item(),
        cmap="viridis",
    )
    axes[0, 3].set_title("$d_U - \\tilde{s}_U$")

    # plotting d_Q, s_Q_opt, s_Q_opt_noisy, and residual_Q
    im_Q1 = axes[1, 0].imshow(
        d_Q.cpu().numpy(), vmin=vmin_Q, vmax=vmax_Q, cmap="viridis"
    )
    axes[1, 0].set_title("$d_Q$")

    im_Q2 = axes[1, 1].imshow(
        s_Q_opt.cpu().numpy(), vmin=vmin_Q, vmax=vmax_Q, cmap="viridis"
    )
    axes[1, 1].set_title("$\\tilde{s}_Q$")

    im_Q3 = axes[1, 2].imshow(
        s_Q_opt_noisy.cpu().numpy(), vmin=vmin_Q, vmax=vmax_Q, cmap="viridis"
    )
    axes[1, 2].set_title("$\\tilde{s}_Q + c_Q$")

    im_Q4 = axes[1, 3].imshow(
        residual_Q.cpu().numpy(),
        vmin=residual_Q.min().item(),
        vmax=residual_Q.max().item(),
        cmap="viridis",
    )
    axes[1, 3].set_title("$d_Q - \\tilde{s}_Q$")

    # layout adjustments
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_name, filename))
    plt.close()


# Full phase (5-channel phase)
def baseline_comp_sep(
    DATA_PATH,
    N=10,
    max_iter=25,
    batch_size=None,
    resampling_period=None,
    epoch_period=None,
    STL_DataClass=STL_2D_FFT_Torch,
):

    # Set default values for optional parameters
    if batch_size is None:
        batch_size = N  # Default to full batch if not specified
    if resampling_period is None:
        resampling_period = max_iter  # Default to no resampling if not specified
    if epoch_period is None:
        epoch_period = max_iter  # Default to single epoch if not specified

    # Sanity checks
    assert batch_size <= N, "Stochastic batch size must be less than or equal to N."
    assert (
        resampling_period <= max_iter
    ), "Resampling period must be less than or equal to max iterations."

    # Load target map
    s_U, s_Q, d_I = np.load(DATA_TEST_PATH + "/" + "Turb_6.npy")[:3, :, :]
    s_U = torch.from_numpy(s_U).float()  # Target map of the U component
    s_Q = torch.from_numpy(s_Q).float()  # Target map of the Q component
    d_I = torch.from_numpy(d_I).float()  # Ancillary data

    # Build ensemble of contamination maps
    H, W = s_U.shape
    c_U = torch.randn(N + 2, H, W) * (0.5**0.5)
    c_Q = torch.randn(N + 2, H, W) * (0.5**0.5)

    # Observed maps
    d_U = s_U + c_U[0]  # Observed map of the U component
    d_Q = s_Q + c_Q[0]  # Observed map of the Q component

    target_maps = torch.stack(
        [
            d_U.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
            c_U[2:],  # [N, H, W]
            d_Q.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
            c_Q[2:],  # [N, H, W]
            d_I.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
        ],
        dim=1,
    )  # [N, 5, H, W]

    compute_PS = not target_maps.isnan().any()

    running_maps = torch.stack([d_U.clone(), d_Q.clone()], dim=0)  # [2, H, W]
    running_maps.requires_grad_(True)

    compute_cross_matrix = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, False, False, False],
            [False, False, True, False, True],
            [False, False, False, True, False],
            [
                False,
                False,
                False,
                False,
                True,
            ],  # Auto-stats for ancillary data computed for normalization concerns.
        ],
        dtype=torch.bool,
    )  # [5, 5]

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
        ).to_flatten(keep_batch_dim=True)

    # Split total iterations into epoch iterations
    n_epochs = max_iter // epoch_period
    n_iter_per_epoch = [epoch_period] * n_epochs + (
        [max_iter % epoch_period] if max_iter % epoch_period > 0 else []
    )

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
                running_maps[0].unsqueeze(0) + c_U[2:][indices],  # [batch_size, H, W]
                (d_U - running_maps[0])
                .unsqueeze(0)
                .repeat(batch_size, 1, 1),  # [batch_size, H, W]
                running_maps[1].unsqueeze(0) + c_Q[2:][indices],  # [batch_size, H, W]
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
        )  # [batch_size, n_stats]

        loss = (stats_rm - stats_target_maps[indices]).abs().square().sum(dim=1).mean()

        loss.backward()
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
        )  # [N, n_stats]

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
                            running_maps[0].unsqueeze(0) + c_U[2:],  # [N, H, W]
                            (d_U - running_maps[0])
                            .unsqueeze(0)
                            .repeat(N, 1, 1),  # [N, H, W]
                            running_maps[1].unsqueeze(0) + c_Q[2:],  # [N, H, W]
                            (d_Q - running_maps[1])
                            .unsqueeze(0)
                            .repeat(N, 1, 1),  # [N, H, W]
                            d_I.unsqueeze(0).repeat(N, 1, 1),  # [N, H, W]
                        ],
                        dim=1,
                    ),
                    pbc=True,
                )
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

    # Compute the optimal noisy maps and the residuals for validation
    s_U_opt = running_maps[0]
    s_Q_opt = running_maps[1]
    s_U_opt_noisy = s_U_opt + c_U[2]
    s_Q_opt_noisy = s_Q_opt + c_Q[2]
    residual_U = d_U - s_U_opt
    residual_Q = d_Q - s_Q_opt

    return (
        d_U,
        d_Q,
        s_U_opt,
        s_Q_opt,
        s_U_opt_noisy,
        s_Q_opt_noisy,
        residual_U,
        residual_Q,
    )


if __name__ == "__main__":

    # Find path to test dataset
    PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
    sys.path.append(PARENT_DIR)
    print("Parent directory added to sys.path:", ".../" + os.path.basename(PARENT_DIR))

    DATA_TEST_PATH = PARENT_DIR + "/data" + "/test"
    print(
        "Dataset directory used:",
        ".../"
        + os.path.basename(PARENT_DIR)
        + DATA_TEST_PATH.split(os.path.basename(PARENT_DIR))[-1],
    )

    # Full phase (5-channel phase) component separation
    d_U, d_Q, s_U_opt, s_Q_opt, s_U_opt_noisy, s_Q_opt_noisy, residual_U, residual_Q = (
        baseline_comp_sep(
            DATA_TEST_PATH,
            N=100,
            max_iter=25,
            batch_size=10,
            resampling_period=5,
            epoch_period=10,
            STL_DataClass=STL_2D_FFT_Torch,
        )
    )
    output_dir_name = "comp_sep_5channels_results"
    os.makedirs(output_dir_name, exist_ok=True)
    save_comp_sep(
        d_U,
        d_Q,
        s_U_opt,
        s_Q_opt,
        s_U_opt_noisy,
        s_Q_opt_noisy,
        residual_U,
        residual_Q,
        output_dir_name,
        filename="comp_sep_5channels_results.png",
    )
