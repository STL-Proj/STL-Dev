import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch


def save_comp_sep(d_U, d_Q, s_U_opt_noisy, s_Q_opt_noisy, output_dir_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # common bounds for U
    vmin_U = d_U.min().item()
    vmax_U = d_U.max().item()

    # common bounds for Q
    vmin_Q = d_Q.min().item()
    vmax_Q = d_Q.max().item()

    # plotting d_U and s_U_opt_noisy
    im_U1 = axes[0, 0].imshow(
        d_U.cpu().numpy(), vmin=vmin_U, vmax=vmax_U, cmap="viridis"
    )
    axes[0, 0].set_title("$d_U$")

    im_U2 = axes[0, 1].imshow(
        s_U_opt_noisy.cpu().numpy(), vmin=vmin_U, vmax=vmax_U, cmap="viridis"
    )
    axes[0, 1].set_title("$\\tilde{s}_U + c_U$")

    # plotting d_Q and s_Q_opt_noisy
    im_Q1 = axes[1, 0].imshow(
        d_Q.cpu().numpy(), vmin=vmin_Q, vmax=vmax_Q, cmap="viridis"
    )
    axes[1, 0].set_title("$d_Q$")

    im_Q2 = axes[1, 1].imshow(
        s_Q_opt_noisy.cpu().numpy(), vmin=vmin_Q, vmax=vmax_Q, cmap="viridis"
    )
    axes[1, 1].set_title("$\\tilde{s}_Q + c_Q$")

    # layout adjustments
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_name, "comp_sep_result.png"))
    plt.close()


def baseline_comp_sep(DATA_TEST_PATH, N, STL_DataClass):

    # Load target map
    s_U, s_Q, d_I = np.load(DATA_TEST_PATH + "/" + "Turb_6.npy")[:3, :, :]
    s_U = torch.from_numpy(s_U).float()  # Target map of the U component
    s_Q = torch.from_numpy(s_Q).float()  # Target map of the Q component
    d_I = torch.from_numpy(d_I).float()  # Ancillary data

    # Build ensemble of contamination maps
    H, W = s_U.shape
    c_U = torch.randn(N + 1, H, W) * (0.5**0.5)
    c_Q = torch.randn(N + 1, H, W) * (0.5**0.5)

    # Observed maps
    d_U = s_U + c_U[0]  # Observed map of the U component
    d_Q = s_Q + c_Q[0]  # Observed map of the Q component

    target_maps = (
        torch.stack([d_U, d_Q, d_I], dim=0).unsqueeze(0).repeat(N, 1, 1, 1)
    )  # [N, 3, H, W]

    running_maps = torch.stack(
        [d_U.clone(), d_Q.clone(), d_I.clone()], dim=0
    )  # [3, H, W]
    running_maps.requires_grad_()

    mask = torch.ones_like(running_maps)
    mask[2, :, :] = 0  # freeze d_I

    running_maps.register_hook(lambda grad: grad * mask)

    optimizer = torch.optim.LBFGS(
        [running_maps],
        lr=1,
        max_iter=25,
        tolerance_grad=1e-17,
        tolerance_change=1e-17,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    # Utils function to compute stats and flatten them for the loss computation
    def stats_flatten(data, st_op, norm):
        return st_op.apply(data, norm=norm).to_flatten(keep_batch_dim=True)

    loss_history = []
    print_iter = 5

    with torch.no_grad():

        stl_target_maps = STL_DataClass(target_maps, pbc=True)
        st_op = stl_target_maps.get_ST_op()
        stats_target_maps = stats_flatten(
            stl_target_maps, st_op, norm="store_ref"
        )  # [N, n_stats]

        # Compute wavelet transform of contamination maps once for the entire optimization
        contamination_maps = torch.cat(
            [
                c_U[1:].unsqueeze(1),
                c_Q[1:].unsqueeze(1),
                torch.zeros_like(c_U[1:]).unsqueeze(1),
            ],
            dim=1,
        )  # [N, 3, H, W]

    def closure():
        optimizer.zero_grad()

        # Compute wavelet transform of running maps once at each iteration
        rm = torch.cat(
            [
                running_maps[0].unsqueeze(0),
                running_maps[1].unsqueeze(0),
                d_I.unsqueeze(0),
            ],
            dim=0,
        )  # [3, H, W]

        stl_running_plus_contamination_maps = STL_DataClass(
            array=rm.unsqueeze(0) + contamination_maps, pbc=True
        )

        stats_running_plus_contamination_maps = stats_flatten(
            stl_running_plus_contamination_maps,
            st_op,
            norm="load_ref",
            precomputed_W_data=False,
        )  # [N, n_stats]

        loss = (
            (stats_running_plus_contamination_maps - stats_target_maps)
            .abs()
            .square()
            .sum(dim=1)
            .mean()
        )

        loss.backward()
        loss_history.append(loss.item())

        if len(loss_history) % print_iter == 0:
            print(f"[LBFGS] iter {len(loss_history)}, loss = {loss.item():.6e}")

        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    end = time.perf_counter()

    print(f"{len(loss_history)} iterations of synthesis.")
    print(f"Execution time: {end - start:.3f} s")

    running_maps = running_maps.detach()

    s_U_opt_noisy = running_maps[0] + c_U[4]
    s_Q_opt_noisy = running_maps[1] + c_Q[4]

    return d_U, d_Q, s_U_opt_noisy, s_Q_opt_noisy


def precomp_wavelets_comp_sep(DATA_TEST_PATH, N, STL_DataClass):
    # Load target map
    s_U, s_Q, d_I = np.load(DATA_TEST_PATH + "/" + "Turb_6.npy")[:3, :, :]
    s_U = torch.from_numpy(s_U).float()  # Target map of the U component
    s_Q = torch.from_numpy(s_Q).float()  # Target map of the Q component
    d_I = torch.from_numpy(d_I).float()  # Ancillary data

    # Build ensemble of contamination maps
    H, W = s_U.shape
    c_U = torch.randn(N + 1, H, W) * (0.5**0.5)
    c_Q = torch.randn(N + 1, H, W) * (0.5**0.5)

    # Observed maps
    d_U = s_U + c_U[0]  # Observed map of the U component
    d_Q = s_Q + c_Q[0]  # Observed map of the Q component

    target_maps = (
        torch.stack([d_U, d_Q, d_I], dim=0).unsqueeze(0).repeat(N, 1, 1, 1)
    )  # [N, 3, H, W]

    running_maps = torch.stack(
        [d_U.clone(), d_Q.clone(), d_I.clone()], dim=0
    )  # [3, H, W]
    running_maps.requires_grad_()

    mask = torch.ones_like(running_maps)
    mask[2, :, :] = 0  # freeze d_I

    running_maps.register_hook(lambda grad: grad * mask)

    optimizer = torch.optim.LBFGS(
        [running_maps],
        lr=1,
        max_iter=25,
        tolerance_grad=1e-17,
        tolerance_change=1e-17,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_history = []
    print_iter = 5

    def stats_flatten(data, st_op, norm, precomputed_W_data=False):
        """
        data : StlData or Dict of StlData
            - If precomputed_W_data=False : StlData raw data
            - If precomputed_W_data=True  : Dictionary with integer keys/scale j in [0, J-1] and values StlData wavelet transform at resolution j_to_dg[j]
        """
        return st_op.apply(
            data, norm=norm, precomputed_W_data=precomputed_W_data
        ).to_flatten(keep_batch_dim=True)

    with torch.no_grad():

        stl_target_maps = STL_DataClass(target_maps, pbc=True)
        st_op = stl_target_maps.get_ST_op()
        stats_target_maps = stats_flatten(
            stl_target_maps, st_op, norm="store_ref", precomputed_W_data=False
        )  # [N, n_stats]

        # Compute wavelet transform of contamination maps once for the entire optimization
        contamination_maps = torch.cat(
            [
                c_U[1:].unsqueeze(1),
                c_Q[1:].unsqueeze(1),
                torch.zeros_like(c_U[1:]).unsqueeze(1),
            ],
            dim=1,
        )  # [N, 3, H, W]

        stl_contamination_maps = STL_DataClass(contamination_maps, pbc=True)

        # Precompute wavelet transforms of contamination maps once for the entire optimization
        W_contamination_maps = {
            j: st_op.wavelet_op.apply(
                st_op.wavelet_op.downsample(
                    data=stl_contamination_maps,
                    dg_out=st_op.wavelet_op.j_to_dg[j],
                    inplace=False,
                    replace_nan_value=st_op.replace_nan_value,
                ),
                j=j,
            )
            for j in range(st_op.J)
        }  # {j: [N, 3, L, Nj]}

    def closure():
        optimizer.zero_grad()

        # Compute wavelet transform of running maps once at each iteration
        rm = torch.cat(
            [
                running_maps[0].unsqueeze(0),
                running_maps[1].unsqueeze(0),
                d_I.unsqueeze(0),
            ],
            dim=0,
        )  # [3, H, W]

        stl_running_maps = STL_DataClass(rm, pbc=True)

        W_running_maps = {
            j: st_op.wavelet_op.apply(
                st_op.wavelet_op.downsample(
                    data=stl_running_maps,
                    dg_out=st_op.wavelet_op.j_to_dg[j],
                    inplace=False,
                    replace_nan_value=st_op.replace_nan_value,
                ),
                j=j,
            )
            for j in range(st_op.J)
        }  # {j: [3, L, Nj]}

        # Broadcast wavelet transform of running maps to the ensemble of contamination maps
        W_running_maps = {
            j: v.__class__(
                v.array.unsqueeze(0).repeat(N, 1, 1, 1, 1), pbc=v.pbc, N0=v.N0, dg=v.dg
            )
            for j, v in W_running_maps.items()
        }  # {j: [N, 3, L, Nj]}

        W_running_plus_contamination_maps = {
            j: W_running_maps[j].__class__(
                W_running_maps[j].array + W_contamination_maps[j].array,
                pbc=W_running_maps[j].pbc,
                N0=W_running_maps[j].N0,
                dg=W_running_maps[j].dg,
            )
            for j in range(st_op.J)
        }  # {j: [N, 3, L, Nj]}

        stl_running_plus_contamination_maps = STL_DataClass(
            array=stl_running_maps.array.unsqueeze(0) + stl_contamination_maps.array,
            pbc=True,
        )

        stats_running_plus_contamination_maps = stats_flatten(
            (stl_running_plus_contamination_maps, W_running_plus_contamination_maps),
            st_op,
            norm="load_ref",
            precomputed_W_data=True,
        )  # [N, n_stats]

        loss = (
            (stats_running_plus_contamination_maps - stats_target_maps)
            .abs()
            .square()
            .sum(dim=1)
            .mean()
        )

        loss.backward()
        loss_history.append(loss.item())

        if len(loss_history) % print_iter == 0:
            print(f"[LBFGS] iter {len(loss_history)}, loss = {loss.item():.6e}")

        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    end = time.perf_counter()

    print(f"{len(loss_history)} iterations of synthesis.")
    print(f"Execution time: {end - start:.3f} s")

    running_maps = running_maps.detach()

    s_U_opt_noisy = running_maps[0] + c_U[4]
    s_Q_opt_noisy = running_maps[1] + c_Q[4]

    return d_U, d_Q, s_U_opt_noisy, s_Q_opt_noisy


def biais_method_comp_sep(DATA_TEST_PATH, N, STL_DataClass, sub_epochs_size=10):

    # Load target map
    s_U, s_Q, d_I = np.load(DATA_TEST_PATH + "/" + "Turb_6.npy")[:3, :, :]
    s_U = torch.from_numpy(s_U).float()  # Target map of the U component
    s_Q = torch.from_numpy(s_Q).float()  # Target map of the Q component
    d_I = torch.from_numpy(d_I).float()  # Ancillary data

    # Build ensemble of contamination maps
    H, W = s_U.shape
    c_U = torch.randn(N + 1, H, W) * (0.5**0.5)
    c_Q = torch.randn(N + 1, H, W) * (0.5**0.5)

    # Observed maps
    d_U = s_U + c_U[0]  # Observed map of the U component
    d_Q = s_Q + c_Q[0]  # Observed map of the Q component

    target_maps = torch.stack([d_U, d_Q, d_I], dim=0)  # [3, H, W]

    running_maps = torch.stack(
        [d_U.clone(), d_Q.clone(), d_I.clone()], dim=0
    )  # [3, H, W]
    running_maps.requires_grad_()

    mask = torch.ones_like(running_maps)
    mask[2, :, :] = 0  # freeze d_I

    running_maps.register_hook(lambda grad: grad * mask)

    # Setup Optimiseur
    optimizer = torch.optim.LBFGS(
        [running_maps],
        lr=1,
        history_size=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-17,
        tolerance_change=1e-17,
    )

    # Function to compute stats from maps and flatten them for the loss computation
    def stats_flatten(data, st_op, norm):
        return st_op.apply(data, norm=norm).to_flatten(keep_batch_dim=True)

    # Split total iterations into sub-epochs
    total_iters = 25
    num_sub_epochs = total_iters // sub_epochs_size
    num_iter_per_sub_epoch = [sub_epochs_size] * num_sub_epochs + (
        [total_iters % sub_epochs_size] if total_iters % sub_epochs_size > 0 else []
    )

    loss_history = []
    print_iter = 2

    def closure():
        optimizer.zero_grad()

        current_running_maps = torch.stack(
            [running_maps[0], running_maps[1], d_I], dim=0
        )  # [3, H, W]
        stl_running_maps = STL_DataClass(array=current_running_maps, pbc=True)
        stats_running_maps = stats_flatten(
            stl_running_maps, st_op, norm="load_ref"
        )  # [1, n_stats]

        # running_biais_term is computed once per sub-epoch
        loss = (
            (stats_target_maps - stats_running_maps - running_biais_term)
            .abs()
            .square()
            .sum(dim=1)
            .mean()
        )

        loss.backward()
        loss_history.append(loss.item())

        if len(loss_history) % print_iter == 0:
            print(f"[LBFGS] iter {len(loss_history)}, loss = {loss.item():.6e}")
        return loss

    start = time.perf_counter()
    # Pre-computations (Target stats and contamination maps)
    with torch.no_grad():
        stl_target_maps = STL_DataClass(target_maps, pbc=True)
        st_op = stl_target_maps.get_ST_op()
        stats_target_maps = stats_flatten(
            stl_target_maps, st_op, norm="store_ref"
        )  # [1, n_stats]

        contamination_maps = torch.cat(
            [
                c_U[1:].unsqueeze(1),
                c_Q[1:].unsqueeze(1),
                torch.zeros_like(c_U[1:]).unsqueeze(1),
            ],
            dim=1,
        )  # [N, 3, H, W]

    for sub_epoch_index, n_iters in enumerate(num_iter_per_sub_epoch):
        print(
            f"--- Sub-epoch {sub_epoch_index+1}/{len(num_iter_per_sub_epoch)} ({n_iters} iters) ---"
        )

        optimizer.param_groups[0]["max_iter"] = n_iters

        # compute running_biais_term once per sub-epoch
        with torch.no_grad():
            current_running_maps = torch.stack(
                [running_maps[0], running_maps[1], d_I], dim=0
            )  # [3, H, W]

            # Stats of running maps with contamination
            stl_running_noisy = STL_DataClass(
                array=current_running_maps.unsqueeze(0) + contamination_maps, pbc=True
            )  # [N, 3, H, W]
            stats_running_noisy = stats_flatten(
                stl_running_noisy, st_op, norm="load_ref"
            )

            # Stats of running maps without contamination
            stl_running = STL_DataClass(array=current_running_maps, pbc=True)
            stats_running = stats_flatten(stl_running, st_op, norm="load_ref")

            # Biais term (change slowly and is then computed once per sub-epoch)
            running_biais_term = (stats_running_noisy - stats_running).mean(
                dim=0, keepdim=True
            )  # [1, n_stats]

        optimizer.step(closure)

    end = time.perf_counter()
    print(f"Total iterations: {len(loss_history)}. Time: {end - start:.3f} s")

    running_maps = running_maps.detach()
    return d_U, d_Q, running_maps[0] + c_U[4], running_maps[1] + c_Q[4]


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

    STL_DataClass = STL_2D_FFT_Torch

    # Baseline component separation
    # for N in [10, 20, 50, 100]:
    #    _, _, _, _ = baseline_comp_sep(DATA_TEST_PATH, N=N, STL_DataClass=STL_DataClass)

    # Component separation with precomputed wavelet transforms of contamination maps optimization
    # for N in [10, 20, 50, 100]:
    #    _, _, _, _ = precomp_wavelets_comp_sep(DATA_TEST_PATH, N=N, STL_DataClass=STL_DataClass)

    # output_dir_name = "precomp_wavelets_comp_sep_results"
    # os.makedirs(output_dir_name, exist_ok=True)
    # save_comp_sep(d_U, d_Q, s_tilde_U, s_tilde_Q, output_dir_name)

    # Component separation with biais method optimization
    d_U, d_Q, s_U_opt_noisy, s_Q_opt_noisy = biais_method_comp_sep(
        DATA_TEST_PATH, N=80, STL_DataClass=STL_DataClass, sub_epochs_size=10
    )
    output_dir_name = "biais_method_comp_sep_results"
    os.makedirs(output_dir_name, exist_ok=True)
    save_comp_sep(d_U, d_Q, s_U_opt_noisy, s_Q_opt_noisy, output_dir_name)
