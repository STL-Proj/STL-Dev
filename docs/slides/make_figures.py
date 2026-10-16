"""Figures for the S3 patch-wise synthesis slides (sober, two colours)."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
INK, ACCENT, LIGHT, GREY = "#36454F", "#028090", "#D9EEF0", "#B8C2CC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": INK,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def chain():
    """The S3 chain: fine wavelet -> modulus -> decimation -> coarse wavelet -> covariance."""
    fig, ax = plt.subplots(figsize=(12.5, 3.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 3.4); ax.axis("off")
    w = 2.0
    boxes = [
        (0.2, "map x\n(nside 2048)", INK),
        (2.5, "ψ_j1 ⋆ x\nfine wavelet\ncomplex, L orient.", ACCENT),
        (4.8, "| · |\nmodulus", INK),
        (7.1, "D^(n−d)\nsmooth decimation\n÷2 per level", ACCENT),
        (9.4, "ψ_j3 ⋆ ( · )\ncoarse\nwavelet", ACCENT),
    ]
    for x, label, color in boxes:
        ax.add_patch(Rectangle((x, 1.4), w, 1.3, facecolor=LIGHT if color == ACCENT else "white",
                               edgecolor=color, linewidth=1.5))
        ax.text(x + w / 2, 2.05, label, ha="center", va="center", fontsize=10.5)
    for x in (2.2, 4.5, 6.8, 9.1):
        ax.add_patch(FancyArrowPatch((x, 2.05), (x + 0.3, 2.05), arrowstyle="-|>", mutation_scale=14, color=INK))
    ax.add_patch(FancyArrowPatch((11.4, 2.05), (11.7, 2.05), arrowstyle="-|>", mutation_scale=14, color=INK))
    ax.text(12.1, 2.05, "S3 =\nmean of\nthe product", ha="center", va="center", fontsize=10.5)
    # the direct branch
    ax.add_patch(FancyArrowPatch((1.2, 1.4), (10.4, 0.55), arrowstyle="-|>", mutation_scale=14, color=GREY,
                                 connectionstyle="arc3,rad=0.22"))
    ax.text(5.8, 0.4, "ψ_j3 ⋆ x   (direct branch, at the coarse resolution)", ha="center", fontsize=10, color=INK)
    ax.text(6.3, 3.15, "fine resolution  ──────────────────────▶  coarse resolution (4^(n−d) times fewer pixels)",
            ha="center", fontsize=10, color=ACCENT)
    fig.savefig(os.path.join(OUT, "s3_chain.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def pyramid():
    """Fine blocks with halos feeding one coarse block through the linear chain."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    # fine level: a row of blocks
    y0 = 0.6
    for k in range(6):
        x = 0.3 + k * 1.1
        ax.add_patch(Rectangle((x - 0.13, y0 - 0.13), 1.16, 1.16, facecolor="none", edgecolor=GREY, linestyle="--", linewidth=1.2))
        ax.add_patch(Rectangle((x, y0), 0.9, 0.9, facecolor=LIGHT if k == 2 else "white", edgecolor=ACCENT if k == 2 else INK, linewidth=1.5))
        ax.text(x + 0.45, y0 + 0.45, f"F{k+1}", ha="center", va="center", fontsize=11)
    ax.text(0.2, 0.12, "fine level d: blocks F_k of constant size, each with its halo (wavelet stencil + decimation cone)", fontsize=10)
    # coarse level: one block
    ax.add_patch(Rectangle((2.2, 3.5), 2.6, 0.9, facecolor="white", edgecolor=INK, linewidth=1.5))
    ax.add_patch(Rectangle((1.95, 3.25), 3.1, 1.4, facecolor="none", edgecolor=GREY, linestyle="--", linewidth=1.2))
    ax.text(3.5, 3.95, "coarse level n\nm[d][n] on the block + halo", ha="center", va="center", fontsize=10.5)
    # arrows from fine blocks to coarse block
    for k in range(6):
        x = 0.75 + k * 1.1
        ax.add_patch(FancyArrowPatch((x, y0 + 1.05), (2.5 + k * 0.4, 3.45), arrowstyle="-|>", mutation_scale=12,
                                     color=ACCENT if k == 2 else GREY, linewidth=1.3 if k == 2 else 0.9))
    ax.text(8.3, 3.0, "m = Σ_k D^(n−d)( |ψ_j1 ⋆ x| · 1_{F_k} )", ha="center", va="center", fontsize=11, color=ACCENT)
    ax.text(8.3, 2.0, "each fine block pushes its\ncontribution upward;\nthe sum is exact because\nthe chain is linear\nafter the modulus", ha="center", va="center", fontsize=10.5)
    ax.text(5.0, 4.95, "memory: one fine block + its halo, whatever the scale distance n − d", ha="center", fontsize=10.5, color=ACCENT)
    fig.savefig(os.path.join(OUT, "pyramid.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def sweeps():
    """Ascending (statistics) and descending (gradient) sweeps over levels."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_xlim(0, 8); ax.set_ylim(0, 4.4); ax.axis("off")
    levels = ["d = 0   (nside 2048: 768 blocks)", "d = 1   (192 blocks)", "d = 2   (48 blocks)", "…", "d = n   (a single block: the whole level)"]
    for i, name in enumerate(levels):
        y = 0.4 + i * 0.85
        w = 5.4 - i * 0.3
        ax.add_patch(Rectangle((4.0 - w / 2, y), w, 0.55, facecolor="white" if i % 2 else LIGHT, edgecolor=INK, linewidth=1.2))
        ax.text(4.0, y + 0.275, name, ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((1.2, 0.5), (1.2, 4.1), arrowstyle="-|>", mutation_scale=16, color=ACCENT, linewidth=2))
    ax.text(1.0, 2.3, "upward\nsweep:\nstatistics,\ndecimated moduli\nm[d][n] stored", ha="right", va="center", fontsize=10, color=ACCENT)
    ax.add_patch(FancyArrowPatch((6.8, 4.1), (6.8, 0.5), arrowstyle="-|>", mutation_scale=16, color=INK, linewidth=2))
    ax.text(7.0, 2.3, "downward\nsweep:\ncotangents,\ntransposes,\nmodulus derivative,\nwavelet adjoint", ha="left", va="center", fontsize=10)
    fig.savefig(os.path.join(OUT, "sweeps.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def loss_curve():
    """Loss of the nside 256 WebSky run (values from the GPU run JSON)."""
    first = [0.025686992714027232, 0.025686953622987903, 0.025686601810590418, 0.025683084375320173,
             0.025647978917935042, 0.025303838732054183, 0.02257878680084631, 0.0164079932991177,
             0.010291449128949382, 0.003682216307506647, 0.0014399333670247795, 0.0007583372843771964,
             9.865536104181238e-05, 5.7622105477082585e-05, 2.5084206094543828e-05, 2.1947117398253382e-05,
             1.881741833252236e-05, 1.7761408691687455e-05, 1.6215235416271617e-05, 1.501133943938451e-05,
             1.36735280640856e-05, 1.2404084202334602e-05, 1.1021290846375777e-05, 9.28955768079933e-06,
             8.461693773904854e-06, 8.192893355938405e-06, 7.2388088628223625e-06, 8.1037723173739e-06,
             6.456998239467743e-06, 5.547106189530301e-06, 4.383668160686425e-06, 3.742871538888288e-06,
             3.3442772109766523e-06, 3.046878619672679e-06, 2.4759157705515037e-06, 3.2311553130222076e-06,
             2.3494011713698934e-06, 2.212171614287191e-06, 2.0633895290251705e-06, 2.0025477922187865e-06]
    later = {50: 1.5495309750512463e-06, 60: 9.893694809937698e-07, 70: 4.0573897025149785e-07,
             80: 2.0173464932567043e-07, 90: 1.4741474818084043e-07, 100: 1.3107571459155857e-07,
             120: 1.1365582908208043e-07, 140: 9.132678931951788e-08, 160: 7.130274448347676e-08,
             180: 5.651526643116224e-08, 200: 4.6552009632819564e-08, 230: 3.7705160320003784e-08,
             260: 3.419634048556939e-08, 300: 3.0606535673672605e-08, 350: 2.7762917168182807e-08,
             400: 2.5073435440557776e-08, 450: 2.3327201867093357e-08, 500: 2.2838549746800507e-08,
             550: 2.2396586168154676e-08, 625: 2.1512262385784758e-08}
    xs = list(range(1, len(first) + 1)) + sorted(later)
    ys = first + [later[k] for k in sorted(later)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogy(xs, ys, color=ACCENT, lw=2, marker=".", ms=5)
    ax.set_xlabel("loss evaluation (LBFGS, 500 iterations)")
    ax.set_ylabel("Σ |S(x) − S(target)|²")
    ax.set_title("WebSky kappa, nside 256, L = 4, 5,882 coefficients", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_256.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    chain(); pyramid(); sweeps(); loss_curve()
    print("figures in", OUT, os.listdir(OUT))
