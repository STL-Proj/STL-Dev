"""Report the largest local field per level for the patch engine.

python -m benchmarks.scripts.patch_field_sizes --nside 16 --J 3 --patch-pixels 32
"""

import argparse

import torch

from STL_main.PatchScattering import vjp_patched
from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import MemoryTracker
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--patch-pixels", type=int, default=32)
    parser.add_argument("--families", default="S2,S3,S4")
    args = parser.parse_args()
    torch.manual_seed(0)
    data = STL_Healpix_Kernel_Torch(
        torch.randn(12 * args.nside**2, dtype=torch.float64)
    )
    op = data.get_ST_op(
        J=args.J,
        L=args.L,
        norm="vanilla",
        compute_PS=False,
        wavelet_op_kwargs=dict(kernel_size=args.kernel_size, ellipsoid="sphere"),
    )
    p = HealpixPyramid.from_map(data, op.wavelet_op)
    plan = tuple(
        c for c in op.build_plan() if c.family in args.families.split(",")
    )
    tracker = MemoryTracker()
    g = torch.randn(1, len(plan), dtype=torch.complex128, device=data.device)
    vjp_patched(op, p, g, plan=plan, patch_pixels=args.patch_pixels, tracker=tracker)
    for dg in range(args.J):
        npix = p.levels[dg].shape[-1]
        sizes = [
            e["shape"][-1]
            for e in tracker.events
            if e["operation"] == "first_wavelet" and e["dg"] == dg
        ]
        if sizes:
            print(
                f"dg={dg}: pixels={npix} largest first-wavelet field={max(sizes)} "
                f"({100 * max(sizes) / npix:.1f}% of the level), "
                f"budget={args.patch_pixels}"
            )


if __name__ == "__main__":
    main()
