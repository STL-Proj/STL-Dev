"""Memory of the S3(0, J-1) gradient versus the fine sub-patch size.

Run each size in a fresh process; psutil samples the process RSS.

python -m benchmarks.scripts.benchmark_s3_multiscale --nside 128 --J 5 --fine-patch 4096
python -m benchmarks.scripts.benchmark_s3_multiscale --nside 128 --J 5 --mode full
"""

import argparse
import gc
import json
import threading
import time

import psutil
import torch

from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import MemoryTracker
from STL_main.S3Multiscale import s3_pair_plan, s3_scale_pair
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "multiscale"), default="multiscale")
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--J", type=int, default=5)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--j1", type=int, default=0)
    parser.add_argument("--j3", type=int, default=None)
    parser.add_argument("--fine-patch", type=int, default=4096)
    parser.add_argument(
        "--geometry",
        choices=("full", "local"),
        default="full",
        help="'local': partial operators per sub-patch, no full-sky fine geometry",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None, help="local geometry: exact disk cache")
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="repeat the gradient; the last iteration shows the amortised cost",
    )
    args = parser.parse_args()
    if args.geometry == "local" and args.mode == "full":
        raise SystemExit("--geometry local requires --mode multiscale")
    torch.set_num_threads(args.threads)
    torch.manual_seed(3)
    j3 = args.J - 1 if args.j3 is None else args.j3
    data = STL_Healpix_Kernel_Torch(torch.randn(12 * args.nside**2, dtype=torch.float64))
    op = data.get_ST_op(
        J=args.J, L=args.L, norm="vanilla", compute_PS=False,
        wavelet_op_kwargs=dict(kernel_size=5, ellipsoid="sphere"),
    )
    from STL_main.PatchOps import _down_csr, _down_csr_t

    if args.geometry == "local":
        from STL_main.LocalGeometry import LocalGeometry

        geometry = LocalGeometry(
            op.wavelet_op, block_pixels=args.fine_patch, cache_dir=args.cache_dir
        )
        p = geometry.pyramid_from_map(data)
        # Only the coarse level keeps a full-sky operator; build it outside
        # the timed region like the full geometry of the other modes.
        coarse = p.templates[j3]
        op.wavelet_op._get_conv(coarse.dg, coarse.cell_ids, args.L, op.wavelet_op._wav_weights)
    else:
        geometry = None
        p = HealpixPyramid.from_map(data, op.wavelet_op)
        # Geometry and sparse index structures are data independent: build
        # them before the timed region so that the increment measures fields.
        for template in p.templates:
            op.wavelet_op._get_conv(
                template.dg, template.cell_ids, args.L, op.wavelet_op._wav_weights
            )
            if template.dg < args.J - 1:
                down = op.wavelet_op._get_down(template.dg, template.cell_ids)
                _down_csr(down)
                _down_csr_t(down)
    # The convolution block bounds the per-row index temporaries (about 6.4 kB
    # per row of a block, 200x a complex L=2 field value); keep it small and
    # independent of the sub-patch, which bounds the fields.
    parser_block = min(4096, args.fine_patch) if args.mode == "multiscale" else 4096
    op.wavelet_op.pyramid_pixel_batch_size = parser_block
    plan = s3_pair_plan(op, 1, args.j1, j3)
    # Operator construction consumes the global RNG (parameter init), so the
    # cotangent gets its own generator: identical in every mode.
    generator = torch.Generator(device=data.device).manual_seed(7)
    g = torch.randn(
        1, len(plan), dtype=torch.complex128, device=data.device, generator=generator
    )
    tracker = MemoryTracker()
    cuda = data.array.is_cuda
    gc.collect()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_baseline = torch.cuda.memory_allocated()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak, stop = [baseline], threading.Event()

    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    timings, cuda_iteration_peaks = [], []
    for _ in range(max(1, args.iterations)):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            iteration_allocated = torch.cuda.memory_allocated()
        iteration_start = time.perf_counter()
        if args.mode == "full":
            bands = op.vjp_pyramid(p, g, plan=plan, tracker=tracker)
        elif args.geometry == "local":
            from STL_main.LocalGeometry import s3_scale_pair_local

            _, maps_gradient = s3_scale_pair_local(
                op, p, args.j1, j3, gradient=g, geometry=geometry,
                fine_patch_pixels=args.fine_patch, tracker=tracker,
            )
            bands = geometry.analysis_vjp(p, maps_gradient)
        else:
            _, maps_gradient = s3_scale_pair(
                op, p, args.j1, j3, gradient=g, fine_patch_pixels=args.fine_patch, tracker=tracker
            )
            bands = p.analysis_vjp(maps_gradient)
        if cuda:
            torch.cuda.synchronize()
            cuda_iteration_peaks.append(
                dict(
                    allocated_at_start=iteration_allocated,
                    peak_increment=torch.cuda.max_memory_allocated() - iteration_allocated,
                )
            )
        timings.append(round(time.perf_counter() - iteration_start, 2))
    elapsed = time.perf_counter() - started
    peak[0] = max(peak[0], process.memory_info().rss)
    stop.set()
    thread.join()
    fine = [e for e in tracker.events if e["operation"] == "first_wavelet"]
    print(json.dumps(dict(
        mode=args.mode, geometry=args.geometry, nside=args.nside, J=args.J, pair=[args.j1, j3],
        partial_operators_built=None if geometry is None else dict(geometry.built),
        fine_patch=args.fine_patch if args.mode == "multiscale" else None,
        coefficients=len(plan), seconds=round(elapsed, 2), seconds_per_iteration=timings,
        gradient_norm=float(sum(b.square().sum() for b in bands).sqrt()),
        largest_fine_field_pixels=max(e["shape"][-1] for e in fine),
        largest_fine_field_bytes=max(e["bytes"] for e in fine),
        sky_pixels=12 * args.nside**2, device=str(data.device),
        rss_baseline=baseline, rss_increment=peak[0] - baseline,
        cuda_baseline_allocated=cuda_baseline if cuda else None,
        cuda_allocated_after=torch.cuda.memory_allocated() if cuda else None,
        cuda_iterations=cuda_iteration_peaks if cuda else None,
        cuda_peak_reserved=torch.cuda.max_memory_reserved() if cuda else None,
    )))


if __name__ == "__main__":
    main()
