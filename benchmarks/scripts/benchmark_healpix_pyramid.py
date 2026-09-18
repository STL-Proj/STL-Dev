"""Run each mode in a fresh process; psutil is needed for sampled CPU RSS.

python -m benchmarks.scripts.benchmark_healpix_pyramid --mode legacy --nside 64
python -m benchmarks.scripts.benchmark_healpix_pyramid --mode pyramid --nside 64
python -m benchmarks.scripts.benchmark_healpix_pyramid --mode fused --nside 64

Geometry and target setup are outside the timed interval. CPU RSS is sampled
every 2 ms; CUDA allocator peaks are exact. 'pyramid' reports separate forward
and VJP timings (the latter recomputes fields). 'fused' is the synthesis path.
"""

import argparse
import gc
import json
import threading
import time

import psutil
import torch

from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import PyramidStatistics
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "pyramid", "fused"), required=True)
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(11)
    data = STL_Healpix_Kernel_Torch(
        torch.randn(12 * args.nside**2, dtype=torch.float64, device=args.device)
    )
    op = data.get_ST_op(
        J=args.J,
        L=args.L,
        norm="vanilla",
        compute_PS=False,
        wavelet_op_kwargs=dict(kernel_size=5, ellipsoid="sphere"),
    )
    p = HealpixPyramid.from_map(data, op.wavelet_op)
    # Identical fixed zero target and cached geometry in every fresh process.
    # Do not warm up with a complete forward: CPU allocators retain that memory
    # and would hide the graph's incremental RSS in the baseline.
    for template in p.templates:
        op.wavelet_op._get_conv(
            template.dg, template.cell_ids, args.L, op.wavelet_op._wav_weights
        )
    plan = op.build_plan()
    target = PyramidStatistics(
        plan, torch.zeros(1, len(plan), dtype=torch.complex128, device=data.device)
    )
    cuda = data.array.is_cuda

    def sync():
        if cuda:
            torch.cuda.synchronize()

    gc.collect()
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    if args.mode == "legacy":
        leaves = [band.clone().requires_grad_() for band in p.levels]
        reference = HealpixPyramid(leaves, p.templates, op.wavelet_op)
        stats = op.apply(p.wrap(reference.reconstruct(), 0))
        values = torch.stack(
            [getattr(stats, c.family)[c.legacy_index] for c in target.plan], -1
        )
        loss = (values - target.values).abs().square().sum()
        sync()
        middle = time.perf_counter()
        gradients = torch.autograd.grad(loss, leaves)
    elif args.mode == "pyramid":
        stats = op.apply_pyramid(p, coefficient_batch_size=args.batch_size)
        residual = stats.values - target.values
        loss = residual.abs().square().sum()
        sync()
        middle = time.perf_counter()
        gradients = op.vjp_pyramid(
            p, 2 * residual, plan=target.plan, coefficient_batch_size=args.batch_size
        )
    else:
        middle = started
        loss, gradients = op.loss_and_grad_pyramid(
            p, target, coefficient_batch_size=args.batch_size
        )
    sync()
    ended = time.perf_counter()
    peak[0] = max(peak[0], process.memory_info().rss)
    stop.set()
    thread.join()
    print(
        json.dumps(
            dict(
                mode=args.mode,
                nside=args.nside,
                J=args.J,
                L=args.L,
                batch_size=args.batch_size,
                device=str(data.device),
                torch_version=torch.__version__,
                loss=float(loss.detach()),
                gradient_norm=float(sum(g.square().sum() for g in gradients).sqrt()),
                forward_seconds=middle - started if args.mode != "fused" else None,
                gradient_seconds=ended - middle if args.mode != "fused" else None,
                total_seconds=ended - started,
                cpu_rss_baseline=baseline,
                cpu_rss_peak_sampled=peak[0],
                cpu_rss_increment=peak[0] - baseline,
                cuda_peak_allocated=torch.cuda.max_memory_allocated() if cuda else None,
                cuda_peak_reserved=torch.cuda.max_memory_reserved() if cuda else None,
            )
        )
    )


if __name__ == "__main__":
    main()
