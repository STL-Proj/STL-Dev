"""Full-sky legacy synthesis versus explicit patch-wise synthesis, same objective.

Modes (run each in a fresh process):

* ``legacy``   - the map is the variable, ScatCov through ``ST_Operator.apply``
                 with autograd, ``torch.optim.LBFGS`` (strong Wolfe).
* ``pyramid``  - Laplacian bands are the variables, explicit full-sky VJP
                 (``loss_and_grad_pyramid``), ``PyramidLBFGS``.
* ``patched``  - same, but the loss and the VJP are accumulated patch by
                 patch (``loss_and_grad_patched``, ``--patch-pixels``).

All modes minimise ``sum |S(x) - S(target)|**2`` over the same coefficient
plan (vanilla normalisation, no power spectrum). The geometry of every
level is built before the measurement, so the memory increments compare
the working sets; the geometry baseline itself is shared and reported.

python -m benchmarks.scripts.benchmark_synthesis --mode legacy --nside 64 --niter 50
python -m benchmarks.scripts.benchmark_synthesis --mode patched --nside 64 --niter 50 --patch-pixels 4096
"""

import argparse
import gc
import json
import threading
import time

import psutil
import torch

from STL_main.LossOptim import PyramidLBFGS, PyramidSynthesis
from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import PyramidStatistics
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("legacy", "pyramid", "patched", "local"), required=True
    )
    parser.add_argument(
        "--block-pixels", type=int, default=16384, help="local mode: block size per level"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="local mode: directory for the exact (operator dtype) partial geometry cache",
    )
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--J", type=int, default=None, help="default log2(nside) - 1")
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--niter", type=int, default=50)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--patch-pixels", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16, help="coefficient batch")
    parser.add_argument("--fewer", action="store_true", help="has_fewer_convolutions")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    J = args.J if args.J is not None else max(1, args.nside.bit_length() - 2)
    generator = torch.Generator().manual_seed(args.seed)
    npix = 12 * args.nside**2
    target_map = torch.randn(npix, dtype=torch.float64, generator=generator)
    initial_map = torch.randn(npix, dtype=torch.float64, generator=generator)
    kwargs = {} if args.device is None else dict(device=args.device)
    target = STL_Healpix_Kernel_Torch(target_map, **kwargs)
    device = target.device
    initial_map = initial_map.to(device)
    op = target.get_ST_op(
        J=J,
        L=args.L,
        norm="vanilla",
        compute_PS=False,
        has_fewer_convolutions=args.fewer,
        wavelet_op_kwargs=dict(kernel_size=args.kernel_size, ellipsoid="sphere"),
    )
    w = op.wavelet_op
    plan = op.build_plan()
    if args.mode == "local":
        from STL_main.LocalGeometry import LocalGeometry

        geometry = LocalGeometry(w, block_pixels=args.block_pixels, cache_dir=args.cache_dir)
        target_stats = op.apply_local(geometry.pyramid_from_map(target), geometry=geometry)
    else:
        geometry = None
        target_pyramid = HealpixPyramid.from_map(target, w)
        # Geometry (all levels) and sparse structures before the baseline.
        from STL_main.PatchOps import _down_csr, _down_csr_t

        for template in target_pyramid.templates:
            w._get_conv(template.dg, template.cell_ids, args.L, w._wav_weights)
            if template.dg < target_pyramid.dg_max:
                down = w._get_down(template.dg, template.cell_ids)
                _down_csr(down)
                _down_csr_t(down)
        target_stats = op.apply_pyramid(target_pyramid, coefficient_batch_size=args.batch_size)
        del target_pyramid
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    gc.collect()
    sync()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_baseline = torch.cuda.memory_allocated()
    process = psutil.Process()
    rss_baseline = process.memory_info().rss
    peak, stop = [rss_baseline], threading.Event()

    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    history = []
    started = time.perf_counter()

    if args.mode == "legacy":
        u = initial_map.reshape(1, 1, npix).clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [u],
            lr=1.0,
            max_iter=args.niter,
            history_size=args.history_size,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
        )

        def closure():
            optimizer.zero_grad()
            stats = op.apply(target.new_like(u))
            values = torch.stack(
                [getattr(stats, c.family)[c.legacy_index] for c in plan], -1
            )
            loss = (values - target_stats.values).abs().square().sum()
            loss.backward()
            history.append(float(loss.detach()))
            return loss

        optimizer.step(closure)
        result = u.detach()
    else:
        solver = PyramidSynthesis(
            op,
            target_stats,
            optimizer=PyramidLBFGS(lr=1.0, history_size=args.history_size),
            coefficient_batch_size=args.batch_size,
            patch_pixels=args.patch_pixels if args.mode == "patched" else None,
            geometry=geometry,
        )
        initial = target.new_like(initial_map)
        if geometry is not None:
            pyramid = geometry.pyramid_from_map(initial)
        else:
            pyramid = HealpixPyramid.from_map(initial, w)
        history = solver.run(pyramid, niter=args.niter)
        result = solver.reconstruct()

    sync()
    elapsed = time.perf_counter() - started
    peak[0] = max(peak[0], process.memory_info().rss)
    stop.set()
    thread.join()
    with torch.no_grad():
        final_data = target.new_like(result.reshape(1, 1, npix))
        if geometry is not None:
            final = op.apply_local(geometry.pyramid_from_map(final_data), geometry=geometry)
        else:
            final = op.apply_pyramid(
                HealpixPyramid.from_map(final_data, w), coefficient_batch_size=args.batch_size
            )
        final_loss = float((final.values - target_stats.values).abs().square().sum())
    print(
        json.dumps(
            dict(
                mode=args.mode,
                nside=args.nside,
                J=J,
                L=args.L,
                coefficients=len(plan),
                niter=args.niter,
                evaluations=len(history),
                patch_pixels=args.patch_pixels if args.mode == "patched" else None,
                block_pixels=args.block_pixels if args.mode == "local" else None,
                partial_operators_built=None if geometry is None else dict(geometry.built),
                device=str(device),
                seconds=round(elapsed, 2),
                seconds_per_evaluation=round(elapsed / max(1, len(history)), 3),
                loss_first=history[0] if history else None,
                loss_last=history[-1] if history else None,
                loss_final_map=final_loss,
                rss_baseline=rss_baseline,
                rss_peak_increment=peak[0] - rss_baseline,
                cuda_baseline_allocated=cuda_baseline if cuda else None,
                cuda_peak_increment=(
                    torch.cuda.max_memory_allocated() - cuda_baseline if cuda else None
                ),
                cuda_peak_reserved=torch.cuda.max_memory_reserved() if cuda else None,
            )
        )
    )


if __name__ == "__main__":
    main()
