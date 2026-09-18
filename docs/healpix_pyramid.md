# Explicit HEALPix scattering covariance synthesis

The opt-in pyramid engine computes the existing `HealpixKernel_torch` ScatCov
coefficients with explicit VJPs. `ST_Operator.apply()` remains the reference and
retains its existing behavior. Other data types do not enter the new engine.

```python
from STL_main.Pyramid import HealpixPyramid
from STL_main.LossOptim import PyramidSynthesis, PyramidOptimizer

# data is STL_Healpix_Kernel_Torch, with NESTED pixels and finite real values.
st = data.get_ST_op(
    J=3, L=4, norm="vanilla", compute_PS=False,
    wavelet_op_kwargs={"pyramid_pixel_batch_size": 4096},
)
target = st.apply_pyramid(HealpixPyramid.from_map(data, st.wavelet_op))
solver = PyramidSynthesis(
    st, target,
    optimizer=PyramidOptimizer(lr=[0.01, 0.01, 0.005]),
    coefficient_batch_size=16,
    strategy="recompute",
)
p = solver.initialize_from_noise(data, seed=42)
history = solver.run(p, niter=100)
result = solver.reconstruct()  # tensor [batch, channel, pixel]
```

See `examples/healpix_pyramid_synthesis.py` for an executable example. An existing
map can be used as initialization with `HealpixPyramid.from_map(initial, st.wavelet_op)`.
The same wavelet operator instance must be used for pyramid and statistics.

## Scientific conventions and supported cases

- S1 auto terms are mean modulus; cross terms are the uncentered covariance of
  `z / sqrt(abs(z) + 1e-8)`.
- S2 auto terms are mean squared modulus; cross terms are `mean(a * conj(b))`.
- S3 and S4 retain the channel ordering, orientation indices, triangular scale
  domain, and complex conjugation of the legacy implementation.
- `has_fewer_convolutions=True` is supported, including its asymmetric cross terms.
- `var` retains the legacy uncentered second moment, despite its name.
- Inputs can have batch and channel axes, including cross-channel statistics.
  Finite partial-sky grids, including unsorted cell ids, are supported.
- `j_to_dg[j]` determines resolution; repeated resolutions are supported.

This first engine requires finite real float32/float64 data, NESTED indexing,
`norm="vanilla"`, `compute_PS=False`, and no statistics compression. Masks/NaNs,
automatic reference/self normalization, standardization, power spectra and
isotropic/Fourier compression are not implemented on the explicit path. Use
the existing `apply()` for these options. Unsupported settings fail explicitly.
Fixed reference normalization is supported through loss weights, for example
`SquaredStatisticsLoss(weights=1 / reference_scale.square())`. Reference scales
must be positive; this is not differentiation through self normalization.

## Plans, statistics and gradients

`st.build_plan(channels=Nc, compute_cross_matrix=..., families=...)` returns
immutable `ScatCoefficient` records, each with `native_dg`, channel, scale and
orientation indices. Families default to mean, var, S1, S2, S3 and S4. The upper
triangle of the channel mask follows legacy semantics. Supply a subset using
`plan=` to compute only selected coefficients. Batches never mix resolutions.

`apply_pyramid()` returns `PyramidStatistics(plan, values)` with shape
`[batch, coefficient]`, rather than a large NaN-padded `ST_Statistics` cube.
`stats.family("S3")` selects a family. To migrate vanilla legacy targets:

```python
from STL_main.PyramidScattering import PyramidStatistics
plan = st.build_plan(channels=Nc)
target = PyramidStatistics.from_legacy(st.apply(data), plan)
```

For complex statistics the VJP convention is
`dL = Re sum(conj(grad_stats) * d_stats)`. Thus the gradient of the weighted
squared loss is `2 * weights * (stats - target)`, and includes both real and
imaginary components. `vjp_pyramid(p, grad_stats, plan=...)` returns a tensor
for every independent pyramid band. A `PyramidStatistics` cotangent carries
its own plan. `loss_and_grad_pyramid(p, target, weights=...)` fuses evaluation,
loss and explicit VJP per batch, and is used by the default synthesis solver.

`gradient_callback(dg, gradient)` can consume completed band gradients without
retaining the entire output gradient pyramid. All coefficient contributions
are finished before callbacks run. The function returns `None` in callback mode.
The current implementation still accumulates analysis gradients in memory.

## Correct pyramid differentiation

Analysis uses the existing anti-aliased `HealPixDown` matrix D, not a substitute
four-child average. U replicates NESTED parent values to their children. Bands
are initialized as `P[d] = X[d] - U X[d+1]`, with `X[d+1] = D X[d]`.
Reconstruction is exact: `R[d] = P[d] + U R[d+1]`.

**For this filtered D, D U is not identity.** Further, synthesis bands are
independent variables. After an update, `R[d]` cannot simply replace
`D**d R[0]` in scattering. Doing so would change the optimized statistics and
give gradients inconsistent with the final reconstructed map. The engine
therefore builds a consistent filtered map pyramid from `R[0]` once per call.
It retains O(N) maps (about 4N/3 pixels), without a differentiation graph.
It does not eliminate the fine reconstructed map or all fine-scale work.

Each coefficient VJP starts at its final native spatial field. Modulus branches
use the explicit second wavelet adjoint, D transpose, modulus VJP and first
wavelet adjoint. Map cotangents are accumulated by analysis level. A final
D-transpose sweep gives the fine-map cotangent, followed by U-transpose sums
for the independent reconstruction bands. U transpose is a sum of children;
it is not D transpose. The j1=0, j3=J-1 branch necessarily reaches fine pixels
through the derivative of the fine-scale modulus.

## Memory and performance

- No production `backward()` or `torch.autograd.grad`, and no global tape.
- `coefficient_batch_size` controls reusable fields and branch cotangents.
  Smaller values reduce memory at the cost of recomputation; 16 is a useful
  starting point for large maps, while 64 is the API default.
- First-wavelet phases live only inside the current batch. `strategy="store"`
  keeps them until the batch reverse pass; `"recompute"` releases and rebuilds
  them for modulus VJPs. No cache of map-dependent values survives a call.
- `pyramid_pixel_batch_size` (default 16384) bounds convolution interpolation
  scratch space, both forward and adjoint. Geometry and full output fields
  still occupy memory. This is not spatially distributed synthesis.
- `MemoryTracker` records operation, dg, shape, dtype and bytes for observed
  fields, not an allocator peak. It holds no tensor references.
- Adam/SGD state and learning rates are per band; Adam adds two band pyramids.
  Geometry caches, maps and accumulators remain on the selected device. No
  CPU offload, MPI or multi-GPU is implemented.

The explicit convolution transpose uses the installed healpix-analyse geometry
contract (`_pos_safe`, `_w_norm`, sort orders and kernel weights). The decimation
adjoint uses its sparse smooth matrix. Revalidate adjoints after changing that
dependency; private layout compatibility is checked explicitly.

The checked backend revision is healpix-analyse
`6f56e8ea63532e152d8e4e6f9d49d92a4709d541` (local checkout).

## Validation and benchmark

```sh
python -m pytest tests/test_healpix_pyramid.py -q
python -m benchmarks.scripts.benchmark_healpix_pyramid --mode legacy --nside 64
python -m benchmarks.scripts.benchmark_healpix_pyramid --mode pyramid --nside 64
python -m benchmarks.scripts.benchmark_healpix_pyramid --mode fused --nside 64
```

Run each benchmark in a fresh process. Install `psutil` to measure CPU RSS.
Use `--device cuda` to record allocated and reserved CUDA peaks. CPU RSS is
sampled every 2 ms, so very short-lived peaks may be missed. Geometry setup is
excluded from time and the incremental RSS, but included in total RSS. The
benchmark uses the same fixed zero target and independent pyramid variables
for all modes. `pyramid` reports separate forward/VJP times; `fused` reports
their combined time, as used during synthesis.

Tests compare every coefficient to legacy STL, validate real/complex adjoints,
compare manual gradients to legacy autograd and directional finite differences
in each band (including noncanonical bands), exercise partial-sky grids,
multi-channel and batched maps, both S3/S4 definitions, float32, batch-size
invariance, shared dg values, fused losses, callbacks and Adam. A dedicated
j1=0/j3=J-1 test checks the native gradient size and prohibits calls to autograd.
