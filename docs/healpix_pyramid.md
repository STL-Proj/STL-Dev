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

## Patch-localised statistics and gradients

`STL_main.PatchScattering` computes the same coefficients and the same band
gradients as the engine above, but no wavelet field ever covers the whole
sphere. Select it with `patch_pixels`:

```python
solver = PyramidSynthesis(st, target, patch_pixels=4096, coefficient_batch_size=16)
stats = st.apply_patched(p, patch_pixels=4096)
bands = st.vjp_patched(p, grad_stats, patch_pixels=4096)
```

Every coefficient is a pixel mean, so its value is a sum of partial sums over
disjoint *cores* of the native-resolution grid, and the gradient is the sum
over cores of the adjoint chain applied to the cotangent restricted to that
core. For one core, the chain reads a bounded neighbourhood that is derived
exactly from the index structure of the full-sky operators (`PatchOps`):
the native wavelet stencil, then the decimation rows, then the first wavelet
stencil at each inner scale. Restricted operators reuse the full-sky gather
indices and weights, so no boundary renormalisation occurs and the patch
result equals the full-sky result to roundoff. Tests compare every family,
both S3/S4 definitions, partial unsorted grids and multi-channel batches.

The receptive field of a cross-scale term is inherent: one native pixel of
S3(j1, j3) reads about `4**(dg3 - dg1)` fine pixels times the stencil halo.
`patch_pixels` is therefore a budget at the finest scale of each coefficient
batch; the core at the native level is shrunk by `4**(n - dg_min)` so that
the fine-scale fields stay near that budget (a halo of a few native pixels
around the core adds `halo * 4**(n - dg_min)` fine pixels, which dominates
for very coarse native levels). Choose `patch_pixels` as an aligned power of
four so that cores are NESTED super-pixels.

The analysis maps (`4N/3` pixels, no field), the map cotangent accumulators
and one dense index table per level (5 bytes per pixel) remain global;
everything with an orientation axis is local. Batches that share a native
and a finest level are processed together with the patch loop outside the
batch loop, so the index structures of a patch are built once per level
while fields are recomputed per batch and per patch. The fused loss performs
two patch sweeps per level (statistics, then cotangents). Patch
contributions are additive: `vjp_patched(..., patches=...,
return_maps_gradient=True)` and `apply_patched(..., patches=...)` evaluate a
subset of patches, so disjoint subsets can run on different workers and be
summed before `HealpixPyramid.analysis_vjp`. The driver shipped here is
sequential. `PatchGeometry(op, p, patch_pixels, cache=True)` keeps the
per-patch contexts across iterations when memory allows; without it the
context build costs about a quarter of a VJP at `nside=64`.

`python -m benchmarks.scripts.patch_field_sizes` reports the largest local
field per level. With `kernel_size=3`, `nside=64`, `J=5`, `patch_pixels=1024`
and S3 only, the largest fine-scale field is 18884 of 49152 pixels: the halo
of the coarsest native level (about three native pixels, i.e. 48 fine pixels
on each side) dominates the 1024-pixel core. The saving grows with `nside`
because the halo in fine pixels scales with `2**(n - dg_min)` while the sky
scales with `nside**2`; at `nside=2048`, `J=8` the same halo is about 1.6% of
the sky. The benchmark accepts `--mode patched --patch-pixels N`.

### Measured status (CPU, float64, `J=5`, `L=2`, `kernel_size=5`, one map)

| nside | mode | patch_pixels | time | RSS baseline | RSS increment |
|---|---|---|---|---|---|
| 128 | fused | - | 9.0 s | 1.27 GB | 57 MB |
| 128 | patched | 4096 | 100 s | 1.28 GB | 234 MB |
| 128 | patched | 16384 | 53 s | 1.28 GB | 274 MB |

Loss and gradient norm agree to all printed digits. Two conclusions:

- The patched engine is exact and bounds the wavelet fields, but at these
  resolutions it is slower (halo overlap and per-block index translation)
  and its process increment is dominated by index temporaries, not fields.
  `pyramid_pixel_batch_size` bounds those temporaries.
- The baseline is the full-sky healpix-analyse geometry (`_pos_safe` and
  `_w_norm`: `4 * G * P * 16` bytes, about 3.2 kB per pixel for `L=2`,
  `kernel_size=5`), about one hundred times the size of a complex wavelet
  field. Neither engine avoids it; at `nside=2048` it would be 160 GB. The
  next step for large maps is therefore patch-local geometry: build partial
  `HealPixConv`/`HealPixDown` operators on each patch domain (ring halos from
  `healpix_geo.nested.kth_neighbourhood`, as in `_healpix_patch_domains`)
  instead of restricting the full-sky operators, which the present engine
  requires in memory. The patch domains, the chain structure, the partial
  sums and the additivity of patch contributions carry over unchanged.

## One S3 scale pair at a time, fine scale streamed by sub-patches

`STL_main.S3Multiscale.s3_scale_pair(op, p, j1, j3, gradient=..., fine_patch_pixels=N)`
is the multiscale form of the cross term that reads every pixel,
S3(j1=0, j3=J-1): the coarse level is handled globally (or by coarse cores
with `coarse_patch_pixels`), and the fine modulus `|psi_j1 * x|` is computed
by fine sub-patches and pushed through the decimation chain restricted to
each sub-patch support (`PatchOps.DownPiece`), which is exact because the
chain is linear:

    D^(n-d1) |psi_j1 * x| = sum_k D^(n-d1) ( |psi_j1 * x| . 1_{F_k} )

The reverse pass redistributes the coarse cotangent through the transposed
pieces sub-patch by sub-patch. No field with an orientation axis therefore
exceeds one fine sub-patch plus its stencil halo, whatever `j3 - j1`. The
function returns the statistics of the pair and per-level map cotangents;
several pairs accumulate into the same `maps_gradient` before
`pyramid.analysis_vjp`. Tests compare every pair of a `J=3` operator, both
S3 definitions, partial unsorted grids and coarse cores against the full-sky
engine for sub-patches of 5, 64 and the whole region.

Measured with `benchmarks.scripts.benchmark_s3_multiscale` (CPU, `nside=128`,
`J=5`, pair (0, 4), one map, `L=2`, `kernel_size=5`; sparse structures
prebuilt):

| mode | fine_patch | largest fine field (pixels) | time | RSS increment |
|---|---|---|---|---|
| full-sky explicit | - | 196608 | 0.2 s | 8 MB |
| multiscale | 65536 | 65536 | 0.9 s | 238 MB |
| multiscale | 16384 | 16384 | 1.2 s | 107 MB |
| multiscale | 4096 | 4096 | 1.2 s | 12 MB |
| multiscale | 1024 | 1024 | 1.6 s | 10 MB |

The gradient norm is identical in every row. The fine field follows the
sub-patch size exactly and the process increment follows it once the
per-block index temporaries (bounded by `pyramid_pixel_batch_size`) stop
dominating. An earlier version showed a plateau of several hundred MB in
every mode: the decimation matrices were rebuilt as COO tensors and
transposed at every call of the analysis chain, which at `nside=256` on a
GPU cost about 440 MB of CUDA peak regardless of the patch. They are now
cached CSR tensors (`PatchOps.down_forward_full/down_adjoint_full`) and
whole-level row scans are blocked. The full-sky geometry of healpix-analyse
is still shared by every mode: at `nside=256`, `J=6` it is about 4 GB on
the GPU and exceeded 15 GB of host RAM when built on CPU. Patch-local
geometry is therefore the prerequisite for large maps.

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
