# STL-Dev-MPI: pyramidal scattering-covariance gradients in Rust for HPC

Kick-off note for the new package `jmdelouis/STL-Dev-MPI`. It is written to
start a working session from scratch: read it first, then the STL-Dev files
it points to. Everything below was validated in STL-Dev (branch
`HealpixUpdate`, GitHub `STL-Proj/STL-Dev`) before being written here.

## 1. Goal

Synthesise HEALPix maps from scattering-covariance statistics
(`mean`, `var`, S1, S2, S3, S4) at `nside` 2048 and beyond, on distributed-memory
HPC clusters, with the memory of every rank bounded by its share of the sky
and not by the full sky. The Python/torch engine of STL-Dev already does this
on one GPU (nside 2048 on 24 GB); STL-Dev-MPI re-implements the same
block-wise algorithm in Rust with MPI so that many nodes share the sky and
the wall-clock time per evaluation drops with the number of ranks.

Non-goals for the first version: masks and NaNs, non-spherical ellipsoids
(WGS84), the power-spectrum term, the planar (2D) data types. Those stay in
STL-Dev.

## 2. What already exists in STL-Dev (reference implementation)

| What | Where | Status |
|---|---|---|
| Explicit (no autograd) full-sky engine, VJPs per family | `STL_main/PyramidScattering.py`, `STL_main/AdjointOps.py` | reference, tested against legacy autograd |
| Patch-local geometry: partial operators + ring halos, exact on block interiors | `STL_main/LocalGeometry.py` | tested to 1e-10 vs full sky |
| Complete block-wise engine (all families), level sweeps, stored decimated moduli | `STL_main/LocalScattering.py` | tested to 1e-10 vs full sky |
| Device-side HEALPix geometry in torch (centres, stencil rotation, bilinear interpolation, decimation matrix) | `STL_main/TorchGeometry.py` | matches healpix-analyse to 1e-12 |
| Synthesis driver (LBFGS, map variable, coarse-to-fine init, FITS/PNG outputs) | `STL_main/LossOptim.py`, `benchmarks/scripts/benchmark_synthesis.py` | used for the nside 256 to 2048 WebSky runs |
| Narrative documentation and measurements | `docs/healpix_pyramid.md` | read the sections "Patch-local geometry", "Complete ScatCov with patch-local geometry", "Device-side geometry" |
| Tests that define exactness | `tests/test_local_scattering.py`, `tests/test_local_geometry.py`, `tests/test_torch_geometry.py` | the Rust package must pass equivalent tests against STL-Dev outputs |

Useful facts learnt the hard way:

- healpix-analyse (the Python HEALPix library STL-Dev used) builds geometry at
  16 us/pixel on CPU and stores 3.2 kB/pixel: unusable at scale. The Rust
  package must compute geometry on the fly per block (as `TorchGeometry` does),
  never store it.
- The wavelet kernel is the same at every level, so the first wavelet field
  depends only on the resolution level, not on the scale index.
- With Laplacian bands as variables LBFGS stalls on structured targets; the
  map itself must be the variable (`variables="map"`).
- Coarse-to-fine initialisation (replicate a lower-nside synthesis on the
  children, add white noise with the RMS of the target's missing small scales)
  makes the high-resolution synthesis converge in tens of iterations.

## 3. The model to port (exact formulas)

Notation: `x` the map at level 0 (`nside`); `maps[d] = D^d x` the anti-aliased
decimations (level `d` has `nside / 2^d`); `psi` the complex directional
wavelet with `L` orientations, applied at every level with the same kernel
(`kernel_size` x `kernel_size` stencil in the tangent plane, bilinear
interpolation of the 4 nearest pixels per tap, gauge `cosmo`);
`z_d = psi * maps[d]` of shape `[C, L, N_d]` complex; `dg(j)` the level of
scale `j` (default `dg(j) = j`).

Decimation `D` (one level): for each coarse pixel, Gaussian weights of the
fine pixels whose centre is within `radius = 3 * fine_pixel_size` of the
coarse centre (`sigma = radius / 2`, haversine distance), l1-normalised. Its
exact Rust-independent definition is `TorchGeometry.smooth_decimation_csr`.

Stored decimated moduli for every pair of levels `d < n`:

    m[d][n] = D^(n-d) |z_d|                        real, [C, L, N_n]

Coefficients (all are pixel means at their native level `n`; the "a1" domain
is the block plus the stencil halo):

    mean(c)            = mean(maps[0][c])
    var(c)             = mean(maps[0][c]^2)
    S1(j, c1, c2, l)   = mean(|z_n[c1, l]|)                      if c1 == c2
                       = mean(a * conj(b)), a = z/sqrt(|z|+1e-8)  otherwise
    S2(j, c1, c2, l)   = mean(|z_n[c1, l]|^2)  /  mean(z_n[c1,l] conj(z_n[c2,l]))
    S3(j1, j3, c1, c2, l1, l3)
                       = mean( [psi * m[d1][n]][c1, l1, l3] * conj(z_n[c2, l3]) )
                       (has_fewer_convolutions: a = maps[n][c1], b = [psi * m[d1][n]][c2, l1, l3])
    S4(j1, j2, j3, ...) = mean( [psi * m[d1][n]][c1, l1, l3] * conj([psi * m[d2][n]][c2, l2, l3]) )
                       (has_fewer_convolutions: a = m[d1][n][c1, l1] on the block)

with `n = dg(j3)`, `d1 = dg(j1)`, `d2 = dg(j2)`, and `[psi * m][.., l1, l3]`
the second wavelet (orientation `l3`) of the modulus of orientation `l1`.
`PyramidScattering.build_plan` enumerates the coefficients and their native
level; `LocalScattering._Block.operands/partial/reverse_coefficient` is the
line-by-line reference of the forward and of the cotangents.

Loss: `sum_i w_i |S_i(x) - S_i(target)|^2` (weights real, non-negative;
cotangent `2 w (S - target)`; complex statistics contribute real and
imaginary parts).

Block decomposition (the reason the memory is bounded):

    m[d][n] = sum_k D^(n-d) ( |z_d| . 1_{F_k} )

over fine blocks `F_k` of level `d`; every piece is computed with the fine
wavelet on `F_k + halo` and pushed through the decimation chain restricted to
its support (`LocalGeometry.piece_forward`), then summed at level `n`. The
gradient goes back through the transposed pieces (`piece_adjoint`), the
modulus derivative (`AdjointOps.modulus_vjp`) and the wavelet adjoint
(`AdjointOps.convolution_adjoint`).

Halos (in pixels of the level, `LocalGeometry`): convolution
`(kernel_size - 1) // 2 + 2`; decimation `ceil(radius / pixel) + 1`, doubled
for pieces and transposes (every coarse row touched by the block must see its
whole cone). Fine-level dilations are computed at the parent level
(`k` coarse rings cover at least `2k` fine pixels).

Sweeps (`LocalScattering.LocalEngine`):

1. upward, levels `0..D`: for every block of level `n`: `z_n` on
   `block + halo` through one wavelet evaluation on `block + 2 halos`; second
   wavelets on the block from `|z_n|` (own level) and `m[d][n]` (finer
   levels, read from the stored maps); partial sums; then push `|z_n| . 1_F`
   to every coarser level's `m[n][n']`;
2. downward, levels `D..0`: recompute the block fields, apply the coefficient
   cotangents, accumulate the cotangents of the stored moduli `gm[d][n]`
   (complete once the coarser levels are done), reverse the block's own chain
   from `gm[n][n']`, modulus derivative, wavelet adjoint, accumulate the map
   cotangent of level `n`;
3. the map cotangents of all levels are chained back to level 0 through the
   transposed decimations (`LocalGeometry.analysis_vjp`, `decimate_adjoint`).

Memory: per block, fields of size `|block + halo|` with orientation axes;
globally `maps` (`4N/3`), their cotangents, and `m`/`gm` (about `4N/9`
values times `C * L`, real).

## 4. Target architecture (Rust + MPI)

Repository `jmdelouis/STL-Dev-MPI`, Rust 2021, workspace with:

- `stl-core`: HEALPix geometry (NESTED <-> (x, y, face), pixel centres, ring
  bilinear interpolation of the HEALPix C++ library, `ring2nest`, stencil
  rotation per pixel and gauge, decimation weights), wavelet convolution and
  its adjoint on a partial grid, decimation pieces, the per-block forward and
  reverse of `LocalScattering._Block`, coefficient plans. Pure Rust, no MPI,
  `f64`. Port `TorchGeometry.py` first: it is already a transcription of the
  reference algorithms and its tests give the expected numbers.
- `stl-mpi`: domain decomposition and communication (`rsmpi` crate), the
  level sweeps, global reductions, distributed LBFGS, I/O.
- `stl-cli`: binaries `stl-stats` (statistics of a map), `stl-synth`
  (synthesis from a target map or from saved statistics), `stl-check`
  (compare with STL-Dev dumps).
- `python/` (optional, later): PyO3 bindings so that STL-Dev's
  `PyramidStatistics` plans and targets can be exchanged without files.

Domain decomposition: contiguous NESTED id ranges at level 0, one range per
rank (compact patches on the sphere); every rank owns the parents of its
fine pixels at every coarser level, so the pyramid of a rank is aligned
(`maps[d]`, `m[d][n]`, `gm[d][n]` are all sliced the same way). Blocks of
constant size (`block_pixels`, e.g. 65536) inside a rank's range are the unit
of work; the coarsest levels, smaller than one block, are replicated on
every rank (cheap) or owned by rank 0.

Halos: a rank needs `maps[d]` and the stored moduli on a ring of pixels
around its range at each level (the conv halo twice plus the decimation
halo); neighbours are found with `kth_neighbourhood` (cdshealpix) at the
parent level once, and exchanged with point-to-point messages before each
sweep. Pieces pushed upward and transposed pieces pulled downward touch rows
owned by neighbours: accumulate into local ghost buffers and reduce them
with the owners after the level is done (ghost-cell pattern, additive).

Global operations: coefficient partial sums (`Allreduce` of a vector of
`len(plan)` complex values per level); dot products of the distributed
LBFGS (two-loop recursion with `Allreduce` of scalars, strong Wolfe line
search as in `torch.optim.LBFGS` so that trajectories can be compared with
STL-Dev); loss value.

Per-rank memory budget (nside 2048, 50 M pixels, `P` ranks, `L = 4`,
`history = 10`): maps and cotangents `~0.5 GB / P`, stored moduli and their
cotangents `~1.4 GB / P`, LBFGS history `~10 GB / P`, block transients
`~0.1 GB`, halos a few percent of the slice. 16 ranks bring every rank
under 1 GB.

HEALPix dependencies: `cdshealpix` (Rust crate behind `healpix_geo`) for
`nest2xyf`, neighbourhoods and pixel centres; or the pure-Rust port in
`stl-core` of the functions in `TorchGeometry.py`, which avoids any
dependency and is already validated. Prefer the port for geometry used in
hot loops; use `cdshealpix` for neighbourhood lists.

I/O: NESTED `.npy` (crate `npyz`) and HEALPix FITS binary tables (crate
`fitsio`) for maps; JSON for plans, statistics, targets and run reports with
the same field names as `benchmark_synthesis.py` (`loss_history`,
`memory_floor_bytes`, `timers_seconds`).

## 5. Validation strategy (non-negotiable)

Exactness against STL-Dev is the acceptance test at every step:

1. Geometry: for random blocks at `nside` 8, 32, 128, 1024, dump from
   STL-Dev (`TorchConv._pos_safe`, `_w_norm`, `_complete`, and the decimation
   CSR of `smooth_decimation_csr`) and compare: pixel indices equal, weights
   to 1e-12 on complete rows; convolution outputs on random inputs to 1e-12.
2. Statistics: `apply_local(op, pyramid, geometry)` dumped as JSON (plan +
   values) for a random map at `nside` 16 and 32, `J = 3, 4`, `L = 2, 4`, both
   S3/S4 definitions, 1 and 2 channels; Rust must match to 1e-11 absolute,
   1e-10 relative.
3. Gradients: `vjp_local(..., return_maps_gradient=True)` and the band
   gradients after `analysis_vjp`, same cases, 1e-10.
4. Multi-rank: 1, 2, 4, 8 ranks give identical statistics and gradients
   (bit-level differences only from summation order, below 1e-13).
5. Synthesis: nside 64 WebSky target, 20 LBFGS iterations, the loss
   trajectory must match STL-Dev's `--mode local --variables map` run to the
   printed digits for the first iterations (line search identical).

Write the STL-Dev dump script first (`tests/dump_reference.py` in STL-Dev or
in the new repo's `tools/`), so that every Rust test reads fixtures.

## 6. Milestones

1. `stl-core` geometry port with fixture tests (centres, ring2nest,
   interpolation, stencil rotation, binding, decimation weights).
2. Partial convolution forward/adjoint and decimation pieces on one block;
   fixture tests.
3. Single-process level sweeps: statistics then gradient for the complete
   plan; fixture tests at nside 16/32.
4. `stl-mpi`: decomposition, halo exchange, ghost accumulation; multi-rank
   equality tests (run under `mpirun -n 1,2,4`).
5. Distributed LBFGS with strong Wolfe; synthesis CLI; FITS/npy I/O;
   trajectory comparison with STL-Dev at nside 64.
6. Scaling study on the cluster: nside 1024 and 2048, 1 to 64 ranks, time per
   evaluation and memory per rank; then the nside 4096 WebSky maps
   (`kap`, `ksz`, already converted to NESTED `.npy` by
   `benchmarks/scripts/prepare_websky_kappa.py` in STL-Dev).

## 7. Conventions for the session

- GitHub: `gh repo create jmdelouis/STL-Dev-MPI --private --clone` (the `gh`
  CLI is authenticated in WSL as `jmdelouis`); work on `main`, feature
  branches for milestones, push when a milestone's tests pass.
- Tooling on this machine: Rust toolchain to install in WSL (`rustup`),
  `cargo test` for everything, `mpirun` from OpenMPI (`apt install libopenmpi-dev openmpi-bin`),
  `maturin` only when the PyO3 bindings start. STL-Dev's CPU venv for the
  fixture dumps is `~/stl-deps/venv-cpu` with the project synced in
  `~/stl-work` (see the STL-Dev session notes: run from WSL-native storage,
  never from `/mnt/c`).
- Numerics: `f64` everywhere, no fast-math, summation orders documented
  where they matter for the multi-rank equality test.
- Documentation in English, commit messages in English; discussions can be
  in French.
- Keep the vocabulary of STL-Dev (block, halo, piece, sweep, native level,
  stored moduli, plan, coefficient families) so that both code bases read the
  same.
