"""Patch-local HEALPix geometry from partial healpix-analyse operators.

The full-sky ``HealPixConv``/``HealPixDown`` geometry costs about 3.2 kB per
pixel, one hundred times a wavelet field, and is the actual memory wall of
the HEALPix backend.  This module never builds it at the fine levels: every
operation on a NESTED block of pixels uses a *partial* operator built on the
block plus a ring halo (``healpix_geo.nested.kth_neighbourhood``).

Exactness.  A partial ``HealPixConv`` reproduces the full-sky weights on
every output pixel whose four bilinear neighbours of every stencil tap are
present; a partial ``HealPixDown`` (``smooth``, l1-normalised) reproduces the
full-sky row of every coarse pixel whose whole cone of fine pixels is
present.  Halos are therefore chosen so that the rows of interest never see
a truncated stencil or cone:

* convolution on block ``F``: domain ``D = F + conv_rings``;
* decimation of the map on block ``S``: domain ``E = S + down_rings`` makes
  the rows ``parents(S)`` exact;
* decimation *piece* of a field supported on ``S`` (the multiscale S3 chain)
  and the transposed decimation restricted to ``S``: domain
  ``E = S + 2 * down_rings`` makes every row touched by ``S`` exact.

``conv_rings = (kernel_size - 1) // 2 + 2`` (stencil reach, bilinear
neighbours, distortion margin) and ``down_rings = ceil(radius / pixel) + 1``
(cone radius, inclusive cone coverage).  Tests compare every operation with
the full-sky engine; an insufficient halo fails them.

Fine-pixel dilations are computed at the parent level (``k`` coarse rings
cover at least ``2k`` fine pixels), which keeps ``kth_neighbourhood`` calls
small.  Levels are assumed full sky in canonical NESTED order, so a cell id
is also an array position.

healpix-analyse caches ``HealPixConv`` geometries on disk (``cache_dir``),
keyed by the cell ids; the per-block geometries are therefore rebuilt from
disk after the first iteration.
"""

import math
import time

import numpy as np
import torch
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down import HealPixDown
from healpix_geo.nested import kth_neighbourhood

from STL_main.AdjointOps import convolution_adjoint, downsampling_adjoint, modulus_vjp
from STL_main.PatchOps import _csr_tensor, down_forward_full
from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import PyramidStatistics
from STL_main.S3Multiscale import _join, _split, s3_pair_plan
from STL_main.STL_Healpix_Kernel_Torch import _build_hpa_operator


def positions(sorted_ids, ids):
    """Index of each ``ids`` entry in ``sorted_ids`` (all must be present)."""
    pos = torch.searchsorted(sorted_ids, ids)
    pos.clamp_(max=max(sorted_ids.numel() - 1, 0))
    if sorted_ids.numel() == 0 or not torch.equal(sorted_ids[pos], ids):
        raise ValueError("Some cell ids are missing from the domain")
    return pos


def parents(ids):
    return torch.unique(torch.div(ids, 4, rounding_mode="floor"))


def children(ids):
    """Children of sorted parent ids, sorted (NESTED blocks are contiguous)."""
    return (ids[:, None] * 4 + torch.arange(4, device=ids.device)).reshape(-1)


def dilate(ids, nside, rings):
    """Sorted union of the ``rings``-th neighbourhoods of ``ids`` at ``nside``."""
    if rings <= 0:
        return ids
    if rings >= nside:
        # kth_neighbourhood cannot cross a base cell twice; such a halo
        # covers the (small) level anyway.
        return torch.arange(12 * nside**2, device=ids.device)
    depth = int(round(math.log2(nside)))
    near = kth_neighbourhood(ids.detach().cpu().numpy().astype(np.uint64), depth, rings)
    near = np.asarray(near, dtype=np.int64).reshape(-1)
    near = np.unique(near[near >= 0])
    return torch.as_tensor(near, dtype=torch.long, device=ids.device)


def dilate_fine(ids, nside, rings):
    """Superset of the ``rings``-pixel neighbourhood, built at the parent level."""
    if nside < 4 or rings <= 1:
        return dilate(ids, nside, rings)
    return children(dilate(parents(ids), nside // 2, rings // 2 + 1))


class SlimDown:
    """A partial decimation reduced to its two CSR tensors.

    Quacks like ``HealPixDown`` for ``PatchOps.down_forward_full``,
    ``AdjointOps.downsampling_adjoint`` and ``LocalGeometry.out_ids``.
    """

    mode = "smooth"

    def __init__(self, n_in, n_out, cell_ids_out, device, forward, transposed):
        self.N_in, self.N_out = int(n_in), int(n_out)
        self.cell_ids_out = np.asarray(cell_ids_out, dtype=np.int64)
        self.device = torch.device(device)
        dtype = forward.values().dtype
        self._stl_csr_tensors = {(False, dtype): forward, (True, dtype): transposed}

    def _tensors(self):
        (forward, transposed) = (
            self._stl_csr_tensors[k] for k in sorted(self._stl_csr_tensors, key=lambda k: k[0])
        )
        return forward, transposed

    def nbytes(self):
        total = 0
        for t in self._stl_csr_tensors.values():
            for part in (t.crow_indices(), t.col_indices(), t.values()):
                total += part.numel() * part.element_size()
        return total

    def to_device(self, device):
        device = torch.device(device)
        if device == self.device:
            return self
        forward, transposed = self._tensors()

        def move(t, size):
            return torch.sparse_csr_tensor(
                t.crow_indices().to(device, non_blocking=True),
                t.col_indices().to(device, non_blocking=True),
                t.values().to(device, non_blocking=True),
                size=size,
            )

        return SlimDown(
            self.N_in, self.N_out, self.cell_ids_out, device,
            move(forward, (self.N_out, self.N_in)), move(transposed, (self.N_in, self.N_out)),
        )

    def state(self):
        forward, transposed = self._tensors()
        return dict(
            n_in=self.N_in, n_out=self.N_out, cell_ids_out=torch.as_tensor(self.cell_ids_out),
            f_crow=forward.crow_indices().cpu(), f_col=forward.col_indices().cpu(),
            f_val=forward.values().cpu(), t_crow=transposed.crow_indices().cpu(),
            t_col=transposed.col_indices().cpu(), t_val=transposed.values().cpu(),
        )

    @classmethod
    def from_state(cls, s, device):
        n_in, n_out = int(s["n_in"]), int(s["n_out"])
        forward = torch.sparse_csr_tensor(
            s["f_crow"].to(device), s["f_col"].to(device), s["f_val"].to(device), size=(n_out, n_in)
        )
        transposed = torch.sparse_csr_tensor(
            s["t_crow"].to(device), s["t_col"].to(device), s["t_val"].to(device), size=(n_in, n_out)
        )
        return cls(n_in, n_out, s["cell_ids_out"].numpy(), device, forward, transposed)


class LocalGeometry:
    """Partial operators and block-wise exact pyramid operations."""

    def __init__(
        self, wavelet_op, block_pixels=16384, cache_dir=None, cache_downs=None,
        torch_geometry=True,
    ):
        """``cache_downs``: where the partial decimation matrices live between
        uses. ``"ram"`` keeps them on the device (in total about the full-sky
        decimation matrix: tens of bytes per fine pixel, block independent);
        ``"host"`` keeps them in CPU memory and moves one block at a time to
        the device; ``"disk"`` stores them under ``cache_dir`` and reloads one
        block at a time (slow on network file systems); ``False`` rebuilds
        them every time. Default: ``"host"`` on CUDA, ``"ram"`` otherwise.
        With ``cache_dir``, built decimations are also saved to disk and
        found there in later runs. Convolution geometries are computed on the
        device (``TorchGeometry``) or, in the fallback, reloaded from
        ``cache_dir``. ``timers`` accumulates seconds per operation class."""
        if wavelet_op.mask_full_res is not None or wavelet_op.nan_aware_stats:
            raise NotImplementedError("Local geometry requires unmasked finite data")
        if not isinstance(block_pixels, int) or block_pixels < 1:
            raise ValueError("block_pixels must be a positive integer")
        if cache_downs is None:
            cache_downs = "host" if wavelet_op.device.type == "cuda" else "ram"
        if cache_downs is True:
            cache_downs = "ram"
        if cache_downs not in ("ram", "host", "disk", False):
            raise ValueError("cache_downs must be 'ram', 'host', 'disk' or False")
        if cache_downs == "disk" and cache_dir is None:
            raise ValueError("cache_downs='disk' requires cache_dir")
        self.w = wavelet_op
        self.block_pixels = block_pixels
        self.cache_dir = cache_dir
        self.cache_downs = cache_downs
        self.torch_geometry = bool(torch_geometry)
        self._downs = {}
        self._dilations = {}
        self.timers = {}
        self.conv_rings = (int(wavelet_op.KERNELSZ) - 1) // 2 + 2
        self.device = wavelet_op.device
        self.built = dict(conv=0, down=0)

    # -- timing ------------------------------------------------------------------
    class _timed:
        def __init__(self, geometry, name):
            self.geometry, self.name = geometry, name

        def __enter__(self):
            if self.geometry.device.type == "cuda":
                torch.cuda.synchronize(self.geometry.device)
            self.start = time.perf_counter()

        def __exit__(self, *exc):
            if self.geometry.device.type == "cuda":
                torch.cuda.synchronize(self.geometry.device)
            self.geometry.timers[self.name] = self.geometry.timers.get(self.name, 0.0) + (
                time.perf_counter() - self.start
            )

    def timed(self, name):
        return self._timed(self, name)

    def dilate_fine(self, ids, nside, rings):
        """``dilate_fine`` memoised on the host (geometry only, reused across calls)."""
        key = (nside, rings, ids.numel(), int(ids[0]), int(ids[-1]), int(ids.sum()))
        hit = self._dilations.get(key)
        if hit is not None:
            return hit.to(ids.device, non_blocking=True)
        with self.timed("dilation"):
            out = dilate_fine(ids, nside, rings)
        self._dilations[key] = out.cpu()
        return out

    # -- resolutions and blocks ---------------------------------------------
    def nside(self, dg):
        return int(self.w.nside) >> dg

    def pixels(self, dg):
        return 12 * self.nside(dg) ** 2

    def down_rings(self, dg):
        """Cone radius of the decimation from level ``dg``, in fine pixels, +1."""
        pix_deg = math.degrees(math.sqrt(4 * math.pi / self.pixels(dg)))
        radius = self.w.down_kwargs.get("radius_deg")
        radius = 3.0 * pix_deg if radius is None else float(radius)
        return int(math.ceil(radius / pix_deg)) + 1

    def blocks(self, dg, size=None):
        n, size = self.pixels(dg), size or self.block_pixels
        return [
            torch.arange(start, min(start + size, n), device=self.device)
            for start in range(0, n, size)
        ]

    # -- partial operators -------------------------------------------------
    def torch_geometry_available(self):
        """Device-side geometry (TorchGeometry) covers the sphere and the
        'phi'/'cosmo' gauges; other ellipsoids fall back to healpix-analyse."""
        return str(self.w.ellipsoid).lower() == "sphere" and self.w.gauge_type in (
            "phi",
            "cosmo",
        )

    def conv(self, dg, ids):
        w, nside = self.w, self.nside(dg)
        if self.torch_geometry and self.torch_geometry_available():
            from STL_main.TorchGeometry import TorchConv

            self.built["conv"] += 1
            with self.timed("conv_geometry"):
                return TorchConv(
                    ids, nside, w.KERNELSZ, w.L, w.gauge_type,
                    w._wav_weights.detach().cpu().numpy(), w.dtype,
                    w.pyramid_pixel_batch_size,
                )
        # healpix-analyse's on-disk cache is on by default and stores the
        # interpolation weights in float32 whatever the operator dtype, which
        # silently degrades float64 runs once a geometry has been cached.
        # With ``cache_dir`` set, a dedicated directory is used and the
        # buffers are re-saved in the operator dtype right after a build, so
        # the library loader (a plain ``.to(dtype)``) returns them exactly.
        optional = dict(
            device=w.device, dtype=w.dtype, ellipsoid=w.ellipsoid, cache_dir=self.cache_dir
        )
        conv = _build_hpa_operator(
            HealPixConv,
            nside=nside,
            level=int(round(math.log2(nside))),
            required=dict(
                in_channels=1,
                out_channels=2,
                kernel_sz=w.KERNELSZ,
                n_gauges=w.L,
                gauge_type=w.gauge_type,
                cell_ids=ids.detach().cpu().numpy().astype(np.int64),
                nest=w.nest,
            ),
            optional=optional,
        )
        conv.set_kernel(w._wav_weights.detach().cpu().numpy())
        self.built["conv"] += 1
        if self.cache_dir is not None:
            self._save_exact(conv, ids)
        return conv

    def _save_exact(self, conv, ids):
        """Re-save a freshly built geometry in the operator dtype (library layout)."""
        try:
            from pathlib import Path

            from healpix_analyse import convol

            key = convol._geometry_cache_key(
                nside=conv.nside,
                kernel_sz=conv.kernel_sz,
                G=conv.G,
                gauge_type=conv.gauge_type,
                ref_direction=getattr(conv, "ref_direction", None),
                nest=conv.nest,
                cell_ids=ids.detach().cpu().numpy().astype(np.int64),
                ellipsoid=conv.ellipsoid,
            )
            path = convol._cache_path(Path(self.cache_dir), key)
            if path.exists() and torch.load(path, map_location="cpu", weights_only=True)[
                "w_norm"
            ].dtype == conv._w_norm.dtype:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(
                dict(
                    pos_safe=conv._pos_safe.cpu().to(torch.long),
                    w_norm=conv._w_norm.cpu(),
                    sort_order=conv._sort_order.cpu(),
                    inv_order=conv._inv_order.cpu(),
                ),
                tmp,
            )
            tmp.replace(path)
            self.built["saved"] = self.built.get("saved", 0) + 1
        except Exception:  # cache is an optimisation only; never fail a build
            pass

    def torch_down(self, dg, ids):
        """Partial smooth decimation built on the device (sphere); no cache needed."""
        from STL_main.TorchGeometry import smooth_decimation_csr

        nside = self.nside(dg)
        coarse = parents(ids)
        # Candidate fine pixels: children of the 2-ring coarse neighbourhood
        # (a cone of 3 fine pixels never reaches further).
        near = dilate(coarse, nside // 2, 2) if nside // 2 >= 1 else coarse
        # Per coarse pixel candidate lists through kth_neighbourhood (CPU, small).
        depth = int(round(math.log2(nside // 2)))
        if nside // 2 >= 4 and 2 < nside // 2:
            nb = kth_neighbourhood(coarse.detach().cpu().numpy().astype(np.uint64), depth, 2)
            nb = torch.as_tensor(np.asarray(nb, dtype=np.int64), device=ids.device)
        else:
            nb = torch.arange(12 * (nside // 2) ** 2, device=ids.device).expand(coarse.numel(), -1)
        children_ids = (nb[:, :, None] * 4 + torch.arange(4, device=ids.device)).reshape(nb.shape[0], -1)
        children_ids = torch.where(nb[:, :, None].expand(-1, -1, 4).reshape(nb.shape[0], -1) < 0, -1, children_ids)
        with self.timed("down_build"):
            coarse_ids, forward, transposed = smooth_decimation_csr(
                ids, nside, children_ids, self.w.dtype,
                radius_deg=self.w.down_kwargs.get("radius_deg"),
                sigma_deg=self.w.down_kwargs.get("sigma_deg"),
                weight_norm=self.w.down_kwargs.get("weight_norm", "l1"),
            )
        self.built["down"] += 1
        return SlimDown(ids.numel(), coarse_ids.numel(), coarse_ids.cpu().numpy(), ids.device, forward, transposed)

    def down(self, dg, ids):
        if self.torch_geometry and self.torch_geometry_available():
            return self.torch_down(dg, ids)
        key = (dg, ids.numel(), int(ids[0]), int(ids[-1]), int(ids.sum()))
        if self.cache_downs in ("ram", "host") and key in self._downs:
            hit = self._downs[key]
            return hit if self.cache_downs == "ram" else hit.to_device(self.device)
        if self.cache_downs in ("disk", "host") and self.cache_dir is not None:
            with self.timed("down_load"):
                loaded = self._load_down(key)
            if loaded is not None:
                if self.cache_downs == "host":
                    self._downs[key] = loaded.to_device("cpu")
                    return loaded
                return loaded
        with self.timed("down_build"):
            down = self._build_down(dg, ids)
        if not self.cache_downs:
            return down
        slim = SlimDown(
            down.N_in, down.N_out, np.asarray(down.cell_ids_out), down.device,
            _csr_tensor(down, False, self.w.dtype), _csr_tensor(down, True, self.w.dtype),
        )
        del down
        if self.cache_downs == "ram":
            self._downs[key] = slim
        elif self.cache_downs == "host":
            self._downs[key] = slim.to_device("cpu")
            if self.cache_dir is not None:
                self._save_down(key, slim)
        else:
            self._save_down(key, slim)
        return slim

    def _build_down(self, dg, ids):
        w, nside = self.w, self.nside(dg)
        down = _build_hpa_operator(
            HealPixDown,
            nside=nside,
            level=int(round(math.log2(nside))),
            required=dict(mode="smooth", cell_ids=ids.detach().cpu().numpy().astype(np.int64)),
            optional=dict(
                device=w.device, dtype=w.dtype, ellipsoid=w.ellipsoid, **w.down_kwargs
            ),
        )
        self.built["down"] += 1
        return down

    def _down_path(self, key):
        from pathlib import Path

        name = "stl_down_{}_{}_{}_{}_{}_{}.pt".format(
            self.nside(0), *key
        )
        return Path(self.cache_dir) / name

    def _save_down(self, key, slim):
        try:
            path = self._down_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(slim.state(), tmp)
            tmp.replace(path)
        except Exception:
            pass

    def _load_down(self, key):
        path = self._down_path(key)
        if not path.exists():
            return None
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return None
        return SlimDown.from_state(state, self.device)

    def down_cache_bytes(self):
        """Bytes held by the cached decimation matrices (device for 'ram', host for 'host')."""
        return sum(d.nbytes() for d in self._downs.values())

    @staticmethod
    def out_ids(down):
        cached = getattr(down, "_stl_out_ids", None)
        if cached is None:
            cached = torch.as_tensor(
                np.asarray(down.cell_ids_out), dtype=torch.long, device=down.device
            )
            try:
                down._stl_out_ids = cached
            except AttributeError:
                pass
        return cached

    # -- exact block-wise pyramid operations ---------------------------------
    def decimate(self, x, dg):
        """``maps[dg] -> maps[dg + 1]`` with the full-sky weights, block by block.

        The domain uses the doubled halo of the transposed decimation and of
        the chain pieces so that one partial operator per block serves all.
        """
        out = x.new_zeros(*x.shape[:-1], self.pixels(dg + 1))
        rings = 2 * self.down_rings(dg)
        for block in self.blocks(dg):
            domain = self.dilate_fine(block, self.nside(dg), rings)
            down = self.down(dg, domain)
            with self.timed("decimate"):
                y = down_forward_full(down, x[..., domain])
                rows = parents(block)
                out[..., rows] = y[..., positions(self.out_ids(down), rows)]
        return out

    def decimate_adjoint(self, g, dg):
        """Transpose of ``decimate``: ``[..., N_{dg+1}] -> [..., N_dg]``, exact."""
        out = g.new_zeros(*g.shape[:-1], self.pixels(dg))
        rings = 2 * self.down_rings(dg)
        for block in self.blocks(dg):
            domain = self.dilate_fine(block, self.nside(dg), rings)
            down = self.down(dg, domain)
            with self.timed("decimate"):
                x = downsampling_adjoint(down, g[..., self.out_ids(down)])
                out[..., block] = x[..., positions(domain, block)]
        return out

    def analysis_levels(self, pyramid):
        levels = [pyramid.reconstruct_level(0)]
        for dg in range(pyramid.dg_max):
            levels.append(self.decimate(levels[-1], dg))
        return levels

    def analysis_vjp(self, pyramid, gradients):
        for dg in range(pyramid.dg_max, 0, -1):
            gradients[dg - 1].add_(self.decimate_adjoint(gradients[dg], dg - 1))
            gradients[dg] = None
        g, result = gradients[0], []
        gradients[0] = None
        for dg in range(pyramid.dg_max + 1):
            next_g = pyramid.up_adjoint(g, dg + 1, dg) if dg < pyramid.dg_max else None
            result.append(g)
            g = next_g
        return result

    @torch.no_grad()
    def pyramid_from_map(self, data, dg_max=None):
        """``HealpixPyramid.from_map`` without the full-sky decimation geometry."""
        w = self.w
        if data.DT != "HealpixKernel_torch" or data.dg != 0 or not data.nest:
            raise ValueError("Expected a dg=0 NESTED HealpixKernel_torch map")
        if data.N0 != (w.nside,) or data.array.shape[-1] != self.pixels(0):
            raise ValueError("Local geometry expects a full-sky map at the operator nside")
        if not torch.equal(
            data.cell_ids.cpu(), torch.arange(self.pixels(0))
        ):
            raise ValueError("Local geometry expects canonical NESTED cell ids")
        required = max(w.j_to_dg)
        dg_max = required if dg_max is None else dg_max
        if dg_max < required or 2**dg_max > w.nside:
            raise ValueError("dg_max must cover every wavelet scale and leave nside >= 1")
        array = data.array.detach().clone()
        array = array.reshape((1,) * (3 - array.ndim) + array.shape)
        arrays, templates = [array], []
        for dg in range(dg_max + 1):
            template = data.copy(empty=True)
            template.dg, template.nside = dg, self.nside(dg)
            template.cell_ids = torch.arange(self.pixels(dg), device=data.cell_ids.device)
            templates.append(template)
            if dg < dg_max:
                arrays.append(self.decimate(arrays[-1], dg))
        pyramid = HealpixPyramid(arrays, templates, w)
        pyramid.levels = [
            arrays[d] - pyramid.up(arrays[d + 1], d + 1, d) for d in range(dg_max)
        ] + [arrays[-1]]
        return pyramid

    # -- pieces of the multiscale S3 chain ------------------------------------
    def wave(self, x_level, block, dg):
        """Fine wavelet on ``block`` through a partial convolution: ``(z, domain, conv)``.

        ``z`` is ``[B, C, L, |block|]`` complex; ``domain`` and ``conv`` are
        needed again by ``wave_adjoint``.
        """
        domain = self.dilate_fine(block, self.nside(dg), self.conv_rings)
        conv = self.conv(dg, domain)
        with self.timed("conv_apply"):
            x = x_level[..., domain]
            batch, channels = x.shape[:2]
            y = conv(x.reshape(-1, 1, domain.numel()).to(conv.dtype))
            y = y[..., positions(domain, block)]
            return _join(y, self.w.L).reshape(batch, channels, self.w.L, -1), domain, conv

    def wave_adjoint(self, gz, block, domain, conv):
        """Adjoint of ``wave`` on ``block``: ``[B, C, L, |block|] -> [B, C, |domain|]``."""
        with self.timed("conv_adjoint"):
            batch, channels = gz.shape[:2]
            g = gz.new_zeros(batch * channels, 2 * self.w.L, domain.numel(), dtype=gz.real.dtype)
            g[..., positions(domain, block)] = _split(gz).reshape(batch * channels, 2 * self.w.L, -1)
            gx = convolution_adjoint(conv, g, self.w.pyramid_pixel_batch_size)
            return gx.reshape(batch, channels, -1)

    def piece_forward(self, a, support, dg):
        """Contribution of a field supported on ``support`` to level ``dg + 1``.

        Returns ``(y, next_support)`` with ``y`` exact on ``next_support``.
        """
        rings = self.down_rings(dg)
        domain = self.dilate_fine(support, self.nside(dg), 2 * rings)
        down = self.down(dg, domain)
        with self.timed("pieces"):
            padded = a.new_zeros(*a.shape[:-1], domain.numel())
            padded[..., positions(domain, support)] = a
            y = down_forward_full(down, padded)
            next_support = parents(self.dilate_fine(support, self.nside(dg), rings))
            return y[..., positions(self.out_ids(down), next_support)], next_support

    def piece_adjoint(self, g, support, next_support, dg):
        """Transpose of ``piece_forward``, exact on ``support``."""
        domain = self.dilate_fine(support, self.nside(dg), 2 * self.down_rings(dg))
        down = self.down(dg, domain)
        with self.timed("pieces"):
            out_ids = self.out_ids(down)
            padded = g.new_zeros(*g.shape[:-1], out_ids.numel())
            padded[..., positions(out_ids, next_support)] = g
            x = downsampling_adjoint(down, padded)
            return x[..., positions(domain, support)]


@torch.no_grad()
def s3_scale_pair_local(
    operator,
    pyramid,
    j1,
    j3,
    gradient=None,
    plan=None,
    geometry=None,
    fine_patch_pixels=None,
    tracker=None,
    maps=None,
    maps_gradient=None,
):
    """``S3Multiscale.s3_scale_pair`` with patch-local fine geometry.

    The coarse level ``dg(j3)`` keeps the full-sky operator of that level
    (small), every fine-level operation goes through partial operators built
    on each sub-patch plus halo.  No full-sky geometry of the fine level is
    ever created.  Returns ``(PyramidStatistics, maps_gradient)``.
    """
    w = operator.wavelet_op
    geometry = LocalGeometry(w) if geometry is None else geometry
    if not 0 <= j1 <= j3 < w.J:
        raise ValueError("Expected 0 <= j1 <= j3 < J")
    n, d1 = w.j_to_dg[j3], w.j_to_dg[j1]
    channels = pyramid.levels[0].shape[1]
    plan = s3_pair_plan(operator, channels, j1, j3) if plan is None else tuple(plan)
    if any(c.family != "S3" or c.j1 != j1 or c.j3 != j3 for c in plan):
        raise ValueError("Plan must contain S3 coefficients of the requested pair")
    if maps is None:
        maps = geometry.analysis_levels(pyramid)
    batch = maps[0].shape[0]
    if gradient is not None:
        if gradient.shape != (batch, len(plan)):
            raise ValueError("Gradient must have shape [batch, len(plan)]")
        gradient = gradient.to(maps[0].device)
        if maps_gradient is None:
            maps_gradient = [torch.zeros_like(x) for x in maps]
    L, fewer = w.L, operator.has_fewer_convolutions
    n_total = maps[n].shape[-1]
    fine_patch = fine_patch_pixels or geometry.block_pixels
    pixel_batch = w.pyramid_pixel_batch_size
    values = torch.zeros(
        batch, len(plan),
        dtype=torch.promote_types(maps[0].dtype, torch.complex64),
        device=maps[0].device,
    )

    def trace(name, dg, x):
        if tracker is not None:
            tracker(name, dg, x)

    # ---- forward: decimated fine modulus on the whole coarse level
    m = torch.zeros(batch, channels, L, n_total, dtype=maps[0].dtype, device=maps[0].device)
    blocks = geometry.blocks(d1, fine_patch)
    for block in blocks:
        z, _, _ = geometry.wave(maps[d1], block, d1)
        trace("first_wavelet", d1, z)
        a, support = z.abs(), block
        for level in range(d1, n):
            a, support = geometry.piece_forward(a, support, level)
        m.index_add_(-1, support, a)
        del z, a

    # ---- coarse level: full-sky operator of level n (the "old mode")
    conv_n = w._get_conv(n, pyramid.templates[n].cell_ids, L, w._wav_weights)
    y = conv_n(m.reshape(-1, 1, n_total).to(conv_n.dtype))
    branch = _join(y, L).reshape(batch, channels, L, L, -1)
    trace("second_wavelet", n, branch)
    z3 = _join(conv_n(maps[n].reshape(-1, 1, n_total).to(conv_n.dtype)), L)
    z3 = z3.reshape(batch, channels, L, -1)
    direct = maps[n]
    gbranch = torch.zeros_like(branch) if gradient is not None else None
    gz3 = torch.zeros_like(z3) if gradient is not None else None
    for i, c in enumerate(plan):
        if fewer:
            a, b = direct[:, c.c1], branch[:, c.c2, c.l1, c.l3]
        else:
            a, b = branch[:, c.c1, c.l1, c.l3], z3[:, c.c2, c.l3]
        values[:, i] += (a * b.conj()).sum(-1) / n_total
        if gradient is None:
            continue
        g = gradient[:, i]
        ga, gb = g[..., None] * b / n_total, g.conj()[..., None] * a / n_total
        if fewer:
            maps_gradient[n][:, c.c1].add_(ga.real)
            gbranch[:, c.c2, c.l1, c.l3].add_(gb)
        else:
            gbranch[:, c.c1, c.l1, c.l3].add_(ga)
            gz3[:, c.c2, c.l3].add_(gb)
    if gradient is None:
        return PyramidStatistics(plan, values), None

    gx3 = convolution_adjoint(
        conv_n, _split(gz3).reshape(batch * channels, 2 * L, -1), pixel_batch
    )
    maps_gradient[n].add_(gx3.reshape(batch, channels, -1))
    gm = convolution_adjoint(
        conv_n, _split(gbranch).reshape(batch * channels * L, 2 * L, -1), pixel_batch
    ).reshape(batch, channels, L, -1)
    del gbranch, gz3, branch, z3, y

    # ---- reverse, fine part by sub-patch through the transposed pieces
    for block in blocks:
        z, domain, conv = geometry.wave(maps[d1], block, d1)
        supports, support = [block], block
        for level in range(d1, n):
            support = parents(geometry.dilate_fine(support, geometry.nside(level), geometry.down_rings(level)))
            supports.append(support)
        g = gm[..., supports[-1]]
        for level in range(n - 1, d1 - 1, -1):
            g = geometry.piece_adjoint(g, supports[level - d1], supports[level - d1 + 1], level)
        gz = modulus_vjp(z, g)
        trace("native_gradient", d1, gz)
        maps_gradient[d1].index_add_(-1, domain, geometry.wave_adjoint(gz, block, domain, conv))
        del z, g, gz, conv
    return PyramidStatistics(plan, values), maps_gradient
