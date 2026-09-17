"""HEALPix Laplacian synthesis variables and their exact reconstruction VJP."""

import torch


class HealpixPyramid:
    """P[d] = X[d] - U X[d+1], with the existing filtered D and NESTED U.

    Bands are independent optimization variables. In particular D U is not
    assumed to be identity: scattering analyzes the reconstructed map, not the
    raw coarse reconstruction. This matters after even one optimizer step.
    Inputs may be full sky or a finite partial-sky grid; masks/NaNs are rejected.
    """

    def __init__(self, levels, templates, wavelet_op):
        self.levels = list(levels)
        self.templates = templates
        self.wavelet_op = wavelet_op
        self.dg_max = len(levels) - 1
        self.list_dg = list(range(len(levels)))
        self.N0 = templates[0].N0
        self.device, self.dtype = levels[0].device, levels[0].dtype
        self._parents = []
        for fine, coarse in zip(templates[:-1], templates[1:]):
            ids, order = coarse.cell_ids.sort()
            parents = fine.cell_ids // 4
            pos = torch.searchsorted(ids, parents)
            if bool((pos >= ids.numel()).any()) or not torch.equal(ids[pos], parents):
                raise ValueError(
                    "Every fine pixel must have a parent in the coarse grid"
                )
            self._parents.append(order[pos])

    @classmethod
    @torch.no_grad()
    def from_map(cls, data, wavelet_op, dg_max=None):
        if data.DT != "HealpixKernel_torch" or data.dg != 0 or not data.nest:
            raise ValueError("Expected a dg=0 NESTED HealpixKernel_torch map")
        if data.N0 != (wavelet_op.nside,):
            raise ValueError("Map and wavelet operator resolutions differ")
        if data.N0[0] < 1 or data.N0[0] & (data.N0[0] - 1):
            raise ValueError("NESTED HEALPix nside must be a positive power of two")
        if data.array.is_complex() or data.array.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Pyramid synthesis requires real floating-point maps")
        if wavelet_op.mask_full_res is not None or wavelet_op.nan_aware_stats:
            raise NotImplementedError(
                "Explicit pyramid engine currently requires unmasked finite data"
            )
        if not bool(torch.isfinite(data.array).all()):
            raise ValueError("Pyramid synthesis requires finite data")
        required = max(wavelet_op.j_to_dg)
        dg_max = required if dg_max is None else dg_max
        if not isinstance(dg_max, int) or dg_max < required or 2**dg_max > data.N0[0]:
            raise ValueError(
                "dg_max must cover every wavelet scale and leave nside >= 1"
            )
        if data.array.ndim > 3:
            raise ValueError(
                "Use [pixels], [channels,pixels] or [batch,channels,pixels]"
            )
        current = data.copy(empty=True)
        current.array = data.array.detach().clone()
        current.array = current.array.reshape(
            (1,) * (3 - current.array.ndim) + current.array.shape
        )
        arrays, templates = [], []
        for dg in range(dg_max + 1):
            arrays.append(current.array)
            templates.append(current.copy(empty=True))
            if dg < dg_max:
                current = wavelet_op.downsample(current, dg + 1, inplace=False)
        pyramid = cls(arrays, templates, wavelet_op)
        pyramid.levels = [
            arrays[d] - pyramid.up(arrays[d + 1], d + 1, d) for d in range(dg_max)
        ] + [arrays[-1]]
        return pyramid

    def wrap(self, array, dg, history=()):
        data = self.templates[dg].copy(empty=True)
        data.array, data.dtype, data.device = array, array.dtype, array.device
        data.conv_history = list(history)
        return data

    def down(self, x, dg_from, dg_to):
        return self.wavelet_op.downsample(
            self.wrap(x, dg_from), dg_to, inplace=True
        ).array

    def down_adjoint(self, g, dg_from, dg_to):
        """Adjoint of down(..., dg_from, dg_to): coarse cotangent -> fine."""
        for dg in range(dg_to - 1, dg_from - 1, -1):
            g = self.wavelet_op.downsample_adjoint(g, self.templates[dg])
        return g

    def up(self, x, dg_from, dg_to):
        if not 0 <= dg_to <= dg_from <= self.dg_max:
            raise ValueError("Invalid upsampling levels")
        for dg in range(dg_from - 1, dg_to - 1, -1):
            x = x.index_select(-1, self._parents[dg])
        return x

    def up_adjoint(self, g, dg_from, dg_to):
        """Adjoint of up(..., dg_from, dg_to): fine cotangent -> coarse."""
        for dg in range(dg_to, dg_from):
            out = g.new_zeros(*g.shape[:-1], self.levels[dg + 1].shape[-1])
            out.scatter_add_(-1, self._parents[dg].expand_as(g), g)
            g = out
        return g

    def reconstruct_level(self, dg):
        if not 0 <= dg <= self.dg_max:
            raise ValueError("Invalid reconstruction level")
        x = self.levels[-1]
        for level in range(self.dg_max - 1, dg - 1, -1):
            x = self.levels[level] + self.up(x, level + 1, level)
        return x

    def reconstruct(self):
        """Return [batch, channels, pixels], including singleton axes."""
        return self.reconstruct_level(0)

    def analysis_levels(self):
        """Consistent filtered maps; O(N) storage, no wavelet bank or graph.

        One fine map is necessary for exact legacy coefficients with arbitrary
        independent bands and filtered D. Coarse reconstruction alone would
        change the statistic. Released when the forward/VJP call ends.
        """
        levels = [self.reconstruct_level(0)]
        for dg in range(self.dg_max):
            levels.append(self.down(levels[-1], dg, dg + 1))
        return levels

    def analysis_vjp(self, gradients, callback=None):
        """Chain analysis D* then synthesis U*, retaining only adjacent fields.

        callback(dg, gradient) receives a completed band gradient. With a
        callback no full pyramid of *output* gradients is retained.
        """
        for dg in range(self.dg_max, 0, -1):
            gradients[dg - 1].add_(self.down_adjoint(gradients[dg], dg - 1, dg))
            gradients[dg] = None
        g = gradients[0]
        gradients[0] = None
        result = [] if callback is None else None
        for dg in range(self.dg_max + 1):
            # Compute the next gradient before exposing g to a callback.
            next_g = self.up_adjoint(g, dg + 1, dg) if dg < self.dg_max else None
            if callback is None:
                result.append(g)
            else:
                callback(dg, g)
            g = next_g
        return result


PyramidState = HealpixPyramid
