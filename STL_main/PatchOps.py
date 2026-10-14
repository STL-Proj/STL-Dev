"""Row-restricted HEALPix linear operators and their exact input domains.

The full-sky ``HealPixConv`` and ``HealPixDown`` geometries already describe,
for every output pixel, which input pixels it reads.  Restricting an operator
to a subset of output rows therefore needs no new geometry and no boundary
renormalisation: the result on those rows is bit-for-bit the full-sky result,
provided the input is supplied on the (exact) set of pixels the rows read.

Index conventions follow ``AdjointOps.convolution_forward``:

* a *position* is an index into the data array at a given resolution, which
  is the order of ``template.cell_ids`` at that level;
* ``conv._pos_safe[nb, g, k*P + p]`` indexes the *sorted* input, row ``k``
  corresponds to output position ``i`` with ``k = conv._inv_order[i]``, and
  sorted index ``s`` is array position ``conv._sort_order[s]``;
* ``HealPixDown`` rows are coarse positions and columns fine positions.

All domains are sorted 1-D ``long`` tensors of positions.  Global-to-local
index translation goes through an ``IndexTable`` (one dense scratch table per
resolution) rather than ``searchsorted``/``unique``, which dominated the cost
of building patches.
"""

import torch

from STL_main.AdjointOps import _check_convolution_geometry


class IndexTable:
    """Dense scratch tables for one resolution: global position -> local index.

    ``collect`` turns a multiset of read positions into a sorted domain and
    ``map`` translates positions into indices within a domain.  Both leave the
    tables cleared.  Memory: 5 bytes per pixel of the level.
    """

    def __init__(self, size, device):
        self.table = torch.full((size,), -1, dtype=torch.int32, device=device)
        self.mark = torch.zeros(size, dtype=torch.bool, device=device)

    def collect(self, positions):
        self.mark[positions] = True
        domain = self.mark.nonzero().reshape(-1)
        self.mark[domain] = False
        return domain

    def fill(self, domain):
        self.table[domain] = torch.arange(
            domain.numel(), dtype=torch.int32, device=domain.device
        )

    def clear(self, domain):
        self.table[domain] = -1

    def get(self, positions):
        """Translate positions while the table is filled for a domain."""
        local = self.table[positions]
        if bool((local < 0).any()):
            raise ValueError("Requested positions are not all inside the domain")
        return local.long()

    def get_optional(self, positions):
        """Like ``get`` but returns -1 for positions outside the filled domain."""
        return self.table[positions].long()

    def map(self, domain, positions, optional=False):
        self.fill(domain)
        try:
            return self.get_optional(positions) if optional else self.get(positions)
        finally:
            self.clear(domain)


def lookup(domain, positions):
    """``searchsorted`` fallback of ``IndexTable.map`` for small one-off uses."""
    local = torch.searchsorted(domain, positions)
    local.clamp_(max=max(domain.numel() - 1, 0))
    if domain.numel() == 0 or not torch.equal(domain[local], positions):
        raise ValueError("Requested positions are not all inside the domain")
    return local


def _collect(positions, table):
    return torch.unique(positions) if table is None else table.collect(positions)


def _map(domain, positions, table):
    return lookup(domain, positions) if table is None else table.map(domain, positions)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------


def _conv_reads(conv, rows):
    """Array positions read by geometry rows ``rows``: ``[4, G, |rows|, P]``."""
    positions = conv._pos_safe.reshape(4, conv.G, conv.K, conv.P)[:, :, rows, :]
    return conv._sort_order[positions.reshape(-1)].reshape(positions.shape)


def _blocks(size, pixel_batch_size):
    if pixel_batch_size < 1:
        raise ValueError("pixel_batch_size must be positive")
    for start in range(0, size, pixel_batch_size):
        yield start, min(start + pixel_batch_size, size)


def conv_input_domain(conv, out_positions, table=None, pixel_batch_size=16384):
    """Sorted input positions read by ``out_positions`` (exact stencil union)."""
    _check_convolution_geometry(conv)
    rows = conv._inv_order[out_positions]
    if table is None:
        return torch.unique(_conv_reads(conv, rows).reshape(-1))
    for start, stop in _blocks(rows.numel(), pixel_batch_size):
        table.mark[_conv_reads(conv, rows[start:stop]).reshape(-1)] = True
    domain = table.mark.nonzero().reshape(-1)
    table.mark[domain] = False
    return domain


class ConvRows:
    """One convolution restricted to ``out_positions`` reading ``in_domain``.

    Nothing of size ``4 * G * P`` per row is stored: the forward and the
    adjoint re-read the full-sky geometry one pixel block at a time and
    translate positions through the level's ``IndexTable`` (or, without a
    table, through ``searchsorted``), so the memory of a patch is its fields,
    not a private copy of its stencils.
    """

    def __init__(self, conv, out_positions, in_domain=None, table=None):
        _check_convolution_geometry(conv)
        self.conv = conv
        self.out_positions = out_positions
        self.rows = conv._inv_order[out_positions]
        self.table = table
        if in_domain is None:
            in_domain = conv_input_domain(conv, out_positions, table)
        self.in_domain = in_domain

    @property
    def size(self):
        return self.out_positions.numel()

    def block(self, start, stop, like):
        """Local gather indices and interpolation weights of one row block."""
        conv, rows = self.conv, self.rows[start:stop]
        read = _conv_reads(conv, rows).reshape(-1)
        local = (
            lookup(self.in_domain, read) if self.table is None else self.table.get(read)
        )
        interp = conv._w_norm.reshape(4, conv.G, conv.K, conv.P)[:, :, rows, :]
        return local.reshape(4, conv.G, rows.numel(), conv.P), interp.to(like)

    def __enter__(self):
        if self.table is not None:
            self.table.fill(self.in_domain)
        return self

    def __exit__(self, *exc):
        if self.table is not None:
            self.table.clear(self.in_domain)


def conv_forward_rows(rows, x, pixel_batch_size=16384):
    """Linear HealPixConv on ``rows.out_positions``; ``x`` is ``[B, C, |in|]``.

    Returns ``[B, G * C_out, |out|]`` in the order of ``rows.out_positions``.
    """
    conv = rows.conv
    batch, channels, n_in = x.shape
    if n_in != rows.in_domain.numel() or channels != conv.in_channels:
        raise ValueError("Convolution input shape does not match the patch domain")
    x = x.to(dtype=conv.dtype)
    weights = conv.weight.to(x)
    m = rows.size
    output = x.new_empty(batch, conv.G, conv.out_channels, m)
    with rows:
        for start, stop in _blocks(m, pixel_batch_size):
            local, interp = rows.block(start, stop, x)
            for gauge in range(conv.G):
                gathered = x.new_zeros(batch, channels, stop - start, conv.P)
                for neighbour in range(4):
                    idx = local[neighbour, gauge].reshape(-1)
                    values = x.index_select(-1, idx).reshape_as(gathered)
                    gathered.add_(values * interp[neighbour, gauge])
                output[:, gauge, :, start:stop] = torch.einsum(
                    "bckp,cop->bok", gathered, weights[gauge]
                )
    output = output.reshape(batch, conv.G * conv.out_channels, m)
    output.add_(conv.bias.to(x).reshape(1, -1, 1))
    return output


def conv_adjoint_rows(rows, gradient, pixel_batch_size=16384):
    """Transpose of ``conv_forward_rows``: ``[B, G*C_out, |out|] -> [B, C, |in|]``."""
    conv = rows.conv
    batch = gradient.shape[0]
    m = rows.size
    g = gradient.reshape(batch, conv.G, conv.out_channels, m)
    weights = conv.weight.to(g)
    out = g.new_zeros(batch, conv.in_channels, rows.in_domain.numel())
    with rows:
        for start, stop in _blocks(m, pixel_batch_size):
            local, interp = rows.block(start, stop, g)
            for gauge in range(conv.G):
                taps = torch.einsum(
                    "bok,cop->bckp", g[:, gauge, :, start:stop], weights[gauge]
                )
                for neighbour in range(4):
                    idx = local[neighbour, gauge].reshape(-1)
                    values = taps * interp[neighbour, gauge]
                    out.scatter_add_(
                        2,
                        idx.expand(batch, conv.in_channels, -1),
                        values.reshape(batch, conv.in_channels, -1),
                    )
    return out


# ---------------------------------------------------------------------------
# Anti-aliased decimation (sparse matrix rows)
# ---------------------------------------------------------------------------


def _down_csr(down):
    """CSR view of the decimation matrix, cached on ``down``."""
    if down.mode != "smooth" or not hasattr(down, "_M_indices"):
        raise NotImplementedError(
            "Explicit downsampling requires the smooth sparse operator"
        )
    cached = getattr(down, "_stl_patch_csr", None)
    if cached is None:
        matrix = torch.sparse_coo_tensor(
            down._M_indices, down._M_values, down._M_size
        ).coalesce()
        csr = matrix.to_sparse_csr()
        cached = (csr.crow_indices(), csr.col_indices(), csr.values())
        down._stl_patch_csr = cached
    return cached


def _csr_rows(csr, rows):
    """Entries of the given CSR rows as (row index into ``rows``, column, value)."""
    crow, col, val = csr
    starts, ends = crow[rows], crow[rows + 1]
    counts = ends - starts
    total = int(counts.sum())
    local_rows = torch.repeat_interleave(
        torch.arange(rows.numel(), device=rows.device), counts
    )
    offsets = torch.cumsum(counts, 0) - counts
    entry = torch.arange(total, device=rows.device) + torch.repeat_interleave(
        starts - offsets, counts
    )
    return local_rows, col[entry], val[entry]


def down_input_domain(down, out_positions, table=None, row_batch_size=16384):
    """Sorted fine positions read by the coarse ``out_positions``.

    Rows are visited in blocks so that the entry arrays never exceed one
    block of the sparse matrix, even when ``out_positions`` is a whole level.
    """
    csr = _down_csr(down)
    if table is None:
        parts = [
            _csr_rows(csr, out_positions[start:stop])[1]
            for start, stop in _blocks(out_positions.numel(), row_batch_size)
        ]
        return torch.unique(torch.cat(parts))
    for start, stop in _blocks(out_positions.numel(), row_batch_size):
        table.mark[_csr_rows(csr, out_positions[start:stop])[1]] = True
    domain = table.mark.nonzero().reshape(-1)
    table.mark[domain] = False
    return domain


def _csr_tensor(down, transposed, dtype):
    """Cached sparse CSR tensor of the decimation matrix or its transpose."""
    cache = getattr(down, "_stl_csr_tensors", None)
    if cache is None:
        cache = down._stl_csr_tensors = {}
    key = (transposed, dtype)
    if key not in cache:
        same = [v for (t, _), v in cache.items() if t == transposed]
        if same and not hasattr(down, "_M_indices"):
            # Slimmed operator (LocalGeometry): convert the cached dtype.
            cache[key] = same[0].to(dtype)
            return cache[key]
        crow, col, val = _down_csr_t(down) if transposed else _down_csr(down)
        rows, cols = down._M_size
        if transposed:
            rows, cols = cols, rows
        cache[key] = torch.sparse_csr_tensor(
            crow, col, val.to(dtype), size=(rows, cols), device=crow.device
        )
    return cache[key]


def _csr_apply(matrix, x):
    """``x[..., n] -> y[..., m]`` through a sparse ``[m, n]`` matrix."""
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    if flat.is_complex():
        result = torch.complex((matrix @ flat.real.T).T, (matrix @ flat.imag.T).T)
    else:
        result = (matrix @ flat.T).T
    return result.reshape(*leading, matrix.shape[0])


def down_forward_full(down, x):
    """Whole-level decimation through the cached CSR matrix (no COO rebuild)."""
    if x.shape[-1] != down.N_in:
        raise ValueError("Decimation input does not match the operator")
    return _csr_apply(_csr_tensor(down, False, x.real.dtype), x)


def down_adjoint_full(down, gradient):
    """Whole-level transposed decimation through the cached CSR transpose."""
    if gradient.shape[-1] != down.N_out:
        raise ValueError("Decimation cotangent does not match the operator")
    return _csr_apply(_csr_tensor(down, True, gradient.real.dtype), gradient)


class DownRows:
    """One decimation restricted to coarse ``out_positions``."""

    def __init__(self, down, out_positions, in_domain=None, table=None):
        self.down = down
        self.out_positions = out_positions
        self.local_rows, cols, self.values = _csr_rows(_down_csr(down), out_positions)
        if in_domain is None:
            in_domain = _collect(cols, table)
        self.in_domain = in_domain
        self.local_cols = _map(in_domain, cols, table)


def down_forward_rows(rows, x):
    """``x[..., |in|] -> y[..., |out|]`` using the exact full-sky weights."""
    if x.shape[-1] != rows.in_domain.numel():
        raise ValueError("Decimation input does not match the patch domain")
    values = rows.values.to(x.real.dtype)
    y = x.new_zeros(*x.shape[:-1], rows.out_positions.numel())
    y.index_add_(-1, rows.local_rows, x[..., rows.local_cols] * values)
    return y


def down_adjoint_rows(rows, gradient):
    """Transpose of ``down_forward_rows``: ``[..., |out|] -> [..., |in|]``."""
    if gradient.shape[-1] != rows.out_positions.numel():
        raise ValueError("Decimation cotangent does not match the patch rows")
    values = rows.values.to(gradient.real.dtype)
    out = gradient.new_zeros(*gradient.shape[:-1], rows.in_domain.numel())
    out.index_add_(-1, rows.local_cols, gradient[..., rows.local_rows] * values)
    return out


def _down_csr_t(down):
    """CSR view of the transposed decimation matrix (rows = fine positions)."""
    _down_csr(down)
    cached = getattr(down, "_stl_patch_csr_t", None)
    if cached is None:
        matrix = torch.sparse_coo_tensor(
            down._M_indices, down._M_values, down._M_size
        ).t().coalesce()
        csr = matrix.to_sparse_csr()
        cached = (csr.crow_indices(), csr.col_indices(), csr.values())
        down._stl_patch_csr_t = cached
    return cached


class DownPiece:
    """One decimation restricted to a *fine* support ``in_positions``.

    This is the column-restricted counterpart of ``DownRows``: it maps a
    field supported on a fine sub-patch to the coarse rows it touches, so a
    linear decimation chain can be streamed by fine sub-patches and the
    pieces summed at the coarse level.
    """

    def __init__(self, down, in_positions, table=None):
        self.down = down
        self.in_positions = in_positions
        self.local_in, cols, self.values = _csr_rows(_down_csr_t(down), in_positions)
        self.out_positions = _collect(cols, table)
        self.local_out = _map(self.out_positions, cols, table)

    def forward(self, x):
        """``x[..., |in|] -> y[..., |out|]`` (contribution of this support)."""
        values = self.values.to(x.real.dtype)
        y = x.new_zeros(*x.shape[:-1], self.out_positions.numel())
        y.index_add_(-1, self.local_out, x[..., self.local_in] * values)
        return y

    def adjoint(self, gradient):
        """``[..., |out|] -> [..., |in|]``, exact on ``in_positions``."""
        values = self.values.to(gradient.real.dtype)
        x = gradient.new_zeros(*gradient.shape[:-1], self.in_positions.numel())
        x.index_add_(-1, self.local_in, gradient[..., self.local_out] * values)
        return x
