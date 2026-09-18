"""Explicit Euclidean VJPs (complex gradients use Re <g, dx>)."""

import torch


def modulus_vjp(z, gradient):
    # Exactly the derivative of abs, including its zero subgradient. A fixed
    # epsilon clamp would change the derivative for small nonzero inputs.
    magnitude = z.abs()
    return gradient * z / torch.where(magnitude == 0, 1, magnitude)


def mean_vjp(x, gradient):
    return gradient[..., None].expand_as(x) / x.shape[-1]


def power_mean_vjp(x, gradient):
    return 2 * gradient.real[..., None] * x / x.shape[-1]


def covariance(a, b):
    """STL's uncentered covariance, mean(a * conj(b))."""
    return (a * b.conj()).mean(-1)


def covariance_vjp(a, b, gradient):
    n = a.shape[-1]
    return gradient[..., None] * b / n, gradient.conj()[..., None] * a / n


def root_modulus_vjp(z, gradient, epsilon=1e-8):
    """VJP of z / sqrt(abs(z) + epsilon), used by cross-channel S1."""
    r = z.abs()
    radial = (gradient.conj() * z).real
    return gradient / (r + epsilon).sqrt() - modulus_vjp(
        z, 0.5 * radial / (r + epsilon).pow(1.5)
    )


def _check_convolution_geometry(conv):
    required = (
        "_pos_safe",
        "_w_norm",
        "_sort_order",
        "_inv_order",
        "weight",
        "G",
        "K",
        "P",
    )
    if not all(hasattr(conv, name) for name in required):
        raise NotImplementedError("Unsupported healpix-analyse convolution geometry")
    expected = (4, conv.G, conv.K * conv.P)
    if tuple(conv._pos_safe.shape) != expected or tuple(conv._w_norm.shape) != expected:
        raise NotImplementedError(
            "Unsupported healpix-analyse interpolation layout; expected [4,G,K*P]"
        )
    if conv.use_norm:
        raise NotImplementedError("The explicit engine requires a linear convolution")


def convolution_forward(conv, x, pixel_batch_size=16384):
    """Identical linear HealPixConv, gathering one output-pixel block at a time."""
    _check_convolution_geometry(conv)
    if pixel_batch_size < 1:
        raise ValueError("pixel_batch_size must be positive")
    batch, channels, k = x.shape
    if k != conv.K or channels != conv.in_channels:
        raise ValueError("Convolution input shape does not match cached geometry")
    x = x.to(dtype=conv.dtype)
    sorted_x = x.index_select(-1, conv._sort_order)
    positions = conv._pos_safe.reshape(4, conv.G, k, conv.P)
    interp = conv._w_norm.reshape(4, conv.G, k, conv.P).to(x)
    weights = conv.weight.to(x)
    output = x.new_empty(batch, conv.G, conv.out_channels, k)
    for start in range(0, k, pixel_batch_size):
        stop = min(start + pixel_batch_size, k)
        for gauge in range(conv.G):
            gathered = x.new_zeros(batch, channels, stop - start, conv.P)
            for neighbour in range(4):
                idx = positions[neighbour, gauge, start:stop].reshape(-1)
                values = sorted_x.index_select(-1, idx).reshape_as(gathered)
                gathered.add_(values * interp[neighbour, gauge, start:stop])
            output[:, gauge, :, start:stop] = torch.einsum(
                "bckp,cop->bok", gathered, weights[gauge]
            )
    output = output.reshape(batch, conv.G * conv.out_channels, k)
    output.add_(conv.bias.to(x).reshape(1, -1, 1))
    return output.index_select(-1, conv._inv_order)


def convolution_adjoint(conv, gradient, pixel_batch_size=16384):
    """Transpose HealPixConv's gather/interpolation/kernel contraction.

    Uses the same cached geometry as the forward. Work is bounded by a pixel
    block, without constructing a dense matrix or a B*G*K*P temporary.
    The private geometry contract is checked rather than guessed.
    """
    _check_convolution_geometry(conv)
    if pixel_batch_size < 1:
        raise ValueError("pixel_batch_size must be positive")
    batch = gradient.shape[0]
    k, p, gauges = conv.K, conv.P, conv.G
    g = gradient.reshape(batch, gauges, conv.out_channels, k)
    g = g.index_select(-1, conv._sort_order)
    weights = conv.weight.to(g)
    positions = conv._pos_safe.reshape(4, gauges, k, p)
    interp = conv._w_norm.reshape(4, gauges, k, p).to(g)
    out = g.new_zeros(batch, conv.in_channels, k)
    for start in range(0, k, pixel_batch_size):
        stop = min(start + pixel_batch_size, k)
        for gauge in range(gauges):
            taps = torch.einsum(
                "bok,cop->bckp", g[:, gauge, :, start:stop], weights[gauge]
            )
            for neighbour in range(4):
                idx = positions[neighbour, gauge, start:stop].reshape(-1)
                values = taps * interp[neighbour, gauge, start:stop]
                out.scatter_add_(
                    2,
                    idx.expand(batch, conv.in_channels, -1),
                    values.reshape(batch, conv.in_channels, -1),
                )
    return out.index_select(-1, conv._inv_order)


def downsampling_adjoint(down, gradient):
    """Transpose the actual anti-aliasing matrix, not child replication."""
    if down.mode != "smooth" or not hasattr(down, "_M_indices"):
        raise NotImplementedError(
            "Explicit downsampling requires the smooth sparse operator"
        )
    matrix = torch.sparse_coo_tensor(
        down._M_indices,
        down._M_values.to(gradient.real),
        down._M_size,
        device=gradient.device,
    ).transpose(0, 1)
    leading = gradient.shape[:-1]
    flat = gradient.reshape(-1, gradient.shape[-1])
    if flat.is_complex():
        result = torch.complex(
            torch.sparse.mm(matrix, flat.real.T).T,
            torch.sparse.mm(matrix, flat.imag.T).T,
        )
    else:
        result = torch.sparse.mm(matrix, flat.T).T
    return result.reshape(*leading, down.N_in)
