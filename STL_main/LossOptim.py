#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA

Still WIP, I try in particular to identify the minimum parameters which are
necessary.

I ork only with single channel maps here, need to be extended to multi-channel
ones.
"""

import math

import torch

from STL_main.Pyramid import HealpixPyramid
from STL_main.PyramidScattering import PyramidStatistics


class SquaredStatisticsLoss:
    """Sum w * abs(S-target)**2; weights are fixed, real and nonnegative.

    For fixed reference normalization use weights = 1 / reference_scale**2.
    Complex statistics keep both real and imaginary constraints.
    """

    def __init__(self, weights=1.0):
        self.weights = weights

    def _terms(self, stats, target):
        if stats.plan != target.plan or stats.values.shape != target.values.shape:
            raise ValueError(
                "Statistics and target must have identical plans and shapes"
            )
        residual = stats.values - target.values
        weights = torch.as_tensor(self.weights, device=residual.device)
        if (
            weights.is_complex()
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
        ):
            raise ValueError("Loss weights must be finite, real and nonnegative")
        weights = weights.to(residual.real.dtype)
        try:
            weights = torch.broadcast_to(weights, residual.shape)
        except RuntimeError as exc:
            raise ValueError(
                "Loss weights must broadcast to the statistics shape"
            ) from exc
        return residual, weights

    @torch.no_grad()
    def value(self, stats, target):
        residual, weights = self._terms(stats, target)
        return (weights * residual.abs().square()).sum()

    @torch.no_grad()
    def gradient(self, stats, target):
        residual, weights = self._terms(stats, target)
        return PyramidStatistics(stats.plan, 2 * weights * residual)


class PyramidOptimizer:
    """Explicit SGD/Adam with learning rates and optimizer state per band."""

    def __init__(self, lr=1e-2, method="adam", betas=(0.9, 0.999), eps=1e-8):
        if (
            method not in ("sgd", "adam")
            or not all(0 <= b < 1 for b in betas)
            or len(betas) != 2
            or eps <= 0
        ):
            raise ValueError("Invalid optimizer settings")
        self.lr, self.method, self.betas, self.eps = lr, method, betas, eps
        self.m, self.v, self.iteration = {}, {}, 0
        self._pyramid = None

    @torch.no_grad()
    def step(self, pyramid, gradients):
        if self._pyramid is not None and self._pyramid is not pyramid:
            raise ValueError("Create a separate optimizer for each pyramid")
        if len(gradients) != len(pyramid.levels):
            raise ValueError("One gradient per pyramid band is required")
        rates = [
            (
                self.lr(d)
                if callable(self.lr)
                else self.lr[d] if isinstance(self.lr, (list, tuple, dict)) else self.lr
            )
            for d in pyramid.list_dg
        ]
        for p, g, rate in zip(pyramid.levels, gradients, rates):
            if (
                p.shape != g.shape
                or p.dtype != g.dtype
                or p.device != g.device
                or not bool(torch.isfinite(g).all())
            ):
                raise ValueError(
                    "Gradient shape/dtype/device must match each band and be finite"
                )
            if not math.isfinite(float(rate)) or rate < 0:
                raise ValueError("Learning rates must be finite and nonnegative")
        self._pyramid = pyramid
        self.iteration += 1
        b1, b2 = self.betas
        for d, (p, g, rate) in enumerate(zip(pyramid.levels, gradients, rates)):
            if self.method == "sgd":
                p.add_(g, alpha=-float(rate))
            else:
                if d not in self.m:
                    self.m[d], self.v[d] = torch.zeros_like(p), torch.zeros_like(p)
                self.m[d].mul_(b1).add_(g, alpha=1 - b1)
                self.v[d].mul_(b2).addcmul_(g, g, value=1 - b2)
                m = self.m[d] / (1 - b1**self.iteration)
                v = self.v[d] / (1 - b2**self.iteration)
                p.addcdiv_(m, v.sqrt().add_(self.eps), value=-float(rate))


class PyramidSynthesis:
    """End-to-end explicit synthesis; target statistics are copied and detached."""

    def __init__(
        self,
        scat_operator,
        target_statistics,
        optimizer=None,
        loss=None,
        coefficient_batch_size=64,
        strategy="recompute",
    ):
        self.operator = scat_operator
        self.target = PyramidStatistics(
            tuple(target_statistics.plan), target_statistics.values.detach().clone()
        )
        if not bool(torch.isfinite(self.target.values).all()):
            raise ValueError("Target statistics must be finite")
        self.optimizer = optimizer if optimizer is not None else PyramidOptimizer()
        self.loss = loss if loss is not None else SquaredStatisticsLoss()
        self.options = dict(
            coefficient_batch_size=coefficient_batch_size, strategy=strategy
        )
        self.pyramid = None

    def initialize_from_noise(self, data_example, seed=0, amplitude=1.0):
        generator = torch.Generator(device=data_example.device).manual_seed(seed)
        noise = (
            torch.randn(
                data_example.array.shape,
                dtype=data_example.dtype,
                device=data_example.device,
                generator=generator,
            )
            * amplitude
        )
        self.pyramid = HealpixPyramid.from_map(
            data_example.new_like(noise), self.operator.wavelet_op
        )
        return self.pyramid

    @torch.no_grad()
    def step(self, pyramid):
        if type(self.loss) is SquaredStatisticsLoss:
            value, gradients = self.operator.loss_and_grad_pyramid(
                pyramid, self.target, weights=self.loss.weights, **self.options
            )
        else:
            stats = self.operator.apply_pyramid(
                pyramid, plan=self.target.plan, **self.options
            )
            value = self.loss.value(stats, self.target)
            gradients = self.operator.vjp_pyramid(
                pyramid, self.loss.gradient(stats, self.target), **self.options
            )
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Non-finite synthesis loss")
        self.optimizer.step(pyramid, gradients)
        self.pyramid = pyramid
        return float(value)

    def run(self, pyramid, niter=100, callback=None):
        if not isinstance(niter, int) or niter < 0:
            raise ValueError("niter must be a nonnegative integer")
        history = []
        for iteration in range(niter):
            history.append(self.step(pyramid))
            if callback is not None:
                callback(iteration, history[-1])
        return history

    @torch.no_grad()
    def reconstruct(self):
        if self.pyramid is None:
            raise ValueError("Initialize a pyramid first")
        return self.pyramid.reconstruct()


###############################################################################
###############################################################################
def synth_from_map(
    data_target,
    mode="1to1",
    N_new=None,
    J=None,
    L=None,
    WType=None,
    SC="ScatCov",
):
    """
    Perform a synthesis from a target map, or an ensemble of target map.

    The data_target needs to be a stl_data object, of shape (Nb,Nc,N).

    The synthes

    """

    return None


###############################################################################
###############################################################################
def synth_from_stats(st_target, mode="1to1", N_new=None):

    return None


###############################################################################
###############################################################################
