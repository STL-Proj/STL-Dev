"""Small, executable example: python -m examples.healpix_pyramid_synthesis."""

import torch

from STL_main.LossOptim import PyramidOptimizer, PyramidSynthesis
from STL_main.Pyramid import HealpixPyramid
from STL_main.STL_Healpix_Kernel_Torch import STL_Healpix_Kernel_Torch


def main():
    torch.manual_seed(12)
    data = STL_Healpix_Kernel_Torch(torch.randn(12 * 16**2, dtype=torch.float64))
    st = data.get_ST_op(
        J=3,
        L=2,
        norm="vanilla",
        compute_PS=False,
        wavelet_op_kwargs=dict(ellipsoid="sphere", pyramid_pixel_batch_size=4096),
    )
    target = st.apply_pyramid(HealpixPyramid.from_map(data, st.wavelet_op))
    solver = PyramidSynthesis(
        st,
        target,
        optimizer=PyramidOptimizer(lr=[0.02, 0.01, 0.005]),
        coefficient_batch_size=16,
    )
    pyramid = solver.initialize_from_noise(data, seed=13)
    history = solver.run(pyramid, niter=20)
    print(f"Loss before first/last step: {history[0]:.6g} / {history[-1]:.6g}")
    result = solver.reconstruct()  # [batch=1, channel=1, pixels]
    print(f"Reconstructed map: {tuple(result.shape)}")


if __name__ == "__main__":
    main()
