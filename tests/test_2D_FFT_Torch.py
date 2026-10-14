import sys
from pathlib import Path

import numpy as np
import torch

# set parent directory to sys.path for imports (if executed directly)
NOTEBOOK_DIR = Path.cwd().resolve()
PARENT_DIR = NOTEBOOK_DIR.parents[0]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
    print(f"Parent directory added to sys.path: ...\\{PARENT_DIR.name}")
else:
    print(f"Parent directory already in sys.path: ...\\{PARENT_DIR.name}")
DATA_TEST_PATH = Path(__file__).parent.parent / "data" / "test"

import STL_main
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass


def test_DataClass_mean():

    # test DataClass instantiation over basic data
    data = DataClass(array=np.load(DATA_TEST_PATH / "Turb_6.npy")[0])

    # test wavelet operator building
    wavelet_op = data.get_wavelet_op()

    # test mean method
    assert wavelet_op.mean(data).item() == torch.mean(data.array).item()


def test_DataClass_cov():

    # test DataClass instanciation over basic data
    data = DataClass(array=np.load(DATA_TEST_PATH / "Turb_6.npy")[0])
    data.array -= torch.mean(data.array).item()

    # test wavelet operator building
    wavelet_op = data.get_wavelet_op()

    # test mean method
    assert torch.allclose(
        wavelet_op.cov(data, data),
        torch.var(data.array),
        rtol=1e-3,
    )


def test_DataClass_downsample():

    # test DataClass instantiation over data
    data = DataClass(array=np.load(DATA_TEST_PATH / "Turb_6.npy"), pbc=True)

    # test wavelet operator building
    wavelet_op = data.get_wavelet_op(J=2, L=4)

    # test downsample
    dg_out = 3
    threshold = 3e-2  # 3% error allowed over 20 maps of size 256x256 in Turb_6.npy
    data_downsampled = wavelet_op.downsample(
        data, dg_out, inplace=False, target_fourier_status=False
    )
    diff = np.asarray(
        (data_downsampled.array - data.array[..., :: 2**dg_out, :: 2**dg_out]).cpu()
    )
    assert np.all(
        np.abs(diff) < threshold * np.abs(np.asarray(data_downsampled.array.cpu()))
    )


def test_CS_op_whiten():

    from STL_main.Synthesis import apply_nyquist_filter

    # Nyquist-filtered periodic data (corner modes are not covered by P_k)
    array = torch.tensor(np.load(DATA_TEST_PATH / "Turb_6.npy")[:4, :128, :128])
    data = DataClass(array=apply_nyquist_filter(array)[:, None], pbc=True)
    CS_op = data.get_CS_op()

    whitened, pk = CS_op.whiten(data, mean_field=False)
    assert pk.shape == (4, 1, CS_op.N_max + 1)

    # whitened power spectrum is flat on each radial shell (k=0 excluded)
    fft = torch.fft.rfft2(whitened.array, norm="ortho")
    assert torch.allclose(CS_op._cross_pk(fft, fft)[..., 1:], torch.ones(1).double())

    # whiten / unwhiten round trip
    colored = CS_op.unwhiten(whitened, pk)
    assert torch.allclose(colored.array, data.array)

    # P_k interpolation onto the radial grid of an operator of another shape
    CS_op_64 = DataClass(array=torch.zeros(64, 64), pbc=True).get_CS_op()
    pk_64 = CS_op_64._interp_pk(pk)
    assert pk_64.shape == (4, 1, CS_op_64.N_max + 1)
    assert torch.allclose(pk_64[..., 1:], pk[..., 2::2])


if __name__ == "__main__":
    test_DataClass_mean()
    test_DataClass_cov()
    test_DataClass_downsample()
    test_CS_op_whiten()
