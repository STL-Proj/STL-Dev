from pathlib import Path

import numpy as np

DATA_TEST_PATH = Path(__file__).parent.parent / "data" / "test"

from STL_main.STL_2D_FFT_Torch import (
    CrappyWavelateOperator2D_FFT_torch as WaveletOperator,
)
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass


def test_DataClass_mean():
    assert DataClass(array=np.array([[1, 2], [3, 4]], dtype=np.float64)).mean() == 2.5


def test_DataClass_findN():
    assert DataClass(
        array=np.array([[1, 2, 2], [3, 4, -0.5]], dtype=np.float64)
    ).findN() == (2, 3)


def test_WaveletOperator():

    # test DataClass instanciation over data
    data = DataClass(array=np.load(DATA_TEST_PATH / "Turb_6.npy"))

    # test wavelet operator building
    wavelet_op = data.get_wavelet_op(J=2, L=4, WType="Crappy")

    # test downsample
    dg_out = 3
    threshold = 3e-2  # 3% error allowed over 20 maps of size 256x256 in Turb_6.npy
    data_downsampled = wavelet_op.downsample(
        data, dg_out, inplace=False, target_fourier_status=False
    )
    diff = np.asarray(
        data_downsampled.array - data.array[..., :: 2**dg_out, :: 2**dg_out]
    )
    assert np.all(np.abs(diff) < threshold * np.abs(np.asarray(data_downsampled.array)))
