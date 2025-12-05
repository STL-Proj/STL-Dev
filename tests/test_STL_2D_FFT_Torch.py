import numpy as np

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
