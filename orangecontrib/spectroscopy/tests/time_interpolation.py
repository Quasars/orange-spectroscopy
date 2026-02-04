import time

import numpy as np
from Orange.data import Table

from orangecontrib.spectroscopy.tests.bigdata import dust, spectra20nea
from orangecontrib.spectroscopy.preprocess import (
    Interpolate,
    interp1d_with_unknowns_numpy,
    interp1d_with_unknowns_scipy,
)
from orangecontrib.spectroscopy.data import getx


def test_time():
    fns = ["collagen", dust(), spectra20nea(), "peach_juice.dpt"]
    for fn in fns:
        print(fn)
        data = Table(fn)
        print(data.X.shape)
        data[0, 2] = np.nan
        t = time.time()
        interpolated = Interpolate(getx(data), handle_nans=False)(data)
        print("no nan", time.time() - t)
        t = time.time()
        intp = Interpolate(getx(data), handle_nans=True)
        intp.interpfn = interp1d_with_unknowns_numpy
        interpolated = intp(data)
        print("nan handling with numpy", time.time() - t)
        intp.interpfn = interp1d_with_unknowns_scipy
        interpolated = intp(data)
        print("nan handling with scipy", time.time() - t)
        assert not np.any(np.isnan(interpolated.X))


if __name__ == "__main__":
    test_time()
