import time

import numpy as np
from Orange.data import Table

from orangecontrib.spectroscopy.preprocess import Normalize
from orangecontrib.spectroscopy.tests.bigdata import dust, spectra20nea


def test_normalization_vector():
    fns = ["collagen", dust(), spectra20nea(), "peach_juice.dpt"]
    for fn in fns:
        print(fn)
        data = Table(fn)
        p = Normalize(method=Normalize.Vector)
        print(data.X.shape)
        t = time.time()
        r = p(data)
        print("no interpolate", time.time() - t)
        data[0, 2] = np.nan
        t = time.time()
        r = p(data)
        print("with interpolate", time.time() - t)
        assert np.all(np.argwhere(np.isnan(r.X)) == [[0, 2]])


if __name__ == "__main__":
    test_normalization_vector()
