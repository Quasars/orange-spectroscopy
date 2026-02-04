import unittest

import numpy as np

from Orange.data import Table
from orangecontrib.spectroscopy.preprocess.als import ALSP, ARPLS, AIRPLS
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALLER_COLLAGEN,
)


class TestAls(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [
        ALSP(lam=100e6, itermax=5, p=0.5),
        ARPLS(lam=100e5, itermax=5, ratio=0.5),
        AIRPLS(lam=100, itermax=5, porder=1),
    ]
    data = SMALLER_COLLAGEN

    def test_als_Basic(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 10.0, 5.0], [3.0, 5.0, 9.0, 4.0]])
        method = ALSP(lam=100e6, p=0.5, itermax=100)
        process = method(data)
        check = Table.from_numpy(
            None, [[-0.5, -1.5, 4.5, -2.5], [-1.2, 0.1, 3.4, -2.3]]
        )
        process = np.array(process)
        np.testing.assert_almost_equal(check, process, 2)

    def test_arpls_basic(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 10.0, 5.0], [3.0, 5.0, 9.0, 4.0]])
        method = ARPLS(lam=100e5, ratio=0.5, itermax=100)
        process = method(data)
        check = Table.from_numpy(
            None, [[-0.5, -1.5, 4.5, -2.5], [-1.2, 0.0999999, 3.4, -2.3]]
        )
        np.testing.assert_almost_equal(np.array(check), np.array(process))

    def test_airpls_basic(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 10.0, 5.0], [3.0, 5.0, 9.0, 4.0]])
        method = AIRPLS(lam=100, itermax=10, porder=1)
        process = method(data)
        check = Table.from_numpy(
            None,
            [[-1.15248, -0.155994, 7.83538, 2.82675], [-0.499999, 1.5, 5.5, 0.499999]],
        )
        np.testing.assert_almost_equal(np.array(check), np.array(process), 2)
