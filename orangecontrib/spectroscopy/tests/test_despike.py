import unittest
import numpy as np
from Orange.data import Table
from orangecontrib.spectroscopy.preprocess import Despike
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALL_COLLAGEN,
)


class TestSpikeRemoval(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [Despike(threshold=5, cutoff=60, dis=5)]
    data = SMALL_COLLAGEN

    def test_spikes(self):
        data = Table.from_numpy(
            None,
            [
                [
                    1000,
                    1,
                    1,
                    1,
                    1,
                    10,
                    1,
                    1,
                    1000,
                    1000,
                    1000,
                    1,
                    1000,
                    1,
                    1,
                    1,
                    1000,
                    1000,
                    1000,
                    1000,
                ],
                [
                    1000,
                    1,
                    1,
                    1,
                    1,
                    10,
                    1,
                    1,
                    1000,
                    1000,
                    1000,
                    1,
                    1000,
                    1,
                    1,
                    1,
                    1000,
                    1000,
                    1000,
                    1000,
                ],
                [
                    1000,
                    1000,
                    2,
                    1,
                    1,
                    10,
                    1,
                    2,
                    1000,
                    1000,
                    1000,
                    1,
                    1000,
                    1,
                    1,
                    1,
                    3,
                    1000,
                    1000,
                    1000,
                ],
            ],
        )
        method = Despike(threshold=7, cutoff=100, dis=1)
        process = method(data).X
        check = Table.from_numpy(
            None,
            (
                [1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 1, 1, 10, 1, 2, 2, 1.5, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3],
            ),
        )
        np.testing.assert_almost_equal(check, process)

    def test_nospike(self):
        data = Table.from_numpy(None, [[1, 2, 10, 15], [2, 3, 6, 10]])
        method = Despike(threshold=7, cutoff=100, dis=5)
        changed = method(data)
        check = np.array(data)
        np.testing.assert_almost_equal(changed, check)
