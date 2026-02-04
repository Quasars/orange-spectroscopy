import unittest

import numpy as np

from orangecontrib.spectroscopy.preprocess.npfunc import (
    Function,
    Constant,
    Identity,
    Segments,
    Sum,
)


class TestInflectionPointWeighting(unittest.TestCase):
    def test_constant(self):
        a = np.zeros((2, 3))
        constant = Constant(3)
        r = constant(a)
        np.testing.assert_equal(r, 3)
        np.testing.assert_equal(r.shape, (2, 3))

    def test_cond(self):
        x = np.arange(6)
        segm = Segments(
            (lambda x: x < 3, Identity()), (lambda x: x >= 3, Function(lambda x: -x))
        )
        np.testing.assert_equal(segm(x), [0, 1, 2, -3, -4, -5])

        segm = Segments(
            (lambda x: True, Identity()), (lambda x: x >= 3, Function(lambda x: -x))
        )
        np.testing.assert_equal(segm(x), [0, 1, 2, -3, -4, -5])

        segm = Segments(
            (lambda x: True, Identity()), (lambda x: True, Function(lambda x: -x))
        )
        np.testing.assert_equal(segm(x), [0, -1, -2, -3, -4, -5])

    def test_sum(self):
        x = np.arange(6)
        r = Sum(Constant(3), lambda x: x)(x)
        np.testing.assert_equal(r, [3, 4, 5, 6, 7, 8])
