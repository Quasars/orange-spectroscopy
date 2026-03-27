import numpy as np
import Orange
import unittest
from orangecontrib.spectroscopy.preprocess import Cut
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALL_COLLAGEN,
)


class TestCut(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [Cut(lowlim=1000, highlim=1800)]
    data = SMALL_COLLAGEN

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")
        cls.collagen = Orange.data.Table("collagen")

    def test_cut_both(self):
        d = self.collagen
        dcut = Cut(lowlim=0, highlim=2)(d)
        self.assertEqual(len(getx(dcut)), 0)
        dcut = Cut(lowlim=1000, highlim=1100)(d)
        self.assertGreaterEqual(min(getx(dcut)), 1000)
        self.assertLessEqual(max(getx(dcut)), 1100)

    def test_cut_single(self):
        d = self.collagen
        dcut = Cut(lowlim=1000)(d)
        self.assertGreaterEqual(min(getx(dcut)), 1000)
        self.assertEqual(max(getx(dcut)), max(getx(d)))
        dcut = Cut(highlim=1000)(d)
        self.assertLessEqual(max(getx(dcut)), 1000)
        self.assertEqual(min(getx(dcut)), min(getx(d)))

    def test_cut_both_inverse(self):
        d = self.collagen
        # cutting out of x interval - need all
        dcut = Cut(lowlim=0, highlim=2, inverse=True)(d)
        np.testing.assert_equal(getx(dcut), getx(d))
        # cutting in the middle - edged are the same
        dcut = Cut(lowlim=1000, highlim=1100, inverse=True)(d)
        dcutx = getx(dcut)
        self.assertEqual(min(dcutx), min(getx(d)))
        self.assertEqual(max(dcutx), max(getx(d)))
        self.assertLess(len(dcutx), len(getx(d)))
        np.testing.assert_equal(np.where(dcutx < 1100), np.where(dcutx < 1000))
        np.testing.assert_equal(np.where(dcutx > 1100), np.where(dcutx > 1000))

    def test_cut_single_inverse(self):
        d = self.collagen
        dcut = Cut(lowlim=1000, inverse=True)(d)
        self.assertLessEqual(max(getx(dcut)), 1000)
        self.assertEqual(min(getx(dcut)), min(getx(d)))
        dcut = Cut(highlim=1000, inverse=True)(d)
        self.assertGreaterEqual(min(getx(dcut)), 1000)
        self.assertEqual(max(getx(dcut)), max(getx(d)))
