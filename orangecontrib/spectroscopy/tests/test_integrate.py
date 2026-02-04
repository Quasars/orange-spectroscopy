import unittest

from Orange.data import Table
import numpy as np

from orangecontrib.spectroscopy.preprocess import Integrate
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALL_COLLAGEN,
)


class TestIntegrate(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [
        Integrate(limits=[[900, 100], [1100, 1200], [1200, 1300]]),
        Integrate(methods=Integrate.Simple, limits=[[1100, 1200]]),
        Integrate(methods=Integrate.Baseline, limits=[[1100, 1200]]),
        Integrate(methods=Integrate.PeakMax, limits=[[1100, 1200]]),
        Integrate(methods=Integrate.PeakBaseline, limits=[[1100, 1200]]),
        Integrate(methods=Integrate.PeakAt, limits=[[1100]]),
        Integrate(methods=Integrate.PeakX, limits=[[1100, 1200]]),
        Integrate(methods=Integrate.PeakXBaseline, limits=[[1100, 1200]]),
    ]
    data = SMALL_COLLAGEN

    def test_simple(self):
        data = Table.from_numpy(
            None, [[1, 2, 3, 1, 1, 1], [1, 2, 3, 1, np.nan, 1], [1, 2, 3, 1, 1, np.nan]]
        )
        i = Integrate(methods=Integrate.Simple, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 8)
        self.assertEqual(i[1][0], 8)
        self.assertEqual(i[2][0], 8)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 0)

    def test_baseline(self):
        data = Table.from_numpy(
            None, [[1, 2, 3, 1, 1, 1], [1, 2, 3, 1, np.nan, 1], [1, 2, 3, 1, 1, np.nan]]
        )
        i = Integrate(methods=Integrate.Baseline, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 3)
        self.assertEqual(i[1][0], 3)
        self.assertEqual(i[2][0], 3)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 1)

    def test_peakmax(self):
        d1 = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        d2 = Table.from_numpy(None, [[1, 2, 3, np.nan, 1, 1]])
        for data in d1, d2:
            i = Integrate(methods=Integrate.PeakMax, limits=[[0, 5]])(data)
            self.assertEqual(i[0][0], 3)
            np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 0)

    def test_peakbaseline(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(methods=Integrate.PeakBaseline, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 2)
        np.testing.assert_equal(
            i.domain[0].compute_value.baseline(data)[1], [[1, 1, 1, 1, 1, 1]]
        )

    def test_peakat(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(methods=Integrate.PeakAt, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 1)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 0)
        i = Integrate(methods=Integrate.PeakAt, limits=[[1.4, None]])(data)
        self.assertEqual(i[0][0], 2)
        i = Integrate(methods=Integrate.PeakAt, limits=[[1.6, None]])(data)
        self.assertEqual(i[0][0], 3)

    def test_peakx(self):
        d1 = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        d2 = Table.from_numpy(None, [[1, 2, 3, np.nan, 1, 1]])
        for data in d1, d2:
            i = Integrate(methods=Integrate.PeakX, limits=[[0, 5]])(data)
            self.assertEqual(i[0][0], 2)
            np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 0)

    def test_peakxbaseline(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(methods=Integrate.PeakXBaseline, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 2)
        np.testing.assert_equal(
            i.domain[0].compute_value.baseline(data)[1], [[1, 1, 1, 1, 1, 1]]
        )

    def test_separate_baseline(self):
        data = Table.from_numpy(
            None, [[1, 2, 3, 1, 1, 1], [1, 2, 3, 1, np.nan, 1], [1, 2, 3, 1, 1, np.nan]]
        )

        # baseline spans the whole region
        i = Integrate(methods=Integrate.Separate, limits=[[0, 5, 0, 5]])(data)
        self.assertEqual(i[0][0], 3)
        self.assertEqual(i[1][0], 3)
        self.assertEqual(i[2][0], 3)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 1)

        # baseline is outside of integral
        i = Integrate(methods=Integrate.Separate, limits=[[1, 2, 0, 5]])(data)
        self.assertEqual(i[0][0], 1.5)
        self.assertEqual(i[1][0], 1.5)
        self.assertEqual(i[2][0], 1.5)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 1)

        # baseline is inside the interval
        i = Integrate(methods=Integrate.Separate, limits=[[0, 3, 1, 2]])(data)
        self.assertEqual(i[0][0], -1.5)
        self.assertEqual(i[1][0], -1.5)
        self.assertEqual(i[2][0], -1.5)
        bx, by = i.domain[0].compute_value.baseline(data)
        # baseline is computed on common parts: [min(limits), max(limits)]
        np.testing.assert_equal(bx, [0, 1, 2, 3])
        np.testing.assert_equal(by, [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    def test_empty_interval(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(methods=Integrate.Simple, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 0)
        i = Integrate(methods=Integrate.Baseline, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 0)
        i = Integrate(methods=Integrate.PeakMax, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], np.nan)
        i = Integrate(methods=Integrate.PeakBaseline, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], np.nan)
        i = Integrate(methods=Integrate.PeakAt, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 1)  # get the rightmost one
        i = Integrate(methods=Integrate.Separate, limits=[[10, 16, 10, 16]])(data)
        self.assertEqual(i[0][0], 0)

    def test_different_integrals(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(
            methods=[Integrate.Simple, Integrate.Baseline], limits=[[0, 5], [0, 5]]
        )(data)
        self.assertEqual(i[0][0], 8)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 0)
        np.testing.assert_equal(i.domain[1].compute_value.baseline(data)[1], 1)

    def test_names(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(
            methods=[Integrate.Simple, Integrate.Baseline, Integrate.Separate],
            limits=[[0, 5], [0, 6], [1, 2, 0, 6]],
        )(data)
        self.assertEqual(i.domain[0].name, "0 - 5")
        self.assertEqual(i.domain[1].name, "0 - 6")
        self.assertEqual(i.domain[2].name, "1 - 2 [baseline 0 - 6]")
        i = Integrate(
            methods=[Integrate.Simple, Integrate.Baseline],
            limits=[[0, 5], [0, 6]],
            names=["simple", "baseline"],
        )(data)
        self.assertEqual(i.domain[0].name, "simple")
        self.assertEqual(i.domain[1].name, "baseline")

    def test_repeated(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(
            methods=[Integrate.Simple, Integrate.Baseline],
            limits=[[0, 5], [0, 6]],
            names=["int", "int"],
        )(data)
        self.assertEqual(i.domain[0].name, "int")
        self.assertEqual(i.domain[1].name, "int (1)")

    def test_metas_output(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i = Integrate(
            methods=[Integrate.Simple, Integrate.Baseline],
            limits=[[0, 5], [0, 6]],
            metas=True,
        )(data)
        metavars = [a.name for a in i.domain.metas]
        self.assertTrue("0 - 5" in metavars and "0 - 6" in metavars)
        self.assertEqual(i[0]["0 - 5"], 8)
        self.assertEqual(i[0]["0 - 6"], 3)

    def test_eq(self):
        data = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1]])
        i1 = Integrate(methods=Integrate.Simple, limits=[[0, 5]])(data)
        i2 = Integrate(methods=Integrate.Simple, limits=[[0, 6]])(data)
        self.assertNotEqual(i1.domain[0], i2.domain[0])
        i3 = Integrate(methods=Integrate.Simple, limits=[[0, 6], [0, 5]])(data)
        self.assertNotEqual(i1.domain[0], i3.domain[0])
        self.assertEqual(i1.domain[0], i3.domain[1])
        i4 = Integrate(methods=Integrate.Baseline, limits=[[0, 5]])(data)
        self.assertNotEqual(i1.domain[0], i4.domain[0])

        # different domain should mean different integrals
        data2 = Table.from_numpy(None, [[1, 2, 3, 1, 1, 1, 1]])
        ii1 = Integrate(methods=Integrate.Simple, limits=[[0, 5]])(data2)
        self.assertNotEqual(i1.domain[0], ii1.domain[0])

        # same domain -> same integrals
        data3 = Table.from_numpy(None, [[1, 2, 3, 22, 1, 2]])
        iii1 = Integrate(methods=Integrate.Simple, limits=[[0, 5]])(data3)
        self.assertEqual(i1.domain[0], iii1.domain[0])
