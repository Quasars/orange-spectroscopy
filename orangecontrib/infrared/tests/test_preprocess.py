import unittest

import numpy as np
import random
import Orange
from orangecontrib.infrared.data import getx
from orangecontrib.infrared.preprocess import Absorbance, Transmittance, \
    Integrate, Interpolate, Cut, SavitzkyGolayFiltering, \
    GaussianSmoothing, PCADenoising, RubberbandBaseline, \
    Normalize, Average

# Preprocessors that work per sample and should return the same
# result for a sample independent of the other samples
PREPROCESSORS_INDEPENDENT_SAMPLES = [
    Interpolate(np.linspace(1000, 1700, 100)),
    SavitzkyGolayFiltering(window=9, polyorder=2, deriv=2),
    Cut(lowlim=1000, highlim=1800),
    GaussianSmoothing(sd=3.),
    Absorbance(),
    Transmittance(),
    Integrate(limits=[[900, 100], [1100, 1200], [1200, 1300]]),
    RubberbandBaseline(),
    Normalize(method=Normalize.Vector),
    Normalize(method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=10000),
]

# Preprocessors that use groups of input samples to infer
# internal parameters.
PREPROCESSORS_GROUPS_OF_SAMPLES = [
    PCADenoising(components=2),
    Average(),
]

PREPROCESSORS = PREPROCESSORS_INDEPENDENT_SAMPLES + PREPROCESSORS_GROUPS_OF_SAMPLES


def shuffle_attr(data):
    natts = list(data.domain.attributes)
    random.Random(0).shuffle(natts)
    ndomain = Orange.data.Domain(natts, data.domain.class_vars,
                             metas=data.domain.metas)
    return Orange.data.Table(ndomain, data)


def reverse_attr(data):
    natts = reversed(data.domain.attributes)
    ndomain = Orange.data.Domain(natts, data.domain.class_vars,
                             metas=data.domain.metas)
    return Orange.data.Table(ndomain, data)


class TestTransmittance(unittest.TestCase):

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = Orange.data.Table("collagen.csv")
        transmittance = Transmittance()(data)
        nt = Orange.data.Table.from_table(transmittance.domain, data)
        self.assertEqual(transmittance.domain, nt.domain)
        np.testing.assert_equal(transmittance.X, nt.X)
        np.testing.assert_equal(transmittance.Y, nt.Y)

    def test_roundtrip(self):
        """Test AB -> TR -> AB calculation"""
        data = Orange.data.Table("collagen.csv")
        calcdata = Absorbance()(Transmittance()(data))
        np.testing.assert_allclose(data.X, calcdata.X)


class TestAbsorbance(unittest.TestCase):

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = Transmittance()(Orange.data.Table("collagen.csv"))
        absorbance = Absorbance()(data)
        nt = Orange.data.Table.from_table(absorbance.domain, data)
        self.assertEqual(absorbance.domain, nt.domain)
        np.testing.assert_equal(absorbance.X, nt.X)
        np.testing.assert_equal(absorbance.Y, nt.Y)

    def test_roundtrip(self):
        """Test TR -> AB -> TR calculation"""
        # actually AB -> TR -> AB -> TR
        data = Transmittance()(Orange.data.Table("collagen.csv"))
        calcdata = Transmittance()(Absorbance()(data))
        np.testing.assert_allclose(data.X, calcdata.X)


class TestSavitzkyGolay(unittest.TestCase):

    def test_unknown_no_propagate(self):
        data = Orange.data.Table("iris")
        f = SavitzkyGolayFiltering()
        data = data[:5]
        for i in range(4):
            data.X[i, i] = np.nan
        data.X[4] = np.nan
        fdata = f(data)
        np.testing.assert_equal(np.sum(np.isnan(fdata.X), axis=1), [1, 1, 1, 1, 4])

    def test_simple(self):
        data = Orange.data.Table("iris")
        f = SavitzkyGolayFiltering()
        data = data[:1]
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
            [[4.86857143, 3.47428571, 1.49428571, 0.32857143]])


class TestGaussian(unittest.TestCase):

    def test_unknown_no_propagate(self):
        data = Orange.data.Table("iris")
        f = GaussianSmoothing()
        data = data[:5]
        for i in range(4):
            data.X[i, i] = np.nan
        data.X[4] = np.nan
        fdata = f(data)
        np.testing.assert_equal(np.sum(np.isnan(fdata.X), axis=1), [1, 1, 1, 1, 4])

    def test_simple(self):
        data = Orange.data.Table("iris")
        f = GaussianSmoothing(sd=1.)
        data = data[:1]
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
            [[4.4907066, 3.2794677, 1.7641664, 0.6909083]])


class TestIntegrate(unittest.TestCase):

    def test_simple(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1], [1, 2, 3, 1, np.nan, 1]])
        i = Integrate(method=Integrate.Simple, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 8)
        self.assertEqual(i[1][0], 8)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 0)

    def test_baseline(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1], [1, 2, 3, 1, np.nan, 1]])
        i = Integrate(method=Integrate.Baseline, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 3)
        self.assertEqual(i[1][0], 3)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 1)

    def test_peakmax(self):
        d1 = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        d2 = Orange.data.Table([[1, 2, 3, np.nan, 1, 1]])
        for data in d1, d2:
            i = Integrate(method=Integrate.PeakMax, limits=[[0, 5]])(data)
            self.assertEqual(i[0][0], 3)
            np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1], 0)

    def test_peakbaseline(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.PeakBaseline, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 2)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1],
                                [[1, 1, 1, 1, 1, 1]])

    def test_peakat(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.PeakAt, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 1)
        np.testing.assert_equal(i.domain[0].compute_value.baseline(data)[1],
                                [[0]])
        i = Integrate(method=Integrate.PeakAt, limits=[[1.4, None]])(data)
        self.assertEqual(i[0][0], 2)
        i = Integrate(method=Integrate.PeakAt, limits=[[1.6, None]])(data)
        self.assertEqual(i[0][0], 3)

    def test_empty_interval(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.Simple, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 0)
        i = Integrate(method=Integrate.Baseline, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 0)
        i = Integrate(method=Integrate.PeakMax, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], np.nan)
        i = Integrate(method=Integrate.PeakBaseline, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], np.nan)
        i = Integrate(method=Integrate.PeakAt, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 1)  # get the rightmost one


class TestRubberbandBaseline(unittest.TestCase):

    def test_whole(self):
        """ Every point belongs in the convex region. """
        data = Orange.data.Table([[2, 1, 2]])
        i = RubberbandBaseline()(data)
        np.testing.assert_equal(i.X, 0)
        data = Orange.data.Table([[1, 2, 1]])
        i = RubberbandBaseline(peak_dir=RubberbandBaseline.PeakNegative)(data)
        np.testing.assert_equal(i.X, 0)

    def test_simple(self):
        """ Just one point is not in the convex region. """
        data = Orange.data.Table([[1, 2, 1, 1]])
        i = RubberbandBaseline()(data)
        np.testing.assert_equal(i.X, [[0, 1, 0, 0]])
        data = Orange.data.Table([[1, 2, 1, 1]])
        i = RubberbandBaseline(peak_dir=RubberbandBaseline.PeakNegative)(data)
        np.testing.assert_equal(i.X, [[0, 0, -0.5, 0]])


class TestNormalize(unittest.TestCase):

    def test_vector_norm(self):
        data = Orange.data.Table([[2, 1, 2, 2, 3]])
        p = Normalize(method=Normalize.Vector)(data)
        q = data.X / np.sqrt((data.X * data.X).sum(axis=1))
        np.testing.assert_equal(p.X, q)
        p = Normalize(method=Normalize.Vector, lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, q)
        p = Normalize(method=Normalize.Vector, lower=0, upper=2)(data)
        np.testing.assert_equal(p.X, q)

    def test_vector_norm_nan_correction(self):
        # even though some values are unknown the other values
        # should be normalized to the same results
        data = Orange.data.Table([[2, 2, 2, 2]])
        p = Normalize(method=Normalize.Vector)(data)
        self.assertAlmostEqual(p.X[0, 0], 0.5)
        # unknown in between that can be interpolated does not change results
        data.X[0, 2] = float("nan")
        p = Normalize(method=Normalize.Vector)(data)
        self.assertAlmostEqual(p.X[0, 0], 0.5)
        self.assertTrue(np.isnan(p.X[0, 2]))
        # unknowns at the edges do not get interpolated
        data.X[0, 3] = float("nan")
        p = Normalize(method=Normalize.Vector)(data)
        self.assertAlmostEqual(p.X[0, 0], 2**0.5/2)
        self.assertTrue(np.all(np.isnan(p.X[0, 2:])))

    def test_area_norm(self):
        data = Orange.data.Table([[2, 1, 2, 2, 3]])
        p = Normalize(method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, data.X / 3)
        p = Normalize(method=Normalize.Area, int_method=Integrate.Simple, lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, data.X / 7.5)
        p = Normalize(method=Normalize.Area, int_method=Integrate.Simple, lower=0, upper=2)(data)
        q = Integrate(method=Integrate.Simple, limits=[[0, 2]])(p)
        np.testing.assert_equal(q.X, np.ones_like(q.X))

    def test_attribute_norm(self):
        data = Orange.data.Table([[2, 1, 2, 2, 3]], metas=[[2]])
        p = Normalize(method=Normalize.Attribute)(data)
        np.testing.assert_equal(p.X, data.X)
        p = Normalize(method=Normalize.Attribute, attr=data.domain.metas[0])(data)
        np.testing.assert_equal(p.X, data.X / 2)
        p = Normalize(method=Normalize.Attribute, attr=data.domain.metas[0],
                lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, data.X / 2)
        p = Normalize(method=Normalize.Attribute, attr=data.domain.metas[0],
                lower=2, upper=4)(data)
        np.testing.assert_equal(p.X, data.X / 2)


class TestCommon(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.collagen = Orange.data.Table("collagen")

    def test_no_samples(self):
        """ Preprocessors should not crash when there are no input samples. """
        data = self.collagen[:0]
        for proc in PREPROCESSORS:
            d2 = proc(data)

    def test_no_attributes(self):
        """ Preprocessors should not crash when samples have no attributes. """
        data = self.collagen
        data = Orange.data.Table(Orange.data.Domain([],
                class_vars=data.domain.class_vars,
                metas=data.domain.metas), data)
        for proc in PREPROCESSORS:
            d2 = proc(data)

    def test_unordered_features(self):
        data = self.collagen
        data_reversed = reverse_attr(data)
        data_shuffle = shuffle_attr(data)
        for proc in PREPROCESSORS:
            comparison = np.testing.assert_equal
            # TODO find out why there are small differences for certain preprocessors
            if isinstance(proc, (RubberbandBaseline, Normalize, PCADenoising, Average)):
                comparison = lambda x,y: np.testing.assert_almost_equal(x, y, decimal=5)
            pdata = proc(data)
            X = pdata.X[:, np.argsort(getx(pdata))]
            pdata_reversed = proc(data_reversed)
            X_reversed = pdata_reversed.X[:, np.argsort(getx(pdata_reversed))]
            comparison(X, X_reversed)
            pdata_shuffle = proc(data_shuffle)
            X_shuffle = pdata_shuffle.X[:, np.argsort(getx(pdata_shuffle))]
            comparison(X, X_shuffle)

    def test_unknown_no_propagate(self):
        data = self.collagen.copy()
        # one unknown in line
        for i in range(200):
            data.X[i, i] = np.nan
        for proc in PREPROCESSORS:
            pdata = proc(data)
            sumnans = np.sum(np.isnan(pdata.X), axis=1)
            self.assertFalse(np.any(sumnans > 1))


class TestPCADenoising(unittest.TestCase):

    def test_no_samples(self):
        data = Orange.data.Table("iris")
        proc = PCADenoising()
        d1 = proc(data[:0])
        newdata = Orange.data.Table(d1.domain, data)
        np.testing.assert_equal(newdata.X, np.nan)

class TestAverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.collagen = Orange.data.Table("collagen")

    def test_average_all(self):
        data = Orange.data.Table([[2, 1, 2, 2, 3], [1, 2, 1, 1, 0]])
        p = Average(avar=None, std=False)(data)
        np.testing.assert_equal(p.X, [[1.5, 1.5, 1.5, 1.5, 1.5]])

    def test_std_all(self):
        data = Orange.data.Table([[2, 1, 2, 2, 3], [1, 2, 1, 1, 0]])
        p = Average(avar=None, std=True)(data)
        np.testing.assert_equal(p.X, [[1.5, 1.5, 1.5, 1.5, 1.5],
                                      [0.5, 0.5, 0.5, 0.5, 1.5]])

    def test_var_average(self):
        data = self.collagen
        var = data.domain.class_vars[0]
        p = Average(avar=var, std=False)(data)
        self.assertEqual(p.X.shape[0], len(var.values))
        p = Average(avar=var, std=True)(data)
        self.assertEqual(p.X.shape[0], 2 * len(var.values))

    def test_novar(self):
        data = self.collagen
        var = Orange.data.DiscreteVariable("OldVar")
        p = Average(avar=var, std=False)(data)
        q = Average(avar=None, std=False)(data)
        np.testing.assert_equal(p.X, q.X)
