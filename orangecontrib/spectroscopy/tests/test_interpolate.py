import unittest

import numpy as np

try:
    import dask
    from Orange.tests.test_dasktable import temp_dasktable
except ImportError:
    dask = None

import Orange

from orangecontrib.spectroscopy.preprocess import (
    Interpolate,
    interp1d_with_unknowns_numpy,
    interp1d_with_unknowns_scipy,
    interp1d_wo_unknowns_scipy,
    InterpolateToDomain,
    NotAllContinuousException,
    nan_extend_edges_and_interpolate,
)
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.tests.util import spectra_table
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALL_COLLAGEN,
)


class TestInterpolate(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [Interpolate(np.linspace(1000, 1700, 100))]
    data = SMALL_COLLAGEN

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")[:5].copy()
        cls.collagen = Orange.data.Table("collagen.csv")[:5]
        cls.titanic = Orange.data.Table("titanic")
        ys = np.arange(16, dtype=float).reshape(4, 4)
        np.fill_diagonal(ys, np.nan)
        cls.range16 = spectra_table([0, 1, 2, 3], X=ys)

    def test_nofloatname(self):
        data = self.iris
        interpolated = Interpolate([0.5])(data)
        self.assertIsInstance(interpolated, type(data))
        av1 = interpolated.X.ravel()
        av2 = data.X[:, :2].mean(axis=1)
        np.testing.assert_allclose(av1, av2)

    def test_floatname(self):
        data = self.collagen
        f1, f2 = 20, 21
        c1, c2 = (
            float(data.domain.attributes[f1].name),
            float(data.domain.attributes[f2].name),
        )
        avg = (c1 + c2) / 2
        interpolated = Interpolate([avg])(data)
        self.assertIsInstance(interpolated, type(data))
        av1 = interpolated.X.ravel()
        av2 = data.X[:, [20, 21]].mean(axis=1)
        np.testing.assert_allclose(av1, av2)

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = self.iris
        interpolated = Interpolate([0.5, 1.5])(data)
        nt = data.transform(interpolated.domain)
        self.assertEqual(interpolated.domain, nt.domain)
        np.testing.assert_equal(np.asarray(interpolated.X), np.asarray(nt.X))
        np.testing.assert_equal(np.asarray(interpolated.Y), np.asarray(nt.Y))

    def test_same(self):
        """Interpolate values are original values."""
        data = self.iris
        interpolated = Interpolate(range(len(data.domain.attributes)))(data)
        np.testing.assert_allclose(interpolated.X, data.X)

    def test_permute(self):
        rs = np.random.RandomState(0)
        data = self.iris
        oldX = data.X
        # permute data
        p = rs.permutation(range(len(data.domain.attributes)))
        nattr = [
            Orange.data.ContinuousVariable(str(p[i]))
            for i, a in enumerate(data.domain.attributes)
        ]
        data = Orange.data.Table.from_numpy(Orange.data.Domain(nattr), X=data.X[:, p])
        interpolated = Interpolate(range(len(data.domain.attributes)))(data)
        np.testing.assert_allclose(interpolated.X, oldX)
        # also permute output
        p1 = rs.permutation(range(len(data.domain.attributes)))
        interpolated = Interpolate(p1)(data)
        np.testing.assert_allclose(interpolated.X, oldX[:, p1])

    def test_out_of_band(self):
        data = self.iris
        interpolated = Interpolate(range(-1, len(data.domain.attributes) + 1))(data)
        np.testing.assert_allclose(interpolated.X[:, 1:5], data.X)
        np.testing.assert_equal(np.asarray(interpolated.X[:, [0, -1]]), np.nan)

    def test_unknown_middle(self):
        data = self.iris.copy()
        # whole column in the middle should be interpolated
        with data.unlocked():
            data.X[:, 1] = np.nan
        interpolated = Interpolate(getx(data))(data)
        self.assertFalse(np.any(np.isnan(interpolated.X)))

    def test_unknown_elsewhere(self):
        data = self.iris.copy()
        with data.unlocked():
            data.X[0, 1] = np.nan
            data.X[1, 1] = np.nan
            data.X[1, 2] = np.nan
        im = Interpolate(getx(data))
        interpolated = im(data)
        self.assertAlmostEqual(interpolated.X[0, 1], 3.25)
        self.assertAlmostEqual(interpolated.X[1, 1], 3.333333333333334)
        self.assertAlmostEqual(interpolated.X[1, 2], 1.766666666666667)
        self.assertFalse(np.any(np.isnan(interpolated.X)))

    def test_unknown_elsewhere_different(self):
        data = self.iris.copy()
        with data.unlocked():
            data.X[0, 1] = np.nan
            data.X[1, 1] = np.nan
            data.X[1, 2] = np.nan
        im = Interpolate(getx(data))
        im.interpfn = interp1d_with_unknowns_numpy
        interpolated = im(data)
        self.assertIsInstance(interpolated.X, type(data.X))
        self.assertAlmostEqual(interpolated.X[0, 1], 3.25)
        self.assertAlmostEqual(interpolated.X[1, 1], 3.333333333333334)
        self.assertAlmostEqual(interpolated.X[1, 2], 1.766666666666667)
        self.assertFalse(np.any(np.isnan(interpolated.X)))
        im.interpfn = interp1d_with_unknowns_scipy
        interpolated = im(data)
        self.assertIsInstance(interpolated.X, type(data.X))
        self.assertAlmostEqual(interpolated.X[0, 1], 3.25)
        self.assertAlmostEqual(interpolated.X[1, 1], 3.333333333333334)
        self.assertAlmostEqual(interpolated.X[1, 2], 1.766666666666667)
        self.assertFalse(np.any(np.isnan(interpolated.X)))
        save_X = interpolated.X
        im.interpfn = interp1d_wo_unknowns_scipy
        interpolated = im(data)
        self.assertIsInstance(interpolated.X, type(data.X))
        self.assertTrue(np.any(np.isnan(interpolated.X)))
        # parts without unknown should be the same
        np.testing.assert_allclose(data.X[2:], save_X[2:])

    def test_nan_extend_edges_and_interpolate(self):
        data = self.iris.copy()
        with data.unlocked():
            data.X[0, :] = np.nan
            data.X[1, 1] = np.nan
            data.X[2, 0] = np.nan
            data.X[3, -1] = np.nan
        xs = getx(data)
        interp, unknowns = nan_extend_edges_and_interpolate(xs, data.X)
        self.assertIsInstance(interp, type(data.X))
        nan = float("nan")
        res = np.array(
            [
                [nan, nan, nan, nan],
                [4.9, 3.15, 1.4, 0.2],
                [3.2, 3.2, 1.3, 0.2],
                [4.6, 3.1, 1.5, 1.5],
                [5.0, 3.6, 1.4, 0.2],
            ]
        )
        resu = np.array(
            [
                [True, True, True, True],
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )
        np.testing.assert_allclose(interp, res)
        np.testing.assert_allclose(unknowns, resu)

    def test_nan_extend_edges_and_interpolate_mixed(self):
        data = self.range16

        xs = getx(data)
        ys = data.X
        v, n = nan_extend_edges_and_interpolate(xs, ys)
        exp = np.arange(16, dtype=float).reshape(4, 4)
        exp[0, 0] = 1
        exp[3, 3] = 14
        np.testing.assert_equal(v, exp)

        mix = np.array([0, 2, 1, 3])
        xsm = xs[mix]
        ysm = ys[:, mix]
        v, n = nan_extend_edges_and_interpolate(xsm, ysm)
        np.testing.assert_equal(v[:, mix], exp)

    def test_eq(self):
        data = Orange.data.Table("iris")
        i1 = Interpolate([0, 1])(data)
        i2 = Interpolate([0, 1])(data)
        self.assertEqual(i1.domain, i2.domain)
        i3 = Interpolate([0, 1.1])(data)
        self.assertNotEqual(i1.domain[0], i3.domain[0])
        i4 = Interpolate([0, 1], kind="quadratic")(data)
        self.assertNotEqual(i1.domain[0], i4.domain[0])

        # different domain
        titanic = Orange.data.Table("titanic")
        it1 = Interpolate([0, 1])(titanic)
        self.assertNotEqual(i1.domain[0], it1.domain[0])


class TestInterpolateToDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")
        cls.housing = Orange.data.Table("housing")
        cls.titanic = Orange.data.Table("titanic")

    def test_same_domain(self):
        iris = self.iris
        housing = self.housing
        iiris = InterpolateToDomain(target=housing)(iris)
        self.assertNotEqual(housing.domain.attributes, iris.domain.attributes)
        # needs to have the same attributes
        self.assertEqual(housing.domain.attributes, iiris.domain.attributes)
        # first 4 values are defined, the rest are not
        np.testing.assert_equal(np.isnan(iiris.X[0])[:4], False)
        np.testing.assert_equal(np.isnan(iiris.X[0])[4:], True)

    def test_not_all_continuous(self):
        titanic = self.titanic
        iris = self.iris
        InterpolateToDomain(target=iris)
        with self.assertRaises(NotAllContinuousException):
            InterpolateToDomain(target=titanic)


@unittest.skipUnless(dask, "installed Orange does not support dask")
class TestInterpolateDask(TestInterpolate):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = temp_dasktable(cls.iris)
        cls.collagen = temp_dasktable(cls.collagen)
        cls.range16 = temp_dasktable(cls.range16)


@unittest.skipUnless(dask, "installed Orange does not support dask")
class TestInterpolateToDomainDask(TestInterpolateToDomain):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = temp_dasktable(cls.iris)
        cls.housing = temp_dasktable(cls.housing)
        cls.titanic = temp_dasktable(cls.titanic)
