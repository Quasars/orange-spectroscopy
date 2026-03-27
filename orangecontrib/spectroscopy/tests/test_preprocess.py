import random
import unittest

import numpy as np

import Orange
from Orange.classification import RandomForestLearner
from Orange.data import Table
from Orange.evaluation import TestOnTestData, AUC

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import (
    Absorbance,
    Transmittance,
    Integrate,
    Interpolate,
    SavitzkyGolayFiltering,
    GaussianSmoothing,
    PCADenoising,
    RubberbandBaseline,
    Normalize,
    LinearBaseline,
    ShiftAndScale,
    MissingReferenceException,
    WrongReferenceException,
    NormalizeReference,
    PreprocessException,
    NormalizePhaseReference,
    SpSubtract,
    MNFDenoising,
)
from orangecontrib.spectroscopy.preprocess.utils import replacex
from orangecontrib.spectroscopy.tests.test_conversion import (
    separate_learn_test,
    slightly_change_wavenumbers,
    odd_attr,
)
from orangecontrib.spectroscopy.tests.util import smaller_data


COLLAGEN = Orange.data.Table("collagen")
SMALL_COLLAGEN = smaller_data(COLLAGEN, 2, 2)
SMALLER_COLLAGEN = smaller_data(COLLAGEN[195:621], 40, 4)  # only glycogen and lipids


def add_zeros(data):
    """Every 5th value is zero"""
    s = data.copy()
    with s.unlocked():
        s[:, ::5] = 0
    return s


def make_edges_nan(data):
    s = data.copy()
    with s.unlocked():
        s[:, 0:3] = np.nan
        s[:, s.X.shape[1] - 3 :] = np.nan
    return s


def make_middle_nan(data):
    """Four middle values are NaN"""
    s = data.copy()
    half = s.X.shape[1] // 2
    with s.unlocked():
        s[:, half - 2 : half + 2] = np.nan
    return s


def shuffle_attr(data):
    natts = list(data.domain.attributes)
    random.Random(0).shuffle(natts)
    ndomain = Orange.data.Domain(natts, data.domain.class_vars, metas=data.domain.metas)
    return data.transform(ndomain)


def reverse_attr(data):
    natts = reversed(data.domain.attributes)
    ndomain = Orange.data.Domain(natts, data.domain.class_vars, metas=data.domain.metas)
    return data.transform(ndomain)


def add_edge_case_data_parameter(
    class_, data_arg_name, data_to_modify, *args, **kwargs
):
    modified = [
        data_to_modify,
        shuffle_attr(data_to_modify),
        make_edges_nan(data_to_modify),
        shuffle_attr(make_edges_nan(data_to_modify)),
        make_middle_nan(data_to_modify),
        add_zeros(data_to_modify),
    ]
    for i, d in enumerate(modified):
        kwargs[data_arg_name] = d
        p = class_(*args, **kwargs)
        # 5 is add_zeros
        if i == 5:
            p.skip_add_zeros = True
        yield p


class TestConversionMixin:
    def test_slightly_different_domain(self):
        """If test data has a slightly different domain then (with interpolation)
        we should obtain a similar classification score."""
        learner = RandomForestLearner(random_state=42)

        for proc in self.preprocessors:
            if hasattr(proc, "skip_add_zeros"):
                continue
            with self.subTest(proc):
                # LR that can not handle unknown values
                train, test = separate_learn_test(self.data)
                train1 = proc(train)
                aucorig = AUC(TestOnTestData()(train1, test, [learner]))
                test = slightly_change_wavenumbers(test, 0.00001)
                test = odd_attr(test)
                # a subset of points for training so that all test sets points
                # are within the train set points, which gives no unknowns
                train = Interpolate(points=getx(train)[1:-3])(
                    train
                )  # interpolatable train
                train = proc(train)
                # explicit domain conversion test to catch exceptions that would
                # otherwise be silently handled in TestOnTestData
                _ = test.transform(train.domain)
                aucnow = AUC(TestOnTestData()(train, test, [learner]))
                self.assertAlmostEqual(
                    aucnow, aucorig, delta=0.03, msg="Preprocessor " + str(proc)
                )
                test = Interpolate(points=getx(test) - 1.0)(test)  # also do a shift
                _ = test.transform(train.domain)  # explicit call again
                aucnow = AUC(TestOnTestData()(train, test, [learner]))
                # the difference should be slight
                self.assertAlmostEqual(
                    aucnow, aucorig, delta=0.05, msg="Preprocessor " + str(proc)
                )


class TestConversionIndpSamplesMixin(TestConversionMixin):
    """
    Testing mixin for preprocessors that work per sample and should
    return the same result for a sample independent of the other samples
    """

    def test_whole_and_train_separate(self):
        """Applying a preprocessor before spliting data into train and test
        and applying is just on train data should yield the same transformation of
        the test data."""
        for proc in self.preprocessors:
            with self.subTest(proc):
                data = self.data
                _, test1 = separate_learn_test(proc(data))
                train, test = separate_learn_test(data)
                train = proc(train)
                test_transformed = test.transform(train.domain)
                np.testing.assert_almost_equal(
                    test_transformed.X, test1.X, err_msg="Preprocessor " + str(proc)
                )


class TestStrangeDataMixin:
    def test_no_samples(self):
        """Preprocessors should not crash when there are no input samples."""
        data = self.data[:0]
        for proc in self.preprocessors:
            with self.subTest(proc):
                _ = proc(data)

    def test_no_attributes(self):
        """Preprocessors should not crash when samples have no attributes."""
        data = self.data
        data = data.transform(
            Orange.data.Domain(
                [], class_vars=data.domain.class_vars, metas=data.domain.metas
            )
        )
        for proc in self.preprocessors:
            with self.subTest(proc):
                _ = proc(data)

    def test_all_nans(self):
        """Preprocessors should not crash when there are all-nan samples."""
        for proc in self.preprocessors:
            with self.subTest(proc):
                data = self.data.copy()
                with data.unlocked():
                    data.X[0, :] = np.nan
                try:
                    _ = proc(data)
                except PreprocessException:
                    continue  # allow explicit preprocessor exception

    def test_unordered_features(self):
        for proc in self.preprocessors:
            with self.subTest(proc):
                data = self.data
                data_reversed = reverse_attr(data)
                data_shuffle = shuffle_attr(data)
                pdata = proc(data)
                X = pdata.X[:, np.argsort(getx(pdata))]
                pdata_reversed = proc(data_reversed)
                X_reversed = pdata_reversed.X[:, np.argsort(getx(pdata_reversed))]
                np.testing.assert_almost_equal(
                    X, X_reversed, err_msg="Preprocessor " + str(proc)
                )
                pdata_shuffle = proc(data_shuffle)
                X_shuffle = pdata_shuffle.X[:, np.argsort(getx(pdata_shuffle))]
                np.testing.assert_almost_equal(
                    X, X_shuffle, err_msg="Preprocessor " + str(proc)
                )

    def test_unknown_no_propagate(self):
        for proc in self.preprocessors:
            with self.subTest(proc):
                data = self.data.copy()
                # one unknown in line
                with data.unlocked():
                    for i in range(min(len(data), len(data.domain.attributes))):
                        data.X[i, i] = np.nan

                if hasattr(proc, "skip_add_zeros"):
                    continue
                pdata = proc(data)
                sumnans = np.sum(np.isnan(pdata.X), axis=1)
                self.assertFalse(np.any(sumnans > 1), msg="Preprocessor " + str(proc))

    def test_no_infs(self):
        """Preprocessors should not return (-)inf"""
        for proc in self.preprocessors:
            with self.subTest(proc):
                data = self.data.copy()
                # add some zeros to the dataset
                with data.unlocked():
                    for i in range(min(len(data), len(data.domain.attributes))):
                        data.X[i, i] = 0
                    data.X[0, :] = 0
                    data.X[:, 0] = 0
                try:
                    pdata = proc(data)
                except PreprocessException:
                    continue  # allow explicit preprocessor exception
                anyinfs = np.any(np.isinf(pdata.X))
                self.assertFalse(anyinfs, msg="Preprocessor " + str(proc))


class TestCommonMixin(TestStrangeDataMixin, TestConversionMixin):
    pass


class TestCommonIndpSamplesMixin(TestStrangeDataMixin, TestConversionIndpSamplesMixin):
    pass


class TestSpSubtract(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = list(
        add_edge_case_data_parameter(
            SpSubtract, "reference", SMALLER_COLLAGEN[:1], amount=0.1
        )
    )
    data = SMALLER_COLLAGEN

    def test_simple(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        reference = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        f = SpSubtract(reference, amount=2)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X, [[-1.0, -2.0, -3.0, -4.0]])


class TestTransmittance(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [Transmittance()] + list(
        add_edge_case_data_parameter(Transmittance, "reference", SMALLER_COLLAGEN[0:1])
    )
    data = SMALLER_COLLAGEN

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = self.data
        transmittance = Transmittance()(data)
        nt = Orange.data.Table.from_table(transmittance.domain, data)
        self.assertEqual(transmittance.domain, nt.domain)
        np.testing.assert_equal(transmittance.X, nt.X)
        np.testing.assert_equal(transmittance.Y, nt.Y)

    def test_roundtrip(self):
        """Test AB -> TR -> AB calculation"""
        data = self.data
        calcdata = Absorbance()(Transmittance()(data))
        np.testing.assert_allclose(data.X, calcdata.X)

    def test_eq(self):
        data = self.data
        t1 = Transmittance()(data)
        t2 = Transmittance()(data)
        self.assertEqual(t1.domain, t2.domain)
        data2 = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        t3 = Transmittance()(data2)
        self.assertNotEqual(t1.domain, t3.domain)
        t4 = Transmittance(reference=data2)(data)
        self.assertNotEqual(t1.domain, t4.domain)
        t5 = Transmittance(reference=data2)(data[:1])
        self.assertGreater(len(t4), len(t5))
        self.assertEqual(t4.domain, t5.domain)
        a = Absorbance()(data)
        self.assertNotEqual(a.domain, t1.domain)


class TestAbsorbance(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [Absorbance()] + list(
        add_edge_case_data_parameter(Absorbance, "reference", SMALLER_COLLAGEN[0:1])
    )
    data = SMALLER_COLLAGEN

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = Transmittance()(self.data)
        absorbance = Absorbance()(data)
        nt = Orange.data.Table.from_table(absorbance.domain, data)
        self.assertEqual(absorbance.domain, nt.domain)
        np.testing.assert_equal(absorbance.X, nt.X)
        np.testing.assert_equal(absorbance.Y, nt.Y)

    def test_roundtrip(self):
        """Test TR -> AB -> TR calculation"""
        # actually AB -> TR -> AB -> TR
        data = Transmittance()(self.data)
        calcdata = Transmittance()(Absorbance()(data))
        np.testing.assert_allclose(data.X, calcdata.X)

    def test_eq(self):
        data = self.data
        t1 = Absorbance()(data)
        t2 = Absorbance()(data)
        self.assertEqual(t1.domain, t2.domain)
        data2 = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        t3 = Absorbance()(data2)
        self.assertNotEqual(t1.domain, t3.domain)
        t4 = Absorbance(reference=data2)(data)
        self.assertNotEqual(t1.domain, t4.domain)
        t5 = Absorbance(reference=data2)(data[:1])
        self.assertGreater(len(t4), len(t5))
        self.assertEqual(t4.domain, t5.domain)


class TestSavitzkyGolay(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [SavitzkyGolayFiltering(window=9, polyorder=2, deriv=2)]
    data = SMALL_COLLAGEN

    def test_simple(self):
        data = Orange.data.Table("iris")
        f = SavitzkyGolayFiltering()
        data = data[:1]
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X, [[4.86857143, 3.47428571, 1.49428571, 0.32857143]]
        )

    def test_eq(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        p1 = SavitzkyGolayFiltering(window=5, polyorder=2, deriv=0)(data)
        p2 = SavitzkyGolayFiltering(window=5, polyorder=2, deriv=1)(data)
        p3 = SavitzkyGolayFiltering(window=5, polyorder=3, deriv=0)(data)
        p4 = SavitzkyGolayFiltering(window=7, polyorder=2, deriv=0)(data)
        self.assertNotEqual(p1.domain, p2.domain)
        self.assertNotEqual(p1.domain, p3.domain)
        self.assertNotEqual(p1.domain, p4.domain)

        s1 = SavitzkyGolayFiltering(window=5, polyorder=2, deriv=0)(data)
        self.assertEqual(p1.domain, s1.domain)

        # even if the data set is different features should be the same
        data2 = Table.from_numpy(None, [[2, 1, 3, 4, 3]])
        s2 = SavitzkyGolayFiltering(window=5, polyorder=2, deriv=0)(data2)
        self.assertEqual(p1.domain, s2.domain)


class TestGaussian(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [GaussianSmoothing(sd=3.0)]
    data = SMALL_COLLAGEN

    def test_simple(self):
        data = Orange.data.Table("iris")
        f = GaussianSmoothing(sd=1.0)
        data = data[:1]
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X, [[4.4907066, 3.2794677, 1.7641664, 0.6909083]]
        )


class TestRubberbandBaseline(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [RubberbandBaseline()]
    data = SMALLER_COLLAGEN

    def test_whole(self):
        """Every point belongs in the convex region."""
        data = Table.from_numpy(None, [[2, 1, 2]])
        i = RubberbandBaseline()(data)
        np.testing.assert_equal(i.X, 0)
        data = Table.from_numpy(None, [[1, 2, 1]])
        i = RubberbandBaseline(peak_dir=RubberbandBaseline.PeakNegative)(data)
        np.testing.assert_equal(i.X, 0)

    def test_simple(self):
        """Just one point is not in the convex region."""
        data = Table.from_numpy(None, [[1, 2, 1, 1]])
        i = RubberbandBaseline()(data)
        np.testing.assert_equal(i.X, [[0, 1, 0, 0]])
        data = Table.from_numpy(None, [[1, 2, 1, 1]])
        i = RubberbandBaseline(peak_dir=RubberbandBaseline.PeakNegative)(data)
        np.testing.assert_equal(i.X, [[0, 0, -0.5, 0]])


class TestLinearBaseline(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [LinearBaseline()]
    data = SMALL_COLLAGEN

    def test_whole(self):
        data = Table.from_numpy(None, [[1, 5, 1]])
        i = LinearBaseline()(data)
        np.testing.assert_equal(i.X, [[0, 4, 0]])

        data = Table.from_numpy(None, [[4, 1, 2, 4]])
        i = LinearBaseline(peak_dir=LinearBaseline.PeakNegative)(data)
        np.testing.assert_equal(i.X, [[0, -3, -2, 0]])

    def test_edgepoints(self):
        data = Table.from_numpy(None, [[1, 5, 1]])
        i = LinearBaseline(zero_points=[0, 2])(data)
        np.testing.assert_equal(i.X, [[0, 4, 0]])

    def test_edgepoints_extrapolate(self):
        data = Table.from_numpy(None, [[1, 5, 1]])
        i = LinearBaseline(zero_points=[0, 1])(data)
        np.testing.assert_equal(i.X, [[0, 0, -8]])

    def test_3points(self):
        data = Table.from_numpy(None, [[1, 5, 1, 5]])
        i = LinearBaseline(zero_points=[0, 1, 3])(data)
        np.testing.assert_equal(i.X, [[0, 0, -4, 0]])

    def test_edgepoints_out_of_data(self):
        data = Table.from_numpy(None, [[1, 5, 1]])
        i = LinearBaseline(zero_points=[0, 2.000000001])(data)
        np.testing.assert_almost_equal(i.X, [[0, 4, 0]])


class TestNormalize(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [
        Normalize(method=Normalize.Vector),
        Normalize(
            method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=10000
        ),
        Normalize(method=Normalize.MinMax),
    ]

    data = SMALL_COLLAGEN

    def test_vector_norm(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
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
        data = Table.from_numpy(None, [[2, 2, 2, 2]])
        p = Normalize(method=Normalize.Vector)(data)
        self.assertAlmostEqual(p.X[0, 0], 0.5)
        # unknown in between that can be interpolated does not change results
        with data.unlocked():
            data.X[0, 2] = float("nan")
        p = Normalize(method=Normalize.Vector)(data)
        self.assertAlmostEqual(p.X[0, 0], 0.5)
        self.assertTrue(np.isnan(p.X[0, 2]))
        # unknowns at the edges do not get interpolated
        with data.unlocked():
            data.X[0, 3] = float("nan")
        p = Normalize(method=Normalize.Vector)(data)
        self.assertAlmostEqual(p.X[0, 0], 2**0.5 / 2)
        self.assertTrue(np.all(np.isnan(p.X[0, 2:])))

    def test_area_norm(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        p = Normalize(
            method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=4
        )(data)
        np.testing.assert_equal(p.X, data.X / 3)
        p = Normalize(
            method=Normalize.Area, int_method=Integrate.Simple, lower=0, upper=4
        )(data)
        np.testing.assert_equal(p.X, data.X / 7.5)
        p = Normalize(
            method=Normalize.Area, int_method=Integrate.Simple, lower=0, upper=2
        )(data)
        q = Integrate(methods=Integrate.Simple, limits=[[0, 2]])(p)
        np.testing.assert_equal(q.X, np.ones_like(q.X))

    def test_attribute_norm(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        ndom = Orange.data.Domain(
            data.domain.attributes,
            data.domain.class_vars,
            metas=[Orange.data.ContinuousVariable("f")],
        )
        data = data.transform(ndom)
        with data.unlocked(data.metas):
            data[0]["f"] = 2
        p = Normalize(method=Normalize.Attribute, attr=data.domain.metas[0])(data)
        np.testing.assert_equal(p.X, data.X / 2)
        p = Normalize(
            method=Normalize.Attribute, attr=data.domain.metas[0], lower=0, upper=4
        )(data)
        np.testing.assert_equal(p.X, data.X / 2)
        p = Normalize(
            method=Normalize.Attribute, attr=data.domain.metas[0], lower=2, upper=4
        )(data)
        np.testing.assert_equal(p.X, data.X / 2)

    def test_attribute_norm_unknown(self):
        data = Table.from_numpy(None, X=[[2, 1, 2, 2, 3]], metas=[[2]])
        p = Normalize(method=Normalize.Attribute, attr="unknown")(data)
        self.assertTrue(np.all(np.isnan(p.X)))

    def test_minmax_norm(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        p = Normalize(method=Normalize.MinMax)(data)
        q = (data.X) / (3 - 1)
        np.testing.assert_equal(p.X, q)
        p = Normalize(method=Normalize.MinMax, lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, q)
        p = Normalize(method=Normalize.MinMax, lower=0, upper=2)(data)
        np.testing.assert_equal(p.X, q)

    def test_SNV_norm(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        p = Normalize(method=Normalize.SNV)(data)
        q = (data.X - 2) / 0.6324555320336759
        np.testing.assert_equal(p.X, q)
        p = Normalize(method=Normalize.SNV, lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, q)
        p = Normalize(method=Normalize.SNV, lower=0, upper=2)(data)
        np.testing.assert_equal(p.X, q)

    def test_eq(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        p1 = Normalize(method=Normalize.MinMax)(data)
        p2 = Normalize(method=Normalize.SNV)(data)
        p3 = Normalize(method=Normalize.MinMax)(data)
        self.assertNotEqual(p1.domain, p2.domain)
        self.assertEqual(p1.domain, p3.domain)

        p1 = Normalize(
            method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=4
        )(data)
        p2 = Normalize(
            method=Normalize.Area, int_method=Integrate.Baseline, lower=0, upper=4
        )(data)
        p3 = Normalize(
            method=Normalize.Area, int_method=Integrate.PeakMax, lower=1, upper=4
        )(data)
        p4 = Normalize(
            method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=4
        )(data)
        self.assertNotEqual(p1.domain, p2.domain)
        self.assertNotEqual(p1.domain, p3.domain)
        self.assertEqual(p1.domain, p4.domain)


class TestNormalizeReference(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = list(
        add_edge_case_data_parameter(
            NormalizeReference, "reference", SMALLER_COLLAGEN[:1]
        )
    ) + list(
        add_edge_case_data_parameter(
            NormalizePhaseReference, "reference", SMALLER_COLLAGEN[:1]
        )
    )
    data = SMALLER_COLLAGEN

    def test_reference(self):
        data = Table.from_numpy(None, [[2, 1, 3], [4, 2, 6]])
        reference = data[:1]
        p = NormalizeReference(reference=reference)(data)
        np.testing.assert_almost_equal(p, [[1, 1, 1], [2, 2, 2]])
        s = NormalizePhaseReference(reference=reference)(data)
        np.testing.assert_almost_equal(s, [[0, 0, 0], [2, 1, 3]])

    def test_reference_exceptions(self):
        with self.assertRaises(MissingReferenceException):
            NormalizeReference(reference=None)
        with self.assertRaises(WrongReferenceException):
            NormalizeReference(reference=Table.from_numpy(None, [[2], [6]]))


class TestPCADenoising(unittest.TestCase, TestCommonMixin):
    preprocessors = [PCADenoising(components=2)]
    data = SMALLER_COLLAGEN

    def test_no_samples(self):
        data = self.data
        proc = PCADenoising()
        d1 = proc(data[:0])
        newdata = data.transform(d1.domain)
        np.testing.assert_equal(newdata.X, np.nan)

    def test_iris(self):
        data = Orange.data.Table("iris")
        proc = PCADenoising(components=2)
        d1 = proc(data)
        newdata = data.transform(d1.domain)
        differences = newdata.X - data.X
        self.assertTrue(np.all(np.abs(differences) < 0.6))
        # pin some values to detect changes in the PCA implementation
        # (for example normalization)
        np.testing.assert_almost_equal(
            newdata.X[:2],
            [
                [5.08718247, 3.51315614, 1.40204280, 0.21105556],
                [4.75015528, 3.15366444, 1.46254138, 0.23693223],
            ],
        )

    def test_selected_components(self):
        data = Orange.data.Table("iris")
        proc = PCADenoising(components=[0, 1])
        d1 = proc(data)
        proc2 = PCADenoising(components=2)
        d2 = proc2(data)
        np.testing.assert_equal(d1.X, d2.X)


class TestMNFDenoising(unittest.TestCase, TestCommonMixin):
    preprocessors = [MNFDenoising(components=2)]
    data = SMALL_COLLAGEN

    def test_no_samples(self):
        data = Orange.data.Table("iris")
        proc = MNFDenoising()
        d1 = proc(data[:0])
        newdata = data.transform(d1.domain)
        np.testing.assert_equal(newdata.X, np.nan)

    def test_iris(self):
        data = Orange.data.Table("iris")
        proc = MNFDenoising(components=2)
        d1 = proc(data)
        newdata = data.transform(d1.domain)
        differences = newdata.X - data.X
        self.assertTrue(np.all(np.abs(differences) < 0.6))
        # pin some values to detect changes in the PCA implementation
        # (for example normalization)
        np.testing.assert_almost_equal(
            newdata.X[:2],
            [
                [5.1084779, 3.4893387, 1.4068703, 0.1887913],
                [4.7484942, 3.1913347, 1.427665, 0.2304239],
            ],
        )

    def test_slightly_different_domain(self):
        # test is disabled because this method is too sensitive to small input changes
        pass


class TestShiftAndScale(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [ShiftAndScale(1, 2)]
    data = SMALL_COLLAGEN

    def test_simple(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        f = ShiftAndScale(offset=1.1, scale=2.0)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X, [[3.1, 5.1, 7.1, 9.1]])


class TestUtils(unittest.TestCase):
    def test_replacex(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(list(getx(data)), [0, 1, 2, 3])
        dr = replacex(data, ["a", 1, 2, 3])
        self.assertEqual([a.name for a in dr.domain.attributes], ["a", "1", "2", "3"])
        dr = replacex(data, np.array([0.5, 1, 2, 3]))
        self.assertEqual(list(getx(dr)), [0.5, 1, 2, 3])
        np.testing.assert_equal(data.X, dr.X)

    def test_replacex_transforms(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        dr = replacex(data, np.linspace(5, 8, 4))
        self.assertEqual(list(getx(dr)), [5, 6, 7, 8])
        np.testing.assert_equal(data.X, dr.X)
        dt = data.transform(dr.domain)
        np.testing.assert_equal(data.X, dt.X)

    def test_replacex_invalid(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        with self.assertRaises(AssertionError):
            replacex(data, [1, 2, 3])
