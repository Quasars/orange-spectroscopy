import random
import unittest

import numpy as np

import Orange
from Orange.data import Table
from Orange.preprocess.preprocess import PreprocessorList

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import Absorbance, Transmittance, \
    Integrate, Interpolate, Cut, SavitzkyGolayFiltering, \
    GaussianSmoothing, PCADenoising, RubberbandBaseline, \
    Normalize, LinearBaseline, CurveShift, EMSC, MissingReferenceException, \
    WrongReferenceException, NormalizeReference, XASnormalization, ExtractEXAFS, \
    PreprocessException, NormalizePhaseReference, Despike, SpSubtract
from orangecontrib.spectroscopy.preprocess.als import ALSP, ARPLS, AIRPLS
from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC
from orangecontrib.spectroscopy.preprocess.atm_corr import AtmCorr
from orangecontrib.spectroscopy.preprocess.utils import replacex
from orangecontrib.spectroscopy.tests.util import smaller_data


COLLAGEN = Orange.data.Table("collagen")
SMALL_COLLAGEN = smaller_data(COLLAGEN, 2, 2)
SMALLER_COLLAGEN = smaller_data(COLLAGEN[195:621], 40, 4)  # only glycogen and lipids


def preprocessor_data(preproc):
    """
    Rerturn appropriate test file for a preprocessor.

    Very slow preprocessors should get smaller files.
    """
    if isinstance(preproc, ME_EMSC):
        return SMALLER_COLLAGEN
    return SMALL_COLLAGEN

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
    Integrate(methods=Integrate.Simple, limits=[[1100, 1200]]),
    Integrate(methods=Integrate.Baseline, limits=[[1100, 1200]]),
    Integrate(methods=Integrate.PeakMax, limits=[[1100, 1200]]),
    Integrate(methods=Integrate.PeakBaseline, limits=[[1100, 1200]]),
    Integrate(methods=Integrate.PeakAt, limits=[[1100]]),
    Integrate(methods=Integrate.PeakX, limits=[[1100, 1200]]),
    Integrate(methods=Integrate.PeakXBaseline, limits=[[1100, 1200]]),
    RubberbandBaseline(),
    LinearBaseline(),
    Normalize(method=Normalize.Vector),
    Normalize(method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=10000),
    Normalize(method=Normalize.MinMax),
    CurveShift(1),
    Despike(threshold=5, cutoff=60, dis=5),
    ALSP(lam=100E+6, itermax=5, p=0.5),
    ARPLS(lam=100E+5, itermax=5, ratio=0.5),
    AIRPLS(lam=100, itermax=5, porder=1),
]

xas_norm_collagen = XASnormalization(edge=1630,
                                     preedge_dict={'from': 1000, 'to': 1300, 'deg': 1},
                                     postedge_dict={'from': 1650, 'to': 1700, 'deg': 1})
extract_exafs = ExtractEXAFS(edge=1630, extra_from=1630, extra_to=1800,
                             poly_deg=1, kweight=0, m=0)


class ExtractEXAFSUsage(PreprocessorList):
    """ExtractEXAFS needs previous XAS normalization"""
    def __init__(self):
        super().__init__(preprocessors=[xas_norm_collagen,
                                        extract_exafs])


PREPROCESSORS_INDEPENDENT_SAMPLES += [xas_norm_collagen, ExtractEXAFSUsage()]


def add_zeros(data):
    """ Every 5th value is zero """
    s = data.copy()
    with s.unlocked():
        s[:, ::5] = 0
    return s


def make_edges_nan(data):
    s = data.copy()
    with s.unlocked():
        s[:, 0:3] = np.nan
        s[:, s.X.shape[1]-3:] = np.nan
    return s


def make_middle_nan(data):
    """ Four middle values are NaN """
    s = data.copy()
    half = s.X.shape[1]//2
    with s.unlocked():
        s[:, half-2:half+2] = np.nan
    return s


def shuffle_attr(data):
    natts = list(data.domain.attributes)
    random.Random(0).shuffle(natts)
    ndomain = Orange.data.Domain(natts, data.domain.class_vars,
                                 metas=data.domain.metas)
    return data.transform(ndomain)


def reverse_attr(data):
    natts = reversed(data.domain.attributes)
    ndomain = Orange.data.Domain(natts, data.domain.class_vars,
                                 metas=data.domain.metas)
    return data.transform(ndomain)


def add_edge_case_data_parameter(class_, data_arg_name, data_to_modify, *args, **kwargs):
    modified = [data_to_modify,
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


for p in [Absorbance, Transmittance]:
    # single reference
    PREPROCESSORS_INDEPENDENT_SAMPLES += list(add_edge_case_data_parameter(p, "reference", SMALL_COLLAGEN[0:1]))

# EMSC with different kinds of reference
PREPROCESSORS_INDEPENDENT_SAMPLES += list(
    add_edge_case_data_parameter(EMSC, "reference", SMALL_COLLAGEN[0:1]))
# EMSC with different kinds of bad spectra
PREPROCESSORS_INDEPENDENT_SAMPLES += list(
    add_edge_case_data_parameter(EMSC, "badspectra", SMALL_COLLAGEN[0:2],
                                 reference=SMALL_COLLAGEN[-1:]))

# AtmCorr with different kinds of reference
PREPROCESSORS_INDEPENDENT_SAMPLES += list(
    add_edge_case_data_parameter(AtmCorr, "reference", SMALL_COLLAGEN[0:1],
                                 correct_ranges=[(1300, 2100)], smooth_win=5))

PREPROCESSORS_INDEPENDENT_SAMPLES += \
    list(add_edge_case_data_parameter(NormalizeReference, "reference", SMALL_COLLAGEN[:1]))

PREPROCESSORS_INDEPENDENT_SAMPLES += \
    list(add_edge_case_data_parameter(SpSubtract, "reference", SMALL_COLLAGEN[:1], amount=0.1))


# Preprocessors that use groups of input samples to infer
# internal parameters.
PREPROCESSORS_GROUPS_OF_SAMPLES = [
    PCADenoising(components=2),
]

PREPROCESSORS_INDEPENDENT_SAMPLES += list(
    add_edge_case_data_parameter(ME_EMSC, "reference", SMALLER_COLLAGEN[0:1], max_iter=4))

PREPROCESSORS = PREPROCESSORS_INDEPENDENT_SAMPLES + PREPROCESSORS_GROUPS_OF_SAMPLES


class TestSpSubtract(unittest.TestCase):

    def test_simple(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        reference = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        f = SpSubtract(reference, amount=2)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X, [[-1.0, -2.0, -3.0, -4.0]])


class TestTransmittance(unittest.TestCase):

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = SMALL_COLLAGEN
        transmittance = Transmittance()(data)
        nt = Orange.data.Table.from_table(transmittance.domain, data)
        self.assertEqual(transmittance.domain, nt.domain)
        np.testing.assert_equal(transmittance.X, nt.X)
        np.testing.assert_equal(transmittance.Y, nt.Y)

    def test_roundtrip(self):
        """Test AB -> TR -> AB calculation"""
        data = SMALL_COLLAGEN
        calcdata = Absorbance()(Transmittance()(data))
        np.testing.assert_allclose(data.X, calcdata.X)

    def test_eq(self):
        data = SMALL_COLLAGEN
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


class TestAbsorbance(unittest.TestCase):

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = Transmittance()(SMALL_COLLAGEN)
        absorbance = Absorbance()(data)
        nt = Orange.data.Table.from_table(absorbance.domain, data)
        self.assertEqual(absorbance.domain, nt.domain)
        np.testing.assert_equal(absorbance.X, nt.X)
        np.testing.assert_equal(absorbance.Y, nt.Y)

    def test_roundtrip(self):
        """Test TR -> AB -> TR calculation"""
        # actually AB -> TR -> AB -> TR
        data = Transmittance()(SMALL_COLLAGEN)
        calcdata = Transmittance()(Absorbance()(data))
        np.testing.assert_allclose(data.X, calcdata.X)

    def test_eq(self):
        data = SMALL_COLLAGEN
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


class TestSavitzkyGolay(unittest.TestCase):

    def test_unknown_no_propagate(self):
        data = Orange.data.Table("iris")[:5].copy()
        f = SavitzkyGolayFiltering()
        with data.unlocked():
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


class TestGaussian(unittest.TestCase):

    def test_unknown_no_propagate(self):
        data = Orange.data.Table("iris")[:5].copy()
        f = GaussianSmoothing()
        with data.unlocked():
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


class TestRubberbandBaseline(unittest.TestCase):

    def test_whole(self):
        """ Every point belongs in the convex region. """
        data = Table.from_numpy(None, [[2, 1, 2]])
        i = RubberbandBaseline()(data)
        np.testing.assert_equal(i.X, 0)
        data = Table.from_numpy(None, [[1, 2, 1]])
        i = RubberbandBaseline(peak_dir=RubberbandBaseline.PeakNegative)(data)
        np.testing.assert_equal(i.X, 0)

    def test_simple(self):
        """ Just one point is not in the convex region. """
        data = Table.from_numpy(None, [[1, 2, 1, 1]])
        i = RubberbandBaseline()(data)
        np.testing.assert_equal(i.X, [[0, 1, 0, 0]])
        data = Table.from_numpy(None, [[1, 2, 1, 1]])
        i = RubberbandBaseline(peak_dir=RubberbandBaseline.PeakNegative)(data)
        np.testing.assert_equal(i.X, [[0, 0, -0.5, 0]])


class TestLinearBaseline(unittest.TestCase):

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


class TestNormalize(unittest.TestCase):

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
        self.assertAlmostEqual(p.X[0, 0], 2**0.5/2)
        self.assertTrue(np.all(np.isnan(p.X[0, 2:])))

    def test_area_norm(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        p = Normalize(method=Normalize.Area, int_method=Integrate.PeakMax, lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, data.X / 3)
        p = Normalize(method=Normalize.Area, int_method=Integrate.Simple, lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, data.X / 7.5)
        p = Normalize(method=Normalize.Area, int_method=Integrate.Simple, lower=0, upper=2)(data)
        q = Integrate(methods=Integrate.Simple, limits=[[0, 2]])(p)
        np.testing.assert_equal(q.X, np.ones_like(q.X))

    def test_attribute_norm(self):
        data = Table.from_numpy(None, [[2, 1, 2, 2, 3]])
        ndom = Orange.data.Domain(data.domain.attributes, data.domain.class_vars,
                                  metas=[Orange.data.ContinuousVariable("f")])
        data = data.transform(ndom)
        with data.unlocked(data.metas):
            data[0]["f"] = 2
        p = Normalize(method=Normalize.Attribute, attr=data.domain.metas[0])(data)
        np.testing.assert_equal(p.X, data.X / 2)
        p = Normalize(method=Normalize.Attribute, attr=data.domain.metas[0],
                      lower=0, upper=4)(data)
        np.testing.assert_equal(p.X, data.X / 2)
        p = Normalize(method=Normalize.Attribute, attr=data.domain.metas[0],
                      lower=2, upper=4)(data)
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

        p1 = Normalize(method=Normalize.Area, int_method=Integrate.PeakMax,
                       lower=0, upper=4)(data)
        p2 = Normalize(method=Normalize.Area, int_method=Integrate.Baseline,
                       lower=0, upper=4)(data)
        p3 = Normalize(method=Normalize.Area, int_method=Integrate.PeakMax,
                       lower=1, upper=4)(data)
        p4 = Normalize(method=Normalize.Area, int_method=Integrate.PeakMax,
                       lower=0, upper=4)(data)
        self.assertNotEqual(p1.domain, p2.domain)
        self.assertNotEqual(p1.domain, p3.domain)
        self.assertEqual(p1.domain, p4.domain)


class TestNormalizeReference(unittest.TestCase):

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


class TestCommon(unittest.TestCase):

    def test_no_samples(self):
        """ Preprocessors should not crash when there are no input samples. """
        data = SMALL_COLLAGEN[:0]
        for proc in PREPROCESSORS:
            with self.subTest(proc):
                _ = proc(data)

    def test_no_attributes(self):
        """ Preprocessors should not crash when samples have no attributes. """
        data = SMALL_COLLAGEN
        data = data.transform(Orange.data.Domain([],
                                                 class_vars=data.domain.class_vars,
                                                 metas=data.domain.metas))
        for proc in PREPROCESSORS:
            with self.subTest(proc):
                _ = proc(data)

    def test_one_attribute(self):
        """ Preprocessors should not crash when samples have one attribute. """
        data = SMALL_COLLAGEN
        data = data.transform(Orange.data.Domain([data.domain.attributes[0]],
                                                 class_vars=data.domain.class_vars,
                                                 metas=data.domain.metas))
        for proc in PREPROCESSORS:
            _ = proc(data)

    def test_all_nans(self):
        """ Preprocessors should not crash when there are all-nan samples. """
        for proc in PREPROCESSORS:
            with self.subTest(proc):
                data = preprocessor_data(proc).copy()
                with data.unlocked():
                    data.X[0, :] = np.nan
                try:
                    _ = proc(data)
                except PreprocessException:
                    continue  # allow explicit preprocessor exception

    def test_unordered_features(self):
        for proc in PREPROCESSORS:
            with self.subTest(proc):
                data = preprocessor_data(proc)
                data_reversed = reverse_attr(data)
                data_shuffle = shuffle_attr(data)
                pdata = proc(data)
                X = pdata.X[:, np.argsort(getx(pdata))]
                pdata_reversed = proc(data_reversed)
                X_reversed = pdata_reversed.X[:, np.argsort(getx(pdata_reversed))]
                np.testing.assert_almost_equal(X, X_reversed, err_msg="Preprocessor " + str(proc))
                pdata_shuffle = proc(data_shuffle)
                X_shuffle = pdata_shuffle.X[:, np.argsort(getx(pdata_shuffle))]
                np.testing.assert_almost_equal(X, X_shuffle, err_msg="Preprocessor " + str(proc))

    def test_unknown_no_propagate(self):
        for proc in PREPROCESSORS:
            with self.subTest(proc):
                data = preprocessor_data(proc).copy()
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
        """ Preprocessors should not return (-)inf """
        for proc in PREPROCESSORS:
            with self.subTest(proc):
                data = preprocessor_data(proc).copy()
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


class TestPCADenoising(unittest.TestCase):

    def test_no_samples(self):
        data = Orange.data.Table("iris")
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
        np.testing.assert_almost_equal(newdata.X[:2],
                                       [[5.08718247, 3.51315614, 1.40204280, 0.21105556],
                                        [4.75015528, 3.15366444, 1.46254138, 0.23693223]])


class TestCurveShift(unittest.TestCase):

    def test_simple(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        f = CurveShift(amount=1.1)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
                                       [[2.1, 3.1, 4.1, 5.1]])


class TestUtils(unittest.TestCase):

    def test_replacex(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(list(getx(data)), [0, 1, 2, 3])
        dr = replacex(data, ["a", 1, 2, 3])
        self.assertEqual([a.name for a in dr.domain.attributes],
                         ["a", "1", "2", "3"])
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
