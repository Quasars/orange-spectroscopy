import unittest

import numpy as np

import Orange
from Orange.data import FileFormat, dataset_dirs

from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC
from orangecontrib.spectroscopy.preprocess.emsc import SmoothedSelectionFunction
from orangecontrib.spectroscopy.preprocess.npfunc import Sum
from orangecontrib.spectroscopy.tests.test_preprocess import TestCommonIndpSamplesMixin, \
    SMALLER_COLLAGEN, add_edge_case_data_parameter


def weights_from_inflection_points_legacy(points, kappa, wavenumbers):
    """ As ported from the original Matlab code.

    BUG: results are shifted for 1 wavenumber positions-wise

    Made legacy due to the following drawbacks:
    - Only supports 3 or 4 points.
    - Edges behave differently in 3 or 4 point cases.
    - Transition shape depends on spacing of input wavenumbers.
    - Essentially undefined behaviour when inflection points are too close.
    - The wavenumbers are assumed to be equidistant.

    This functions needs correctly ordered inputs. Example calls (function as f):
    (3 points) f([3700, 2550, 1900, 0], [1, 1, 1, 1], np.arange(0, 5000, 10))
    (4 points) f([3700, 2550, 1900, 1000], [1, 1, 1, 1], np.arange(0, 5000, 10))

    For both calls the weights go towards 0 for wn > 3700. For (3 points)
    wn < 1900 go towards 1, but for (4 points) they go towards 0 when wn < 1000.
    """
    # Hyperbolic tangent function
    hypTan = lambda x_range, kap: 0.5 * (np.tanh(kap * x_range) + 1)  # noqa: E731

    # Calculate position of inflection points
    p1 = np.argmin(np.abs(wavenumbers - points[2]))
    p2 = np.argmin(np.abs(wavenumbers - points[1]))
    p3 = np.argmin(np.abs(wavenumbers - points[0]))

    # Resolution on the x-axis used in the hyperbolic tangent function
    dx = 0.094

    x1 = np.linspace(-(p1 - 1) * dx, 0, p1)
    x2 = np.linspace(dx, np.floor((p2 - p1) / 2) * dx, int(np.floor((p2 - p1) / 2)))

    xp1 = np.hstack([x1, x2])

    x3 = np.linspace(-(np.ceil((p2 - p1) / 2) - 1) * dx, 0, int((np.ceil((p2 - p1) / 2))))
    x4 = np.linspace(dx, np.floor((p3 - p2) / 2) * dx, int(np.floor((p3 - p2) / 2)))

    xp2 = np.hstack([x3, x4])

    x5 = np.linspace(-(np.ceil((p3 - p2) / 2) - 1) * dx, 0, int((np.ceil((p3 - p2) / 2))))
    x6 = np.linspace(dx, (len(wavenumbers) - p3) * dx, int(len(wavenumbers) - p3))

    xp3 = np.hstack([x5, x6])

    patch1 = 1 - hypTan(xp1, kappa[2])
    patch2 = hypTan(xp2, kappa[1])
    patch3 = 1 - hypTan(xp3, kappa[0])

    wei_X = np.hstack([patch1, patch2, patch3])

    if points[3]:
        p0 = np.argmin(np.abs(wavenumbers - points[3]))

        x1a = np.linspace(-(p0 - 1) * dx, 0, p0)
        x2a = np.linspace(dx, np.floor((p1 - p0) / 2) * dx, int(np.floor((p1 - p0) / 2)))
        x3a = np.linspace(-(np.ceil((p1 - p0) / 2) - 1) * dx, 0, int((np.ceil((p1 - p0) / 2))))

        xp0 = np.hstack([x1a, x2a])
        xp1 = np.hstack([x3a, x2])

        patch0 = hypTan(xp0, kappa[3])
        patch1 = 1 - hypTan(xp1, kappa[2])
        wei_X = np.hstack([patch0, patch1, patch2, patch3])

    wei_X = wei_X.reshape(1, -1)
    dom = Orange.data.Domain(
        [Orange.data.ContinuousVariable(name=str(float(a))) for a in wavenumbers])
    data = Orange.data.Table.from_numpy(dom, wei_X)
    return data


class TestME_EMSC(unittest.TestCase, TestCommonIndpSamplesMixin):

    preprocessors = list(
        add_edge_case_data_parameter(ME_EMSC, "reference", SMALLER_COLLAGEN[0:1], max_iter=4))
    data = SMALLER_COLLAGEN

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def locate_dataset(fn):
            return FileFormat.locate(fn, dataset_dirs)

        path2data1 = locate_dataset('emsc/MieStd1_rawSpec.csv')
        path2data2 = locate_dataset('emsc/MieStd2_refSpec.csv')
        path2data3 = locate_dataset('emsc/MieStd3_corr.csv')
        path2data4 = locate_dataset('emsc/MieStd4_param.csv')
        path2data5 = locate_dataset('emsc/MieStd5_residuals.csv')
        path2data6 = locate_dataset('emsc/MieStd6_niter.csv')
        path2data7 = locate_dataset('emsc/MieStd7_RMSE.csv')

        v = np.loadtxt(path2data1, usecols=np.arange(1, 779), delimiter=",")
        cls.wnS = v[0]
        cls.Spectra = np.vstack((v[1], v[1]))

        v = np.loadtxt(path2data2, usecols=np.arange(1, 752), delimiter=",")
        cls.wnM = v[0]
        cls.Matrigel = v[1].reshape(1, -1)

        v = np.loadtxt(path2data3, usecols=np.arange(1, 40), delimiter=",")
        cls.corr_default_20th_elem = v[0]
        cls.corr_14ncomp_20th_elem = v[1]
        cls.corr_fixed_iter3_20th_elem = v[2]

        v = np.loadtxt(path2data4, usecols=np.arange(1, 17), delimiter=",")
        cls.param_default_20th_elem = v[0][~np.isnan(v[0])]
        cls.param_14ncomp_20th_elem = v[1]
        cls.param_fixed_iter3_20th_elem = v[2][~np.isnan(v[2])]

        v = np.loadtxt(path2data5, usecols=np.arange(1, 40), delimiter=",")
        cls.res_default_20th_elem = v[0]
        cls.res_14ncomp_20th_elem = v[1]
        cls.res_fixed_iter3_20th_elem = v[2]

        cls.numiter_std = np.loadtxt(path2data6, usecols=(1,), delimiter=",", dtype="int64")

        cls.RMSE_std = np.loadtxt(path2data7, usecols=(1,), delimiter=",", dtype="float")

        domain_reference = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in cls.wnM])
        cls.reference = Orange.data.Table(domain_reference, cls.Matrigel)

        domain_spectra = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in cls.wnS])
        cls.spectra = Orange.data.Table(domain_spectra, cls.Spectra)

        # inflPoints = [3700, 2550, 1900, 0]
        # kappa = [1, 1, 1, 0]
        #
        # weights = weights_from_inflection_points(inflPoints, kappa, self.wnS)

        f = ME_EMSC(reference=cls.reference, ncomp=False, weights=False, max_iter=45, output_model=True)
        cls.f1data = f(cls.spectra)

        f2 = ME_EMSC(reference=cls.reference, ncomp=14, output_model=True)  # With weights
        cls.f2data = f2(cls.spectra)

        f3 = ME_EMSC(reference=cls.reference, ncomp=False, fixed_iter=1, output_model=True)
        cls.f3data = f3(cls.spectra)

    def disabled_test_plotting(self):
        import matplotlib.pyplot as plt

        # Default parameters
        plt.figure()
        plt.plot(self.wnS[0::20], self.f1data[0, 0::20].X.T, label='python')
        plt.plot(self.wnS[0::20], self.corr_default_20th_elem, label='matlab')
        plt.plot(self.wnS[0::20], self.f1data[0, 0::20].X.T[:, 0] - self.corr_default_20th_elem, label='diff')
        plt.legend()
        plt.title('Comparison Matlab/Python - default parameters')

        # 14 principal components
        plt.figure()
        plt.plot(self.wnS[0::20], self.f2data[0, 0::20].X.T, label='python')
        plt.plot(self.wnS[0::20], self.corr_14ncomp_20th_elem, label='matlab')
        plt.plot(self.wnS[0::20], self.f2data[0, 0::20].X.T[:, 0]-self.corr_14ncomp_20th_elem, label='diff')
        plt.legend()
        plt.title('Comparison Matlab/Python - 14 principal components')

        # Fixed iteration number 3
        plt.figure()
        plt.plot(self.wnS[0::20], self.f3data[0, 0::20].X.T, label='python')
        plt.plot(self.wnS[0::20], self.corr_fixed_iter3_20th_elem, label='matlab')
        plt.plot(self.wnS[0::20], self.f3data[0, 0::20].X.T[:, 0]-self.corr_fixed_iter3_20th_elem, label='diff')
        plt.legend()
        plt.title('Comparison Matlab/Python - fixed iterations 3')
        plt.show()

    def test_correction_output(self):
        np.testing.assert_almost_equal(self.corr_default_20th_elem, self.f1data[0, 0::20].X.T[:, 0])
        np.testing.assert_almost_equal(self.corr_14ncomp_20th_elem, self.f2data[0, 0::20].X.T[:, 0])
        np.testing.assert_almost_equal(self.corr_fixed_iter3_20th_elem, self.f3data[0, 0::20].X.T[:, 0])

    def test_EMSC_parameters(self):
        np.testing.assert_almost_equal(abs(self.f1data.metas[0, :-2]), abs(self.param_default_20th_elem))
        np.testing.assert_almost_equal(abs(self.f2data.metas[0, :-2]), abs(self.param_14ncomp_20th_elem))
        np.testing.assert_almost_equal(abs(self.f3data.metas[0, :-2]), abs(self.param_fixed_iter3_20th_elem))

    def test_number_iterations(self):
        numiter = np.array([self.f1data.metas[0, -2], self.f2data.metas[0, -2], self.f3data.metas[0, -2]])
        np.testing.assert_equal(numiter, self.numiter_std)

    def test_RMSE(self):
        RMSE = np.array([self.f1data.metas[0, -1], self.f2data.metas[0, -1], self.f3data.metas[0, -1]])
        np.testing.assert_equal(RMSE, self.RMSE_std)

    def test_same_data_reference(self):
        # it was crashing before
        ME_EMSC(reference=self.reference)(self.reference)

    def test_short_reference(self):
        wnMshort = self.wnM[0::30]
        Matrigelshort = self.Matrigel[0, 0::30]
        Matrigelshort = Matrigelshort.reshape(-1, 1).T
        domain_reference = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                               for w in wnMshort])
        reference = Orange.data.Table(domain_reference, Matrigelshort)
        # it was crashing before
        ME_EMSC(reference=reference)(self.spectra)

    def test_eq(self):
        ref = self.reference
        spectra = self.spectra
        d1 = ME_EMSC(reference=ref, ncomp=False, weights=False, max_iter=1)(spectra)
        d2 = ME_EMSC(reference=ref, ncomp=False, weights=False, max_iter=1)(spectra)
        self.assertEqual(d1.domain, d2.domain)
        self.assertEqual(hash(d1.domain), hash(d2.domain))

        d2 = ME_EMSC(reference=ref, ncomp=2, weights=False, max_iter=1)(spectra)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        d2 = ME_EMSC(reference=ref, ncomp=False, weights=False, max_iter=1,
                     n0=np.linspace(1.1, 1.4, 11),
                     a=np.linspace(2, 7.1, 11))(spectra)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        d2 = ME_EMSC(reference=ref, ncomp=False, weights=False, max_iter=2)(spectra)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        r2 = ref.copy()
        d2 = ME_EMSC(reference=r2, ncomp=False, weights=False, max_iter=1)(spectra)
        self.assertEqual(d1.domain, d2.domain)
        self.assertEqual(hash(d1.domain), hash(d2.domain))

        with r2.unlocked():
            r2[0][0] = 1
        d2 = ME_EMSC(reference=r2, ncomp=False, weights=False, max_iter=1)(spectra)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))


class TestInflectionPointWeighting(unittest.TestCase):

    def test_weights_from_inflection_points_old_use(self):
        ws = 5  # spacing between wavenumbers
        dx = 0.094  # hardcoded constant from weights_from_inflection_points_legacy

        wns = np.arange(0, 5000, ws)

        # at > 3700 towards 0, at > 1900 towards 1
        weights = weights_from_inflection_points_legacy([3700, 2800, 1900, 0], [1, 1, 1, 1], wns)
        oldx = wns + ws  # shift due to a bug in weights_from_inflection_points_legacy
        oldy = weights.X[0]
        new = Sum(SmoothedSelectionFunction(2800, 3700, ws / dx, 1),
                  SmoothedSelectionFunction(-1000, 1900, ws / dx, 1))
        newy = new(oldx)
        self.assertLess(np.max(np.abs(newy-oldy)), 1e-7)

        # at > 3700 towards 0, at > 1000 towards 0
        weights = weights_from_inflection_points_legacy([3700, 2800, 1900, 1000], [1, 1, 1, 1], wns)
        oldx = wns + ws  # shift due to a bug in weights_from_inflection_points_legacy
        oldy = weights.X[0]
        new = Sum(SmoothedSelectionFunction(2800, 3700, ws / dx, 1),
                  SmoothedSelectionFunction(1000, 1900, ws / dx, 1))
        newy = new(oldx)
        self.assertLess(np.max(np.abs(newy-oldy)), 1e-7)
