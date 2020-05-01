import unittest

import numpy as np
import Orange

from orangecontrib.spectroscopy.data import getx

from orangecontrib.spectroscopy.irfft import (IRFFT, zero_fill, PhaseCorrection,
                                              find_zpd, PeakSearch, ApodFunc,
                                              MultiIRFFT,
                                             )

dx = 1.0 / 15797.337544 / 2.0


class TestIRFFT(unittest.TestCase):

    def setUp(self):
        self.ifg_single = Orange.data.Table("IFG_single.dpt")
        self.ifg_seq_ref = Orange.data.Table("agilent/background_agg256.seq")
        self.sc_dat_ref = Orange.data.Table("agilent/background_agg256.dat")

    def test_zero_fill(self):
        N = 1975
        a = np.zeros(N)
        # Test all zff offered in OWFFT()
        for zff in (2**n for n in range(10)):
            a_zf = zero_fill(a, zff)
            N_zf = a_zf.size
            # Final array must be power of 2
            assert np.log2(N_zf) == int(np.log2(N_zf))
            # Final array should be >= N * zff
            assert N_zf >= N * zff

    def test_simple_fft(self):
        data = self.ifg_single.X[0]
        fft = IRFFT(dx=dx)
        fft(data)

    def test_stored_phase_zpd(self):
        data = self.ifg_single.X[0]
        fft = IRFFT(dx=dx)
        fft(data)
        stored_phase = fft.phase
        zpd = fft.zpd
        fft_stored = IRFFT(dx=dx, phase_corr=PhaseCorrection.STORED)
        fft_stored(data, phase=stored_phase, zpd=zpd)
        np.testing.assert_array_equal(fft.spectrum, fft_stored.spectrum)
        np.testing.assert_array_equal(fft.phase, fft_stored.phase)

    def test_peak_search(self):
        data = self.ifg_single.X[0]
        assert find_zpd(data, PeakSearch.MAXIMUM) == data.argmax()
        assert find_zpd(data, PeakSearch.MINIMUM) == data.argmin()
        assert find_zpd(data, PeakSearch.ABSOLUTE) == abs(data).argmax()
        data *= -1
        assert find_zpd(data, PeakSearch.ABSOLUTE) == abs(data).argmax()

    def test_agilent_fft_sc(self):
        ifg = self.ifg_seq_ref.X[0]
        # dat = self.sc_dat_ref.X[0]  # TODO scaling diffrences fail
        dx_ag = (1 / 1.57980039e+04 / 2) * 4
        fft = IRFFT(dx=dx_ag,
                    apod_func=ApodFunc.BLACKMAN_HARRIS_4,
                    zff=1,
                    phase_res=None,
                    phase_corr=PhaseCorrection.MERTZ,
                    peak_search=PeakSearch.MINIMUM)
        fft(ifg)
        self.assertEqual(fft.zpd, 69)
        dat_x = getx(self.sc_dat_ref)
        limits = np.searchsorted(fft.wavenumbers, [dat_x[0] - 1, dat_x[-1]])
        np.testing.assert_allclose(fft.wavenumbers[limits[0]:limits[1]], dat_x)
        # TODO This fails due to scaling differences
        # np.testing.assert_allclose(fft.spectrum[limits[0]:limits[1]], dat)

    def test_agilent_fft_ab(self):
        ifg_ref = self.ifg_seq_ref.X[0]
        ifg_sam = Orange.data.Table("agilent/4_noimage_agg256.seq").X[0]
        dat_T = Orange.data.Table("agilent/4_noimage_agg256.dat")
        dat = dat_T.X[0]
        dx_ag = (1 / 1.57980039e+04 / 2) * 4
        fft = IRFFT(dx=dx_ag,
                    apod_func=ApodFunc.BLACKMAN_HARRIS_4,
                    zff=1,
                    phase_res=None,
                    phase_corr=PhaseCorrection.MERTZ,
                    peak_search=PeakSearch.MINIMUM)
        fft(ifg_ref)
        rsc = fft.spectrum
        fft(ifg_sam)
        ssc = fft.spectrum
        dat_x = getx(dat_T)
        limits = np.searchsorted(fft.wavenumbers, [dat_x[0] - 1, dat_x[-1]])
        np.testing.assert_allclose(fft.wavenumbers[limits[0]:limits[1]], dat_x)
        # Calculate absorbance from ssc and rsc
        ab = np.log10(rsc / ssc)
        # Compare to agilent absorbance
        # NB 4 mAbs error
        np.testing.assert_allclose(ab[limits[0]:limits[1]], dat, atol=0.004)

    def test_multi(self):
        dx_ag = (1 / 1.57980039e+04 / 2) * 4
        fft = MultiIRFFT(dx=dx_ag,
                         apod_func=ApodFunc.BLACKMAN_HARRIS_4,
                         zff=1,
                         phase_res=None,
                         phase_corr=PhaseCorrection.MERTZ,
                         peak_search=PeakSearch.MINIMUM)
        zpd = 69    # from test_agilent_fft_sc(), TODO replace with value read from file
        fft(self.ifg_seq_ref.X, zpd)

    def test_multi_ab(self):
        ifg_ref = self.ifg_seq_ref.X
        ifg_sam = Orange.data.Table("agilent/4_noimage_agg256.seq").X
        dat_T = Orange.data.Table("agilent/4_noimage_agg256.dat")
        dat = dat_T.X
        dx_ag = (1 / 1.57980039e+04 / 2) * 4
        fft = MultiIRFFT(dx=dx_ag,
                         apod_func=ApodFunc.BLACKMAN_HARRIS_4,
                         zff=1,
                         phase_res=None,
                         phase_corr=PhaseCorrection.MERTZ,
                         peak_search=PeakSearch.MINIMUM)
        zpd = 69  # from test_agilent_fft_sc(), TODO replace with value read from file
        fft(ifg_ref, zpd)
        rsc = fft.spectrum
        fft(ifg_sam, zpd)
        ssc = fft.spectrum
        dat_x = getx(dat_T)
        limits = np.searchsorted(fft.wavenumbers, [dat_x[0] - 1, dat_x[-1]])
        np.testing.assert_allclose(fft.wavenumbers[limits[0]:limits[1]], dat_x)
        # Calculate absorbance from ssc and rsc
        ab = np.log10(rsc / ssc)
        # Compare to agilent absorbance
        # NB 4 mAbs error
        np.testing.assert_allclose(ab[:, limits[0]:limits[1]], dat, atol=0.004)
