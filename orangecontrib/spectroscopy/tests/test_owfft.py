import numpy as np

import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy import irfft
from orangecontrib.spectroscopy.widgets.owfft import OWFFT, CHUNK_SIZE


class TestOWFFT(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWFFT)
        self.ifg_single = Orange.data.Table("IFG_single.dpt")
        self.ifg_seq = Orange.data.Table("agilent/4_noimage_agg256.seq")

    def test_load_unload(self):
        self.send_signal("Interferogram", self.ifg_single)
        self.send_signal("Interferogram", None)

    def test_laser_metadata(self):
        """ Test dx in presence/absence of laser metadata """
        self.send_signal("Interferogram", self.ifg_seq)
        self.assertEqual(self.widget.dx, (1 / 1.57980039e+04 / 2) * 4)
        self.send_signal("Interferogram", self.ifg_single)
        self.assertEqual(self.widget.dx, (1 / self.widget.laser_wavenumber / 2))

    def test_respect_custom_dx(self):
        """ Setting new data should not overwrite custom dx value """
        self.send_signal("Interferogram", self.ifg_single)
        self.widget.dx_HeNe = False
        self.widget.dx = 5
        self.widget.dx_changed()
        self.send_signal("Interferogram", self.ifg_single)
        self.assertEqual(self.widget.dx, 5)

    def test_auto_dx(self):
        self.send_signal("Interferogram", self.ifg_seq)
        self.assertEqual(self.widget.dx, (1 / 1.57980039e+04 / 2) * 4)

    def test_keep_metas(self):
        input = self.ifg_seq
        self.send_signal(self.widget.Inputs.data, input)
        self.commit_and_wait()
        spectra = self.get_output(self.widget.Outputs.spectra)
        phases = self.get_output(self.widget.Outputs.phases)
        np.testing.assert_equal(input.metas, spectra.metas)
        np.testing.assert_equal(input.metas, phases.metas[:, :input.metas.shape[1]])

    def test_custom_zpd(self):
        """ Test setting custom zpd value"""
        custom_zpd = 1844
        self.send_signal(self.widget.Inputs.data, self.ifg_single)
        self.widget.peak_search_enable = False
        self.widget.zpd1 = custom_zpd
        self.widget.peak_search_changed()
        self.commit_and_wait()
        phases = self.get_output(self.widget.Outputs.phases)
        self.assertEqual(phases[0, "zpd_fwd"], custom_zpd)

    def test_chunk_one(self):
        """ Test batching when len(data) < chunk_size """
        self.assertLess(len(self.ifg_seq), CHUNK_SIZE)
        self.send_signal(self.widget.Inputs.data, self.ifg_seq)
        self.widget.peak_search_enable = False
        self.widget.zpd1 = 69 # TODO replace with value read from file
        self.widget.peak_search_changed()
        self.commit_and_wait()

    def test_chunk_many(self):
        """ Test batching when len(data) >> chunk_size """
        data = Orange.data.table.Table.concatenate(5 * (self.ifg_seq,))
        self.assertGreater(len(data), CHUNK_SIZE)
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.peak_search_enable = False
        self.widget.zpd1 = 69 # TODO replace with value read from file
        self.widget.peak_search_changed()
        self.commit_and_wait()

    def test_calculation(self):
        """" Test calculation with custom settings and batching """
        ifg_ref = Orange.data.Table("agilent/background_agg256.seq")
        abs = Orange.data.Table("agilent/4_noimage_agg256.dat")

        self.widget.apod_func = irfft.ApodFunc.BLACKMAN_HARRIS_4
        self.widget.zff = 0  # 2**0 = 1
        self.widget.phase_res_limit = False
        self.widget.phase_corr = irfft.PhaseCorrection.MERTZ
        self.widget.setting_changed()
        self.widget.peak_search_enable = False
        self.widget.zpd1 = 69 # TODO replace with value read from file
        self.widget.peak_search_changed()

        self.send_signal(self.widget.Inputs.data, ifg_ref)
        self.commit_and_wait()
        rsc = self.get_output(self.widget.Outputs.spectra)

        self.send_signal(self.widget.Inputs.data, self.ifg_seq)
        self.commit_and_wait()
        ssc = self.get_output(self.widget.Outputs.spectra)

        # Calculate absorbance from ssc and rsc
        calc_abs = np.log10(rsc.X / ssc.X)
        # Match energy region
        abs_x = getx(abs)
        calc_x = getx(ssc)
        limits = np.searchsorted(calc_x, [abs_x[0] - 1, abs_x[-1]])
        np.testing.assert_allclose(calc_x[limits[0]:limits[1]], abs_x)
        # Compare to agilent absorbance
        # NB 4 mAbs error
        np.testing.assert_allclose(calc_abs[:, limits[0]:limits[1]], abs.X, atol=0.004)
