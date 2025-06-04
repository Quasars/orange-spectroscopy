import numpy as np

import Orange
from Orange.data import dataset_dirs
from Orange.data.io import FileFormat
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.io.neaspec import NeaReaderGSF
from orangecontrib.spectroscopy import irfft
from orangecontrib.spectroscopy.widgets.owfft import OWFFT, CHUNK_SIZE, DEFAULT_HENE


class TestOWFFT(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWFFT)
        self.ifg_single = Orange.data.Table("IFG_single.dpt")
        self.ifg_seq = Orange.data.Table("agilent/4_noimage_agg256.seq")
        fn = 'NeaReaderGSF_test/NeaReaderGSF_test O2A raw.gsf'
        absolute_filename = FileFormat.locate(fn, dataset_dirs)
        self.ifg_gsf = NeaReaderGSF(absolute_filename).read()

    def test_load_unload(self):
        self.send_signal("Interferogram", self.ifg_single)
        self.send_signal("Interferogram", None)

    def test_laser_metadata(self):
        """ Test dx in presence/absence of laser metadata """
        self.send_signal("Interferogram", self.ifg_seq)
        self.assertEqual(self.widget.dx, (1 / 1.57980039e+04 / 2) * 4)
        self.send_signal("Interferogram", self.ifg_single)
        self.assertEqual(self.widget.dx, (1 / DEFAULT_HENE / 2))

    def test_respect_custom_dx(self):
        """ Setting new data should not overwrite custom dx value """
        self.send_signal("Interferogram", self.ifg_single)
        self.widget.dx_auto = False
        self.widget.dx = 5
        self.widget.dx_auto_changed()
        self.send_signal("Interferogram", self.ifg_single)
        self.assertEqual(self.widget.dx, 5)
        self.assertEqual(self.widget.dx_auto, False)
        # Disconnent, reconnect
        self.send_signal("Interferogram", None)
        self.send_signal("Interferogram", self.ifg_single)
        self.assertEqual(self.widget.dx, 5)
        self.assertEqual(self.widget.dx_auto, False)

    def test_auto_dx(self):
        self.send_signal("Interferogram", self.ifg_seq)
        self.assertEqual(self.widget.dx, (1 / 1.57980039e+04 / 2) * 4)
        self.send_signal("Interferogram", self.ifg_gsf)
        self.assertEqual(self.widget.dx, (0.00019550342130987293))

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

    def test_complex_calculation(self):
        """" Test calculation Complex FFT """

        self.widget.zff = 2  # 2**2 = 4
        self.widget.limit_output = False
        self.widget.peak_search = 1 # MINIMUM
        self.widget.apod_func = 0 # boxcar

        self.send_signal(self.widget.Inputs.data, self.ifg_gsf)
        self.commit_and_wait()
        # testing info panel text associated with the input file metadata
        widget_text = self.widget.info_dx.text()
        self.assertIn('Using Calculated Datapoint Spacing (Δx) from metadata', widget_text)
        self.assertTrue(self.widget.use_interleaved_data)
        self.assertTrue(self.widget.complexfft)
        self.assertTrue(self.widget.apod_asym)

        spectra = self.get_output(self.widget.Outputs.spectra)
        phases = self.get_output(self.widget.Outputs.phases)
        np.testing.assert_allclose(spectra.X.size, (2049))
        np.testing.assert_allclose(phases.X.size, (2049))
        np.testing.assert_allclose(spectra.X[0, 396:399], (23.67618359, 25.02051088, 25.82566789))
        np.testing.assert_allclose(phases.X[0, 396:399], (2.61539453, 2.65495979, 2.72814989))

    def test_migrate_HeNe(self):
        settings = {"dx_HeNe": False}
        OWFFT.migrate_settings(settings, 1)
        self.assertEqual(settings["dx_auto"], False)
        settings = {"dx_HeNe": True}
        OWFFT.migrate_settings(settings, 1)
        self.assertEqual(settings["dx_auto"], True)
