import unittest

import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.owsnrregion import (
    OWSNRRegion,
    RMS,
    PEAK_TO_PEAK,
    LINEAR,
    NOISE_PRESETS,
    QUADRATIC,
    compute_snr,
    noise_peak_to_peak,
    noise_rms,
    signal_max,
)


class TestSNRFunctions(unittest.TestCase):
    def test_rms_removes_linear_trend(self):
        x = np.arange(10.0)
        noise = np.array([1, -1] * 5, dtype=float)
        ys = np.vstack([3 * x + 2 + noise, -x + noise])
        rms, fits = noise_rms(x, ys, LINEAR)
        # alternating noise is not fully orthogonal to a line; compare to
        # the residuals of an explicit numpy fit
        for y, r in zip(ys, rms, strict=True):
            res = y - np.polyval(np.polyfit(x, y, 1), x)
            self.assertAlmostEqual(r, np.sqrt(np.sum(res**2) / len(x)))
        np.testing.assert_allclose(rms[0], rms[1])

    def test_rms_quadratic(self):
        x = np.linspace(-1, 1, 21)
        y = 4 * x**2 - x + 1
        rms, fits = noise_rms(x, y, QUADRATIC)
        np.testing.assert_allclose(rms, [0], atol=1e-12)
        np.testing.assert_allclose(fits[0], y)
        rms_lin, _ = noise_rms(x, y, LINEAR)
        self.assertGreater(rms_lin[0], 0.1)

    def test_rms_nan(self):
        x = np.arange(6.0)
        ys = np.array(
            [
                [0, 1, 0, 1, 0, 1],
                [0, 1, np.nan, 1, 0, 1],
                [np.nan, np.nan, np.nan, np.nan, np.nan, 1],
            ]
        )
        rms, _ = noise_rms(x, ys, LINEAR)
        valid = [0, 1, 3, 4, 5]
        y = ys[1, valid]
        res = y - np.polyval(np.polyfit(x[valid], y, 1), x[valid])
        self.assertAlmostEqual(rms[1], np.sqrt(np.sum(res**2) / 5))
        self.assertTrue(np.isfinite(rms[0]))
        self.assertTrue(np.isnan(rms[2]))  # too few points for a fit

    def test_peak_to_peak_and_signal(self):
        ys = np.array([[1, 5, -2], [np.nan, 3, 1], [np.nan, np.nan, np.nan]])
        np.testing.assert_equal(noise_peak_to_peak(ys), [7, 2, np.nan])
        np.testing.assert_equal(signal_max(ys), [5, 3, np.nan])

    def test_compute_snr(self):
        x = np.array([5.0, 4, 3, 2, 1, 0])  # descending order
        X = np.array([[10, 1, 1, 2, 2, 1]], dtype=float)
        signal, noise, snr = compute_snr(x, X, (0, 3), (6, 4), method=PEAK_TO_PEAK)
        np.testing.assert_equal(signal, [10])
        np.testing.assert_equal(noise, [1])
        np.testing.assert_equal(snr, [10])

        signal, noise, snr = compute_snr(x, X, (0, 3), (4, 6), method=RMS)
        self.assertAlmostEqual(snr[0], 10 / noise[0])
        self.assertAlmostEqual(noise[0], 0.5)

    def test_empty_region(self):
        x = np.arange(5.0)
        X = np.ones((2, 5))
        signal, noise, snr = compute_snr(x, X, (10, 11), (0, 4))
        np.testing.assert_equal(noise, [np.nan, np.nan])
        np.testing.assert_equal(signal, [1, 1])
        np.testing.assert_equal(snr, [np.nan, np.nan])


class TestOWSNRRegion(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")
        cls.collagen = Orange.data.Table("collagen")[:20]

    def setUp(self):
        self.widget = self.create_widget(OWSNRRegion)

    def test_load_unload(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_output(self):
        data = self.collagen
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(out), len(data))
        self.assertEqual(out.domain.attributes, data.domain.attributes)
        self.assertEqual(
            [m.name for m in out.domain.metas[-3:]], ["Signal", "Noise", "SNR"]
        )
        x = getx(data)
        w = self.widget
        signal, noise, snr = compute_snr(
            x,
            data.X,
            (w.noise_low, w.noise_high),
            (w.signal_low, w.signal_high),
            RMS,
            LINEAR,
        )
        np.testing.assert_allclose(
            out.metas[:, -3:].astype(float), np.column_stack([signal, noise, snr])
        )
        np.testing.assert_allclose(
            out.metas[:, -1].astype(float),
            out.metas[:, -3].astype(float) / out.metas[:, -2].astype(float),
        )

    def test_default_regions(self):
        self.send_signal(self.widget.Inputs.data, self.collagen)
        x = getx(self.collagen)
        w = self.widget
        self.assertEqual(w.signal_low, x.min())
        self.assertEqual(w.signal_high, x.max())
        self.assertEqual(w.noise_low, x.min())
        self.assertLess(w.noise_high, x.max())

    def test_methods(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.collagen)
        x = getx(self.collagen)
        w.noise_low, w.noise_high = 1500, 1800
        w.signal_low, w.signal_high = 1600, 1700
        results = {}
        for method, fit in [(RMS, 0), (RMS, 1), (PEAK_TO_PEAK, 0)]:
            w.noise_method = method
            w.fit_order = fit
            w._method_changed()
            self.assertEqual(w.fit_combo.isEnabled(), method == RMS)
            out = self.get_output(w.Outputs.data)
            _, noise, _ = compute_snr(
                x, self.collagen.X, (1500, 1800), (1600, 1700), method, fit + 1
            )
            np.testing.assert_allclose(out.metas[:, -2].astype(float), noise)
            results[(method, fit)] = noise
        # quadratic fit can only reduce residuals
        self.assertTrue(np.all(results[(RMS, 1)] <= results[(RMS, 0)] + 1e-12))
        # peak-to-peak is always at least as large as RMS
        self.assertTrue(np.all(results[(PEAK_TO_PEAK, 0)] >= results[(RMS, 0)]))

    def test_line_moves_update_settings(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.collagen)
        line = w.lines["noise_high"]
        line.setValue(1700)
        line.sigMoved.emit(line.value())
        line.sigMoveFinished.emit(line.value())
        self.assertAlmostEqual(float(w.noise_high), 1700, places=0)
        out = self.get_output(w.Outputs.data)
        _, noise, _ = compute_snr(
            getx(self.collagen),
            self.collagen.X,
            (float(w.noise_low), float(w.noise_high)),
            (w.signal_low, w.signal_high),
        )
        np.testing.assert_allclose(out.metas[:, -2].astype(float), noise)

    def test_preview(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.collagen)
        self.assertEqual(len(w.preview_markings), 1)  # fit curve
        self.assertIn("SNR", w.info_label.text())
        w.noise_method = PEAK_TO_PEAK
        w._method_changed()
        self.assertEqual(len(w.preview_markings), 2)  # min, max

    def test_warnings(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.collagen)
        w.noise_low, w.noise_high = 10, 20
        w.settings_changed()
        w.commit.now()
        self.assertTrue(w.Warning.empty_noise.is_shown())
        out = self.get_output(w.Outputs.data)
        self.assertTrue(np.all(np.isnan(out.metas[:, -2].astype(float))))

        x = np.sort(getx(self.collagen))
        w.noise_low, w.noise_high = x[0], x[1]
        w.fit_order = 1
        w.settings_changed()
        w.commit.now()
        self.assertFalse(w.Warning.empty_noise.is_shown())
        self.assertTrue(w.Warning.few_points.is_shown())

    def test_lock_regions(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.collagen)
        x = getx(self.collagen)
        w.noise_low, w.noise_high = 1600, 1700
        w.signal_low, w.signal_high = 1000, 1100
        w.lock_regions = True
        w._lock_changed()
        self.assertFalse(w.editors["signal_low"].isEnabled())
        self.assertFalse(w.lines["signal_high"].isVisible())
        out = self.get_output(w.Outputs.data)
        signal, noise, _ = compute_snr(x, self.collagen.X, (1600, 1700), (1600, 1700))
        np.testing.assert_allclose(out.metas[:, -3].astype(float), signal)
        np.testing.assert_allclose(out.metas[:, -2].astype(float), noise)

        # unlocking restores the separate signal region
        w.lock_regions = False
        w._lock_changed()
        self.assertTrue(w.editors["signal_low"].isEnabled())
        self.assertEqual((w.signal_low, w.signal_high), (1000, 1100))
        out = self.get_output(w.Outputs.data)
        signal, _, _ = compute_snr(x, self.collagen.X, (1600, 1700), (1000, 1100))
        np.testing.assert_allclose(out.metas[:, -3].astype(float), signal)

    def test_presets(self):
        w = self.widget
        rng = np.random.default_rng(0)
        domain = Orange.data.Domain(
            [Orange.data.ContinuousVariable(str(v)) for v in range(0, 6001, 10)]
        )
        data = Orange.data.Table.from_numpy(domain, rng.normal(size=(5, 601)))
        self.send_signal(w.Inputs.data, data)
        self.assertEqual(w.preset_combo.currentText(), "Custom")

        for i, (name, (a, b)) in enumerate(NOISE_PRESETS, start=1):
            w.preset_combo.setCurrentIndex(i)
            w.preset_combo.activated.emit(i)
            self.assertTrue(w.preset_combo.currentText().startswith(name))
            self.assertEqual(
                (float(w.noise_low), float(w.noise_high)), (min(a, b), max(a, b))
            )
            out = self.get_output(w.Outputs.data)
            _, noise, _ = compute_snr(
                getx(data), data.X, (a, b), (w.signal_low, w.signal_high)
            )
            np.testing.assert_allclose(out.metas[:, -2].astype(float), noise)

        # moving a line away from a preset switches back to custom
        line = w.lines["noise_low"]
        line.setValue(4000)
        line.sigMoved.emit(line.value())
        self.assertEqual(w.preset_combo.currentText(), "Custom")

        # typing the preset values selects the preset
        w.noise_low, w.noise_high = 1900, 2100
        w.settings_changed()
        self.assertTrue(w.preset_combo.currentText().startswith("MIR"))

    def test_saved_regions_kept(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.collagen)
        w.noise_low, w.noise_high = 1500, 1800
        settings = self.widget.settingsHandler.pack_data(w)
        w2 = self.create_widget(OWSNRRegion, stored_settings=settings)
        self.send_signal(w2.Inputs.data, self.collagen, widget=w2)
        self.assertEqual((w2.noise_low, w2.noise_high), (1500, 1800))

    def test_no_attributes(self):
        data = self.iris.transform(Orange.data.Domain([], self.iris.domain.class_var))
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.no_x.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))


if __name__ == "__main__":
    unittest.main()
