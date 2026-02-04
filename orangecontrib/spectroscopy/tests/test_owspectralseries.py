from unittest.mock import patch, MagicMock

import numpy as np

from AnyQt.QtCore import QPointF
from AnyQt.QtWidgets import QToolTip

import Orange
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owspectralseries import OWSpectralSeries
from orangecontrib.spectroscopy.preprocess import Interpolate

NAN = float("nan")


class TestOWSpectralSeries(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")
        cls.whitelight = Orange.data.Table("whitelight.gsf")
        cls.whitelight_unknown = cls.whitelight.copy()
        with cls.whitelight_unknown.unlocked():
            cls.whitelight_unknown[0][0] = NAN
        # dataset with a single attribute
        cls.iris1 = cls.iris.transform(Orange.data.Domain(cls.iris.domain[:1]))
        # dataset without any attributes
        iris0 = cls.iris.transform(Orange.data.Domain([]))
        # dataset without rows
        empty = cls.iris[:0]
        # dataset with large blank regions
        irisunknown = Interpolate(np.arange(20))(cls.iris)
        # dataset without any attributes, but XY
        whitelight0 = cls.whitelight.transform(
            Orange.data.Domain([], None, metas=cls.whitelight.domain.metas)
        )
        cls.strange_data = [None, cls.iris1, iris0, empty, irisunknown, whitelight0]

    def setUp(self):
        self.widget = self.create_widget(OWSpectralSeries)

    def try_big_selection(self):
        all_select = None if self.widget.data is None else [1] * len(self.widget.data)
        self.widget.imageplot.make_selection(all_select)
        self.widget.imageplot.make_selection(None)

    def test_strange(self):
        for data in self.strange_data:
            self.send_signal("Data", data)
            self.try_big_selection()

    def test_no_samples(self):
        self.send_signal("Data", self.whitelight[:0])
        self.try_big_selection()

    def test_few_samples(self):
        self.send_signal("Data", self.whitelight[:1])
        self.send_signal("Data", self.whitelight[:2])
        self.send_signal("Data", self.whitelight[:3])
        self.try_big_selection()

    def test_simple(self):
        self.send_signal("Data", self.whitelight)
        self.send_signal("Data", None)
        self.try_big_selection()
        self.assertIsNone(self.get_output("Selection"), None)

    def test_unknown(self):
        self.send_signal("Data", self.whitelight)
        levels = self.widget.imageplot.img.levels
        self.send_signal("Data", self.whitelight_unknown)
        levelsu = self.widget.imageplot.img.levels
        np.testing.assert_equal(levelsu, levels)

    def test_select_click(self):
        self.send_signal("Data", self.whitelight)
        self.widget.imageplot.select_by_click(QPointF(0, 1))
        # work by indices
        out = self.get_output("Selection")
        np.testing.assert_almost_equal(out.metas[:, 0], 53.0693, decimal=3)
        np.testing.assert_almost_equal(out.metas[:, 1], 30.7085, decimal=3)
        np.testing.assert_equal(out.Y, 0)  # selection group
        # select a feature
        self.widget.imageplot.attr_x = "map_x"
        self.widget.imageplot.update_attr()
        self.widget.imageplot.select_by_click(QPointF(53.1, 1))
        out = self.get_output("Selection")
        np.testing.assert_almost_equal(out.metas[:, 0], 53.099327, decimal=3)
        np.testing.assert_almost_equal(
            out.metas[:, 1], self.whitelight[::200].metas[:, 1], decimal=3
        )
        np.testing.assert_equal(out.Y, 0)  # selection group

    def test_single_update_view(self):
        uw = "orangecontrib.spectroscopy.widgets.owspectralseries.LineScanPlot.update_view"
        with patch(uw) as p:
            self.send_signal("Data", self.iris)
            self.assertEqual(p.call_count, 1)

    def test_tooltip(self):
        data = self.iris
        self.send_signal(self.widget.Inputs.data, data)

        event = MagicMock()
        with (
            patch.object(self.widget.imageplot.plot.vb, "mapSceneToView"),
            patch.object(QToolTip, "showText") as show_text,
        ):
            sel = np.zeros(len(data), dtype="bool")

            sel[3] = 1  # a single instance
            with patch.object(
                self.widget.imageplot, "_points_at_pos", return_value=(sel, 2)
            ):
                self.assertTrue(self.widget.imageplot.help_event(event))
                (_, text), _ = show_text.call_args
                self.assertIn("iris = {}".format(data[3, "iris"]), text)
                self.assertIn("value = {}".format(data[3, 2]), text)
                self.assertEqual(1, text.count("iris ="))

            sel[51] = 1  # add a data point
            with patch.object(
                self.widget.imageplot, "_points_at_pos", return_value=(sel, 2)
            ):
                self.assertTrue(self.widget.imageplot.help_event(event))
                (_, text), _ = show_text.call_args
                self.assertIn("iris = {}".format(data[3, "iris"]), text)
                self.assertIn("iris = {}".format(data[51, "iris"]), text)
                self.assertIn("value = {}".format(data[3, 2]), text)
                self.assertIn("value = {}".format(data[51, 2]), text)
                self.assertEqual(2, text.count("iris ="))

    def test_compat_no_group(self):
        settings = {}
        OWSpectralSeries.migrate_settings(settings, 2)
        self.assertEqual(settings, {})
        self.widget = self.create_widget(OWSpectralSeries, stored_settings=settings)
        self.assertFalse(self.widget.compat_no_group)

        settings = {}
        OWSpectralSeries.migrate_settings(settings, 1)
        self.assertEqual(settings, {"compat_no_group": True})
        self.widget = self.create_widget(OWSpectralSeries, stored_settings=settings)
        self.assertTrue(self.widget.compat_no_group)
