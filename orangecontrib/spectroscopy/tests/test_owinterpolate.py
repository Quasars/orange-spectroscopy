import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owinterpolate import OWInterpolate
from orangecontrib.spectroscopy.data import getx


class TestOWInterpolate(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.collagen = Orange.data.Table("collagen.csv")
        cls.peach = Orange.data.Table("peach_juice.dpt")

    def setUp(self):
        self.widget = self.create_widget(OWInterpolate)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def test_autointerpolate(self):
        self.send_signal("Data", self.collagen)
        out = self.get_output("Interpolated data")
        np.testing.assert_equal(getx(self.collagen), getx(out))
        # no auto-interpolation
        non_interp = self.peach.transform(self.collagen.domain)
        self.assertTrue(np.isnan(non_interp.X).all())
        # auto-interpolation
        auto_interp = self.peach.transform(out.domain)
        self.assertFalse(np.isnan(auto_interp.X).all())
        np.testing.assert_equal(getx(self.collagen), getx(auto_interp))

    def test_interpolate_interval(self):
        self.widget.controls.input_radio.buttons[1].click()
        self.send_signal("Data", self.peach)
        out = self.get_output("Interpolated data")
        np.testing.assert_almost_equal(np.arange(499.53234, 4000.1161, 10), getx(out))
        self.widget.dx = 0
        self.widget.commit.now()
        self.assertTrue(self.widget.Error.dxzero.is_shown())
        self.widget.dx = 0.001
        self.widget.commit.now()
        self.assertTrue(self.widget.Error.too_many_points.is_shown())
        self.widget.dx = 10
        self.widget.commit.now()
        self.assertFalse(self.widget.Error.dxzero.is_shown())
        self.assertFalse(self.widget.Error.too_many_points.is_shown())
        self.widget.xmin = 4000.1161
        self.widget.xmax = 499.53234
        self.widget.commit.now()
        out2 = self.get_output("Interpolated data")
        np.testing.assert_almost_equal(getx(out2), getx(out))
        self.send_signal("Data", None)
        self.assertTrue(self.get_output("Interpolated data") is None)

    def test_interpolate_points(self):
        self.assertFalse(self.widget.Warning.reference_data_missing.is_shown())
        self.widget.controls.input_radio.buttons[2].click()
        self.assertTrue(self.widget.Warning.reference_data_missing.is_shown())
        self.send_signal("Data", self.peach)
        self.assertTrue(self.widget.Warning.reference_data_missing.is_shown())
        self.send_signal("Points", self.collagen)
        self.assertFalse(self.widget.Warning.reference_data_missing.is_shown())
        out = self.get_output("Interpolated data")
        np.testing.assert_equal(getx(self.collagen), getx(out))
        self.send_signal("Points", None)
        self.assertTrue(self.widget.Warning.reference_data_missing.is_shown())

    def test_interpolate_invalid(self):
        self.send_signal("Points", Orange.data.Table("iris"))
        self.assertFalse(self.widget.Error.non_continuous.is_shown())
        self.send_signal("Points", Orange.data.Table("titanic"))
        self.assertTrue(self.widget.Error.non_continuous.is_shown())
