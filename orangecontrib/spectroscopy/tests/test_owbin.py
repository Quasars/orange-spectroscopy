import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.utils import get_ndim_hyperspec
from orangecontrib.spectroscopy.widgets.owbin import OWBin


class TestOWBin(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mosaic = Orange.data.Table("agilent/5_mosaic_agg1024.dmt")

    def setUp(self):
        self.widget = self.create_widget(OWBin)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def test_bin(self):
        self.widget.bin_shape = (2, 2)
        self.widget._init_bins()
        self.send_signal(self.widget.Inputs.data, self.mosaic)
        m = self.get_output(self.widget.Outputs.bindata)
        np.testing.assert_equal(len(m.X), len(self.mosaic.X) / 2**2)
        x_coords = self.mosaic[:, "map_x"].metas[:, 0]
        x_coords_binned = np.array([x_coords[0:2].mean(), x_coords[2:4].mean()])
        np.testing.assert_equal(m[:, "map_x"].metas[::4, 0], x_coords_binned)
        y_coords = self.mosaic[:, "map_y"].metas[:, 0]
        y_coords_binned = np.array([y_coords[0:8].mean(), y_coords[8:16].mean(),
                                    y_coords[16:24].mean(), y_coords[24:32].mean()])
        np.testing.assert_equal(m[:, "map_y"].metas[0:4, 0], y_coords_binned)

    def test_bin_changed(self):
        self.send_signal(self.widget.Inputs.data, self.mosaic)
        self.widget.bin_0 = 2
        self.widget.bin_1 = 2
        self.widget._bin_changed()
        m = self.get_output(self.widget.Outputs.bindata)
        np.testing.assert_equal(len(m.X), len(self.mosaic.X) / 2**2)

    def test_nonsquare_bin(self):
        self.widget.bin_shape = (2, 4)
        self.widget._init_bins()
        self.send_signal(self.widget.Inputs.data, self.mosaic)
        m = self.get_output(self.widget.Outputs.bindata)
        np.testing.assert_equal(len(m.X), len(self.mosaic.X) / (2 * 4))
        x_coords = self.mosaic[:, "map_x"].metas[:, 0]
        x_coords_binned = np.array([x_coords[0:2].mean(), x_coords[2:4].mean()])
        np.testing.assert_equal(m[:, "map_x"].metas[::2, 0], x_coords_binned)
        y_coords = self.mosaic[:, "map_y"].metas[:, 0]
        y_coords_binned = np.array([y_coords[0:16].mean(), y_coords[16:32].mean()])
        np.testing.assert_equal(m[:, "map_y"].metas[0:2, 0], y_coords_binned)

    def test_no_bin(self):
        self.widget.bin_shape = (1, 1)
        self.widget._init_bins()
        self.send_signal(self.widget.Inputs.data, self.mosaic)
        m = self.get_output(self.widget.Outputs.bindata)

        # Comparing hypercube data and axes here instead of Tables because
        # self.mosaic is built (row, column) i.e. (map_y, map_x)
        # but bin_hyperspectra always returns (attr_0, attr_1) i.e. (map_x, map_y)
        # so the resulting tables are arranged differently (but contain the same data).
        xat = [v for v in m.domain.metas if v.name == "map_x"][0]
        yat = [v for v in m.domain.metas if v.name == "map_y"][0]
        attrs = [xat, yat]
        hyper_orig = get_ndim_hyperspec(self.mosaic, attrs)
        hyper_m = get_ndim_hyperspec(m, attrs)
        np.testing.assert_equal(hyper_orig, hyper_m)

    def test_invalid_bin(self):
        self.widget.bin_shape = (3, 3)
        self.widget._init_bins()
        self.send_signal(self.widget.Inputs.data, self.mosaic)
        self.assertTrue(self.widget.Error.invalid_block.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.bindata))

    def test_invalid_axis(self):
        data = self.mosaic.copy()
        with data.unlocked():
            data.metas[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.invalid_axis.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.invalid_axis.is_shown())

    def test_nan_in_image(self):
        data = self.mosaic.copy()
        with data.unlocked():
            data.X[1, 2] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.nan_in_image.is_shown())
        self.send_signal(self.widget.Inputs.data, self.mosaic)
        self.assertFalse(self.widget.Warning.nan_in_image.is_shown())
