import os
import unittest
from unittest.mock import patch
import io
from base64 import b64decode

import numpy as np
from PIL import Image

try:
    import dask
    from Orange.tests.test_dasktable import temp_dasktable
except ImportError:
    dask = None

from AnyQt.QtCore import QPointF, Qt, QRectF
from AnyQt.QtTest import QSignalSpy
import Orange
from Orange.data import DiscreteVariable, Domain, Table
from Orange.widgets.tests.base import WidgetTest
from Orange.util import OrangeDeprecationWarning

from orangecontrib.spectroscopy.data import _spectra_from_image, build_spec_table
from orangecontrib.spectroscopy.io.util import VisibleImage
from orangecontrib.spectroscopy.preprocess.integrate import IntegrateFeaturePeakSimple, \
    Integrate, IntegrateFeatureSimple
from orangecontrib.spectroscopy.widgets import owhyper
from orangecontrib.spectroscopy.widgets.owhyper import \
    OWHyper
from orangecontrib.spectroscopy.preprocess import Interpolate
from orangecontrib.spectroscopy.widgets.line_geometry import in_polygon, is_left
from orangecontrib.spectroscopy.tests.util import hold_modifiers, set_png_graph_save
from orangecontrib.spectroscopy.utils import values_to_linspace, \
    index_values, location_values, index_values_nan

NAN = float("nan")


def wait_for_image(widget, timeout=5000):
    spy = QSignalSpy(widget.imageplot.image_updated)
    assert spy.wait(timeout), "Failed update image in the specified timeout"


class TestReadCoordinates(unittest.TestCase):

    def test_linspace(self):
        v = values_to_linspace(np.array([1, 2, 3]))
        np.testing.assert_equal(np.linspace(*v), [1, 2, 3])
        v = values_to_linspace(np.array([1, 2, 3, np.nan]))
        np.testing.assert_equal(np.linspace(*v), [1, 2, 3])
        v = values_to_linspace(np.array([1]))
        np.testing.assert_equal(np.linspace(*v), [1])
        v = values_to_linspace(np.array([1.001, 2, 3.002]))
        np.testing.assert_equal(np.linspace(*v), [1.001, 2.0015, 3.002])

    def test_index(self):
        a = np.array([1, 2, 3])
        v = values_to_linspace(a)
        iv = index_values(a, v)
        np.testing.assert_equal(iv, [0, 1, 2])
        a = np.array([1, 2, 3, 4])
        v = values_to_linspace(a)
        iv = index_values(a, v)
        np.testing.assert_equal(iv, [0, 1, 2, 3])
        a = np.array([1, 2, 3, 6, 5])
        v = values_to_linspace(a)
        iv = index_values(a, v)
        np.testing.assert_equal(iv, [0, 1, 2, 5, 4])

    def test_index_nan(self):
        a = np.array([1, 2, 3, np.nan])
        v = values_to_linspace(a)
        iv, nans = index_values_nan(a, v)
        np.testing.assert_equal(iv[:-1], [0, 1, 2])
        np.testing.assert_equal(nans, [0, 0, 0, 1])

    def test_location(self):
        lsc = values_to_linspace(np.array([1, 1, 1]))  # a constant
        lv = location_values([0, 1, 2], lsc)
        np.testing.assert_equal(lv, [-1, 0, 1])


class TestPolygonSelection(unittest.TestCase):

    def test_is_left(self):
        self.assertGreater(is_left(0, 0, 0, 1, -1, 0), 0)
        self.assertLess(is_left(0, 0, 0, -1, -1, 0), 0)

    def test_point(self):
        poly = [(0, 1), (1, 0), (2, 1), (3, 0), (3, 2), (0, 1)]  # non-convex

        self.assertFalse(in_polygon([0, 0], poly))
        self.assertTrue(in_polygon([1, 1.1], poly))
        self.assertTrue(in_polygon([1, 1], poly))
        self.assertTrue(in_polygon([1, 0.5], poly))
        self.assertFalse(in_polygon([2, 0], poly))
        self.assertFalse(in_polygon([0, 2], poly))

        # multiple points at once
        np.testing.assert_equal([False, True], in_polygon([[0, 0], [1, 1]], poly))

    def test_order(self):
        poly = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]  # square
        self.assertTrue(in_polygon([0.5, 0.5], poly))
        self.assertTrue(in_polygon([0.5, 0.5], list(reversed(poly))))


class TestOWHyper(WidgetTest):

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
        irisunknown = Interpolate(np.arange(20))(cls.iris)[:20]
        # dataset without any attributes, but XY
        whitelight0 = cls.whitelight.transform(
            Orange.data.Domain([], None, metas=cls.whitelight.domain.metas))[:100]
        unknowns = cls.iris[::10].copy()
        with unknowns.unlocked():
            unknowns.X[:, :] = float("nan")
        single_pixel = cls.iris[:50]  # all image coordinates map to one spot
        unknown_pixel = single_pixel.copy()
        with unknown_pixel.unlocked():
            unknown_pixel.Y[:] = float("nan")
        cls.strange_data = [None, cls.iris1, iris0, empty, irisunknown, whitelight0,
                            unknowns, single_pixel, unknown_pixel]

    def setUp(self):
        self.widget = self.create_widget(OWHyper)  # type: OWHyper

    def test_feature_init(self):
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.attr_value, self.iris.domain.class_var)
        attr1, attr2, attr3 = self.iris.domain.attributes[:3]
        self.assertEqual(self.widget.rgb_red_value, attr1)
        self.assertEqual(self.widget.rgb_green_value, attr2)
        self.assertEqual(self.widget.rgb_blue_value, attr3)
        self.send_signal("Data", self.iris1)
        self.assertEqual(self.widget.attr_value, attr1)
        self.assertEqual(self.widget.rgb_red_value, attr1)
        self.assertEqual(self.widget.rgb_green_value, attr1)
        self.assertEqual(self.widget.rgb_blue_value, attr1)

    def test_integral_lines(self):
        w = self.widget
        self.send_signal(OWHyper.Inputs.data, self.iris)
        icombo = w.controls.integration_method
        for i in range(icombo.count()):
            icombo.setCurrentIndex(i)
            icombo.activated.emit(i)
            wait_for_image(w)
            imethod = w.integration_methods[w.integration_method]
            if imethod == Integrate.PeakAt:
                correct = [False, False, True, False, False]
            elif imethod == Integrate.Separate:
                correct = [True, True, False, True, True]
            else:
                correct = [True, True, False, False, False]
            visible = [a.isVisible() for a in [w.line1, w.line2, w.line3, w.line4, w.line5]]
            self.assertEqual(visible, correct)

    def try_big_selection(self):
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(100, 100))
        self.widget.imageplot.make_selection(None)

    def test_strange(self):
        for data in self.strange_data:
            self.widget = self.create_widget(OWHyper)
            self.send_signal("Data", data)
            wait_for_image(self.widget)
            self.try_big_selection()

    def test_context_not_open_invalid(self):
        self.send_signal("Data", self.iris1)
        self.assertIsNone(self.widget.imageplot.attr_x)
        self.send_signal("Data", self.iris)
        self.assertIsNotNone(self.widget.imageplot.attr_x)

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
        self.send_signal("Data", self.whitelight[:10])
        wait_for_image(self.widget)
        levels = self.widget.imageplot.img.levels
        self.send_signal("Data", self.whitelight_unknown[:10])
        wait_for_image(self.widget)
        levelsu = self.widget.imageplot.img.levels
        np.testing.assert_equal(levelsu, levels)

    def test_select_all(self):
        self.send_signal("Data", self.whitelight)
        wait_for_image(self.widget)

        out = self.get_output("Selection")
        self.assertIsNone(out, None)

        # select all
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(1000, 1000))
        out = self.get_output("Selection")
        self.assertEqual(len(self.whitelight), len(out))

        # test if mixing increasing and decreasing works
        self.widget.imageplot.select_square(QPointF(1000, -100), QPointF(-100, 1000))
        out = self.get_output("Selection")
        self.assertEqual(len(self.whitelight), len(out))

        # deselect
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(-100, -100))
        out = self.get_output("Selection")
        self.assertIsNone(out, None)

        # select specific points
        self.widget.imageplot.select_square(QPointF(53.20, 30.0), QPointF(53.21, 30.03))
        out = self.get_output("Selection")
        np.testing.assert_almost_equal(out.metas,
                                       [[53.2043, 30.0185], [53.2043, 30.0085]],
                                       decimal=3)
        np.testing.assert_equal([o[out.domain["Group"]].value for o in out],
                                ["G1", "G1"])

    def test_select_polygon_as_rectangle(self):
        # rectangle and a polygon need to give the same results
        self.send_signal("Data", self.whitelight)
        wait_for_image(self.widget)
        self.widget.imageplot.select_square(QPointF(53, 30), QPointF(54, 31))
        out = self.get_output("Selection")
        self.widget.imageplot.select_polygon([QPointF(53, 30), QPointF(53, 31), QPointF(54, 31),
                                              QPointF(54, 30), QPointF(53, 30)])
        outpoly = self.get_output("Selection")
        self.assertEqual(list(out), list(outpoly))

    def test_select_click(self):
        self.send_signal("Data", self.whitelight)
        wait_for_image(self.widget)
        self.widget.imageplot.select_by_click(QPointF(53.2443, 30.6984))
        out = self.get_output("Selection")
        np.testing.assert_almost_equal(out.metas, [[53.2443, 30.6984]], decimal=3)

    def test_select_line_change_file(self):
        self.send_signal("Data", self.whitelight)
        wait_for_image(self.widget)
        # select whole image row
        self.widget.imageplot.select_line(QPointF(50, 30.6), QPointF(55, 30.6))
        out = self.get_output("Selection")
        self.assertEqual(len(out), 200)
        self.send_signal("Data", self.iris)
        wait_for_image(self.widget)
        out = self.get_output("Selection")
        self.assertIsNone(out, None)

    def test_select_click_multiple_groups(self):
        data = self.whitelight
        self.send_signal("Data", data)
        wait_for_image(self.widget)
        self.widget.imageplot.select_by_click(QPointF(53.2, 30))
        with hold_modifiers(self.widget, Qt.ShiftModifier):
            self.widget.imageplot.select_by_click(QPointF(53.4, 30))
        with hold_modifiers(self.widget, Qt.ShiftModifier):
            self.widget.imageplot.select_by_click(QPointF(53.6, 30))
        with hold_modifiers(self.widget, Qt.ControlModifier):
            self.widget.imageplot.select_by_click(QPointF(53.8, 30))
        out = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(out), 20000)  # have a data table at the output
        newvars = out.domain.variables + out.domain.metas
        oldvars = data.domain.variables + data.domain.metas
        group_at = [a for a in newvars if a not in oldvars][0]
        unselected = group_at.to_val("Unselected")
        out = out[np.asarray(np.flatnonzero(
            out.transform(Orange.data.Domain([group_at])).X != unselected))]
        self.assertEqual(len(out), 4)
        np.testing.assert_almost_equal([o["map_x"].value for o in out],
                                       [53.1993, 53.3993, 53.5993, 53.7993],
                                       decimal=3)
        np.testing.assert_equal([o[group_at].value for o in out], ["G1", "G2", "G3", "G3"])
        out = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_equal([o[out.domain["Group"]].value for o in out],
                                ["G1", "G2", "G3", "G3"])

        # remove one element
        with hold_modifiers(self.widget, Qt.AltModifier):
            self.widget.imageplot.select_by_click(QPointF(53.2, 30))
        out = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_equal(len(out), 3)
        np.testing.assert_equal([o[out.domain["Group"]].value for o in out],
                                ["G2", "G3", "G3"])

    def test_select_a_curve(self):
        self.send_signal("Data", self.iris)
        self.widget.curveplot.make_selection([0])

    def test_settings_curves(self):
        self.send_signal("Data", self.iris)
        self.widget.curveplot.feature_color = self.iris.domain.class_var
        self.send_signal("Data", self.whitelight)
        self.assertEqual(self.widget.curveplot.feature_color, None)
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.feature_color.name, "iris")

    def test_set_variable_color(self):
        data = self.iris
        ndom = Orange.data.Domain(data.domain.attributes[:-1], data.domain.class_var,
                                  metas=[data.domain.attributes[-1]])
        data = data.transform(ndom)
        self.send_signal("Data", data)
        self.widget.controls.value_type.buttons[1].click()
        with patch("orangecontrib.spectroscopy.widgets.owhyper.ImageItemNan.setLookupTable") as p:
            # a discrete variable
            self.widget.attr_value = data.domain["iris"]
            self.widget.imageplot.update_color_schema()
            self.widget.update_feature_value()
            wait_for_image(self.widget)
            np.testing.assert_equal(len(p.call_args[0][0]), 3)  # just 3 colors for 3 values
            # a continuous variable
            self.widget.attr_value = data.domain["petal width"]
            self.widget.imageplot.update_color_schema()
            self.widget.update_feature_value()
            wait_for_image(self.widget)
            np.testing.assert_equal(len(p.call_args[0][0]), 256)  # 256 for a continuous variable

    def test_color_variable_levels(self):
        class_values = ["a"], ["a", "b", "c"]
        correct_levels = [0, 0], [0, 2]
        for values, correct in zip(class_values, correct_levels):
            domain = Domain([], DiscreteVariable("c", values=values))
            data = Table.from_numpy(domain, X=[[]], Y=[[0]])
            self.send_signal("Data", data)
            self.widget.controls.value_type.buttons[1].click()
            self.widget.attr_value = data.domain.class_var
            self.widget.update_feature_value()
            wait_for_image(self.widget)
            np.testing.assert_equal(self.widget.imageplot.img.levels, correct)

    def test_single_update_view(self):
        with patch("orangecontrib.spectroscopy.widgets.owhyper.ImagePlot.update_view") as p:
            self.send_signal("Data", self.iris)
            self.assertEqual(p.call_count, 1)

    def test_correct_legend(self):
        self.send_signal("Data", self.iris)
        wait_for_image(self.widget)
        self.assertTrue(self.widget.imageplot.legend.isVisible())
        self.widget.controls.value_type.buttons[1].click()
        wait_for_image(self.widget)
        self.assertFalse(self.widget.imageplot.legend.isVisible())

    def test_migrate_selection(self):
        c = QPointF()  # some we set an attribute to
        setattr(c, "selection", [False, True, True, False])
        settings = {"context_settings": [c]}
        OWHyper.migrate_settings(settings, 2)
        self.assertEqual(settings["imageplot"]["selection_group_saved"], [(1, 1), (2, 1)])

    def test_color_no_data(self):
        self.send_signal("Data", None)
        self.widget.controls.value_type.buttons[1].click()
        self.widget.imageplot.update_color_schema()

    def test_image_too_big_error(self):
        oldval = owhyper.IMAGE_TOO_BIG
        try:
            owhyper.IMAGE_TOO_BIG = 3
            self.send_signal("Data", self.iris)
            wait_for_image(self.widget)
            self.assertTrue(self.widget.Error.image_too_big.is_shown())
        finally:
            owhyper.IMAGE_TOO_BIG = oldval

    def test_save_graph(self):
        self.send_signal("Data", self.iris)
        with set_png_graph_save() as fname:
            self.widget.save_graph()
            self.assertGreater(os.path.getsize(fname), 1000)

    def test_unknown_values_axes(self):
        data = self.iris.copy()
        with data.unlocked():
            data.Y[0] = np.nan
        self.send_signal("Data", data)
        wait_for_image(self.widget)
        self.assertTrue(self.widget.Information.not_shown.is_shown())

    def test_migrate_context_feature_color(self):
        # avoid class_vars in tests, because the setting does not allow them
        iris = self.iris.transform(
            Domain(self.iris.domain.attributes, None, self.iris.domain.class_vars))
        c = self.widget.settingsHandler.new_context(iris.domain, None, iris.domain.metas)
        c.values["curveplot"] = {"feature_color": ("iris", 1)}
        self.widget = self.create_widget(OWHyper,
                                         stored_settings={"context_settings": [c]})
        self.send_signal("Data", iris)
        self.assertIsInstance(self.widget.curveplot.feature_color, DiscreteVariable)

    def test_image_computation(self):
        spectra = [[[0, 0, 2, 0],
                    [0, 0, 1, 0]],
                   [[1, 2, 2, 0],
                    [0, 1, 1, 0]]]
        wns = [0, 1, 2, 3]
        x_locs = [0, 1]
        y_locs = [0, 1]
        data = build_spec_table(*_spectra_from_image(spectra, wns, x_locs, y_locs))

        def last_called_array(m):
            arrays = [a[0][0] for a in m.call_args_list
                      if a and a[0] and isinstance(a[0][0], np.ndarray)]
            return arrays[-1]

        wrap = self.widget.imageplot.img

        # integral from zero
        self.widget.integration_method = \
            self.widget.integration_methods.index(IntegrateFeatureSimple)
        self.send_signal("Data", data)
        with patch.object(wrap, 'setImage', wraps=wrap.setImage) as m:
            wait_for_image(self.widget)
            called = last_called_array(m)
            target = [[2, 1], [4.5, 2]]
            np.testing.assert_equal(called.squeeze(), target)

        # peak from zero
        self.widget.integration_method = \
            self.widget.integration_methods.index(IntegrateFeaturePeakSimple)
        self.widget._change_integral_type()
        with patch.object(wrap, 'setImage', wraps=wrap.setImage) as m:
            wait_for_image(self.widget)
            called = last_called_array(m)
            target = [[2, 1], [2, 1]]
            np.testing.assert_equal(called.squeeze(), target)

        # single wavenumber (feature)
        self.widget.controls.value_type.buttons[1].click()
        self.widget.attr_value = data.domain.attributes[1]
        self.widget.update_feature_value()
        with patch.object(wrap, 'setImage', wraps=wrap.setImage) as m:
            wait_for_image(self.widget)
            called = last_called_array(m)
            target = [[0, 0], [2, 1]]
            np.testing.assert_equal(called.squeeze(), target)

        # RGB
        self.widget.controls.value_type.buttons[2].click()
        self.widget.rgb_red_value = data.domain.attributes[0]
        self.widget.rgb_green_value = data.domain.attributes[1]
        self.widget.rgb_blue_value = data.domain.attributes[2]
        self.widget.update_rgb_value()
        with patch.object(wrap, 'setImage', wraps=wrap.setImage) as m:
            wait_for_image(self.widget)
            called = last_called_array(m)
            # first three wavenumbers (features) should be passed to setImage
            target = [data.X[0, :3], data.X[1, :3]], [data.X[2, :3], data.X[3, :3]]
            np.testing.assert_equal(called, target)

    def test_migrate_visual_setttings(self):
        settings = {"curveplot":
                        {"label_title": "title",
                         "label_xaxis": "x",
                         "label_yaxis": "y"}
                    }
        OWHyper.migrate_settings(settings, 6)
        self.assertEqual(settings["visual_settings"],
                         {('Annotations', 'Title', 'Title'): 'title',
                          ('Annotations', 'x-axis title', 'Title'): 'x',
                          ('Annotations', 'y-axis title', 'Title'): 'y'})
        settings = {}
        OWHyper.migrate_settings(settings, 6)
        self.assertNotIn("visual_settings", settings)

    def test_compat_no_group(self):
        settings = {}
        OWHyper.migrate_settings(settings, 6)
        self.assertEqual(settings, {})
        self.widget = self.create_widget(OWHyper, stored_settings=settings)
        self.assertFalse(self.widget.compat_no_group)

        settings = {}
        OWHyper.migrate_settings(settings, 5)
        self.assertEqual(settings, {"compat_no_group": True})
        self.widget = self.create_widget(OWHyper, stored_settings=settings)
        self.assertTrue(self.widget.compat_no_group)



@unittest.skipUnless(dask, "installed Orange does not support dask")
class TestOWHyperWithDask(TestOWHyper):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = temp_dasktable("iris")
        cls.whitelight = temp_dasktable("whitelight.gsf")
        cls.whitelight_unknown = temp_dasktable(cls.whitelight_unknown)
        cls.iris1 = temp_dasktable(cls.iris1)
        cls.strange_data = [temp_dasktable(d) if d is not None else None
                            for d in cls.strange_data]


class _VisibleImageStream(VisibleImage):
    """ Do not use this class in practice because too many things
    will get copied when transforming tables."""

    def __init__(self, name, pos_x, pos_y, size_x, size_y, stream):
        super().__init__(name, pos_x, pos_y, size_x, size_y)
        self.stream = stream

    @property
    def image(self):
        return Image.open(self.stream)


class TestVisibleImage(WidgetTest):

    @classmethod
    def mock_visible_image_data_oldformat(cls):
        red_img = io.BytesIO(b64decode("iVBORw0KGgoAAAANSUhEUgAAAA"
                                       "oAAAAKCAYAAACNMs+9AAAAFUlE"
                                       "QVR42mP8z8AARIQB46hC+ioEAG"
                                       "X8E/cKr6qsAAAAAElFTkSuQmCC"))
        black_img = io.BytesIO(b64decode("iVBORw0KGgoAAAANSUhEUgAAA"
                                         "AoAAAAKCAQAAAAnOwc2AAAAEU"
                                         "lEQVR42mNk+M+AARiHsiAAcCI"
                                         "KAYwFoQ8AAAAASUVORK5CYII="))

        return [
            {
                "name": "Image 01",
                "image_ref": red_img,
                "pos_x": 100,
                "pos_y": 100,
                "pixel_size_x": 1.7,
                "pixel_size_y": 2.3
            },
            {
                "name": "Image 02",
                "image_ref": black_img,
                "pos_x": 0.5,
                "pos_y": 0.5,
                "pixel_size_x": 1,
                "pixel_size_y": 0.3
            },
            {
                "name": "Image 03",
                "image_ref": red_img,
                "pos_x": 100,
                "pos_y": 100,
                "img_size_x": 17.0,
                "img_size_y": 23.0
            },
        ]

    @classmethod
    def mock_visible_image_data(cls):
        red_img = io.BytesIO(b64decode("iVBORw0KGgoAAAANSUhEUgAAAA"
                                       "oAAAAKCAYAAACNMs+9AAAAFUlE"
                                       "QVR42mP8z8AARIQB46hC+ioEAG"
                                       "X8E/cKr6qsAAAAAElFTkSuQmCC"))
        black_img = io.BytesIO(b64decode("iVBORw0KGgoAAAANSUhEUgAAA"
                                         "AoAAAAKCAQAAAAnOwc2AAAAEU"
                                         "lEQVR42mNk+M+AARiHsiAAcCI"
                                         "KAYwFoQ8AAAAASUVORK5CYII="))

        return [
            _VisibleImageStream("Image 01", 100, 100, 17., 23., red_img),
            _VisibleImageStream("Image 02", 0.5, 0.5, 10, 3, black_img),
            _VisibleImageStream("Image 03", 100, 100, 17., 23., red_img),
        ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data_with_visible_images = Orange.data.Table(
            "agilent/4_noimage_agg256.dat"
        )
        cls.data_with_visible_images.attributes["visible_images"] = \
            cls.mock_visible_image_data()

    def setUp(self):
        self.widget = self.create_widget(OWHyper)  # type: OWHyper

    def assert_same_visible_image(self, img_info, vis_img, mock_rect):
        img = img_info.image.convert('RGBA')
        img = np.array(img)[::-1]
        rect = QRectF(img_info.pos_x, img_info.pos_y,
                      img_info.size_x, img_info.size_y)
        self.assertTrue((vis_img.image == img).all())
        mock_rect.assert_called_with(rect)

    def test_no_visible_image(self):
        data = Orange.data.Table("agilent/4_noimage_agg256.dat")
        self.send_signal("Data", data)
        wait_for_image(self.widget)

        self.assertFalse(self.widget.visbox.isEnabled())

    def test_controls_enabled_when_visible_image(self):
        w = self.widget
        self.send_signal("Data", self.data_with_visible_images)
        wait_for_image(w)

        self.assertTrue(w.visbox.isEnabled())

    def test_controls_enabled_by_show_chkbox(self):
        w = self.widget
        self.send_signal("Data", self.data_with_visible_images)
        wait_for_image(w)

        self.assertTrue(w.controls.show_visible_image.isEnabled())
        self.assertFalse(w.show_visible_image)
        controls = [w.controls.visible_image,
                    w.controls.visible_image_composition,
                    w.controls.visible_image_opacity]
        for control in controls:
            self.assertFalse(control.isEnabled())

        w.controls.show_visible_image.setChecked(True)
        for control in controls:
            self.assertTrue(control.isEnabled())

    def test_first_visible_image_selected_in_combobox_by_default(self):
        w = self.widget
        vis_img = w.imageplot.vis_img
        with patch.object(vis_img, 'setRect', wraps=vis_img.setRect) as mock_rect:
            data = self.data_with_visible_images
            self.send_signal("Data", data)
            wait_for_image(w)

            w.controls.show_visible_image.setChecked(True)
            self.assertEqual(len(w.visible_image_model),
                             len(data.attributes["visible_images"]))
            self.assertEqual(w.visible_image, data.attributes["visible_images"][0])
            self.assertEqual(w.controls.visible_image.currentIndex(), 0)
            self.assertEqual(w.controls.visible_image.currentText(), "Image 01")

            self.assert_same_visible_image(data.attributes["visible_images"][0],
                                           w.imageplot.vis_img,
                                           mock_rect)

    def test_visible_image_displayed(self):
        w = self.widget
        data = self.data_with_visible_images
        self.send_signal("Data", data)
        wait_for_image(w)

        self.assertNotIn(w.imageplot.vis_img, w.imageplot.plot.items)

        w.controls.show_visible_image.setChecked(True)
        self.assertIn(w.imageplot.vis_img, w.imageplot.plot.items)
        w.controls.show_visible_image.setChecked(False)
        self.assertNotIn(w.imageplot.vis_img, w.imageplot.plot.items)

    def test_hide_visible_image_after_no_image_loaded(self):
        w = self.widget
        data = self.data_with_visible_images
        self.send_signal("Data", data)
        wait_for_image(w)

        w.controls.show_visible_image.setChecked(True)
        data = Orange.data.Table("agilent/4_noimage_agg256.dat")
        self.send_signal("Data", data)
        wait_for_image(w)

        self.assertFalse(w.visbox.isEnabled())
        self.assertFalse(w.show_visible_image)
        self.assertNotIn(w.imageplot.vis_img, w.imageplot.plot.items)

    def test_select_another_visible_image(self):
        w = self.widget
        data = self.data_with_visible_images
        self.send_signal("Data", data)
        wait_for_image(w)

        w.controls.show_visible_image.setChecked(True)
        vis_img = w.imageplot.vis_img
        with patch.object(vis_img, 'setRect', wraps=vis_img.setRect) as mock_rect:
            w.controls.visible_image.setCurrentIndex(1)
            # since activated signal emitted only by visual interaction
            # we need to trigger it by hand here.
            w.controls.visible_image.activated.emit(1)

            self.assert_same_visible_image(data.attributes["visible_images"][1],
                                           w.imageplot.vis_img,
                                           mock_rect)

    def test_visible_image_opacity(self):
        w = self.widget
        data = self.data_with_visible_images
        self.send_signal("Data", data)
        wait_for_image(w)

        with patch.object(w.imageplot.vis_img, 'setOpacity') as m:
            w.controls.visible_image_opacity.setValue(20)
            self.assertEqual(w.visible_image_opacity, 20)
            m.assert_called_once_with(w.visible_image_opacity / 255)

    def test_visible_image_composition_mode(self):
        w = self.widget
        self.assertEqual(w.controls.visible_image_composition.currentText(), 'Normal')

        data = self.data_with_visible_images
        self.send_signal("Data", data)
        wait_for_image(w)

        for i in range(len(w.visual_image_composition_modes)):
            with patch.object(w.imageplot.vis_img, 'setCompositionMode') as m:
                w.controls.visible_image_composition.setCurrentIndex(i)
                # since activated signal emitted only by visual interaction
                # we need to trigger it by hand here
                w.controls.visible_image_composition.activated.emit(i)
                name = w.controls.visible_image_composition.currentText()
                mode = w.visual_image_composition_modes[name]
                m.assert_called_once_with(mode)

    def test_visible_image_img_size(self):
        w = self.widget
        data = self.data_with_visible_images
        self.send_signal("Data", data)
        wait_for_image(w)

        w.controls.show_visible_image.setChecked(True)
        vis_img = w.imageplot.vis_img
        with patch.object(vis_img, 'setRect', wraps=vis_img.setRect) as mock_rect:
            w.controls.visible_image.setCurrentIndex(2)
            # since activated signal emitted only by visual interaction
            # we need to trigger it by hand here.
            w.controls.visible_image.activated.emit(2)

            self.assert_same_visible_image(data.attributes["visible_images"][0],
                                           w.imageplot.vis_img,
                                           mock_rect)

    def test_oldformat(self):
        data = Orange.data.Table(
            "agilent/4_noimage_agg256.dat"
        )
        data.attributes["visible_images"] = \
            self.mock_visible_image_data_oldformat()
        w = self.widget

        with self.assertWarns(OrangeDeprecationWarning):
            self.send_signal("Data", data)

        wait_for_image(w)

        self.assertNotIn(w.imageplot.vis_img, w.imageplot.plot.items)

        w.controls.show_visible_image.setChecked(True)
        self.assertIn(w.imageplot.vis_img, w.imageplot.plot.items)
        w.controls.show_visible_image.setChecked(False)
        self.assertNotIn(w.imageplot.vis_img, w.imageplot.plot.items)


class TestVectorPlot(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.whitelight = Orange.data.Table("whitelight.gsf")
        cls.iris = Orange.data.Table("iris")

    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWHyper)  # type: OWHyper

    def test_enable_disable(self):
        w = self.widget
        for data in [None, self.whitelight, self.iris]:
            self.send_signal(w.Inputs.data, data)
            self.assertFalse(w.imageplot.show_vector_plot)
            self.assertFalse(w.controls.imageplot.vector_magnitude.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_angle.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_color_index.isEnabled())
            self.assertFalse(w.controls.imageplot.vcol_byval_index.isEnabled())
            self.assertFalse(w.controls.imageplot.vcol_byval_feat.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_scale.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_width.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_opacity.isEnabled())
            self.assertFalse(w.controls.imageplot.v_bin.isEnabled())

            w.controls.imageplot.show_vector_plot.click()
            self.assertTrue(w.imageplot.show_vector_plot)
            self.assertTrue(w.controls.imageplot.vector_magnitude.isEnabled())
            self.assertTrue(w.controls.imageplot.vector_angle.isEnabled())
            self.assertTrue(w.controls.imageplot.vector_color_index.isEnabled())
            self.assertTrue(w.controls.imageplot.vector_scale.isEnabled())
            self.assertTrue(w.controls.imageplot.vector_width.isEnabled())
            self.assertTrue(w.controls.imageplot.vector_opacity.isEnabled())
            self.assertTrue(w.controls.imageplot.v_bin.isEnabled())

            w.imageplot.vector_color_index = 8
            w.imageplot._update_vector()
            self.assertTrue(w.controls.imageplot.vcol_byval_index.isEnabled())
            self.assertTrue(w.controls.imageplot.vcol_byval_feat.isEnabled())

            w.imageplot.vector_color_index = 3
            w.imageplot._update_vector()
            self.assertFalse(w.controls.imageplot.vcol_byval_index.isEnabled())
            self.assertFalse(w.controls.imageplot.vcol_byval_feat.isEnabled())

            w.imageplot.vector_color_index = 8
            w.imageplot._update_vector()
            self.assertTrue(w.controls.imageplot.vcol_byval_index.isEnabled())
            self.assertTrue(w.controls.imageplot.vcol_byval_feat.isEnabled())

            w.controls.imageplot.show_vector_plot.click()
            self.assertFalse(w.controls.imageplot.vector_magnitude.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_angle.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_color_index.isEnabled())
            self.assertFalse(w.controls.imageplot.vcol_byval_index.isEnabled())
            self.assertFalse(w.controls.imageplot.vcol_byval_feat.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_scale.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_width.isEnabled())
            self.assertFalse(w.controls.imageplot.vector_opacity.isEnabled())
            self.assertFalse(w.controls.imageplot.v_bin.isEnabled())

    def test_legend(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.controls.imageplot.show_vector_plot.setChecked(True)
        self.widget.imageplot.enable_vector()
        self.widget.imageplot.vector_color_index = 8
        self.widget.imageplot._update_vector()
        self.assertFalse(self.widget.imageplot.vect_legend.isVisible())
        self.widget.imageplot.vcol_byval_feat = self.iris.domain.attributes[0]
        self.widget.imageplot._update_cbyval()
        self.assertTrue(self.widget.imageplot.vect_legend.isVisible())
        self.widget.imageplot.vcol_byval_feat = None
        self.widget.imageplot._update_cbyval()
        self.assertFalse(self.widget.imageplot.vect_legend.isVisible())
        self.widget.controls.imageplot.show_vector_plot.setChecked(False)
        self.widget.imageplot.enable_vector()
    
    def test_vect_color(self):
        feat = self.iris.get_column(self.iris.domain.attributes[0])
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.controls.imageplot.show_vector_plot.setChecked(True)
        self.widget.imageplot.enable_vector()
        for i in range(8):
            self.widget.imageplot.vector_color_index = i
            self.widget.imageplot._update_vector()
            self.assertEqual(len(self.widget.imageplot.get_vector_color(feat)), 4)
        self.widget.imageplot.vector_color_index = 8
        self.widget.imageplot._update_vector()
        self.assertEqual(self.widget.imageplot.get_vector_color(feat)[0].shape, (feat.shape[0], 4))
        self.widget.controls.imageplot.show_vector_plot.setChecked(False)
        self.widget.imageplot.enable_vector()

    def test_vect_bin(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.controls.imageplot.show_vector_plot.setChecked(True)
        self.widget.imageplot.enable_vector()
        self.widget.imageplot.vector_angle = self.iris.domain.attributes[0]
        self.widget.imageplot.vector_magnitude = self.iris.domain.attributes[0]
        self.widget.imageplot._update_vector_params()

        self.widget.imageplot.v_bin = 0
        self.widget.imageplot._update_binsize()
        self.widget.imageplot.update_view()
        wait_for_image(self.widget)
        print(self.widget.imageplot.vector_plot.params[0].shape)
        self.assertEqual(self.widget.imageplot.vector_plot.params[0].shape[0],
                         self.iris.X.shape[0]*2)
        self.assertEqual(self.widget.imageplot.vector_plot.params[1].shape[0],
                         self.iris.X.shape[0]*2)

        self.widget.imageplot.v_bin = 1
        self.widget.imageplot._update_binsize()
        self.widget.imageplot.update_view()
        wait_for_image(self.widget)
        self.assertEqual(self.widget.imageplot.vector_plot.params[0].shape[0], 2)
        self.assertEqual(self.widget.imageplot.vector_plot.params[1].shape[0], 2)

        self.widget.imageplot.v_bin = 2
        self.widget.imageplot._update_binsize()
        self.widget.imageplot.update_view()
        wait_for_image(self.widget)
        self.assertEqual(self.widget.imageplot.vector_plot.params[0].shape[0], 2)
        self.assertEqual(self.widget.imageplot.vector_plot.params[1].shape[0], 2)
        
        self.widget.imageplot.v_bin = 3
        self.widget.imageplot._update_binsize()
        self.widget.imageplot.update_view()
        wait_for_image(self.widget)
        self.assertTrue(self.widget.Warning.bin_size_error.is_shown())
        self.assertEqual(self.widget.imageplot.vector_plot.params[0].shape[0], 2)
        self.assertEqual(self.widget.imageplot.vector_plot.params[1].shape[0], 2)
