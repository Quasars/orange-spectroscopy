import unittest
import numpy as np

from unittest.mock import patch

from AnyQt.QtCore import QPointF
import Orange
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.infrared.widgets.owhyperspectra import values_to_linspace, \
    index_values, OWHyper, location_values
from orangecontrib.infrared.preprocess import Interpolate
from orangecontrib.infrared.widgets.line_geometry import in_polygon, is_left


NAN = float("nan")

class TestReadCoordinates(unittest.TestCase):

    def test_linspace(self):
        v = values_to_linspace(np.array([1, 2, 3]))
        np.testing.assert_equal(np.linspace(*v), [1, 2, 3])
        v = values_to_linspace(np.array([1, 2, 3, float("nan")]))
        np.testing.assert_equal(np.linspace(*v), [1, 2, 3])
        v = values_to_linspace(np.array([1]))
        np.testing.assert_equal(np.linspace(*v), [1])
        v = values_to_linspace(np.array([1.001, 2, 3.002]))
        np.testing.assert_equal(np.linspace(*v), [1.001, 2.0015, 3.002])

    def test_index(self):
        a = np.array([1,2,3])
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

    def test_location(self):
        lsc = values_to_linspace(np.array([1, 1, 1]))  # a constant
        lv = location_values([0,1,2], lsc)
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
        cls.whitelight_unknown[0]["value"] = NAN
        # dataset with a single attribute
        iris1 = Orange.data.Table(Orange.data.Domain(cls.iris.domain[:1]), cls.iris)
        # dataset without any attributes
        iris0 = Orange.data.Table(Orange.data.Domain([]), cls.iris)
        # dataset with large blank regions
        irisunknown = Interpolate(np.arange(20))(cls.iris)
        # dataset without any attributes, but XY
        whitelight0 = Orange.data.Table(Orange.data.Domain([], None,
            metas=cls.whitelight.domain.metas), cls.whitelight)
        cls.strange_data = [None, iris1, iris0, irisunknown, whitelight0]

    def setUp(self):
        self.widget = self.create_widget(OWHyper)

    def try_big_selection(self):
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(100, 100), False)
        self.widget.imageplot.make_selection(None, False)
        self.widget.imageplot.make_selection(None, True)

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

    def test_select_all(self):
        self.send_signal("Data", self.whitelight)

        out = self.get_output("Selection")
        self.assertIsNone(out, None)

        # select all
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(1000, 1000), False)
        out = self.get_output("Selection")
        self.assertEqual(len(self.whitelight), len(out))

        # test if mixing increasing and decreasing works
        self.widget.imageplot.select_square(QPointF(1000, -100), QPointF(-100, 1000), False)
        out = self.get_output("Selection")
        self.assertEqual(len(self.whitelight), len(out))

        # deselect
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(-100, -100), False)
        out = self.get_output("Selection")
        self.assertIsNone(out, None)

        # select specific points
        self.widget.imageplot.select_square(QPointF(9.4, 9.4), QPointF(11.6, 10.6), False)
        out = self.get_output("Selection")
        np.testing.assert_equal(out.metas, [[10, 10], [11, 10]])

    def test_select_polygon_as_rectangle(self):
        # rectangle and a polygon need to give the same results
        self.send_signal("Data", self.whitelight)
        self.widget.imageplot.select_square(QPointF(5, 5), QPointF(15, 10), False)
        out = self.get_output("Selection")
        self.widget.imageplot.select_polygon([QPointF(5, 5), QPointF(15, 5), QPointF(15, 10),
                                              QPointF(5, 10), QPointF(5, 5)], False)
        outpoly = self.get_output("Selection")
        self.assertEqual(list(out), list(outpoly))

    def test_select_click(self):
        self.send_signal("Data", self.whitelight)
        self.widget.imageplot.select_by_click(QPointF(1, 2), False)
        out = self.get_output("Selection")
        np.testing.assert_equal(out.metas, [[1, 2]])

    def test_select_a_curve(self):
        self.send_signal("Data", self.iris)
        self.widget.curveplot.make_selection([0], False)

    def test_settings_curves(self):
        self.send_signal("Data", self.iris)
        self.widget.curveplot.feature_color = "iris"
        self.send_signal("Data", self.whitelight)
        self.assertEqual(self.widget.curveplot.feature_color, None)
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.feature_color, "iris")

    def test_single_update_view(self):
        with patch("orangecontrib.infrared.widgets.owhyper.ImagePlot.update_view") as p:
            self.send_signal("Data", self.iris)
            self.assertEqual(p.call_count, 1)
