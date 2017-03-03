import unittest
import numpy as np

from AnyQt.QtCore import QPointF
import Orange
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.infrared.widgets.owhyper import values_to_linspace, \
    index_values, OWHyper, location_values

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


class TestOWCurves(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")
        cls.whitelight = Orange.data.Table("whitelight.gsf")
        cls.whitelight_unknown = cls.whitelight.copy()
        cls.whitelight_unknown[0]["value"] = NAN

    def setUp(self):
        self.widget = self.create_widget(OWHyper)

    def try_big_selection(self):
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(100, 100), False)
        self.widget.imageplot.make_selection(None, False)
        self.widget.imageplot.make_selection(None, True)

    def test_empty(self):
        self.send_signal("Data", None)
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

        # deselect
        self.widget.imageplot.select_square(QPointF(-100, -100), QPointF(-100, -100), False)
        out = self.get_output("Selection")
        self.assertIsNone(out, None)

    def test_select_a_curve(self):
        self.send_signal("Data", self.iris)
        self.widget.curveplot.make_selection([0], False)
