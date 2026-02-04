import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owreshape import OWReshape


class TestOWReshape(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.collagen = Orange.data.Table("collagen.csv")

    def setUp(self):
        self.widget = self.create_widget(OWReshape)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def test_no_data_warning(self):
        self.assertTrue(self.widget.Warning.nodata.is_shown())
        self.send_signal("Data", self.collagen)
        self.assertFalse(self.widget.Warning.nodata.is_shown())

    def test_line(self):
        self.send_signal("Data", self.collagen)
        self.widget.xpoints = 1
        self.widget.ypoints = 731
        self.widget.commit.now()
        m = self.get_output("Map data")
        np.testing.assert_equal(m[:, "X"].metas[:, 0], 0)
        np.testing.assert_equal(m[:, "Y"].metas[:, 0], range(len(self.collagen)))
        # the other direction
        self.widget.ypoints = 1
        self.widget.xpoints = 731
        self.widget.commit.now()
        m = self.get_output("Map data")
        np.testing.assert_equal(m[:, "Y"].metas[:, 0], 0)
        np.testing.assert_equal(m[:, "X"].metas[:, 0], range(len(self.collagen)))

    def test_wrong_dimensions(self):
        self.send_signal("Data", self.collagen)
        self.widget.xpoints = 2
        self.widget.ypoints = 400
        self.widget.le1_changed()  # get warnings activated
        self.widget.commit.now()
        self.assertTrue(self.widget.Warning.wrong_div.is_shown())
        self.assertIsNone(self.get_output("Map data"))

    def test_rect(self):
        self.send_signal("Data", self.collagen[:500])
        self.widget.xpoints = 5
        self.widget.le1_changed()
        self.assertEqual(self.widget.ypoints, 100)
        self.widget.commit.now()
        m = self.get_output("Map data")
        self.assertEqual(m[1]["X"].value, 1)
        self.assertEqual(m[1]["Y"].value, 0)
        self.assertEqual(m[102]["X"].value, 2)
        self.assertEqual(m[102]["Y"].value, 20)

    def test_invert(self):
        self.send_signal("Data", self.collagen[:500])
        self.widget.xpoints = 5
        self.widget.le1_changed()
        self.assertEqual(self.widget.ypoints, 100)
        self.widget.invert_x = True
        self.widget.invert_y = True
        self.widget.commit.now()
        m = self.get_output("Map data")
        self.assertEqual(m[1]["X"].value, 3)
        self.assertEqual(m[1]["Y"].value, 99)
        self.assertEqual(m[102]["X"].value, 2)
        self.assertEqual(m[102]["Y"].value, 79)

    def test_var_name_exists(self):
        self.send_signal("Data", self.collagen[:500])
        self.widget.xpoints = 5
        self.widget.le1_changed()
        self.widget.commit.now()
        m = self.get_output("Map data")
        self.send_signal("Data", m)
        self.widget.commit.now()
        m = self.get_output("Map data")
        np.testing.assert_equal(m[:, "X"].metas[:, 0], m[:, "X (1)"].metas[:, 0])
        np.testing.assert_equal(m[:, "Y"].metas[:, 0], m[:, "Y (1)"].metas[:, 0])
