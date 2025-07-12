import unittest
from unittest.mock import patch
import numpy as np

import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owoverlay import OWOverlay
from orangecontrib.spectroscopy.tests.test_owhyper import wait_for_image

NAN = float("nan")


class TestOWOverlay(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # cls.iris = Orange.data.Table("iris")
        cls.whitelight = Orange.data.Table("whitelight.gsf")
        cls.whitelight_unknown = cls.whitelight.copy()
        with cls.whitelight_unknown.unlocked():
            cls.whitelight_unknown[0][0] = NAN

    def setUp(self):
        self.widget = self.create_widget(OWOverlay)  # type: OWOverlay

    def test_feature_init(self):
        self.send_signal("Overlay Data", self.whitelight)
        self.assertEqual(self.widget.attr_value.name, "1.000000")

    def test_context_not_open_invalid(self):
        self.send_signal("Overlay Data", None)
        self.assertIsNone(self.widget.imageplot.attr_x)
        self.send_signal("Overlay Data", self.whitelight)
        self.assertIsNotNone(self.widget.imageplot.attr_x)

    def test_unknown(self):
        self.send_signal("Overlay Data", self.whitelight[:10])
        wait_for_image(self.widget)
        levels = self.widget.imageplot.img.levels
        self.send_signal("Overlay Data", self.whitelight_unknown[:10])
        wait_for_image(self.widget)
        levelsu = self.widget.imageplot.img.levels
        np.testing.assert_equal(levelsu, levels)

    def test_set_variable_color(self):
        data = self.whitelight
        self.send_signal("Overlay Data", data)
        with patch(
            "orangecontrib.spectroscopy.widgets.owhyper.ImageItemNan.setLookupTable"
        ) as p:
            self.widget.attr_value = data.domain["1.000000"]
            self.widget.imageplot.update_color_schema()
            self.widget._update_feature_value()
            wait_for_image(self.widget)
            np.testing.assert_equal(
                len(p.call_args[0][0]), 256
            )  # 256 for a continuous variable

    def test_add_visible_image(self):
        data = self.whitelight
        self.send_signal("Data", data)
        self.send_signal("Overlay Data", data)
        self.widget._attr_changed()
        self.widget._update_feature_value()
        wait_for_image(self.widget)

        # recommit to avoid a bug because commit functionality is not implemented asynchronously
        # this should be removed when properly implemented
        self.widget.commit.now()

        out = self.get_output("Decorated Data")
        self.assertIsNotNone(out.attributes['visible_images'])

        out = self.get_output("Decorated Data")
        self.assertIsNotNone(out.attributes["visible_images"])


if __name__ == "__main__":
    unittest.main()
