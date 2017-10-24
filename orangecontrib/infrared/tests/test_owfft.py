import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owifg_to_sp import OWFFT


class TestOWFFT(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWFFT)

    def test_load_unload(self):
        self.send_signal("Interferogram", Orange.data.Table("IFG_single.dpt"))
        self.send_signal("Interferogram", None)
