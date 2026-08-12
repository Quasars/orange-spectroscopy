from Orange.data import Table
import numpy as np
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owunwrap import OWUnwrap

class TestOWReplace(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWUnwrap)
        self.t = np.linspace(0, 25, 801)
        self.signal = 1.5 * np.sin(1.1 * self.t + 0.26) * (1 - self.t / 6 + (self.t / 23) ** 3)
    
    def test_expected_wrap(self):
        input = np.mod(self.signal, 2*np.pi) - 1
        self.input_data = [input]
        self.send_signal("Data", Table.from_numpy(None, self.input_data))
        self.widget.commit.now()
        self.wait_until_finished(timeout=10000)
        data = self.get_output(self.widget.Outputs.data, wait=10000)
        expected = np.unwrap(input, discont=np.pi, period=2*np.pi)
        self.assertTrue(np.all(np.isclose(data, expected)))
    
    def test_expected_no_wrap(self):
        input = self.signal - 1
        self.input_data = [input]
        self.send_signal("Data", Table.from_numpy(None, self.input_data))
        self.widget.commit.now()
        self.wait_until_finished(timeout=10000)
        data = self.get_output(self.widget.Outputs.data, wait=10000)
        expected = input
        self.assertTrue(np.all(np.isclose(data, expected)))
