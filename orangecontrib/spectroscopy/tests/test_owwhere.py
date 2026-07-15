from Orange.data import Table
import numpy as np
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owwhere import OWWhere


class TestOWWhere(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWWhere)
        # self.data = COLLAGEN_1
        
        self.input_data = [[1, 10, 10, 10, 1]]

    
    def test_tolerance_replacement(self):
        self.widget.old_value=10
        self.widget.tolerance=1
        self.widget.new_value=5
        self.widget.mode = 0

        self.send_signal("Data", Table.from_numpy(None, self.input_data))
        self.widget.commit.now()
        self.wait_until_finished(timeout=10000)
        data = self.get_output(self.widget.Outputs.data, wait=10000)
        self.assertTrue(np.all(np.isclose(data, [[1,5,5,5,1]])))
       
    def test_minimum_replacement(self):
        self.widget.old_min_value=1
        self.widget.new_min_value=5
        self.widget.mode = 1

        self.send_signal("Data", Table.from_numpy(None, self.input_data))
        self.widget.commit.now()
        self.wait_until_finished(timeout=10000)
        data = self.get_output(self.widget.Outputs.data, wait=10000)
        self.assertTrue(np.all(np.isclose(data, [[5,10,10,10,5]])))
       
    def test_maximum_replacement(self):
        self.widget.old_max_value=10
        self.widget.new_max_value=7
        self.widget.mode = 2

        self.send_signal("Data", Table.from_numpy(None, self.input_data))
        self.widget.commit.now()
        self.wait_until_finished(timeout=10000)
        data = self.get_output(self.widget.Outputs.data, wait=10000)
        self.assertTrue(np.all(np.isclose(data, [[1,7,7,7,1]])))
