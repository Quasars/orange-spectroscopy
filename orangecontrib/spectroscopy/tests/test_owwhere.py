from Orange.data import Table
import numpy as np
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owwhere import OWWhere


class TestOWWhere(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWWhere)
        # self.data = COLLAGEN_1
        self.widget.old_value=10
        self.widget.tolerance=1
        self.widget.new_value=5
        
        self.input_data = [[1, 10, 10, 10, 1]]

    
    def test_outputs(self):
        self.send_signal("Data", Table.from_numpy(None, self.input_data))
        self.widget.commit()
        self.wait_until_finished(timeout=10000)
        data = self.get_output(self.widget.Outputs.data, wait=10000)
        self.assertTrue(np.all(np.isclose(data, [[1,5,5,5,1]])))
   