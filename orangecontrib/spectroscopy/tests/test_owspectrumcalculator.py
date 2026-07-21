from Orange.data import Table
import numpy as np
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owoperations import OWOperations


class TestOWOperations(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWOperations)
        
        self.input_data_p0 = [[1, 2, 3, 4, 5]]
        self.input_data_p1 = [[10, 20, 30, 40, 50]]
        self.input_data_s0 = [[5, 4, 3, 2, 1]]
        self.input_data_s1 = [[50, 40, 30, 20, 10]]

    
    def test_addition(self):
        self.send_signal("Primary Data", Table.from_numpy(None, self.input_data_p0))
        self.send_signal("Secondary Data", Table.from_numpy(None, self.input_data_s0))
        
        # set addition
        self.widget.operation_index = 0

        self.commit_and_wait()
        data = self.get_output()
        self.assertTrue(np.all(np.isclose(data, [[6,6,6,6,6]])))
    
    def test_subtraction(self):
        self.send_signal("Primary Data", Table.from_numpy(None, self.input_data_p1))
        self.send_signal("Secondary Data", Table.from_numpy(None, self.input_data_s0))
        
        # set subtraction
        self.widget.operation_index = 1

        self.commit_and_wait()
        data = self.get_output()
        self.assertTrue(np.all(np.isclose(data, [[5,16,27,38,49]])))

    def test_multiplication(self):
        self.send_signal("Primary Data", Table.from_numpy(None, self.input_data_p1))
        self.send_signal("Secondary Data", Table.from_numpy(None, self.input_data_s0))
        
        # set multiplication
        self.widget.operation_index = 2

        self.commit_and_wait()
        data = self.get_output()
        self.assertTrue(np.all(np.isclose(data, [[50,80,90,80,50]])))
    
    def test_division(self):
        self.send_signal("Primary Data", Table.from_numpy(None, self.input_data_s1))
        self.send_signal("Secondary Data", Table.from_numpy(None, self.input_data_s0))
        
        # set division
        self.widget.operation_index = 3

        self.commit_and_wait()
        data = self.get_output()
        self.assertTrue(np.all(np.isclose(data, [[10,10,10,10,10]])))
   
    def test_scaled_multiplication(self):
        self.send_signal("Primary Data", Table.from_numpy(None, self.input_data_p1))
        self.send_signal("Secondary Data", Table.from_numpy(None, self.input_data_s0))
        
        # set scaled multiplication
        self.widget.operation_index = 4
        self.widget.factor = 3

        self.commit_and_wait()
        data = self.get_output()
        self.assertTrue(np.all(np.isclose(data, [[150, 240, 270, 240, 150]])))
    

