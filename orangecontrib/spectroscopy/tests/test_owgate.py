import unittest

from Orange.data import Table

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owgate import OWGate

class TestOWGate(WidgetTest):
    def setUp(self):
        self.iris = Table("iris")
        self.zoo = Table("zoo")
        self.widget = self.create_widget(OWGate)

    
    def _get_output(self):
        return self.get_output(self.widget.Outputs.data)
    
    def _get_not_connected(self):
        return self.widget.Warning.not_connected.is_shown()
    
    def _close_gate(self):
        self.widget.controls.autocommit.setChecked(False)

    def _open_gate(self):
        self.widget.controls.autocommit.setChecked(True)



    def test_autocommit_changes(self):
        from orangecontrib.flow.tests.utils import checkbox_linked_test

        checkbox_linked_test(self, self.widget,
                             "autocommit", "autocommit")

    
    def test_default(self):
        self.assertIsNone(
            self.widget.in_data,
            "in_data should be None"
        )

        self.assertIsNone(
            self.widget.out_data,
            "out_data should be None"
        )

        self.assertIsNone(
            self._get_output(),
            "output should be None"
        )

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )

    
    def test_closed_input(self):
        self._close_gate()

        self.send_signal(self.widget.Inputs.data, self.iris)

        self.assertEqual(
            self.widget.in_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertIsNone(
            self.widget.out_data,
            "out_data should be None"
        )

        self.assertIsNone(
            self._get_output(),
            "output should be None"
        )

        self.assertTrue(
            self._get_not_connected(),
            "not_connected should be raised"
        )

        # Disconnect

        self.send_signal(self.widget.Inputs.data, None)

        self.assertIsNone(
            self.widget.in_data,
            "in_data should be None"
        )

        self.assertIsNone(
            self.widget.out_data,
            "out_data should be None"
        )

        self.assertIsNone(
            self._get_output(),
            "output should be None"
        )

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )



    def test_commit_input(self):
        self._close_gate()

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.commit_and_wait()

        self.assertEqual(
            self.widget.in_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self.widget.out_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self._get_output(),
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )

        # Change data
        self.send_signal(self.widget.Inputs.data, self.zoo)

        self.assertEqual(
            self.widget.in_data,
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self.widget.out_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self._get_output(),
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertTrue(
            self._get_not_connected(),
            "not_connected should be raised"
        )

        # Revert back to previous dataset
        self.send_signal(self.widget.Inputs.data, self.iris)

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )

        # Change data again
        self.send_signal(self.widget.Inputs.data, self.zoo)

        self.assertTrue(
            self._get_not_connected(),
            "not_connected should be raised"
        )

        self.commit_and_wait()

        self.assertEqual(
            self.widget.in_data,
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self.widget.out_data,
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self._get_output(),
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )

    def test_open_input(self):
        self._open_gate()

        self.send_signal(self.widget.Inputs.data, self.iris)

        self.assertEqual(
            self.widget.in_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self.widget.out_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self._get_output(),
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )

        # Change data
        self.send_signal(self.widget.Inputs.data, self.zoo)

        self.assertEqual(
            self.widget.in_data,
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self.widget.out_data,
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self._get_output(),
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )

        # Close gate
        self._close_gate()

        # Change data again
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(
            self.widget.in_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self.widget.out_data,
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self._get_output(),
            self.zoo,
            "in_data doesn't have the correct data"
        )

        self.assertTrue(
            self._get_not_connected(),
            "not_connected should be raised"
        )

        # Open gate
        self._open_gate()

        self.assertEqual(
            self.widget.in_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self.widget.out_data,
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertEqual(
            self._get_output(),
            self.iris,
            "in_data doesn't have the correct data"
        )

        self.assertFalse(
            self._get_not_connected(),
            "not_connected shouldn't be raised"
        )



if __name__ == "__main__":
    unittest.main()
