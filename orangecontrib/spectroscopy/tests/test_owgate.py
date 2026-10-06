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

    def _close_gate(self):
        self.widget.controls.autocommit.setChecked(False)

    def _open_gate(self):
        self.widget.controls.autocommit.setChecked(True)


    def test_autocommit_changes(self):
        from orangecontrib.spectroscopy.tests.util import checkbox_linked_test

        checkbox_linked_test(self, self.widget,
                             "autocommit", "autocommit")

    def test_default(self):
        self.assertIsNone(self.widget.in_data)
        self.assertIsNone(self.widget.out_data)
        self.assertIsNone(self._get_output())

    def test_closed_input(self):
        self._close_gate()

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.in_data, self.iris)
        self.assertIsNone(self.widget.out_data)
        self.assertIsNone(self._get_output())
        self.assertTrue(self.widget.Warning.not_connected.is_shown())

        # Disconnect
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.in_data)
        self.assertIsNone(self.widget.out_data)
        self.assertIsNone(self._get_output())

    def test_commit_input(self):
        self._close_gate()

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.commit_and_wait()

        self.assertEqual(self.widget.in_data, self.iris)
        self.assertEqual(self.widget.out_data, self.iris)
        self.assertEqual(self._get_output(), self.iris)
        self.assertFalse(self.widget.Warning.not_connected.is_shown())

        # Change data
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertEqual(self.widget.in_data, self.zoo)
        self.assertEqual(self.widget.out_data, self.iris)
        self.assertEqual(self._get_output(), self.iris)
        self.assertTrue(self.widget.Warning.not_connected.is_shown())

        # Revert back to previous dataset
        self.send_signal(self.widget.Inputs.data, self.iris)

        # Change data again
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertTrue(self.widget.Warning.not_connected.is_shown())

        self.commit_and_wait()

        self.assertEqual(self.widget.in_data, self.zoo)
        self.assertEqual(self.widget.out_data, self.zoo)
        self.assertEqual(self._get_output(), self.zoo)
        self.assertFalse(self.widget.Warning.not_connected.is_shown())

    def test_open_input(self):
        self._open_gate()

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.in_data, self.iris)
        self.assertEqual(self.widget.out_data, self.iris)
        self.assertEqual(self._get_output(), self.iris)
        self.assertFalse(self.widget.Warning.not_connected.is_shown())

        # Change data
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertEqual(self.widget.in_data, self.zoo)
        self.assertEqual(self.widget.out_data, self.zoo)
        self.assertEqual(self._get_output(), self.zoo)
        self.assertFalse(self.widget.Warning.not_connected.is_shown())

        self._close_gate()

        # Change data again
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.in_data, self.iris)
        self.assertEqual(self.widget.out_data, self.zoo)
        self.assertEqual(self._get_output(), self.zoo)
        self.assertTrue(self.widget.Warning.not_connected.is_shown())

        # Open gate
        self._open_gate()
        self.assertEqual(self.widget.in_data, self.iris)
        self.assertEqual(self.widget.out_data, self.iris)
        self.assertEqual(self._get_output(), self.iris)
        self.assertFalse(self.widget.Warning.not_connected.is_shown())


if __name__ == "__main__":
    unittest.main()
