import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owsnr import OWSNR


class TestOWSNR(WidgetTest):

    @classmethod
    def setUpClass(cls): # carregando dado de teste
        super().setUpClass()
        cls.file_test = Orange.data.Table("three_coordinates_data.csv")

    def setUp(self):
        self.widget = self.create_widget(OWSNR)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def test_no_data_warning(self):
        self.assertTrue(self.widget.Warning.nodata.is_shown())
        self.send_signal("Data", self.file_test)
        self.assertFalse(self.widget.Warning.nodata.is_shown())

    def test_domain(self):
        self.send_signal("Data", self.file_test)
        out = self.get_output("SNR")
        self.assertEqual(self.file_test.domain.attributes, out.domain.attributes)

    def test_1coordinate_snr(self):
        self.send_signal("Data", self.file_test)
        self.widget.out_choiced = 0
        self.widget.out_choice_changed()
        self.widget.group_x = self.widget.group_axis_x[1]
        self.widget.out_choice_changed()
        out = self.get_output("SNR")
        ref = [1.0726, 1.46003, 1.28718]
        np.testing.assert_equal(out.X.shape, (5, 10))
        np.testing.assert_allclose(out.X[0, :3], ref, rtol=1e-05, atol=1e-05)

    def test_1coordinate_average(self):
        self.send_signal("Data", self.file_test)
        self.widget.out_choiced = 1
        self.widget.group_y = self.widget.group_axis_y[2]
        self.widget.out_choice_changed()
        out = self.get_output("SNR")
        ref = [0.08537, 0.0684601, 0.0553439]
        np.testing.assert_equal(out.X.shape, (5, 10))
        np.testing.assert_allclose(out.X[0, :3], ref, rtol=1e-06, atol=1e-06)
        np.testing.assert_equal(out.metas[:3, :2].astype(float),
                                [[np.nan, 0], [np.nan, 1], [np.nan, 2]])

    def test_none_coordinate_std(self):
        self.send_signal("Data", self.file_test)
        self.widget.out_choiced = 2
        self.widget.out_choice_changed()
        out = self.get_output("SNR")
        ref = [0.0598001, 0.0440469, 0.0508131]
        np.testing.assert_equal(out.X.shape, (1, 10))
        np.testing.assert_allclose(out.X[0, :3], ref, rtol=1e-06, atol=1e-06)

    def test_2coordinates_std(self):
        self.send_signal("Data", self.file_test)
        self.widget.out_choiced = 0
        self.widget.group_x = self.widget.group_axis_x[1]
        self.widget.group_y = self.widget.group_axis_y[2]
        self.widget.out_choice_changed()
        out = self.get_output("SNR")
        ref = [2.89239, 1.44214, 1.97268]
        np.testing.assert_equal(out.X.shape, (25, 10))
        np.testing.assert_allclose(out.X[0, :3], ref, rtol=1e-05, atol=1e-05)
        np.testing.assert_equal(out.metas[:3, :2], [[0, 0], [1, 0], [2, 0]])
        np.testing.assert_equal(out.metas[-3:, :2], [[2, 4], [3, 4], [4, 4]])
        print(out.metas)