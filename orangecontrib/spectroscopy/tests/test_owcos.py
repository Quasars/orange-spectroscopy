import numpy
import numpy as np

import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.owcos import OWCos, calc_cos, sort_data


class TestOWCOS(WidgetTest):
    # make test data
    WN = Orange.data.Domain([Orange.data.ContinuousVariable(str(w)) for w in [3, 1, 2]])
    X1 = np.array([[1, 2, 3], [4, 5, 6]])
    X2 = np.array([[2, 3, 4], [6, 7, 8]])

    DATA1 = Orange.data.Table.from_numpy(WN, X1)
    DATA2 = Orange.data.Table.from_numpy(WN, X2)

    def setUp(self):
        self.widget = self.create_widget(OWCos)

    def test_calc_cos(self):
        cos = calc_cos(self.DATA1, self.DATA2)
        numpy.testing.assert_array_equal(
            cos[0], [[6.0, 6.0, 6.0], [6.0, 8.0, 4.0], [6.0, 4.0, 8.0]]
        )

        numpy.testing.assert_array_almost_equal(
            cos[1],
            [
                [0.0, -0.95492966, 0.95492966],
                [1.27323954, 0.31830989, 2.2281692],
                [-1.27323954, -2.2281692, -0.31830989],
            ],
        )

        numpy.testing.assert_array_equal(cos[2], [[-1.5, -0.5, -2.5], [1.5, 2.5, 0.5]])

        numpy.testing.assert_array_equal(cos[3], [[-2.0, -1.0, -3.0], [2.0, 3.0, 1.0]])

        numpy.testing.assert_array_equal(cos[4], [1.0, 2.0, 3.0])

        numpy.testing.assert_array_equal(cos[5], [1.0, 2.0, 3.0])

    def test_sort_data(self):
        sorted_data = sort_data(self.DATA1)
        numpy.testing.assert_array_equal(sorted_data, [[2, 3, 1], [5, 6, 4]])
        numpy.testing.assert_array_equal(getx(sorted_data), [1, 2, 3])
