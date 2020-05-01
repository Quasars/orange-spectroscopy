from unittest import TestCase

import numpy as np

from sklearn.cross_decomposition import PLSRegression

from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, ParameterMapping

from orangecontrib.spectroscopy.widgets.owpls import OWPLS
from orangecontrib.spectroscopy.models.pls import PLSRegressionLearner


def table(rows, attr, vars):
    attr_vars = [ContinuousVariable(name="Feature %i" % i) for i in range(attr)]
    class_vars = [ContinuousVariable(name="Class %i" % i) for i in range(vars)]
    domain = Domain(attr_vars, class_vars, [])
    X = np.ones((rows, attr))
    Y = np.ones((rows, vars))
    return Table.from_numpy(domain, X=X, Y=Y)


class TestPLS(TestCase):

    def test_allow_y_dim(self):
        """ The current PLS version allows only a single Y dimension. """
        d = table(10, 5, 1)
        learner = PLSRegressionLearner(n_components=2)
        learner(d)
        for n_class_vars in [0, 2]:
            d = table(10, 5, n_class_vars)
            with self.assertRaises(ValueError):
                learner(d)

    def test_compare_to_sklearn(self):
        d = table(10, 5, 1)
        d.X = np.random.RandomState(0).rand(*d.X.shape)
        d.Y = np.random.RandomState(0).rand(*d.Y.shape)
        orange_model = PLSRegressionLearner()(d)
        scikit_model = PLSRegression().fit(d.X, d.Y)
        np.testing.assert_almost_equal(scikit_model.predict(d.X).ravel(),
                                       orange_model(d))
        np.testing.assert_almost_equal(scikit_model.coef_,
                                       orange_model.coefficients)


class TestOWPLS(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWPLS,
                                         stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            ParameterMapping('max_iter', self.widget.n_iters),
            ParameterMapping('n_components', self.widget.ncomps_spin)]
