from unittest import TestCase, skipIf

import numpy as np

from sklearn.cross_decomposition import PLSRegression

from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.tests.base import (
    WidgetTest,
    WidgetLearnerTestMixin,
    ParameterMapping,
)

import Orange.version
from orangecontrib.spectroscopy.widgets.owpls import OWPLS
from orangecontrib.spectroscopy.models.pls import PLSRegressionLearner


def table(rows, attr, vars):
    attr_vars = [ContinuousVariable(name="Feature %i" % i) for i in range(attr)]
    class_vars = [ContinuousVariable(name="Class %i" % i) for i in range(vars)]
    domain = Domain(attr_vars, class_vars, [])
    X = np.random.RandomState(0).random((rows, attr))
    Y = np.random.RandomState(1).random((rows, vars))
    return Table.from_numpy(domain, X=X, Y=Y)


def coefficients(sklmodel):
    coef = sklmodel.coef_
    return coef


class TestPLS(TestCase):
    def test_allow_y_dim(self):
        """The current PLS version allows only a single Y dimension."""
        learner = PLSRegressionLearner(n_components=2)
        d = table(10, 5, 0)
        with self.assertRaises(ValueError):
            learner(d)
        for n_class_vars in [1, 2, 3]:
            d = table(10, 5, n_class_vars)
            learner(d)  # no exception

    def test_compare_to_sklearn(self):
        d = table(10, 5, 1)
        orange_model = PLSRegressionLearner()(d)
        scikit_model = PLSRegression().fit(d.X, d.Y)
        np.testing.assert_almost_equal(
            scikit_model.predict(d.X).ravel(), orange_model(d)
        )
        np.testing.assert_almost_equal(
            coefficients(scikit_model), orange_model.coefficients
        )

    def test_compare_to_sklearn_multid(self):
        d = table(10, 5, 3)
        orange_model = PLSRegressionLearner()(d)
        scikit_model = PLSRegression().fit(d.X, d.Y)
        np.testing.assert_almost_equal(scikit_model.predict(d.X), orange_model(d))
        np.testing.assert_almost_equal(
            coefficients(scikit_model), orange_model.coefficients
        )

    def test_too_many_components(self):
        # do not change n_components
        d = table(5, 5, 1)
        model = PLSRegressionLearner(n_components=4)(d)
        self.assertEqual(model.skl_model.n_components, 4)
        # need to use fewer components; column limited
        d = table(6, 5, 1)
        model = PLSRegressionLearner(n_components=6)(d)
        self.assertEqual(model.skl_model.n_components, 4)
        # need to use fewer components; row limited
        d = table(5, 6, 1)
        model = PLSRegressionLearner(n_components=6)(d)
        self.assertEqual(model.skl_model.n_components, 4)

    def test_scores(self):
        for d in [table(10, 5, 1), table(10, 5, 3)]:
            orange_model = PLSRegressionLearner()(d)
            scikit_model = PLSRegression().fit(d.X, d.Y)
            scores = orange_model.project(d)
            sx, sy = scikit_model.transform(d.X, d.Y)
            np.testing.assert_almost_equal(sx, scores.X)
            np.testing.assert_almost_equal(sy, scores.metas)

    def test_components(self):
        def t2d(m):
            return m.reshape(-1, 1) if len(m.shape) == 1 else m

        for d in [table(10, 5, 1), table(10, 5, 3)]:
            orange_model = PLSRegressionLearner()(d)
            scikit_model = PLSRegression().fit(d.X, d.Y)
            components = orange_model.components()
            np.testing.assert_almost_equal(scikit_model.x_loadings_, components.X.T)
            np.testing.assert_almost_equal(
                scikit_model.y_loadings_, t2d(components.Y).T
            )

    def test_coefficients(self):
        for d in [table(10, 5, 1), table(10, 5, 3)]:
            orange_model = PLSRegressionLearner()(d)
            scikit_model = PLSRegression().fit(d.X, d.Y)
            coef_table = orange_model.coefficients_table()
            np.testing.assert_almost_equal(coefficients(scikit_model).T, coef_table.X)


class TestOWPLS(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWPLS, stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            ParameterMapping('max_iter', self.widget.n_iters),
            ParameterMapping('n_components', self.widget.ncomps_spin),
        ]

    skip_reason = (
        "orange-widget-base changed apply button, which resulted in "
        "test failures (but was otherwise benign)"
    )

    @skipIf(Orange.version.version < "3.35.0", skip_reason)
    def test_output_learner(self):
        super().test_output_learner()

    @skipIf(Orange.version.version < "3.35.0", skip_reason)
    def test_output_learner_name(self):
        super().test_output_learner_name()
