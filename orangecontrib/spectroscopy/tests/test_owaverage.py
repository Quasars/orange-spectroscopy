import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owaverage import OWAverage


class TestOWAverage(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.collagen = Orange.data.Table("collagen.csv")

    def setUp(self):
        self.widget = self.create_widget(OWAverage)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def test_average(self):
        self.send_signal("Data", self.collagen)
        out = self.get_output("Averages")
        self.assertTrue(out.X.shape[0] == 1)
        self.assertEqual(out.X.shape[1], self.collagen.X.shape[1])
        avg = np.mean(self.collagen.X[:, :3], axis=0, keepdims=True)
        np.testing.assert_equal(out.X[:, :3], avg)
        # Other variables are unknown if not all the same value
        self.assertTrue(np.isnan(out.Y[0]))

    def test_domain(self):
        self.send_signal("Data", self.collagen)
        out = self.get_output("Averages")
        self.assertEqual(self.collagen.domain.attributes, out.domain.attributes)

    def test_nan_propagation(self):
        copy = self.collagen.copy()
        with copy.unlocked():
            copy[:, :2] = np.nan
            copy[3, 3] = np.nan
        self.send_signal("Data", copy)
        out = self.get_output("Averages")
        self.assertTrue(np.all(np.isnan(out[:, :2])))
        self.assertFalse(np.any(np.isnan(out.X[:, 2:])))

    def test_average_by_group(self):
        self.send_signal("Data", self.collagen)
        gvar = self.widget.group_var = self.collagen.domain.class_var
        self.widget.grouping_changed()
        out = self.get_output("Averages")
        self.assertEqual(out.X.shape[0], len(gvar.values))
        self.assertEqual(out.X.shape[1], self.collagen.X.shape[1])
        # First 195 rows are labelled "collagen"
        collagen_avg = np.mean(self.collagen.X[:195], axis=0)
        np.testing.assert_equal(out.X[1,], collagen_avg)

    def test_average_by_group_metas(self):
        # Alter collagen domain to have Continuous/String/TimeVariables in metas
        c_domain = self.collagen.domain
        str_var = Orange.data.StringVariable.make(name="stringtest")
        time_var = Orange.data.TimeVariable.make(name="timetest")
        n_domain = Orange.data.Domain(
            c_domain.attributes,
            c_domain.class_vars,
            [Orange.data.ContinuousVariable("con"), str_var, time_var],
        )
        collagen = self.collagen.transform(n_domain)
        with collagen.unlocked(collagen.metas):
            collagen.metas[:, 0] = np.atleast_2d(collagen.X[:, 0])
            collagen.metas[:, 1] = ["string"] * len(collagen)
            collagen.metas[:, 2] = [1560.3] * len(collagen)

        self.send_signal("Data", collagen)
        gvar = self.widget.group_var = collagen.domain.class_var
        self.widget.grouping_changed()
        out = self.get_output("Averages")
        self.assertEqual(out.X.shape[0], len(gvar.values))
        self.assertEqual(out.X.shape[1], collagen.X.shape[1])
        # First 195 rows are labelled "collagen"
        collagen_avg = np.mean(collagen.X[:195], axis=0)
        np.testing.assert_equal(out.X[1,], collagen_avg)
        # ContinuousVariable averaging in metas
        # assert_allclose is due to float rounding error
        np.testing.assert_allclose(out[0, 0], out[0, -1])
        # Other variables keep first if all the same
        self.assertEqual(collagen[0, gvar], out[1, gvar])
        self.assertEqual(collagen[0, str_var], out[1, str_var])
        np.testing.assert_allclose(collagen[0, time_var], out[1, time_var])

    def test_average_by_group_unknown(self):
        # Alter collagen to have some unknowns in "type" variable
        collagen = self.collagen.copy()
        gvar = collagen.domain.class_var
        index_unknowns = [3, 15, 100, 500, 650]
        with collagen.unlocked():
            collagen[index_unknowns, gvar] = Orange.data.Unknown

        self.send_signal("Data", collagen)
        self.widget.group_var = gvar
        self.widget.grouping_changed()
        out = self.get_output("Averages")
        self.assertEqual(out.X.shape[0], len(gvar.values) + 1)
        unknown_avg = np.mean(collagen.X[index_unknowns], axis=0)
        np.testing.assert_equal(out.X[4,], unknown_avg)

    def test_average_by_group_missing(self):
        # Alter collagen to have a "type" variable value with no members
        gvar = self.collagen.domain.class_var
        svfilter = Orange.data.filter.SameValue(gvar, gvar.values[0], negate=True)
        collagen = svfilter(self.collagen)

        self.send_signal("Data", collagen)
        self.widget.group_var = gvar
        self.widget.grouping_changed()
        out = self.get_output("Averages")
        self.assertEqual(out.X.shape[0], len(gvar.values) - 1)

    def test_average_by_group_objectvar(self):
        # Test with group_var in metas (object array)
        gvar = self.collagen.domain.class_var
        c_domain = self.collagen.domain
        str_var = Orange.data.StringVariable.make(name="stringtest")
        n_domain = Orange.data.Domain(
            c_domain.attributes, None, [c_domain.class_var, str_var]
        )
        collagen = self.collagen.transform(n_domain)
        # collagen.metas[:, 1] = np.atleast_2d(self.collagen.Y)
        self.send_signal("Data", collagen)
        self.widget.group_var = gvar
        self.widget.grouping_changed()
        out = self.get_output("Averages")
        # First 195 rows are labelled "collagen"
        collagen_avg = np.mean(self.collagen.X[:195], axis=0)
        np.testing.assert_equal(out.X[1,], collagen_avg)
