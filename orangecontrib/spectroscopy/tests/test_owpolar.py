import unittest
import numpy as np
from AnyQt.QtCore import QItemSelectionModel, QItemSelection, QItemSelectionRange
import Orange
from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owpolar import OWPolar

class TestOWPolar(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.multifile = Orange.data.Table("polar/4-angle-ftir_multifile.tab")
        cls.in1 = Orange.data.Table("polar/4-angle-ftir_multiin1.tab")
        cls.in2 = Orange.data.Table("polar/4-angle-ftir_multiin2.tab")
        cls.in3 = Orange.data.Table("polar/4-angle-ftir_multiin3.tab")
        cls.in4 = Orange.data.Table("polar/4-angle-ftir_multiin4.tab")
        cls.multifile_polar = Orange.data.Table("polar/4-angle-ftir_multifile_polar-results.tab")
        cls.multifile_model = Orange.data.Table("polar/4-angle-ftir_multifile_model-results.tab")
        cls.multiin_polar = Orange.data.Table("polar/4-angle-ftir_multiin_polar-results.tab")
        cls.multiin_model = Orange.data.Table("polar/4-angle-ftir_multiin_model-results.tab")

    def setUp(self):
        self.widget = self.create_widget(OWPolar)

    def test_multifile_init(self):
        self.send_signal("Data", self.multifile, 0)

        testfeats = [ft for ft in self.multifile.domain.metas
                     if isinstance(ft, ContinuousVariable)]
        testfeats = testfeats + [ft for ft in self.multifile.domain.attributes
                     if isinstance(ft, ContinuousVariable)]
        polfeats = [ft for ft in self.widget.featureselect[:]
                    if isinstance(ft, ContinuousVariable)]
        self.assertEqual(polfeats, testfeats)
        testinputs = [inp for inp in self.multifile.domain
                      if isinstance(inp, DiscreteVariable)]
        self.assertEqual(self.widget.anglemetas[:], testinputs)
        testxy = [xy for xy in self.multifile.domain.metas
                  if isinstance(xy, ContinuousVariable)]
        self.assertEqual(self.widget.x_axis[:], testxy)
        self.assertEqual(self.widget.y_axis[:], testxy)

    def test_multifile_in(self):
        self.send_signal("Data", self.multifile, 0)

        self.assertTrue(self.widget.isEnabled())
        for i in self.widget.multiin_labels:
            self.assertFalse(i.isEnabled())
        for i in self.widget.multiin_lines:
            self.assertFalse(i.isEnabled())
        self.widget.angles = self.widget.anglemetas[0]
        self.assertEqual(self.widget.angles, self.multifile.domain.metas[2])
        self.widget._change_angles()
        self.assertEqual(len(self.widget.labels), 4)
        self.assertEqual(len(self.widget.lines), 4)
        self.assertEqual(self.widget.polangles, list(np.linspace(0, 180, 5)[:4]))
        for i in self.widget.labels:
            self.assertTrue(i.isEnabled())
        for i in self.widget.lines:
            self.assertTrue(i.isEnabled())
        self.widget.map_x = self.widget.x_axis[0]
        self.assertEqual(self.widget.map_x, self.multifile.domain.metas[0])
        self.widget.map_y = self.widget.y_axis[1]
        self.assertEqual(self.widget.map_y, self.multifile.domain.metas[1])
        self.widget.feats = [self.widget.feat_view.model()[:][2],
                                 self.widget.feat_view.model()[:][3]]
        self.assertEqual(self.widget.feats[0], self.multifile.domain.metas[3])
        self.assertEqual(self.widget.feats[1], self.multifile.domain.metas[4])
        self.widget.alphas = [0, 0]
        self.widget.invert_angles = True
        self.widget.autocommit = True
        self.commit_and_wait(self.widget, 20000)

        polar = self.get_output("Polar Data")
        model = self.get_output("Curve Fit model data")

        np.testing.assert_allclose(np.asarray(self.multifile_polar.metas, dtype=float),
                                   np.asarray(polar.metas, dtype=float), rtol=4e-06)
        np.testing.assert_allclose(np.asarray(self.multifile_polar.X, dtype=float),
                                   np.asarray(polar.X, dtype=float), rtol=5e-06)
        np.testing.assert_allclose(np.asarray(self.multifile_model.metas, dtype=float),
                                   np.asarray(model.metas, dtype=float), rtol=4e-06)
        np.testing.assert_allclose(np.asarray(self.multifile_model.X, dtype=float),
                                   np.asarray(model.X, dtype=float), rtol=5e-06)

    def test_multi_inputs(self):
        self.send_signal("Data", self.in1, 0, widget=self.widget)
        self.send_signal("Data", self.in2, 1, widget=self.widget)

        self.assertFalse(self.widget.anglesel.isEnabled())
        for i in self.widget.multiin_labels:
            self.assertFalse(i.isEnabled())
        for i in self.widget.multiin_lines:
            self.assertFalse(i.isEnabled())
        self.send_signal("Data", self.in3, 2, widget=self.widget)
        self.send_signal("Data", self.in4, 3, widget=self.widget)
        self.assertFalse(self.widget.anglesel.isEnabled())
        for i in self.widget.multiin_labels:
            self.assertTrue(i.isEnabled())
        for i in self.widget.multiin_lines:
            self.assertTrue(i.isEnabled())

        self.widget.map_x = self.widget.x_axis[0]
        self.assertEqual(self.widget.map_x, self.in1.domain.metas[0])
        self.widget.map_y = self.widget.y_axis[1]
        self.assertEqual(self.widget.map_y, self.in1.domain.metas[1])

        self.widget.feats = [self.widget.feat_view.model()[:][2],
                                 self.widget.feat_view.model()[:][3]]
        self.assertEqual(self.widget.feats[0],
                         self.in1.domain.metas[2].copy(compute_value=None))
        self.assertEqual(self.widget.feats[1],
                         self.in1.domain.metas[3].copy(compute_value=None))
        self.widget.alphas = [0, 0]
        self.widget.invert_angles = True
        self.widget.autocommit = True
        self.commit_and_wait(self.widget, 20000)

        polar = self.get_output("Polar Data")
        model = self.get_output("Curve Fit model data")

        np.testing.assert_allclose(
            np.asarray(self.multiin_polar.metas[:,np.r_[0:2,3:7]], dtype=float),
            np.asarray(polar.metas[:,np.r_[0:2,3:7]], dtype=float), rtol=2e-06)
        np.testing.assert_allclose(
            np.asarray(self.multiin_polar.metas[:,7:], dtype=float),
            np.asarray(polar.metas[:,7:], dtype=float), rtol=4e-06)
        np.testing.assert_allclose(self.multiin_polar.X, polar.X, rtol=5e-06)
        np.testing.assert_allclose(
            np.asarray(self.multiin_model.metas[:,np.r_[0:2,3:7]], dtype=float),
            np.asarray(model.metas[:,np.r_[0:2,3:7]], dtype=float), rtol=4e-06)
        np.testing.assert_allclose(
            np.asarray(self.multiin_model.metas[:,7:], dtype=float),
            np.asarray(model.metas[:,7:], dtype=float), rtol=4e-06)
        np.testing.assert_allclose(self.multiin_model.X, model.X, rtol=5e-06)

    def test_pixelsubset(self):
        #Test multi in with subset of pixels selected
        rng = np.random.default_rng()
        sub_idx = rng.choice(4, size=(2), replace=False)
        subset = self.in1[sub_idx]

        self.send_signal("Data", subset, 0, widget=self.widget)
        self.send_signal("Data", self.in2, 1, widget=self.widget)
        self.send_signal("Data", self.in3, 2, widget=self.widget)
        self.send_signal("Data", self.in4, 3, widget=self.widget)

        self.widget.map_x = self.widget.x_axis[0]
        self.widget.map_y = self.widget.y_axis[1]
        self.widget.feats = [self.widget.feat_view.model()[:][2],
                                 self.widget.feat_view.model()[:][3]]
        self.widget.alphas = [0, 0]
        self.widget.invert_angles = True
        self.widget.autocommit = True
        self.commit_and_wait(self.widget, 20000)

        polar = self.get_output("Polar Data")
        model = self.get_output("Curve Fit model data")

        self.assertEqual(len(polar), len(sub_idx)*4)
        self.assertEqual(len(model), len(sub_idx)*4)

    def test_multiin_mismatched_domain(self):

        metadom = self.in1.domain.metas
        metadom = [i for i in metadom if isinstance(i, ContinuousVariable)]
        attdom = self.in1.domain.attributes
        attdom = attdom[0::2]
        mismatched_domain = Domain(attdom, metas = metadom)
        mismatched_table = self.in1.transform(mismatched_domain)

        self.send_signal("Data", mismatched_table, 0, widget=self.widget)
        self.send_signal("Data", self.in2, 1, widget=self.widget)
        self.send_signal("Data", self.in3, 2, widget=self.widget)
        self.send_signal("Data", self.in4, 3, widget=self.widget)

        feat_len = len(metadom) + len(attdom) + 1
        XY_len = len(metadom)
        self.assertEqual(feat_len, len(self.widget.feat_view.model()[:]))
        self.assertEqual(XY_len, len(self.widget.x_axis[:]))
        self.assertEqual(XY_len, len(self.widget.y_axis[:]))

        self.send_signal("Data", self.in2, 0, widget=self.widget)
        self.send_signal("Data", self.in3, 1, widget=self.widget)
        self.send_signal("Data", mismatched_table, 2, widget=self.widget)
        self.send_signal("Data", self.in4, 3, widget=self.widget)

        feat_len = len(metadom) + len(attdom) + 1
        XY_len = len(metadom)
        self.assertEqual(feat_len, len(self.widget.feat_view.model()[:]))
        self.assertEqual(XY_len, len(self.widget.x_axis[:]))
        self.assertEqual(XY_len, len(self.widget.y_axis[:]))

    def test_custom_angles(self):
        # test inputting custom angles (multin and multifile)
        self.send_signal("Data", self.multifile, 0, widget=self.widget)
        angles = np.array([0, 22.5, 45.0, 90])

        for i, j in enumerate(self.widget.lines):
            j.setText(str(angles[i]))
        self.widget._send_angles()
        for i, j in enumerate(self.widget.polangles):
            self.assertEqual(j, angles[i])

        self.send_signal("Data", self.in1, 0, widget=self.widget)
        self.send_signal("Data", self.in2, 1, widget=self.widget)
        self.send_signal("Data", self.in3, 2, widget=self.widget)
        self.send_signal("Data", self.in4, 3, widget=self.widget)

        for i, j in enumerate(self.widget.multiin_lines):
            j.setText(str(angles[i]))
        self.widget._send_ind_angles()
        for i, j in enumerate(self.widget.polangles):
            self.assertEqual(j, angles[i])

    def test_warnings(self):
        #test all warnings
        self.send_signal("Data", self.multifile, 0, widget=self.widget)
        self.widget.autocommit = True

        self.commit_and_wait(self.widget)
        self.assertTrue(self.widget.Warning.nofeat.is_shown())

        self.widget.feats = [self.widget.feat_view.model()[:][4]]
        self.widget.map_x = None
        self.widget.map_y = None
        self.commit_and_wait(self.widget)
        self.assertTrue(self.widget.Warning.noxy.is_shown())

        self.widget.map_x = self.widget.x_axis[0]
        self.widget.map_y = self.widget.y_axis[1]
        self.widget.polangles = []
        self.commit_and_wait(self.widget)
        self.assertTrue(self.widget.Warning.pol.is_shown())
        self.widget.polangles = [0.0,45.0,'hi',135.0]
        self.commit_and_wait(self.widget)
        self.assertTrue(self.widget.Warning.pol.is_shown())

        self.widget.polangles = [0.0,45.0,90.0,135.0]
        self.widget.feats = [self.widget.feat_view.model()[:][0]]
        self.widget.alphas = [0]
        self.commit_and_wait(self.widget)
        self.assertTrue(self.widget.Warning.XYfeat.is_shown())

        self.send_signal("Data", self.in1, 0, widget=self.widget)
        self.send_signal("Data", self.in2, 1, widget=self.widget)
        self.assertTrue(self.widget.Warning.notenough.is_shown())

        self.send_signal("Data", self.in3, 2, widget=self.widget)
        self.assertTrue(self.widget.Warning.notenough.is_shown())

        self.send_signal("Data", self.in4, 3, widget=self.widget)
        self.assertFalse(self.widget.Warning.notenough.is_shown())

    def test_disconnect(self):
        self.send_signal("Data", self.multifile, 0, widget=self.widget)
        self.widget.angles = self.widget.anglemetas[0]
        self.widget.map_x = self.widget.x_axis[0]
        self.widget.map_y = self.widget.y_axis[1]
        self.widget.alphas = [0, 0]
        self.widget.invert_angles = True
        self.widget.autocommit = True
        self.widget.feats = [self.widget.feat_view.model()[:][2],
                            self.widget.feat_view.model()[:][3]]
        self.widget.handleNewSignals()
        self.wait_until_stop_blocking()
        self.send_signal("Data", None, 0, widget=self.widget)

    def test_alpha_changes(self):
        self.send_signal("Data", self.multifile, 0, widget=self.widget)
        self.widget.angles = self.widget.anglemetas[0]
        self.widget.map_x = self.widget.x_axis[0]
        self.widget.map_y = self.widget.y_axis[1]
        self.widget.alpha = 0
        vars = [self.widget.feat_view.model()[:][2],
                self.widget.feat_view.model()[:][3]]
        view = self.widget.feat_view
        model = self.widget.featureselect
        update_selection(model, view, vars)
        self.widget.change_alphas()
        self.assertEqual(self.widget.alphas, [0, 0])
        self.widget.alpha = 90
        update_selection(model, view, [vars[1]])
        self.widget.change_alphas()
        self.assertEqual(self.widget.alphas, [0, 90])
        update_selection(model, view, [vars[0]])
        self.widget.change_alphas()
        self.assertEqual(self.widget.alphas, [90, 90])

def update_selection(model, view, setting):
    selection = QItemSelection()
    sel_model = view.selectionModel()
    model_values = model[:]
    for var in setting:
        index = model_values.index(var)
        model_index = view.model().index(index, 0)
        selection.append(QItemSelectionRange(model_index))
    sel_model.select(selection, QItemSelectionModel.ClearAndSelect)
    # def test_clearangles(self):
    #     #test clearing angles
    #     pass

if __name__ == "__main__":
    unittest.main()
