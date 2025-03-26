import unittest
from collections import OrderedDict
from functools import reduce

import Orange
import lmfit
import numpy as np
from Orange.widgets.data.utils.preprocess import DescriptionRole
from Orange.widgets.tests.base import WidgetTest
from orangewidget.tests.base import GuiTest

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import Cut, LinearBaseline, Integrate
from orangecontrib.spectroscopy.tests.spectral_preprocess import wait_for_preview
from orangecontrib.spectroscopy.widgets.gui import MovableVline
import orangecontrib.spectroscopy.widgets.owpeakfit as owpeakfit
from orangecontrib.spectroscopy.widgets.owpeakfit import OWPeakFit, fit_peaks, PREPROCESSORS, \
    create_model, prepare_params, unique_prefix, create_composite_model, pack_model_editor
from orangecontrib.spectroscopy.widgets.peak_editors import ParamHintBox, VoigtModelEditor, \
    PseudoVoigtModelEditor, ExponentialGaussianModelEditor, PolynomialModelEditor, \
    GaussianModelEditor
import orangecontrib.spectroscopy.widgets.peakfit_compute as peakfit_compute

# shorter initializations in tests
owpeakfit.N_PROCESSES = 1


COLLAGEN = Orange.data.Table("collagen")[0:3]
COLLAGEN_2 = LinearBaseline()(Cut(lowlim=1500, highlim=1700)(COLLAGEN))
COLLAGEN_1 = LinearBaseline()(Cut(lowlim=1600, highlim=1700)(COLLAGEN_2))


class TestOWPeakFit(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPeakFit)
        self.data = COLLAGEN_1

    def test_load_unload(self):
        self.send_signal("Data", Orange.data.Table("iris.tab"))
        self.send_signal("Data", None)

    def test_allint_indv(self):
        for p in PREPROCESSORS:
            with self.subTest(msg=f"Testing model {p.name}"):
                settings = None
                if p.viewclass == PolynomialModelEditor:
                    self.skipTest("Polynomial Model does not converge on this data")
                if p.viewclass == ExponentialGaussianModelEditor:
                    self.skipTest("Exponential Gaussian Model does not converge on this data")
                elif p.viewclass == PseudoVoigtModelEditor:
                    settings = {'storedsettings':
                                {'name': '',
                                 'preprocessors':
                                 [('orangecontrib.spectroscopy.widgets.peak_editors.pv',
                                   {'center': OrderedDict([('value', 1650.0)]),
                                    'fraction': OrderedDict([('vary', "fixed")]),
                                    })]}}
                self.widget = self.create_widget(OWPeakFit, stored_settings=settings)
                self.send_signal("Data", self.data)
                if settings is None:
                    self.widget.add_preprocessor(p)
                wait_for_preview(self.widget, 10000)
                self.widget.onDeleteWidget()

    def test_outputs(self):
        self.send_signal("Data", self.data)
        self.widget.add_preprocessor(PREPROCESSORS[0])
        wait_for_preview(self.widget)
        self.widget.commit.now()
        self.wait_until_finished(timeout=10000)
        fit_params = self.get_output(self.widget.Outputs.fit_params, wait=10000)
        fits = self.get_output(self.widget.Outputs.fits)
        residuals = self.get_output(self.widget.Outputs.residuals)
        data = self.get_output(self.widget.Outputs.annotated_data)
        # fit_params
        self.assertEqual(len(fit_params), len(self.data))
        np.testing.assert_array_equal(fit_params.Y, self.data.Y)
        np.testing.assert_array_equal(fit_params.metas, self.data.metas)
        # fits
        self.assertEqual(len(fits), len(self.data))
        self.assert_domain_equal(fits.domain, self.data.domain)
        np.testing.assert_array_equal(fits.Y, self.data.Y)
        np.testing.assert_array_equal(fits.metas, self.data.metas)
        # residuals
        self.assertEqual(len(residuals), len(self.data))
        self.assert_domain_equal(residuals.domain, self.data.domain)
        np.testing.assert_array_equal(residuals.X, self.data.X - fits.X)
        np.testing.assert_array_equal(residuals.Y, self.data.Y)
        np.testing.assert_array_equal(residuals.metas, self.data.metas)
        # annotated data
        self.assertEqual(len(data), len(self.data))
        np.testing.assert_array_equal(data.X, self.data.X)
        np.testing.assert_array_equal(data.Y, self.data.Y)
        join_metas = np.asarray(np.hstack((self.data.metas, fit_params.X)), dtype=object)
        np.testing.assert_array_equal(data.metas, join_metas)

    def test_saving_models(self):
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual([], settings['storedsettings']['preprocessors'])
        self.widget.add_preprocessor(PREPROCESSORS[0])
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual(PREPROCESSORS[0].qualname,
                         settings['storedsettings']['preprocessors'][0][0])
        self.widget = self.create_widget(OWPeakFit, stored_settings=settings)
        vc = self.widget.preprocessormodel.item(0).data(DescriptionRole).viewclass
        self.assertEqual(PREPROCESSORS[0].viewclass, vc)

    def test_migrate_refactor1(self):
        i1 = ("xyz", {'amplitude': {'value': 42.0, 'vary': False},
                      'center': {'value': 1349.984, 'min': 1307.984, 'max': 1391.984},
                      'sigma': {'min': 0, 'value': 1.0}})
        i2 = ("gam", {'gamma1': {'expr': 'sigma'},
                      'gamma2': {'expr': '', 'value': 0.0, 'min': -1.0, 'max': 1.0}})
        settings = {"storedsettings": {"preprocessors": [i1, i2]}}
        OWPeakFit.migrate_settings(settings, 1)
        o1 = ('xyz', {'amplitude': {'value': 42.0, 'vary': 'fixed'},
                      'center': {'max': 1391.984, 'min': 1307.984,
                                 'value': 1349.984, 'vary': 'limits'},
                      'sigma': {'min': 0, 'value': 1.0, 'vary': 'limits'}})
        o2 = ('gam', {'gamma1': {'expr': 'sigma', 'vary': 'expr'},
                      'gamma2': {'expr': '', 'max': 1.0, 'min': -1.0,
                                 'value': 0.0, 'vary': 'limits'}})
        self.assertEqual(settings["storedsettings"]["preprocessors"], [o1, o2])

    def test_bug_iris_crash(self):
        # bug with override wavenumbers:
        # TypeError: Object of type 'float32' is not JSON serializable
        data = Orange.data.Table('iris')
        self.send_signal("Data", data)
        # fixing getx output type fixes the bug
        self.assertEqual(getx(data).dtype, np.float64)
        self.widget.add_preprocessor(pack_model_editor(GaussianModelEditor))
        self.widget.commit.now()
        wait_for_preview(self.widget, 10000)

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()


class TestPeakFit(unittest.TestCase):

    def setUp(self):
        self.data = COLLAGEN_2

    def test_fit_peaks(self):
        model = lmfit.models.VoigtModel(prefix="v1_")
        params = model.make_params(center=1655)
        out = fit_peaks(self.data, model, params)
        assert len(out) == len(self.data)

    def test_peakfit_compute_dumpload(self):
        model = lmfit.models.VoigtModel(prefix="v1_")
        params = model.make_params(center=1655)
        x = getx(self.data)
        peakfit_compute.pool_initializer(model.dumps(), params, x)
        assert peakfit_compute.lmfit_model[0].dumps() == model.dumps()

    def test_table_output(self):
        pcs = [1547, 1655]
        mlist = [lmfit.models.VoigtModel(prefix=f"v{i}_") for i in range(len(pcs))]
        model = reduce(lambda x, y: x + y, mlist)
        params = model.make_params()
        for i, center in enumerate(pcs):
            p = f"v{i}_"
            dx = 20
            params[p + "center"].set(value=center, min=center-dx, max=center+dx)
            params[p + "sigma"].set(max=50)
            params[p + "amplitude"].set(min=0.0001)
        out_result = model.fit(self.data.X[0], params, x=getx(self.data))
        out_table = fit_peaks(self.data, model, params)
        out_row = out_table[0]
        self.assertEqual(out_row.x.shape[0], len(pcs) + len(out_result.var_names) + 5)
        attrs = [a.name for a in out_table.domain.attributes[:4]]
        self.assertEqual(attrs, ["v0 area", "v0 amplitude", "v0 center", "v0 sigma"])
        self.assertNotEqual(0, out_row["v0 area"].value)
        self.assertEqual(out_result.best_values["v0_amplitude"], out_row["v0 amplitude"].value)
        self.assertEqual(out_result.best_values["v0_center"], out_row["v0 center"].value)
        self.assertEqual(out_result.best_values["v0_sigma"], out_row["v0 sigma"].value)
        self.assertEqual(out_result.chisqr, out_row["Chi-square"].value)
        self.assertEqual(out_result.redchi, out_row["Reduced chi-square"].value)
        self.assertEqual(out_result.aic, out_row["Akaike info crit"].value)
        self.assertEqual(out_result.bic, out_row["Bayesian info crit"].value)
        self.assertEqual(out_result.rsquared, out_row["R-squared"].value)
        self.assertEqual(out_row.id, self.data.ids[0])


class TestBuildModel(GuiTest):

    def test_model_from_editor(self):
        self.editor = VoigtModelEditor()
        self.editor.set_hint('center', 'value', 1655)
        self.editor.edited.emit()

        m = self.editor.createinstance(prefix=unique_prefix(self.editor, 0))
        self.assertIsInstance(m, self.editor.model)
        editor_params = self.editor.parameters()
        for name, hints in editor_params.items():
            m.set_param_hint(name, **hints)
        params = m.make_params()
        self.assertEqual(params['v0_center'], 1655)


class ModelEditorTest(WidgetTest):
    EDITOR = None

    def setUp(self):
        self.widget = self.create_widget(OWPeakFit)
        if self.EDITOR is not None:
            self.editor = self.add_editor(self.EDITOR, self.widget)
            self.data = COLLAGEN_1
            self.send_signal(self.widget.Inputs.data, self.data)
        else:
            # Test adding all the editors
            for p in self.widget.PREPROCESSORS:
                self.add_editor(p.viewclass, self.widget)

    def wait_until_finished(self, widget=None, timeout=None):
        super().wait_until_finished(widget,
                                    timeout=timeout if timeout is not None else 10000)

    def wait_for_preview(self):
        wait_for_preview(self.widget, timeout=10000)

    def add_editor(self, cls, widget):  # type: (Type[T], object) -> T
        widget.add_preprocessor(pack_model_editor(cls))
        editor = widget.flow_view.widgets()[-1]
        self.process_events()
        return editor

    def get_model_single(self):
        m_def = self.widget.preprocessormodel.item(0)
        return create_model(m_def, 0)

    def get_params_single(self, model):
        m_def = self.widget.preprocessormodel.item(0)
        return prepare_params(m_def, model)

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()


class TestVoigtEditor(ModelEditorTest):
    EDITOR = VoigtModelEditor

    def test_no_interaction(self):
        self.widget.commit.now()
        self.wait_until_finished()
        self.assertIsInstance(self.editor, self.EDITOR)
        m = self.get_model_single()
        self.assertIsInstance(m, self.EDITOR.model)

    def test_create_model(self):
        m = self.get_model_single()
        params = self.get_params_single(m)
        for p in self.editor.parameters():
            self.assertIn(f"{m.prefix}{p}", params)

    def test_set_param(self):
        e = self.editor
        e.set_hint('center', 'value', 1623)
        e.set_hint('center', 'min', 1603)
        e.set_hint('center', 'max', 1643)
        p_set = e.parameters()['center']
        self.assertIsInstance(p_set, dict)
        self.assertEqual(p_set['value'], 1623)
        self.assertEqual(p_set['min'], 1603)
        self.assertEqual(p_set['max'], 1643)

    def test_set_center(self):
        e = self.editor
        e.set_hint('center', 'value', 1655)
        e.edited.emit()
        m = self.get_model_single()
        params = self.get_params_single(m)
        c_p = f'{m.prefix}center'
        self.assertEqual(1655, params[c_p].value)

    def test_only_spec_lines(self):
        self.editor.activateOptions()
        model_lines = self.editor.model_lines()
        lines = [l.label.toPlainText().strip() for l in self.widget.curveplot.markings
                 if isinstance(l, MovableVline)]
        for ml in model_lines:
            self.assertIn(ml, lines)
        no_lines = [p for p in self.editor.model_parameters() if p not in model_lines]
        for nl in no_lines:
            self.assertNotIn(nl, lines)

    def test_move_line(self):
        self.editor.activateOptions()
        l = self.widget.curveplot.markings[0]
        self.assertIsInstance(l, MovableVline)
        l.setValue(1673)
        l.sigMoved.emit(l.value())
        self.assertEqual(1673, self.editor.parameters()['center']['value'])


class TestVoigtEditorMulti(ModelEditorTest):

    def setUp(self):
        self.pcs = [1547, 1655]
        self.widget = self.create_widget(OWPeakFit)
        self.editors = [self.add_editor(VoigtModelEditor, self.widget)
                        for _ in range(len(self.pcs))]
        self.data = COLLAGEN_2
        self.send_signal(self.widget.Inputs.data, self.data)
        self.model, self.params = self.matched_models()

    def test_no_change(self):
        self.widget.commit.now()
        self.wait_until_finished()

    def matched_models(self):
        mlist = [lmfit.models.VoigtModel(prefix=f"v{i}_") for i in range(len(self.pcs))]
        model = reduce(lambda x, y: x + y, mlist)
        params = model.make_params()
        for i, center in enumerate(self.pcs):
            p = f"v{i}_"
            dx = 20
            params[p + "center"].set(value=center, min=center - dx, max=center + dx)
            params[p + "sigma"].set(max=50)
            params[p + "amplitude"].set(min=0.0001)
            # Set editor to same values
            e = self.editors[i]
            e_params = e.parameters()
            e_params['center'].update({'value': center, 'min': center - dx, 'max': center + dx})
            e_params['sigma']['max'] = 50
            e_params['amplitude']['min'] = 0.0001
            e.setParameters(e_params)
            e.edited.emit()
        return model, params

    def test_same_params(self):
        m_def = [self.widget.preprocessormodel.item(i)
                 for i in range(self.widget.preprocessormodel.rowCount())]
        ed_model, ed_params = create_composite_model(m_def)

        self.assertEqual(self.model.name, ed_model.name)
        self.assertEqual(set(self.params), set(ed_params))
        for k, v in self.params.items():
            self.assertEqual(v, ed_params[k])

    def test_same_output(self):
        out_fit = fit_peaks(self.data, self.model, self.params)
        out = self.get_output(self.widget.Outputs.fit_params, wait=10000)

        self.assertEqual(out_fit.domain.attributes, out.domain.attributes)
        np.testing.assert_array_equal(out_fit.X, out.X)

    def test_saving_model_params(self):
        settings = self.widget.settingsHandler.pack_data(self.widget)
        restored_widget = self.create_widget(OWPeakFit, stored_settings=settings)
        m_def = [restored_widget.preprocessormodel.item(i)
                 for i in range(restored_widget.preprocessormodel.rowCount())]
        sv_model, sv_params = create_composite_model(m_def)

        self.assertEqual(self.model.name, sv_model.name)
        self.assertEqual(set(self.params), set(sv_params))

    def test_total_area(self):
        """ Test v0 + v1 area == total fit area """
        fit_params = self.get_output(self.widget.Outputs.fit_params, wait=10000)
        fits = self.get_output(self.widget.Outputs.fits)
        xs = getx(fits)
        total_areas = Integrate(methods=Integrate.Simple, limits=[[xs.min(), xs.max()]])(fits)
        total_area = total_areas.X[0, 0]
        v0_area = fit_params[0]["v0 area"].value
        v1_area = fit_params[0]["v1 area"].value
        self.assertAlmostEqual(total_area, v0_area + v1_area)


class TestParamHintBox(GuiTest):

    def test_defaults(self):
        defaults = {
            'value': 0,
            'vary': 'limits',
            'min': float('-inf'),
            'max': float('-inf'),
            'delta': 1,
            'expr': "",
        }
        hb = ParamHintBox(defaults)
        e_vals = {
            'value': hb.val_e.value(),
            'vary': hb.vary_e.currentText(),
            'min': hb.min_e.value(),
            'max': hb.max_e.value(),
            'delta': hb.delta_e.value(),
            'expr': hb.expr_e.text(),
        }
        self.assertEqual(defaults, e_vals)
        self.assertEqual({'value': 0.0}, VoigtModelEditor.translate("center", hb.hints))

    def test_keep_delta(self):
        h = {'vary': 'limits'}
        hb = ParamHintBox(h)
        hb.vary_e.setCurrentText('delta')
        hb.setValues()  # an Editor should run this
        self.assertEqual('delta', h['vary'])
        self.assertEqual((-1, 1), (h['min'], h['max']))
        self.assertEqual((-1, 1), (hb.min_e.value(), hb.max_e.value()))
        hb.vary_e.setCurrentText('limits')
        hb.setValues()  # an Editor should run this
        self.assertEqual('limits', h['vary'])
        self.assertEqual((-1, 1), (h['min'], h['max']))
        self.assertEqual((-1, 1), (hb.min_e.value(), hb.max_e.value()))
        hb.vary_e.setCurrentText('delta')
        hb.setValues()  # an Editor should run this
        self.assertEqual((-1, 1), (h['min'], h['max']))
        self.assertEqual((-1, 1), (hb.min_e.value(), hb.max_e.value()))

    def test_delta_update_limits(self):
        h = {'vary': 'limits'}
        hb = ParamHintBox(h)
        hb.vary_e.setCurrentText('delta')
        hb.setValues()  # an Editor should run this
        self.assertEqual((-1, 1), (hb.min_e.value(), hb.max_e.value()))
        hb.val_e.setValue(10)
        hb.setValues()
        self.assertEqual((9, 11), (hb.min_e.value(), hb.max_e.value()))
        hb.vary_e.setCurrentText('limits')
        hb.setValues()
        self.assertEqual((9, 11), (hb.min_e.value(), hb.max_e.value()))
        hb.vary_e.setCurrentText('delta')
        h['value'] = 20
        hb.update_min_max_for_delta()  # should be called after line move
        self.assertEqual((19, 21), (hb.min_e.value(), hb.max_e.value()))

    def test_expr_change_to_vary(self):
        h = {'expr': 'test', 'vary': 'expr'}
        hb = ParamHintBox(h)
        self.assertEqual({}, VoigtModelEditor.translate("gamma", hb.hints))  # default
        hb.vary_e.setCurrentText('delta')
        self.assertEqual('delta', hb.vary_e.currentText())
        self.assertEqual("", VoigtModelEditor.translate("gamma", hb.hints)["expr"])
        hb.vary_e.setCurrentText('expr')
        self.assertEqual('expr', hb.vary_e.currentText())
        self.assertEqual({}, VoigtModelEditor.translate("gamma", hb.hints))  # default

    def test_expr_set_hint(self):
        h = {'expr': 'test', 'vary': 'expr'}
        hb = ParamHintBox(h)
        self.assertEqual('expr', hb.vary_e.currentText())
        self.assertEqual({}, VoigtModelEditor.translate("gamma", hb.hints))  # default
