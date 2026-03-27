from orangecontrib.spectroscopy.preprocess.atm_corr import AtmCorr

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.atm_corr import AtmCorrEditor


class TestAtmCorrEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(AtmCorrEditor, self.widget)
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_for_preview()  # ensure initialization with preview data

    def test_no_interaction(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN)
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, AtmCorr)

    def test_interaction(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN)
        self.editor.smooth_button.click()
        self.editor.smooth_win_spin.setValue(17)
        self.commit_get_preprocessor()

    def test_controls(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN[:1])
        self.editor.controls.smooth_win.setValue(5)
        self.editor.controls.bridge_win.setValue(3)
        p = self.commit_get_preprocessor()
        self.assertEqual(p.smooth_win, 5)
        self.assertEqual(p.spline_base_win, 3)
        self.editor.controls.smooth.click()
        self.editor.controls.mean_reference.click()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.smooth_win, 0)
        self.assertEqual(p.spline_base_win, 3)

    def test_controls_load(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN[:1])
        self.editor.controls.smooth_win.setValue(5)
        self.editor.controls.bridge_win.setValue(3)
        self.editor.controls.smooth.click()
        self.editor.controls.mean_reference.click()
        pars = self.editor.parameters()
        self.setUp()
        self.editor.setParameters(pars.copy())
        pars2 = self.editor.parameters()
        self.assertEqual(pars, pars2)
