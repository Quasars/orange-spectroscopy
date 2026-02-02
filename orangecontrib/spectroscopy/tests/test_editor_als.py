from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.preprocess.als import ALSP, ARPLS, AIRPLS
from orangecontrib.spectroscopy.widgets.preprocessors.als import ALSEditor

import unittest
class A(unittest.TestCase): pass  # noqa: E701

class TestALSEditor(PreprocessorEditorTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(ALSEditor, self.widget)
        self.data = SMALL_COLLAGEN[1:3]
        self.send_signal(self.widget.Inputs.data, self.data)

    def test_no_interaction(self):
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, ALSP)
        self.assertEqual(p.itermax, 10)
        self.assertEqual(p.p, 0.1)

    def test_disable_types(self):
        self.assertTrue(self.editor.palsspin.isEnabled())
        self.assertFalse(self.editor.ratior.isEnabled())
        self.assertFalse(self.editor.porderairplsspin.isEnabled())
        self.editor.alst_combo.setCurrentIndex(1)
        self.editor.alst_combo.activated.emit(1)
        self.assertTrue(self.editor.ratior.isEnabled())
        self.assertFalse(self.editor.palsspin.isEnabled())
        self.editor.alst_combo.setCurrentIndex(2)
        self.editor.alst_combo.activated.emit(2)
        self.assertTrue(self.editor.porderairplsspin.isEnabled())

    def test_arpls(self):
        self.editor.als_type = 1
        self.editor.ratio = 0.75
        self.editor.lam = 42
        self.editor.itermax = 2
        self.editor.edited.emit()
        p = self.commit_get_preprocessor()
        self.process_events()
        self.assertIsInstance(p, ARPLS)
        self.assertEqual(p.itermax, 2)
        self.assertEqual(p.lam, 42)
        self.assertEqual(p.ratio, 0.75)

    def test_airpls(self):
        self.editor.als_type = 2
        self.editor.porder = 2
        self.editor.lam = 41
        self.editor.itermax = 3
        self.editor.edited.emit()
        p = self.commit_get_preprocessor()
        self.process_events()
        self.assertIsInstance(p, AIRPLS)
        self.assertEqual(p.itermax, 3)
        self.assertEqual(p.lam, 41)
        self.assertEqual(p.porder, 2)
