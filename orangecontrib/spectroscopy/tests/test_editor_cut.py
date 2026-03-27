from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.preprocessors.misc import CutEditor
from orangecontrib.spectroscopy.preprocess import Cut


class TestCutEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(CutEditor, self.widget)
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_for_preview()  # ensure initialization with preview data

    def test_no_interaction(self):
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, Cut)
        self.assertEqual(p.lowlim, 995.902)
        self.assertEqual(p.highlim, 1711.779)
        self.assertEqual(p.inverse, False)

    def test_basic(self):
        self.editor.lowlim = 5
        self.editor.highlim = 10
        self.editor.inverse = True
        self.editor.edited.emit()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.lowlim, 5)
        self.assertEqual(p.highlim, 10)
        self.assertEqual(p.inverse, True)
