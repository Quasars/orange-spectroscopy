from orangecontrib.spectroscopy.preprocess import GaussianSmoothing

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.preprocessors.misc import (
    GaussianSmoothingEditor,
)
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess


class TestGaussianEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(GaussianSmoothingEditor, self.widget)  # type: GaussianSmoothingEditor
        data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, data)

    def test_no_interaction(self):
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, GaussianSmoothing)
        self.assertEqual(p.sd, GaussianSmoothingEditor.DEFAULT_SD)

    def test_zero(self):
        self.editor.sd = 0
        self.editor.edited.emit()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.sd, 0)

    def test_basic(self):
        self.editor.sd = 1.5
        self.editor.edited.emit()
        p = self.commit_get_preprocessor()  # type: GaussianSmoothing
        self.assertEqual(1.5, p.sd)
