from orangecontrib.spectroscopy.preprocess import Normalize, NormalizeReference

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.normalize import NormalizeEditor

NORMALIZE_REFERENCE_INDEX = NormalizeEditor.Normalizers.index(
    ("Normalize by Reference", 42)
)


class TestNormalizeEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(NormalizeEditor, self.widget)
        data = SMALL_COLLAGEN
        self.send_signal(OWPreprocess.Inputs.data, data)

    def test_no_interaction(self):
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, Normalize)
        self.assertIs(p.method, Normalize.Vector)

    def test_normalize_by_reference(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(self.widget.Inputs.reference, reference)
        self.editor._group.buttons()[NORMALIZE_REFERENCE_INDEX].click()
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, NormalizeReference)
        self.assertIs(p.reference, reference)

    def test_normalize_by_reference_no_reference(self):
        self.editor._group.buttons()[NORMALIZE_REFERENCE_INDEX].click()
        self.widget.commit.now()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.applying.is_shown())
        self.assertTrue(self.editor.Error.exception.is_shown())
        out = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsNone(out)

    def test_normalize_by_reference_wrong_reference(self):
        reference = SMALL_COLLAGEN[:2]
        self.send_signal(self.widget.Inputs.reference, reference)
        self.editor._group.buttons()[NORMALIZE_REFERENCE_INDEX].click()
        self.widget.commit.now()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.applying.is_shown())
        self.assertTrue(self.editor.Error.exception.is_shown())
        out = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsNone(out)
