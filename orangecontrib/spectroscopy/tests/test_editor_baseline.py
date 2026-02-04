from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import LinearBaseline

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.baseline import BaselineEditor
from orangecontrib.spectroscopy.widgets.preprocessors.utils import layout_widgets


class TestBaselineEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(BaselineEditor, self.widget)
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_for_preview()  # ensure initialization with preview data

    def test_no_interaction(self):
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, LinearBaseline)
        self.assertIs(p.zero_points, None)

    def test_disable_checkbox(self):
        self.assertFalse(self.editor.peakcb.isEnabled())
        self.editor.baselinecb.activated.emit(1)
        self.editor.baselinecb.setCurrentIndex(1)
        self.assertTrue(self.editor.peakcb.isEnabled())
        self.editor.baselinecb.activated.emit(0)
        self.editor.baselinecb.setCurrentIndex(0)
        self.assertFalse(self.editor.peakcb.isEnabled())

    def test_add_limit(self):
        dmin, dmax = min(getx(self.data)), max(getx(self.data))
        # first addition adds two limits
        self.editor.range_button.click()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.zero_points, [dmin, dmax])
        # the second addition adds one limit
        self.editor.range_button.click()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.zero_points, [dmin, dmax, (dmin + dmax) / 2])

    def test_remove_limit(self):
        dmin, dmax = min(getx(self.data)), max(getx(self.data))
        self.test_add_limit()
        # if there are three entries, remove one element
        second = list(layout_widgets(self.editor.ranges_box))[1]
        button = list(layout_widgets(second))[1]
        button.click()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.zero_points, [dmin, (dmin + dmax) / 2])
        # if there are two entries, remove both
        second = list(layout_widgets(self.editor.ranges_box))[1]
        button = list(layout_widgets(second))[1]
        button.click()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.zero_points, None)
