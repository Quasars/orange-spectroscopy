import numpy as np

from orangecontrib.spectroscopy.preprocess.emsc import EMSC

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.emsc import EMSCEditor


class TestEMSCEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(EMSCEditor, self.widget)  # type: EMSCEditor
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_for_preview()  # ensure initialization with preview data

    def test_no_interaction(self):
        reference = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.reference, reference)
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, EMSC)

    def test_add_range(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN)
        self.assertEqual(0, len(self.editor.parameters()["ranges"]))
        self.editor.range_button.click()
        ranges = self.editor.parameters()["ranges"]
        self.assertEqual(1, len(ranges))
        np.testing.assert_almost_equal([906.4177, 1801.264], ranges[0][:2])

    def test_activate_options(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN)
        self.editor.activateOptions()
        self.assertEqual(0, len(self.editor.weight_curve.getData()[1]))
        self.editor.range_button.click()
        self.editor.activateOptions()
        self.assertEqual(1, np.max(self.editor.weight_curve.getData()[1]))

    def test_migrate_smoothing(self):
        name = "orangecontrib.spectroscopy.preprocess.emsc"
        settings = {
            "storedsettings": {"preprocessors": [(name, {"ranges": [[0, 1, 2]]})]}
        }
        OWPreprocess.migrate_settings(settings, 6)
        self.assertEqual(
            settings["storedsettings"]["preprocessors"][0],
            (name, {"ranges": [[0, 1, 2, 0]]}),
        )
