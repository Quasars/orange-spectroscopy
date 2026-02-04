import numpy as np

from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.me_emsc import MeEMSCEditor


class TestMeEMSCEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(MeEMSCEditor, self.widget)  # type: MeEMSCEditor
        self.editor.controls.max_iter.setValue(3)
        # FIXME: current ME-EMSC can not handle same data and reference
        self.data = SMALL_COLLAGEN[1:3]
        self.send_signal(self.widget.Inputs.data, self.data)

    def test_no_interaction(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(self.widget.Inputs.reference, reference)
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, ME_EMSC)

    def test_ncomp(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(self.widget.Inputs.reference, reference)

        p = self.commit_get_preprocessor()
        self.assertFalse(self.editor.controls.ncomp.isEnabled())
        self.assertEqual(p.ncomp, 6)  # for this data set

        self.editor.controls.autoset_ncomp.click()
        self.editor.controls.ncomp.setValue(4)
        p = self.commit_get_preprocessor()
        self.assertTrue(self.editor.controls.ncomp.isEnabled())
        self.assertEqual(p.ncomp, 4)  # for this data set

        self.editor.controls.autoset_ncomp.click()
        p = self.commit_get_preprocessor()
        self.assertFalse(self.editor.controls.ncomp.isEnabled())
        self.assertEqual(p.ncomp, 6)  # for this data set

    def _change_le(self, le, val):
        le.setText(val)
        le.textEdited.emit(val)
        le.editingFinished.emit()

    def test_refractive_index(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(self.widget.Inputs.reference, reference)

        p = self.commit_get_preprocessor()
        np.testing.assert_equal(p.n0, np.linspace(1.1, 1.4, 10))

        self._change_le(self.editor.controls.n0_low, str(1.2))
        self._change_le(self.editor.controls.n0_high, str(1.5))
        p = self.commit_get_preprocessor()
        np.testing.assert_equal(p.n0, np.linspace(1.2, 1.5, 10))

    def test_spherical_radius(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(self.widget.Inputs.reference, reference)

        p = self.commit_get_preprocessor()
        np.testing.assert_equal(p.a, np.linspace(2, 7.1, 10))

        self._change_le(self.editor.controls.a_low, str(3))
        self._change_le(self.editor.controls.a_high, str(42))
        p = self.commit_get_preprocessor()
        np.testing.assert_equal(p.a, np.linspace(3, 42, 10))

    def test_iterations(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(self.widget.Inputs.reference, reference)

        p = self.commit_get_preprocessor()
        self.assertEqual(p.maxNiter, 3)
        self.assertEqual(p.fixedNiter, False)

        self.editor.controls.max_iter.setValue(4)
        self.editor.controls.fixed_iter.click()
        p = self.commit_get_preprocessor()
        self.assertEqual(p.maxNiter, 4)
        self.assertEqual(p.fixedNiter, 4)

    def test_migrate_smoothing(self):
        name = "orangecontrib.spectroscopy.preprocess.me_emsc.me_emsc"
        settings = {
            "storedsettings": {"preprocessors": [(name, {"ranges": [[0, 1, 2]]})]}
        }
        OWPreprocess.migrate_settings(settings, 6)
        self.assertEqual(
            settings["storedsettings"]["preprocessors"][0],
            (name, {"ranges": [[0, 1, 2, 0]]}),
        )
