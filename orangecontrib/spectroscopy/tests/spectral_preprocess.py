from AnyQt.QtTest import QSignalSpy

from Orange.data import Table
from Orange.widgets.data.owpreprocess import PreprocessAction, Description
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import Msg
from orangecontrib.spectroscopy.preprocess.utils import PreprocessException
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange


def pack_editor(editor):
    return PreprocessAction("", "", "", Description("Packed"), editor)


def wait_for_preview(widget, timeout=5000):
    spy = QSignalSpy(widget.preview_runner.preview_updated)
    assert spy.wait(timeout), "Failed to get output in the specified timeout"


class WarningEditor(BaseEditorOrange):
    class Warning(BaseEditorOrange.Warning):
        some_warning = Msg("Warn.")

    def __init__(self):
        super().__init__()
        self.warn_editor = False
        self.warn_widget = False
        self.raise_exception = False

    def setParameters(self, p):
        pass

    def parameters(self):
        return {"raise_exception": self.raise_exception}

    def set_preview_data(self, data):
        if self.warn_editor:
            self.Warning.some_warning()
        else:
            self.Warning.some_warning.clear()

        if self.warn_widget:
            self.parent_widget.Warning.preprocessor()

    @staticmethod
    def createinstance(params):
        if params.get("raise_exception", False):
            raise PreprocessException("42")
        return lambda x, **kwargs: x


class TestWarning(WidgetTest):
    widget_cls = None

    def setUp(self):
        self.widget = self.create_widget(self.widget_cls)
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.add_preprocessor(pack_editor(WarningEditor))
        self.editor = self.widget.flow_view.widgets()[0]
        self.assertIsInstance(self.editor, WarningEditor)

    def test_no_warnings(self):
        self.widget.show_preview()
        wait_for_preview(self.widget)
        self.assertFalse(self.widget.Warning.preprocessor.is_shown())
        self.assertFalse(self.editor.Warning.some_warning.is_shown())
        self.assertTrue(self.editor.message_bar.isHidden())

    def test_warn_widget(self):
        self.editor.warn_widget = True
        self.widget.show_preview()
        wait_for_preview(self.widget)
        self.assertTrue(self.widget.Warning.preprocessor.is_shown())
        self.editor.warn_widget = False
        self.widget.show_preview()
        wait_for_preview(self.widget)
        self.assertFalse(self.widget.Warning.preprocessor.is_shown())

    def test_warn_editor(self):
        self.editor.warn_editor = True
        self.widget.show_preview()
        wait_for_preview(self.widget)
        self.assertTrue(self.editor.Warning.some_warning.is_shown())
        self.assertFalse(self.editor.message_bar.isHidden())
        self.editor.warn_editor = False
        self.widget.show_preview()
        wait_for_preview(self.widget)
        self.assertFalse(self.editor.Warning.some_warning.is_shown())
        self.assertTrue(self.editor.message_bar.isHidden())

    def test_exception_preview(self):
        self.editor.raise_exception = True
        self.editor.edited.emit()
        wait_for_preview(self.widget)
        self.assertTrue(self.editor.Error.exception.is_shown())
        self.assertTrue(self.widget.Error.preview.is_shown())
        self.assertFalse(self.editor.message_bar.isHidden())

        self.editor.raise_exception = False
        self.editor.edited.emit()
        wait_for_preview(self.widget)
        self.assertFalse(self.editor.Error.exception.is_shown())
        self.assertFalse(self.widget.Error.preview.is_shown())
        self.assertTrue(self.editor.message_bar.isHidden())

    def test_exception_apply(self):
        self.editor.raise_exception = True
        self.editor.edited.emit()
        self.widget.commit.now()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.applying.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.preprocessed_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.preprocessor))

        self.editor.raise_exception = False
        self.editor.edited.emit()
        self.widget.commit.now()
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.applying.is_shown())
        self.assertIsNotNone(self.get_output(self.widget.Outputs.preprocessed_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.preprocessor))
