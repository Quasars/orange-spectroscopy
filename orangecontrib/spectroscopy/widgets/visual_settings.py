from AnyQt.QtWidgets import QLineEdit

from orangewidget.utils.visual_settings_dlg import _add_control, _set_control_value

from orangecontrib.spectroscopy.widgets.gui import (
    FloatOrEmptyValidator,
    floatornone,
    str_or_empty,
)


class FloatOrUndefined:
    pass


class FloatOrEmptyLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validator = FloatOrEmptyValidator(self, allow_empty=True)
        self.setValidator(validator)

    def setValue(self, value):
        self.setText(str_or_empty(value))


@_add_control.register(FloatOrUndefined)
def _(_: FloatOrUndefined, value, key, signal):
    line_edit = FloatOrEmptyLineEdit()
    line_edit.setValue(value)
    line_edit.textChanged.connect(lambda text: signal.emit(key, floatornone(text)))
    return line_edit


@_set_control_value.register(FloatOrEmptyLineEdit)
def _(edit: FloatOrEmptyLineEdit, value: str):
    edit.setValue(value)
