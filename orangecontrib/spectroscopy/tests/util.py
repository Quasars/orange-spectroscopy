from contextlib import contextmanager
from unittest.mock import patch

from AnyQt.QtTest import QTest
from AnyQt.QtCore import Qt

import Orange
from Orange.tests import named_file


@contextmanager
def hold_modifiers(widget, modifiers):
    # use some unexisting key
    QTest.keyPress(widget, Qt.Key_F35, modifiers)
    try:
        yield
    finally:
        QTest.keyRelease(widget, Qt.Key_F35)


def smaller_data(data, nth_instance, nth_feature):
    natts = [a for i, a in enumerate(data.domain.attributes) if i % nth_feature == 0]
    data = data[::nth_instance]
    ndomain = Orange.data.Domain(natts, data.domain.class_vars, metas=data.domain.metas)
    return data.transform(ndomain)


@contextmanager
def set_png_graph_save():
    with named_file("", suffix=".png") as fname:
        with patch(
            "AnyQt.QtWidgets.QFileDialog.getSaveFileName",
            lambda *x: (fname, 'Portable Network Graphics (*.png)'),
        ):
            yield fname


def spectra_table(wavenumbers, *args, **kwargs):
    domain = Orange.data.Domain(
        [Orange.data.ContinuousVariable(str(w)) for w in wavenumbers]
    )
    data = Orange.data.Table.from_numpy(domain, *args, **kwargs)
    return data


def checkbox_linked_test(self, widget, checkbox_name, setting_name):
    def _get_checkbox():
        return getattr(widget.controls, checkbox_name)

    def get_checkbox():
        checkbox = _get_checkbox()
        return checkbox.isChecked()
    
    def set_checkbox(flag):
        checkbox = _get_checkbox()
        checkbox.setChecked(flag)
    
    def get_setting():
        return getattr(widget, setting_name)
    
    def set_setting(flag):
        setattr(widget, setting_name, flag)


    default_setting = get_setting()
    default_checkbox = get_checkbox()

    # Setting and checkbox checked should be the same.
    self.assertEqual(
        default_setting,
        default_checkbox,
        "setting and checkbox checked don't have the same initial value"
    )

    # Changing checkbox checked should change the setting.
    set_checkbox(not default_setting)

    self.assertEqual(
        get_checkbox(),
        not default_setting,
        "'checkbox.setChecked(flag)' didn't check checkbox"
    )

    self.assertEqual(
        get_setting(),
        not default_setting,
        "setting didn't change when checkbox checked did"
    )

    # Changing the setting should change checkbox checked.
    set_setting(default_setting)

    self.assertEqual(
        get_setting(),
        default_setting,
        "'setting=flag' didn't change setting to flag"
    )

    self.assertEqual(
        get_checkbox(),
        default_setting,
        "checkbox checked didn't change when setting did"
    )
