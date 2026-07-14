import numpy as np

from AnyQt import QtWidgets

import Orange.data
from Orange.widgets import gui, settings, widget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin


class UnknownValueException(Exception):
    pass

class OWWhere(widget.OWWidget, ConcurrentWidgetMixin):
    name = "Elementwise Where"
    description = "Change attribute values where they equal some value."
    icon = "icons/where.svg"
    id = "orangecontrib.spectroscopy.widgets.owwhere"
    priority = 20


    class Inputs:
        data = widget.Input("Data", Orange.data.Table, default=True)

    class Outputs:
        data = widget.Output("Data", Orange.data.Table, default=True)

    class Error(widget.OWWidget.Error):
        invalid_value = widget.Msg("Can't parse '{}'.")

    class Information(widget.OWWidget.Information):
        changed_count = widget.Msg("{} values have been changed from '{}' to '{}'.")

    settingsHandler = settings.DomainContextHandler()

    old_value = settings.Setting("")
    new_value = settings.Setting("")
    tolerance = settings.Setting("0")
    
    want_main_area = False
    resizing_enabled = False


    def __init__(self):
        widget.OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data = None
        
        where = gui.widgetBox(self.controlArea, "Set Where")

        gui.lineEdit(where, self, "old_value", label="Old Value: ", callback=self.value_changed)
        gui.lineEdit(where, self, "new_value", label="New Value: ", callback=self.value_changed)
        gui.lineEdit(where, self, "tolerance", label="Tolerance: ", callback=self.value_changed)

    
    @staticmethod
    def where(data, old_value, new_value, tol=0.0, want_count=False):
    
        mask = np.logical_and(data.X >= old_value - tol, data.X <= old_value + tol)

        data.X[mask] = new_value

        if want_count:
            return data, np.sum(mask)
        
        return data
    
    @staticmethod
    def get_value(text):
        try:
            return np.float64(text)
        except ValueError:
            pass
        
        try:
            return np.float64(eval(text))
        except (AttributeError):
            raise UnknownValueException()

        
    def get_outdata(self):
        if self.data is None:
            return None
        
        if self.old_value == "" or self.new_value == "":
            return self.data
        
        if self.tolerance == "":
            self.tolerance = "0.0"

        
        try:
            old_val = OWWhere.get_value(self.old_value)
        except UnknownValueException:
            self.Error.invalid_value(self.old_value)
            return self.data

        try:
            new_val = OWWhere.get_value(self.new_value)
        except UnknownValueException:
            self.Error.invalid_value(self.new_value)
            return self.data
        
        try:
            tolerance = OWWhere.get_value(self.tolerance)
        except UnknownValueException:
            self.Error.invalid_value(self.tolerance)
            return self.data

        data, n = OWWhere.where(self.data.copy(), old_val, new_val, tol=tolerance, want_count=True)

        self.Information.changed_count(n, f"{old_val - tolerance} <= x <= {old_val + tolerance}", self.new_value)

        return data


    def value_changed(self):
        self.commit()


    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.commit()


    def commit(self):
        self.Error.clear()
        self.Information.clear()

        data = self.get_outdata()
        
        self.Outputs.data.send(data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWWhere).run()
