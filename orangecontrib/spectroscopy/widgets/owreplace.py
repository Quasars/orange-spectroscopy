import numpy as np
import sys

import Orange.data
from Orange.widgets import gui, settings, widget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from AnyQt.QtWidgets import QFormLayout, QWidget
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatOrNone


class UnknownValueException(Exception):
    pass


REPLACEMENT_MODES = ["Tolerance", "Minimum value", "Maximum value"]

class OWReplace(widget.OWWidget, ConcurrentWidgetMixin):
    name = "Replace Values"
    description = "Change attribute values where they equal some value."
    icon = "icons/replace.svg"
    id = "orangecontrib.spectroscopy.widgets.owreplace"
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

    mode = settings.Setting(0)
    old_value = settings.Setting("")
    new_value = settings.Setting("")
    tolerance = settings.Setting("")
    
    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False


    def __init__(self):
        widget.OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data = None
        

        gui.comboBox(
            self.controlArea, self, "mode",
            contentsLength=12, searchable=True,
            callback=self._change_input, items=REPLACEMENT_MODES
        )

        box = gui.widgetBox(self.controlArea, "Replace")

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        box.layout().addWidget(form)

        self.old_value_edit = lineEditFloatOrNone(
            box, self, "old_value",
            callback=self.commit.deferred
            )
        self.new_value_edit = lineEditFloatOrNone(
            box, self, "new_value",
            callback=self.commit.deferred
            )
        self.tolerance_edit = lineEditFloatOrNone(
            box, self, "tolerance",
            callback=self.commit.deferred
            )
        
        formlayout.addRow("Old value: ", self.old_value_edit)
        formlayout.addRow("New value: ", self.new_value_edit)
        formlayout.addRow("Tolerance: ", self.tolerance_edit)

        self._update_input()
        gui.auto_apply(self.buttonsArea,            
                       self,            
                       "autocommit",            
                       commit=self.commit,        
                       )

    def _update_input(self):
        check = self.mode == 0
        self.tolerance_edit.setEnabled(check)

    def _change_input(self):
        self._update_input()
        self.commit.deferred()

    
    @staticmethod
    def tolerance_replace(data, old_value, new_value, **kwargs):
    
        mask = np.logical_and(data.X >= old_value - kwargs["tol"], 
                              data.X <= old_value + kwargs["tol"])

        data.X[mask] = new_value

        if kwargs["want_count"]:
            return data, np.sum(mask)
        
        return data

    @staticmethod
    def min_replace(data, old_value, new_value, **kwargs):
    
        mask = np.where(data.X <= old_value)
        data.X[mask] = new_value

        if kwargs["want_count"]:
            return data, np.sum(mask)
        
        return data
    
    @staticmethod
    def max_replace(data, old_value, new_value, **kwargs):
    
        mask = np.where(data.X >= old_value)
        data.X[mask] = new_value

        if kwargs["want_count"]:
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
        
        out = None
        if self.data:
            if self.mode == 0:
                
                if self.old_value == "" or self.new_value == "":
                    return self.data
                
                if self.tolerance == "":
                    self.tolerance = "0.0"

                old_value = self.get_value(self.old_value)
                new_value = self.get_value(self.new_value)
                tolerance_value = self.get_value(self.tolerance)
                kwargs={"tol": tolerance_value,
                        "want_count": True}
                out, n = self.tolerance_replace(self.data.copy(),
                                                old_value, new_value,
                                                **kwargs)
                self.Information.changed_count(n, f"{old_value - tolerance_value :.3f} <= x <= {old_value + tolerance_value :.3f}", new_value)

                
            elif self.mode == 1:
                
                if self.old_value == "" or self.new_value == "":
                    return self.data
                
                old_value = self.get_value(self.old_value)
                new_value = self.get_value(self.new_value)
                kwargs={"want_count": True}
                out, n = self.min_replace(self.data.copy(),
                                              old_value, new_value,
                                              **kwargs)
                self.Information.changed_count(n, f"{old_value:.3f} <= x", new_value)

                
            elif self.mode == 2:
                if self.old_value == "" or self.new_value == "":
                    return self.data

                old_value = self.get_value(self.old_value)
                new_value = self.get_value(self.new_value)
                kwargs={"want_count": True}
                out, n = self.max_replace(self.data.copy(),
                                              old_value, new_value,
                                              **kwargs)
                self.Information.changed_count(n, f"x <= {old_value:.3f}", new_value)
                

        return out

    def value_changed(self):
        self.commit()

    @Inputs.data
    def set_data(self, data):
        self.data = data

    @gui.deferred
    def commit(self):
        self.Error.clear()
        self.Information.clear()

        data = None
        try:
            data = self.get_outdata()
        except UnknownValueException:
            self.Error.invalid_value()

        self.Outputs.data.send(data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWWhere).run()
