import numpy as np

import Orange.data
from Orange.widgets import gui, settings, widget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from AnyQt.QtWidgets import QFormLayout, QWidget
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatOrNone


class UnknownValueException(Exception):
    pass

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
    tolerance = settings.Setting("0")
    old_min_value = settings.Setting("")
    new_min_value = settings.Setting("")
    old_max_value = settings.Setting("")
    new_max_value = settings.Setting("")
    
    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False


    def __init__(self):
        widget.OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data = None
        
        box = gui.widgetBox(self.controlArea, "Info")
        gui.widgetLabel(box, 'WARNING: elementwise operation.\nTake care using this widget if the order of elements in the input data have been changed.')

        gui.separator(self.controlArea)

        where = gui.widgetBox(self.controlArea, "Set Where")

        rbox = gui.radioButtons(where, self,
                                "mode",
                                callback=self._change_input)

        # section for the tolerance mode
        gui.appendRadioButton(rbox, "Tolerance value replacement.\n'New Value' will replace any cell where Old Value +/- Tolerance.")
        tol_box = gui.indentedBox(rbox)

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        tol_box.layout().addWidget(form)

        self.old_value_edit = lineEditFloatOrNone(
            tol_box, self, "old_value", 
            callback=self.commit.deferred
            )
        formlayout.addRow("Old Value: ", self.old_value_edit)
        self.new_value_edit = lineEditFloatOrNone(
            tol_box, self, "new_value", 
            callback=self.commit.deferred
            )
        formlayout.addRow("New Value: ", self.new_value_edit)
        self.tolerance_edit = lineEditFloatOrNone(
            tol_box, self, "tolerance", 
            callback=self.commit.deferred
            )
        formlayout.addRow("Tolerance: ", self.tolerance_edit)

        # section for the minimum value replacement
        gui.appendRadioButton(rbox, "Minimum value replacement.\n'New Value' will replace any cell currently < Old Value.")
        min_box = gui.indentedBox(rbox)

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        min_box.layout().addWidget(form)

        self.old_min_value_edit = lineEditFloatOrNone(
            min_box, self, "old_min_value", 
            callback=self.commit.deferred
            )
        formlayout.addRow("Old Value: ", self.old_min_value_edit)
        self.new_min_value_edit = lineEditFloatOrNone(
            min_box, self, "new_min_value", 
            callback=self.commit.deferred
            )
        formlayout.addRow("New Value: ", self.new_min_value_edit)

        # section for the maximum value replacement
        gui.appendRadioButton(rbox, "Maximum value replacement.\n'New Value' will replace any cell currently > Old Value.")
        max_box = gui.indentedBox(rbox)

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        max_box.layout().addWidget(form)

        self.old_max_value_edit = lineEditFloatOrNone(
            max_box, self, "old_max_value", 
            callback=self.commit.deferred
            )
        formlayout.addRow("Old Value: ", self.old_max_value_edit)
        self.new_max_value_edit = lineEditFloatOrNone(
            max_box, self, "new_max_value", 
            callback=self.commit.deferred
            )
        formlayout.addRow("New Value: ", self.new_max_value_edit)

        gui.auto_commit(self.controlArea, self, "autocommit", "Replace values")
        self._change_input()


    def _update_input(self):
        # reset unused min/max values when we change which mode we use.
        if self.mode == 0:
            self.old_min_value = ""
            self.new_min_value = ""
            self.old_max_value = ""
            self.new_max_value = ""
        elif self.mode == 1:
            self.old_value = ""
            self.new_value = ""
            self.old_max_value = ""
            self.new_max_value = ""
        elif self.mode == 2:
            self.old_value = ""
            self.new_value = ""
            self.old_min_value = ""
            self.new_min_value = ""


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
                self.Information.changed_count(n, f"{old_value - tolerance_value} <= x <= {old_value + tolerance_value}", new_value)

                
            elif self.mode == 1:
                
                if self.old_min_value == "" or self.new_min_value == "":
                    return self.data
                
                old_value = self.get_value(self.old_min_value)
                new_value = self.get_value(self.new_min_value)
                kwargs={"want_count": True}
                out, n = self.min_replace(self.data.copy(),
                                              old_value, new_value,
                                              **kwargs)
                self.Information.changed_count(n, f"{old_value} <= x", new_value)

                
            elif self.mode == 2:
                if self.old_max_value == "" or self.new_max_value == "":
                    return self.data

                old_value = self.get_value(self.old_max_value)
                new_value = self.get_value(self.new_max_value)
                kwargs={"want_count": True}
                out, n = self.max_replace(self.data.copy(),
                                              old_value, new_value,
                                              **kwargs)
                self.Information.changed_count(n, f"x <= {old_value}", new_value)
                

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

        data = self.get_outdata()
        
        self.Outputs.data.send(data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWWhere).run()
