import numpy as np

import Orange.data
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from AnyQt.QtWidgets import QFormLayout, QWidget


class CalculationError(Exception):
    pass


REPLACEMENT_MODES = ["Tolerance", "Minimum value", "Maximum value"]

class Replacement:

    def __init__(self, data, mode, old_value, new_value, 
                 tolerance=None, return_count = True):

        self.function_map = {0: self.tolerance_replace,
                             1: self.min_replace, 
                             2: self.max_replace
                             }

        self.data = data
        self.mode = mode
        self.old_value = old_value
        self.new_value = new_value
        self.tolerance = tolerance
        self.return_count = return_count
        self.message = None
        self.data_out = None

    def calculate(self):
        self.function_map.get(self.mode, None)(self.data,
                                               self.old_value,
                                               self.new_value
                                               )
        return self.data_out

    def tolerance_replace(self, data, old_value, new_value):
    
        mask = np.logical_and(data.X >= old_value - self.tolerance, 
                              data.X <= old_value + self.tolerance)
        data.X[mask] = new_value
        self.message = (np.sum(mask), f"{old_value - self.tolerance :.3f} <= x <= {old_value + self.tolerance :.3f}", new_value)
        self.data_out = data

    def min_replace(self, data, old_value, new_value):
    
        mask = np.where(data.X <= old_value)
        data.X[mask] = new_value
        self.message = (np.sum(mask), f"{old_value:.3f} <= x", new_value)
        self.data_out = data
    
    def max_replace(self, data, old_value, new_value):
    
        mask = np.where(data.X >= old_value)
        data.X[mask] = new_value
        self.message = (np.sum(mask), f"x <= {old_value:.3f}", new_value)        
        self.data_out = data
    

class OWReplace(OWWidget):
    name = "Replace Values"
    description = "Change attribute values where they equal some value."
    icon = "icons/replace.svg"
    id = "orangecontrib.spectroscopy.widgets.owreplace"
    priority = 20


    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        data = Output("Data", Orange.data.Table, default=True)

    class Error(OWWidget.Error):
        invalid_value = Msg("Can't parse '{}'.")
        calculation_error = Msg("Calculation error")

    class Information(OWWidget.Information):
        changed_count = Msg("{} values have been changed from '{}' to '{}'.")

    mode = settings.Setting(0)
    old_value = settings.Setting(None)
    new_value = settings.Setting(None)
    tolerance = settings.Setting(None)
    
    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False


    def __init__(self):
        OWWidget.__init__(self)

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

        self.old_value_edit = gui.lineEdit(
            box, self, "old_value",
            callback=self.commit.deferred
            )
        self.new_value_edit = gui.lineEdit(
            box, self, "new_value",
            callback=self.commit.deferred
            )
        self.tolerance_edit = gui.lineEdit(
            box, self, "tolerance",
            callback=self.commit.deferred
            )
        
        formlayout.addRow("Old value: ", self.old_value_edit)
        formlayout.addRow("New value: ", self.new_value_edit)
        formlayout.addRow("Tolerance: ", self.tolerance_edit)

        gui.auto_apply(self.buttonsArea,            
                       self,            
                       "autocommit",            
                       )
        self._change_input()

    def _update_input(self):
        check = self.mode == 0
        self.tolerance_edit.setEnabled(check)

    def _change_input(self):
        self._update_input()
        self.commit.deferred()
    
    @staticmethod
    def get_value(text):
        try: # in case something like 'nan'
            return np.float64(text)
        except ValueError:
            try: # in case something like 'np.nan' is given
                return np.float64(exec(text))
            except NameError:
                return None

    def get_outdata(self):
        if self.data is None:
            return None
        
        out = None

        if self.old_value is None or self.new_value is None:
            return self.data

        if self.get_value(self.old_value) is None:
            self.Error.invalid_value(self.old_value)
            return self.data
            
        if self.get_value(self.new_value) is None:
            self.Error.invalid_value(self.new_value)
            return self.data

        if self.data:
            replacer = Replacement(data=self.data.copy(),
                                   mode=self.mode,
                                   old_value=self.get_value(self.old_value),
                                   new_value=self.get_value(self.new_value),
                                   tolerance=self.tolerance)
            out = replacer.calculate()

            if replacer.message is not None:
                # Would rather just 
                # self.Information.changed_count(replacer.message) 
                # but can't figure out how
                self.Information.changed_count(replacer.message[0],
                                               replacer.message[1],
                                               replacer.message[2])
            if replacer.data_out is None:
                self.Error.calculation_error()
                # out = None

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

    def handleNewSignals(self):
        self.commit.now()

if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWWhere).run()
