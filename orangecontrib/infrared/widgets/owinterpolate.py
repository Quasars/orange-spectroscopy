import sys
import math

import numpy as np
import Orange.data
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui, settings

from AnyQt.QtCore import Qt

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.preprocess import Interpolate
from orangecontrib.infrared.widgets.gui import lineEditFloatOrNone


class OWInterpolate(OWWidget):
    name = "Interpolate"
    description = "Interpolate spectra"
    icon = "icons/interpolate.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Points", Orange.data.Table, "set_points")]
    outputs = [("Interpolated data", Orange.data.Table)]

    # how are the interpolation points given
    input_radio = settings.Setting(0)

    # specification of linear space
    xmin = settings.Setting(None)
    xmax = settings.Setting(None)
    dx = settings.Setting(10.)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        reference_data_missing = Msg("Missing separate reference data input.")
        reference_data_unused = Msg("Reference data is present but unused.")

    class Error(OWWidget.Error):
        dxzero = Msg("Step should be higher than 0.0.")
        too_many_points = Msg("More than 10000 points with your current setting.")

    def __init__(self):
        super().__init__()

        self.data_points = None

        dbox = gui.widgetBox(self.controlArea, "Interpolation")

        rbox = gui.radioButtons(
            dbox, self, "input_radio", callback=self._change_input)

        gui.appendRadioButton(rbox, "Enable automatic interpolation")

        gui.appendRadioButton(rbox, "Linear interval")
        ibox = gui.indentedBox(rbox)

        self.xmin_edit = lineEditFloatOrNone(ibox, self, "xmin",
            label="Min", labelWidth=50, orientation=Qt.Horizontal,
            callback=self.commit)
        self.xmax_edit = lineEditFloatOrNone(ibox, self, "xmax",
            label="Max", labelWidth=50, orientation=Qt.Horizontal,
            callback=self.commit)
        self.dx_edit = lineEditFloatOrNone(ibox, self, "dx",
            label="Δ", labelWidth=50, orientation=Qt.Horizontal,
            callback=self.commit)

        gui.appendRadioButton(rbox, "Reference data")

        self.data = None

        gui.auto_commit(self.controlArea, self, "autocommit", "Interpolate")
        self._change_input()

    def commit(self):
        out = None
        self.Error.dxzero.clear()
        self.Error.too_many_points.clear()
        if self.data:
            if self.input_radio == 0:
                points = getx(self.data)
                out = Interpolate(points)(self.data)
            elif self.input_radio == 1:
                xs = getx(self.data)
                if not self.dx > 0:
                    self.Error.dxzero()
                else:
                    xmin = self.xmin if self.xmin is not None else np.min(xs)
                    xmax = self.xmax if self.xmax is not None else np.max(xs)
                    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
                    reslength = abs(math.ceil((xmax - xmin)/self.dx))
                    if reslength < 10002:
                        points = np.arange(xmin, xmax, self.dx)
                        out = Interpolate(points)(self.data)
                    else:
                        self.Error.too_many_points(reslength)
            elif self.input_radio == 2 and self.data_points is not None:
                out = Interpolate(self.data_points)(self.data)
        self.send("Interpolated data", out)

    def _change_input(self):
        if self.input_radio == 2 and self.data_points is None:
            self.Warning.reference_data_missing()
        else:
            self.Warning.reference_data_missing.clear()
        self.xmin_edit.setDisabled(self.input_radio != 1)
        self.xmax_edit.setDisabled(self.input_radio != 1)
        self.dx_edit.setDisabled(self.input_radio != 1)
        self.commit()

    def set_data(self, data):
        self.data = data
        if self.data and len(getx(data)):
            points = getx(data)
            self.xmin_edit.setPlaceholderText(str(np.min(points)))
            self.xmax_edit.setPlaceholderText(str(np.max(points)))
        else:
            self.xmin_edit.setPlaceholderText("")
            self.xmax_edit.setPlaceholderText("")
        self.commit()

    def set_points(self, data):
        if data:
            self.data_points = getx(data)
        else:
            self.data_points = None
        self._change_input()


def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    ow = OWInterpolate()
    ow.show()
    ow.raise_()

    dataset = Orange.data.Table("collagen")
    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    return 0

if __name__=="__main__":
    sys.exit(main())
