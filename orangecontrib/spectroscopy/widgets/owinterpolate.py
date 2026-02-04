import math

import numpy as np
import Orange.data
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings

from AnyQt.QtWidgets import QFormLayout, QWidget

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import (
    Interpolate,
    InterpolateToDomain,
    NotAllContinuousException,
)
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatOrNone


class OWInterpolate(OWWidget):
    name = "Interpolate"
    description = "Interpolate spectra"
    icon = "icons/interpolate.svg"
    priority = 990
    replaces = ["orangecontrib.infrared.widgets.owinterpolate.OWInterpolate"]

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)
        points = Input("Points", Orange.data.Table)

    class Outputs:
        interpolated_data = Output("Interpolated data", Orange.data.Table, default=True)

    # how are the interpolation points given
    input_radio = settings.Setting(0)

    # specification of linear space
    xmin = settings.Setting(None)
    xmax = settings.Setting(None)
    dx = settings.Setting(10.0)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        reference_data_missing = Msg("Missing separate reference data input.")
        reference_data_unused = Msg("Reference data is present but unused.")

    class Error(OWWidget.Error):
        dxzero = Msg("Step should be higher than 0.0.")
        too_many_points = Msg("More than 10000 points with your current setting.")
        non_continuous = Msg("Points input contains non-continuous features.")

    def __init__(self):
        super().__init__()

        self.data_points_interpolate = None

        dbox = gui.widgetBox(self.controlArea, "Interpolation")

        rbox = gui.radioButtons(dbox, self, "input_radio", callback=self._change_input)

        gui.appendRadioButton(rbox, "Enable automatic interpolation")

        gui.appendRadioButton(rbox, "Linear interval")

        ibox = gui.indentedBox(rbox)

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        ibox.layout().addWidget(form)

        self.xmin_edit = lineEditFloatOrNone(
            ibox, self, "xmin", callback=self.commit.deferred
        )
        formlayout.addRow("Min", self.xmin_edit)
        self.xmax_edit = lineEditFloatOrNone(
            ibox, self, "xmax", callback=self.commit.deferred
        )
        formlayout.addRow("Max", self.xmax_edit)
        self.dx_edit = lineEditFloatOrNone(
            ibox, self, "dx", callback=self.commit.deferred
        )
        formlayout.addRow("Δ", self.dx_edit)

        gui.appendRadioButton(rbox, "Reference data")

        self.data = None

        gui.auto_commit(self.controlArea, self, "autocommit", "Interpolate")
        self._change_input()

    @gui.deferred
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
                    reslength = abs(math.ceil((xmax - xmin) / self.dx))
                    if reslength < 10002:
                        points = np.arange(xmin, xmax, self.dx)
                        out = Interpolate(points)(self.data)
                    else:
                        self.Error.too_many_points(reslength)
            elif self.input_radio == 2 and self.data_points_interpolate is not None:
                out = self.data_points_interpolate(self.data)
        self.Outputs.interpolated_data.send(out)

    def _update_input_type(self):
        if self.input_radio == 2 and self.data_points_interpolate is None:
            self.Warning.reference_data_missing()
        else:
            self.Warning.reference_data_missing.clear()
        self.xmin_edit.setDisabled(self.input_radio != 1)
        self.xmax_edit.setDisabled(self.input_radio != 1)
        self.dx_edit.setDisabled(self.input_radio != 1)

    def _change_input(self):
        self._update_input_type()
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if self.data and len(getx(data)):
            points = getx(data)
            self.xmin_edit.setPlaceholderText(str(np.min(points)))
            self.xmax_edit.setPlaceholderText(str(np.max(points)))
        else:
            self.xmin_edit.setPlaceholderText("")
            self.xmax_edit.setPlaceholderText("")

    @Inputs.points
    def set_points(self, data):
        self.Error.non_continuous.clear()
        if data:
            try:
                self.data_points_interpolate = InterpolateToDomain(target=data)
            except NotAllContinuousException:
                self.data_points_interpolate = None
                self.Error.non_continuous()
        else:
            self.data_points_interpolate = None
        self._update_input_type()

    def handleNewSignals(self):
        self.commit.now()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWInterpolate).run(Orange.data.Table("collagen"))
