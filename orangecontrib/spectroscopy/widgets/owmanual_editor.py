import numpy as np
import pyqtgraph as pg
from sys import float_info

import Orange.data

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QWidget, QSplitter

from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, OWBaseWidget, Msg
from Orange.widgets.settings import Setting, DomainContextHandler, SettingProvider
from Orange.widgets.utils.concurrent import  ConcurrentWidgetMixin
from orangecontrib.spectroscopy.util import getx
from orangecontrib.spectroscopy.widgets.owspectra import CurvePlot
from orangecontrib.spectroscopy.preprocess import DegTilt

def calc_slope_extrema(data: Orange.data.Table):
    """calc_slope_extrema Finds the maximum and minimum slope in relation 
    to a referenc point in the beginnning (on the x axis) of the data arrays 
    and the mean of the first element of each data arrays.

    Args:
        data (Orange.data.Table): Standard Orange data table input. 
    """
    def slope(dx, dy):
        if dx != 0:
            return np.degrees(np.arctan(dy/dx))
        else:
            return np.sign(dy)*90 # dy<0 -> -90, dy=0 -> 0, dy>0 -> 90
    
    vslope = np.vectorize(slope)
        
    xdom = getx(data)
    xref = xdom[0]

    y = data.X
    yref = np.mean(y[:,0])

    dx = xdom - xref
    dy = y - yref

    slopes = vslope(dx, dy)
    extrema = np.min(slopes), np.max(slopes)
                
    return {'refpos': {'x':xref,'y':yref}, 'slope': extrema} 


class OWManualEditor(OWWidget, ConcurrentWidgetMixin):
    """OWManualEditor _summary_

    Attributes:
        Inputs (Orange.data.Table): Default OWWidget Input containing multiple spectra
    """

    name = "ManualEditor"

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        data_edited = Output("Edited Data", Orange.data.Table, default=True)

    icon = "icons/hyper.svg"
    priority = 200 # change this number to an appropriate one
    keywords = ["image", "spectral", "chemical", "imaging"]

    settings_version = 6
    settingsHandler = DomainContextHandler()

    plot_in = SettingProvider(CurvePlot)
    plot_out = SettingProvider(CurvePlot)

    lowlim = Setting(None)
    highlim = Setting(None)

    autocommit = settings.Setting(True)

    class Warning(OWBaseWidget.Warning):
        out_of_range = Msg("Limits are out of range.")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        # data control variable
        self.data = None

        # init slope control variables (in degrees)

        self.slope = 0.
        self.slope_max = 90. 
        self.slope_min = -90. 
        self.slope_step = 0.01*(self.slope_max-self.slope_min)/2

        # grid control variable for input controllers (slider, spin and buttons)
        box = gui.widgetBox(self.controlArea, "Map grid")

        # slope_controls box
        slope_controls = gui.vBox(box, "Slope Controls")

        # self.slope_slider_hbox elements 
        # TODO: Solve confictinn updating values in slope_spin_val and slope 
        self.slope_slider_hbox = gui.hBox(slope_controls)
        self.slope_slider = gui.hSlider(self.slope_slider_hbox, self, "slope", 0., minValue=self.slope_min, maxValue=self.slope_max, step=self.slope_step, label="Slope", 
                    callback=self._update_slope, intOnly=False, labelFormat=" %.4f", createLabel=False)
        self.slope_spin_val = 0.
        self.slope_spin = gui.spin(self.slope_slider_hbox, self, "slope_spin_val", self.slope_min, self.slope_max, step=self.slope_step, label=None,
                callback=self._update_slope_spin, spinType=float, callbackOnReturn=True)
        # self.slope_spin = gui.lineEdit(self.slope_slider_hbox, self, "slope_spin_val", callback=self._update_slope_spin, valueType=float, )
        
        # slope_buttons elements
        slope_buttons = gui.hBox(slope_controls)
        gui.button(slope_buttons, self, "-10x", callback=self._buttonSlope10Down, autoDefault=False)
        gui.button(slope_buttons, self, "-1x", callback=self._buttonSlope1Down, autoDefault=False)
        gui.button(slope_buttons, self, "+1x", callback=self._buttonSlope1Up, autoDefault=False)
        gui.button(slope_buttons, self, "+10x", callback=self._buttonSlope10Up, autoDefault=False)

        # slope_range edit elements
        slope_range = gui.hBox(slope_controls)
        gui.label(slope_range, self,"Slope Range")
        gui.spin(slope_range, self, "slope_min", -90., 90., step=1., label="Min",
                 callback=self._updateSlopeRange, spinType=float)
        gui.spin(slope_range, self, "slope_max", -90., 90., step=1., label="Max",
                 callback=self._updateSlopeRange, spinType=float)

        # shift control in radians
        self.shift = 0.
        gui.spin(box, self, "shift", -float_info.max, float_info.max, step=0.0001, label="Shift Spin",
                 callback=self._update_shift, spinType=float)

        # setting plot control variables

        # data plot variables
        self.plot_in = CurvePlot(self)
        self.plot_out = CurvePlot(self)
        # same padding used in other orange-spectroscopy widgets with CurvePlots
        self.plot_in.plot.vb.x_padding = 0.1005   
        self.plot_out.plot.vb.y_padding = 0.1005 
        # setting labels
        self.plot_in.label_xaxis = "Wavenumber"
        self.plot_out.label_xaxis = "Wavenumber"
        self.plot_in.label_yaxis = "Phase"
        self.plot_out.label_yaxis = "Phase"
        self.plot_in.label_title = "in_data"
        self.plot_out.label_title = "out_data"
        self.plot_in.labels_changed()
        self.plot_out.labels_changed()

        # auxiliary controllable line plot in the same view of plot_in
        self.linepos = pg.Point(0,0) 
        red = (255,0,0)#(128,128,128)
        pen = pg.mkPen(color=red, width=2, style=Qt.DashLine)
        self.diagonal_line = pg.InfiniteLine(pos=self.linepos, angle=float(self.slope), pen=pen, hoverPen=None, movable=False, span=(0, 2))
        self.plot_in.add_marking(self.diagonal_line)

        # setting plot views
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        splitter.addWidget(self.plot_in)
        splitter.addWidget(self.plot_out)
        self.mainArea.layout().addWidget(splitter)

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")
        # self._update_slope()

    @Inputs.data
    def set_data(self, data):
        print(">>>> on set_data (Input)")
        self.data = data
        self.plot_in.set_data(data)
        self._init_slope_params()
        self._update_slope()

    def _init_slope_params(self):
        print(">>>> setting slope limits according to data")
        if self.data is not None:
            params = calc_slope_extrema(self.data)
            print(params)
            self.slope_min, self.slope_max = params['slope']
            slope = np.sum(self.slope_min + self.slope_max )/2
            self.slope = slope
            self.slope_spin_val = slope
            x0, y0 = params['refpos'].values()
            self.linepos = pg.Point(x0, y0)
            self.shift = -y0
            self.slope_step = .01 * (self.slope_max - self.slope_min)
            self._updateSlopeRange()
            # print("setting... ", y0, x0, self.slope, self.slope_step, self.slope_max, self.slope_min)

    def _updateSlopeRange(self):
        # TODO: validate min < max
        # validate slope in range
        if self.slope < self.slope_min:
            self.slope = self.slope_min
        elif self.slope > self.slope_max:
            self.slope = self.slope_max
        print('Slope after update range {}'.format(self.slope))
        self.slope_spin_val = self.slope
        self.slope_slider.setValue(self.slope)
        self.slope_slider.setScale(minValue=self.slope_min, maxValue=self.slope_max, step=0.01*(self.slope_max-self.slope_min)/2)
        self._update_slope()

    def _buttonSlope1Up(self):
        print("button increment +1")
        self._incrementSlope(1)

    def _buttonSlope10Up(self):
        print("button increment +1")
        self._incrementSlope(10)
    
    def _buttonSlope1Down(self):
        print("button increment -1")
        self._incrementSlope(-1)

    def _buttonSlope10Down(self):
        print("button increment -10")
        self._incrementSlope(-10)

    def _incrementSlope(self, ammount):
        self.slope += ammount*self.slope_step
        self._update_slope()
    
    def _update_lines(self):
        print(">>>> updating lines", self.slope)
        self.plot_in.clear_markings() 
        self.plot_out.clear_markings()
        self.plot_in.add_marking(self.diagonal_line)
        self.diagonal_line.setAngle(float(self.slope))
        self.diagonal_line.setPos(self.linepos)
        print(">>>> updating lines end", self.slope)
    
    def _update_slope_spin(self):
        print(">>>> update_slope_spin: ", self.slope_spin_val)
        self.slope = self.slope_spin_val
        self.slope_slider.setValue(self.slope_spin_val)
        if self.data is not None:
            self._update_lines()
            self.commit.now()

    def _update_slope(self):
        print(">>>> update_slope", self.slope)
        self.slope_spin_val = self.slope
        # self._update_slope_spin()
        if self.data is not None:
            self._update_lines()
            self.commit.deferred()
        else:
            self.slope = 0.

    def _update_shift(self):
        print(">>>> update_slope")
        if self.data is not None:
            xax_min = getx(self.data)[0]
            self.linepos = pg.Point(xax_min,-self.shift) 
            self._update_lines()
            self.commit.deferred()
            print(self.shift)
        
    @gui.deferred
    def commit(self):
        print(">>>> on commit")
        if self.data is not None:
            # calculate out_data
            out_data = DegTilt(slope=float(self.slope), shift=float(self.shift))(self.data)
            self.on_done(out_data)

    def on_done(self, out_data):
        print(">>>> on_done", self.slope, self.slope_spin_val)
        self.plot_out.set_data(out_data) # set data to plot_out
        self.Outputs.data_edited.send(out_data) # send data to Output
        print(">>>> on_done end", self.slope, self.slope_spin_val)

    def handleNewSignals(self):
        print(">>>> on handleNewSignals")
        self.commit.now()

if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWManualEditor).run(Orange.data.Table("iris.tab"))