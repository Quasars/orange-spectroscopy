import numpy as np
import pyqtgraph as pg
from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets import widget
import sys
from Orange.widgets import gui
from PyQt4 import QtGui
from PyQt4.QtGui import QDoubleValidator
from .owcurves import CurvePlot, SelectRegion
import gc


class OWCut(widget.OWWidget):
    name = "Cut"
    inputs = [("Data", Orange.data.Table, 'set_data', Default)]
    outputs = [("Cut spectra", Orange.data.Table)]
    priority = 300
    icon = "icons/unknown.svg"


    def __init__(self):
        super().__init__()
        self.box = gui.hBox(self.mainArea)
        self.selected_indices = set() #parents of CurvePlot need this property
        self.controlArea.hide()
        self.limit1 = 0
        self.limit2 = 0
        self.data = None
        self.plotview = CurvePlot(parent=self)
        self.inversecut = False
        self.mainArea.layout().addWidget(self.plotview)
        gui.lineEdit(self.box, self, "limit1", valueType=float,
            validator=QDoubleValidator(), callback=self.submit)
        gui.lineEdit(self.box, self, "limit2", valueType=float,
            validator=QDoubleValidator(), callback=self.submit)
        gui.checkBox(self.box, self, "inversecut", "Reverse selection",
            callback=self.submit)
        self.resize(900, 700)
        self.region = SelectRegion()
        self.region.sigRegionChanged.connect(self.regionChanged)
        self.plotview.add_marking(self.region)
        self.blockchanges = False

    def selection_changed(self):
        pass

    def regionChanged(self):
        if not self.blockchanges:
            minX, maxX = self.region.getRegion()
            self.limit1 = minX
            self.limit2 = maxX
            self.submit()

    def set_data(self, data):
        self.data = data
        self.plotview.set_data(data)

    def submit(self):
        values =  [ float(self.limit1), float(self.limit2) ]
        minX, maxX = min(values), max(values)
        self.limit1 = minX
        self.limit2 = maxX
        self.blockchanges = True
        self.region.setRegion([minX, maxX])
        self.blockchanges = False
        if self.data:
            data = self.data
            try:
                x = np.array([ float(a.name) for a in data.domain.attributes ])
            except:
                x = np.arange(len(data.domain.attributes))
            if not self.inversecut:
                okattrs = [ at for at, v in zip(data.domain.attributes, x) if minX <= v <= maxX ]
            else:
                okattrs = [ at for at, v in zip(data.domain.attributes, x) if v <= minX or v >= maxX ]
            domain = Orange.data.Domain(okattrs, data.domain.class_vars, metas=data.domain.metas)
            self.send("Cut spectra", Orange.data.Table(domain, data))


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    w = OWCut()
    w.show()
    data = Orange.data.Table("2012.11.09-11.45_Peach juice colorful spot.dpt")
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval

if __name__ == "__main__":
    sys.exit(main())

