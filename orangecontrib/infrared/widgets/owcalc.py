import sys

import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Default
from Orange.widgets import gui, settings

from orangecontrib.infrared.preprocess import Absorbance, Transmittance, Subtract

import orangecontrib.infrared #TODO only used to get access to datasets directory in standalone

class OWCalc(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Spectral Calculator"

    # Short widget description
    description = (
        "Perform simple calculations on spectra")

    # An icon resource file path for this widget
    # (a path relative to the module where this widget is defined)
    icon = "icons/fft.svg" #TODO

    # Define inputs and outputs
    inputs = [("Spectra", Orange.data.Table, "set_data", Default),
              ("Reference", Orange.data.Table, "set_data_ref")]
    outputs = [("Spectra", Orange.data.Table)]

    # Define widget settings
    autocommit = settings.Setting(False)
    calc = settings.Setting(0)

    calculators = [("AB (-log(DataSC/RefSC))", Absorbance),
                   ("TR (DataSC/RefSC)", Transmittance),
                   ("Subtract (Data - Ref)", Subtract)]

    # GUI definition:
    #   a simple 'single column' GUI layout
    want_main_area = False
    #   with a fixed non resizable geometry.
    resizing_enabled = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.data_ref = None

        # An info box
        infoBox = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(infoBox, "No spectra on input.")
        self.infob = gui.widgetLabel(infoBox, "No reference data on input.")

        # Select calculation
        self.calcBox = gui.widgetBox(self.controlArea, "Calculate")
        gui.radioButtons(self.calcBox, self, "calc",
                         btnLabels=(x[0] for x in self.calculators),
                         callback=self.setting_changed)

        gui.auto_commit(self.controlArea, self, "autocommit", "Calculate", box=False)

        # Disable the controls initially (no data)
        self.calcBox.setDisabled(True)

    def set_data(self, dataset):
        if dataset is not None:
            self.data = dataset
            self.infoa.setText("{0} spectra, {1} data points".format(
                                    dataset.X.shape[0],
                                    dataset.X.shape[1]))
            self.calcBox.setDisabled(False)
            self.commit()
        else:
            self.data = None
            self.spectra_out = None
            self.calcBox.setDisabled(True)
            self.infoa.setText("No spectra on input.")
            self.send("Spectra", self.spectra_out)

    def set_data_ref(self, dataset):
        if dataset is not None and dataset.X.shape[0] == 1:
            try:
                self.data_ref = Orange.data.Table.from_table(self.data.domain, dataset)
            except AttributeError:
                self.data_ref = None
                self.infob.setText("No reference data on input.")
                self.commit()
            else:
                self.infob.setText("{0} reference data points used (of {1})".format(
                                    self.data_ref.X.shape[1],
                                    dataset.X.shape[1]))
                self.commit()
        elif dataset is not None:
            self.data_ref = None
            self.infob.setText("Only one reference spectra allowed")
        else:
            self.data_ref = None
            self.infob.setText("No reference data on input.")
            self.commit()

    def commit(self):
        if self.data is not None:
            self.spectra_out = self.calculators[self.calc][1](ref=self.data_ref)(self.data)
            self.send("Spectra", self.spectra_out)

    def setting_changed(self):
        """Required by auto_commit button"""
        self.commit()

# Simple main stub function in case being run outside Orange Canvas
def main(argv=sys.argv):
    from PyQt4.QtGui import QApplication
    app = QApplication(list(argv))
    args = app.argv()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "peach_juice.0"

    ow = OWCalc()
    ow.show()
    ow.raise_()

    dataset = Orange.data.Table(filename)
    ow.set_data(dataset)
    ow.set_data_ref(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    return 0

if __name__=="__main__":
    sys.exit(main())
