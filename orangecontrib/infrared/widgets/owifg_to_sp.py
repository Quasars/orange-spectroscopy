import sys
import numpy as np

from AnyQt.QtWidgets import QGridLayout, QApplication

import Orange.data
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui, settings

from orangecontrib.infrared.data import build_spec_table
from orangecontrib.infrared import irfft

class OWFFT(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Interferogram to Spectrum"

    # Short widget description
    description = (
        "Performs Fast Fourier Transform on an interferogram, including "
        "zero filling, apodization and phase correction.")

    # An icon resource file path for this widget
    # (a path relative to the module where this widget is defined)
    icon = "icons/fft.svg"

    # Define inputs and outputs
    inputs = [("Interferogram", Orange.data.Table, "set_data")]
    outputs = [("Spectra", Orange.data.Table),
               ("Phases", Orange.data.Table)]

    # Define widget settings
    laser_wavenumber = settings.Setting(15797.337544)
    dx_HeNe = settings.Setting(True)
    dx = settings.Setting(1.0)
    apod_func = settings.Setting(1)
    zff = settings.Setting(1)
    phase_corr = settings.Setting(0)
    phase_res_limit = settings.Setting(True)
    phase_resolution = settings.Setting(32)
    limit_output = settings.Setting(True)
    out_limit1 = settings.Setting(400)
    out_limit2 = settings.Setting(4000)
    autocommit = settings.Setting(False)

    apod_opts = ("Boxcar (None)",
                 "Blackman-Harris (3-term)",
                 "Blackman-Harris (4-term)",
                 "Blackman Nuttall (EP)")

    phase_opts = ("Mertz",)

    # GUI definition:
    #   a simple 'single column' GUI layout
    want_main_area = False
    #   with a fixed non resizable geometry.
    resizing_enabled = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.spectra = None
        self.spectra_table = None
        self.wavenumbers = None
        self.sweeps = None
        if self.dx_HeNe is True:
            self.dx = 1.0 / self.laser_wavenumber / 2.0

        # GUI
        # An info box
        infoBox = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(infoBox, "No data on input.")
        self.infob = gui.widgetLabel(infoBox, "")

        # Input Data control area
        self.dataBox = gui.widgetBox(self.controlArea, "Input Data")

        gui.widgetLabel(self.dataBox, "Datapoint spacing (Δx):")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.dx_edit = gui.lineEdit(
                    self.dataBox, self, "dx",
                    callback=self.setting_changed,
                    valueType=float,
                    controlWidth=100, disabled=self.dx_HeNe
                    )
        cb = gui.checkBox(
                    self.dataBox, self, "dx_HeNe",
                    label="HeNe laser",
                    callback=self.dx_changed,
                    )
        lb = gui.widgetLabel(self.dataBox, "cm")
        grid.addWidget(cb, 0, 0)
        grid.addWidget(self.dx_edit, 0, 1)
        grid.addWidget(lb, 0, 2)
        self.dataBox.layout().addLayout(grid)

        # FFT Options control area
        self.optionsBox = gui.widgetBox(self.controlArea, "FFT Options")

        box = gui.comboBox(
            self.optionsBox, self, "apod_func",
            label="Apodization function:",
            items=self.apod_opts,
            callback=self.setting_changed
            )

        box = gui.comboBox(
            self.optionsBox, self, "zff",
            label="Zero Filling Factor:",
            items=(2**n for n in range(10)),
            callback=self.setting_changed
            )

        box = gui.comboBox(
            self.optionsBox, self, "phase_corr",
            label="Phase Correction:",
            items=self.phase_opts,
            callback=self.setting_changed
            )

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)

        le1 = gui.lineEdit(
                    self.optionsBox, self, "phase_resolution",
                    callback=self.setting_changed,
                    valueType=int, controlWidth=30
                    )
        cb1 = gui.checkBox(
                    self.optionsBox, self, "phase_res_limit",
                    label="Limit phase resolution to ",
                    callback=self.setting_changed,
                    disables=le1
                    )
        lb1 = gui.widgetLabel(self.optionsBox, "cm<sup>-1<sup>")

        grid.addWidget(cb1, 0, 0)
        grid.addWidget(le1, 0, 1)
        grid.addWidget(lb1, 0, 2)

        self.optionsBox.layout().addLayout(grid)

        # Output Data control area
        self.outputBox = gui.widgetBox(self.controlArea, "Output")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        le2 = gui.lineEdit(
                    self.outputBox, self, "out_limit1",
                    callback=self.out_limit_changed,
                    valueType=float, controlWidth=50
                    )
        le3 = gui.lineEdit(
                    self.outputBox, self, "out_limit2",
                    callback=self.out_limit_changed,
                    valueType=float, controlWidth=50
                    )
        cb2 = gui.checkBox(
                    self.outputBox, self, "limit_output",
                    label="Limit spectral region:",
                    callback=self.setting_changed,
                    disables=[le2,le3]
                    )
        lb2 = gui.widgetLabel(self.outputBox, "-")
        lb3 = gui.widgetLabel(self.outputBox, "cm<sup>-1</sup>")
        grid.addWidget(cb2, 0, 0, 1, 6)
        grid.addWidget(le2, 1, 1)
        grid.addWidget(lb2, 1, 2)
        grid.addWidget(le3, 1, 3)
        grid.addWidget(lb3, 1, 4)
        self.outputBox.layout().addLayout(grid)

        gui.auto_commit(self.outputBox, self, "autocommit", "Calculate", box=False)

        # Disable the controls initially (no data)
        self.dataBox.setDisabled(True)
        self.optionsBox.setDisabled(True)

    def set_data(self, dataset):
        """
        Receive input data.
        """
        if dataset is not None:
            self.data = dataset
            self.determine_sweeps()
            self.infoa.setText('%d %s interferogram(s)' %
                               (dataset.X.shape[0],
                                ["Single", "Forward-Backward"][self.sweeps]))
            self.infob.setText('%d points each' % dataset.X.shape[1])
            self.dataBox.setDisabled(False)
            self.optionsBox.setDisabled(False)
            self.commit()
        else:
            self.data = None
            self.spectra_table = None
            self.dataBox.setDisabled(True)
            self.optionsBox.setDisabled(True)
            self.infoa.setText("No data on input.")
            self.infob.setText("")
            self.send("Spectra", self.spectra_table)

    def setting_changed(self):
        self.commit()

    def out_limit_changed(self):
        values = [ float(self.out_limit1), float(self.out_limit2) ]
        minX, maxX = min(values), max(values)
        self.out_limit1 = minX
        self.out_limit2 = maxX
        self.commit()

    def dx_changed(self):
        self.dx_edit.setDisabled(self.dx_HeNe)
        if self.dx_HeNe is True:
            self.dx = 1.0 / self.laser_wavenumber / 2.0
        self.commit()

    def commit(self):
        if self.data is not None:
            self.calculateFFT()

    def calculateFFT(self):
        """
        Calculate FFT from input interferogram(s).
        This is a handler method for
          - bad data / data shape
          - splitting the array in the case of two interferogram sweeps per dataset.
          - multiple input interferograms

        Based on mertz module by Eric Peach, 2014
        """

        self.wavenumbers = None
        self.spectra = None
        self.phases = None

        # Reset info, error and warning dialogs
        self.error(1)   # FFT ValueError, usually wrong sweep number
        self.error(2)   # vsplit ValueError, odd number of data points
        self.warning(4) # Phase resolution limit too low

        for row in self.data.X:
            # Check to see if interferogram is single or double sweep
            if self.sweeps == 0:
                try:
                    spectrum_out, phase_out, self.wavenumbers = self.fft_single(row)
                except ValueError as e:
                    self.error(1, "FFT error: %s" % e)
                    return

            elif self.sweeps == 1:
                # Double sweep interferogram is split, solved independently and the
                # two results are averaged.
                try:
                    data = np.hsplit(row, 2)
                except ValueError as e:
                    self.error(2, "%s" % e)
                    return

                fwd = data[0]
                # Reverse backward sweep to match fwd sweep
                back = data[1][::-1]

                # Calculate spectrum for both forward and backward sweeps
                try:
                    spectrum_fwd, phase_fwd, self.wavenumbers = self.fft_single(fwd)
                    spectrum_back, phase_back, self.wavenumbers = self.fft_single(back)
                except ValueError as e:
                    self.error(1, "FFT error: %s" % e)
                    return

                # Calculate the average of the forward and backward sweeps
                spectrum_out = np.mean( np.array([spectrum_fwd, spectrum_back]), axis=0)
                phase_out = np.mean(np.array([phase_fwd, phase_back]), axis=0)

            else:
                return

            if self.spectra is not None:
                self.spectra = np.vstack((self.spectra, spectrum_out))
                self.phases = np.vstack((self.phases, phase_out))
            else:
                self.spectra = spectrum_out
                self.phases = phase_out

        if self.limit_output is True:
            limits = np.searchsorted(self.wavenumbers,
                                     [self.out_limit1, self.out_limit2])
            self.wavenumbers = self.wavenumbers[limits[0]:limits[1]]
            # Handle 1D array if necessary
            if self.spectra.ndim == 1:
                self.spectra = self.spectra[None,limits[0]:limits[1]]
                self.phases = self.phases[None,limits[0]:limits[1]]
            else:
                self.spectra = self.spectra[:,limits[0]:limits[1]]
                self.phases = self.phases[:,limits[0]:limits[1]]

        self.spectra_table = build_spec_table(self.wavenumbers, self.spectra)
        self.phases_table = build_spec_table(self.wavenumbers, self.phases)
        self.send("Spectra", self.spectra_table)
        self.send("Phases", self.phases_table)


    def determine_sweeps(self):
        """
        Determine if input interferogram is single-sweep or
        double-sweep (Forward-Backward).
        """
        # Just testing 1st row for now
        # assuming all in a group were collected the same way
        data = self.data.X[0]
        zpd = irfft.peak_search(data)
        middle = data.shape[0] // 2
        # Allow variation of +/- 0.4 % in zpd location
        var = middle // 250
        if zpd >= middle - var and zpd <= middle + var:
            # single, symmetric
            self.sweeps = 0
        else:
            try:
                data = np.hsplit(data, 2)
            except ValueError:
                # odd number of data points, probably single
                self.sweeps = 0
                return
            zpd1 = irfft.peak_search(data[0])
            zpd2 = irfft.peak_search(data[1][::-1])
            # Forward / backward zpds never perfectly match
            if zpd1 >= zpd2 - var and zpd1 <= zpd2 + var:
                # forward-backward, symmetric and asymmetric
                self.sweeps = 1
            else:
                # single, asymetric
                self.sweeps = 0

    def fft_single(self, Ix):
        """
        Handle FFT options and call irfft.fft_single_sweep.

        Args:
            Ix (np.array): 1D array with a single-sweep interferogram

        Returns:
            spectrum: 1D array of frequency domain amplitude intensities
            phase: 1D array of frequency domain phase intensities
            wavenumbers: 1D array of corresponding wavenumber set
        """
        if self.phase_res_limit is True:
            phase_res = self.phase_resolution
        else:
            phase_res = None

        spectrum, phase, wavenumbers = irfft.fft_single_sweep(Ix, self.dx,
                                        phase_res, self.apod_func, self.zff)

        return spectrum, phase, wavenumbers

# Simple main stub function in case being run outside Orange Canvas
def main(argv=sys.argv):
    app = QApplication(list(argv))
    filename = "IFG_single.dpt"

    ow = OWFFT()
    ow.show()
    ow.raise_()

    dataset = Orange.data.Table(filename)
    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    return 0

if __name__=="__main__":
    sys.exit(main())
