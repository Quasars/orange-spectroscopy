import numpy as np

from AnyQt.QtWidgets import QGridLayout

import Orange.data
from Orange.data import ContinuousVariable, Domain, Table
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings

from orangecontrib.spectroscopy.data import build_spec_table
from orangecontrib.spectroscopy import irfft
from orangecontrib.spectroscopy.irfft import PhaseCorrection


def add_meta_to_table(data, var, values):
    """
    Take an existing Table and add a meta variable
    """
    metas = data.domain.metas + (var,)
    newdomain = Domain(data.domain.attributes, data.domain.class_vars, metas)
    newtable = data.transform(newdomain)
    with newtable.unlocked(newtable.metas):
        newtable[:, var] = np.atleast_1d(values).reshape(-1, 1)
    return newtable

DEFAULT_HENE = 15797.337544
CHUNK_SIZE = 100

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
    keywords = ["Fast Fourier Transform", "FFT", "IFG"]

    # Define inputs and outputs
    class Inputs:
        data = Input("Interferogram", Orange.data.Table, default=True)
        stored_phase = Input("Stored Phase", Orange.data.Table)

    class Outputs:
        spectra = Output("Spectra", Orange.data.Table, default=True)
        phases = Output("Phases", Orange.data.Table)

    replaces = ["orangecontrib.infrared.widgets.owfft.OWFFT"]

    settings_version = 2
    # Define widget settings
    dx_auto = settings.Setting(True)
    dx = settings.Setting(1.0 / DEFAULT_HENE / 2.0)
    auto_sweeps = settings.Setting(True)
    sweeps = settings.Setting(0)
    peak_search = settings.Setting(irfft.PeakSearch.MAXIMUM)
    peak_search_enable = settings.Setting(True)
    zpd1 = settings.Setting(0)
    zpd2 = settings.Setting(0)
    apod_func = settings.Setting(1)
    zff = settings.Setting(1)  # an exponent for zero-filling factor, IRFFT() needs 2**zff
    phase_corr = settings.Setting(0)
    phase_res_limit = settings.Setting(True)
    phase_resolution = settings.Setting(32)
    limit_output = settings.Setting(True)
    out_limit1 = settings.Setting(400)
    out_limit2 = settings.Setting(4000)
    autocommit = settings.Setting(False)
    complexfft = settings.Setting(False)

    sweep_opts = ("Single",
                  "Forward-Backward",
                  "Forward",
                  "Backward",
                 )

    apod_opts = ("Boxcar (None)",               # <irfft.ApodFunc.BOXCAR: 0>
                 "Blackman-Harris (3-term)",    # <irfft.ApodFunc.BLACKMAN_HARRIS_3: 1>
                 "Blackman-Harris (4-term)",    # <irfft.ApodFunc.BLACKMAN_HARRIS_4: 2>
                 "Blackman Nuttall (EP)")       # <irfft.ApodFunc.BLACKMAN_NUTTALL: 3>

    phase_opts = ("Mertz",              # <irfft.PhaseCorrection.MERTZ: 0>
                  "Mertz Signed",       # <irfft.PhaseCorrection.MERTZSIGNED: 1>
                  "Stored Phase",       # <irfft.PhaseCorrection.STORED: 2>
                  "None (real/imag)",   # <irfft.PhaseCorrection.NONE: 3>
                 )

    # GUI definition:
    #   a simple 'single column' GUI layout
    want_main_area = False
    #   with a fixed non resizable geometry.
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        # This is not actuully called anywhere at the moment
        phase_res_limit_low = Msg("Phase resolution limit too low")
        complex_data_phase_zero = Msg("Phase data is not connected, thus it is considered zero.")

    class Error(OWWidget.Error):
        fft_error = Msg("FFT error:\n{}")
        ifg_split_error = Msg("IFG Forward-Backward split error:\n{}")

    def __init__(self):
        super().__init__()

        self.data = None
        self.stored_phase = None
        self.spectra_table = None
        self.phases_table = None
        self.reader = None
        self.use_interleaved_data = False

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        gui.widgetBox(self.controlArea, orientation=layout)

        # GUI
        # An info box
        infoBox = gui.widgetBox(None, "Info")
        layout.addWidget(infoBox, 0, 0, 2, 1)
        self.info_type = gui.widgetLabel(infoBox, "No data on input.")
        self.info_pts = gui.widgetLabel(infoBox, "")
        self.info_dx = gui.widgetLabel(infoBox, "")
        gui.rubber(infoBox)

        # Input Data control area
        self.dataBox = gui.widgetBox(None, "Input Data")
        layout.addWidget(self.dataBox, 2, 0, 3, 1)

        gui.widgetLabel(self.dataBox, "Datapoint spacing (Δx):")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.dx_auto_cb = gui.checkBox(
            self.dataBox, self, "dx_auto",
            label="Auto",
            callback=self.dx_auto_changed,
            )
        self.dx_edit = gui.lineEdit(
            self.dataBox, self, "dx",
            callback=self.setting_changed,
            valueType=float,
            disabled=self.dx_auto,
            tooltip="Pathlength Step Size",
        )
        lb = gui.widgetLabel(self.dataBox, "cm")
        grid.addWidget(self.dx_auto_cb, 0, 0)
        grid.addWidget(self.dx_edit, 0, 1, 1, 2)
        grid.addWidget(lb, 0, 3)

        wl = gui.widgetLabel(self.dataBox, "Sweep Direction:")
        box = gui.comboBox(
            self.dataBox, self, "sweeps",
            label=None,
            items=self.sweep_opts,
            callback=self.sweeps_changed,
            disabled=self.auto_sweeps
            )
        cb2 = gui.checkBox(
            self.dataBox, self, "auto_sweeps",
            label="Auto",
            callback=self.sweeps_changed,
            )
        grid.addWidget(wl, 1, 0, 1, 3)
        grid.addWidget(cb2, 2, 0)
        grid.addWidget(box, 2, 1, 1, 2)

        wl = gui.widgetLabel(self.dataBox, "ZPD Peak Search:")
        box = gui.comboBox(
            self.dataBox, self, "peak_search",
            label=None,
            items=[name.title() for name, _ in irfft.PeakSearch.__members__.items()],
            callback=self.peak_search_changed,
            enabled=self.peak_search_enable,
            )
        le1 = gui.lineEdit(
            self.dataBox, self, "zpd1",
            callback=self.peak_search_changed,
            valueType=int,
            controlWidth=50,
            disabled=self.peak_search_enable,
            )
        le2 = gui.lineEdit(
            self.dataBox, self, "zpd2",
            callback=self.peak_search_changed,
            valueType=int,
            controlWidth=50,
            disabled=self.peak_search_enable,
            )
        cb = gui.checkBox(
            self.dataBox, self, "peak_search_enable",
            label=None,
            callback=self.peak_search_changed,
        )
        grid.addWidget(wl, 3, 0, 1, 3)
        grid.addWidget(cb, 4, 0)
        grid.addWidget(box, 4, 1, 1, 2)
        grid.addWidget(gui.widgetLabel(self.dataBox, "    Manual ZPD:"), 5, 0)
        grid.addWidget(le1, 5, 1)
        grid.addWidget(le2, 5, 2)

        self.dataBox.layout().addLayout(grid)

        # FFT Options control area
        self.optionsBox = gui.widgetBox(None, "FFT Options")
        layout.addWidget(self.optionsBox, 0, 1, 3, 1)

        self.complexfft_cb = gui.checkBox(
            self.optionsBox, self, "complexfft",
            label="Complex FFT",
            callback=self.complex_fft_changed
            )

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
        self.outputBox = gui.widgetBox(None, "Output")
        layout.addWidget(self.outputBox, 3, 1, 2, 1)

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
            disables=[le2, le3]
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

    @Inputs.data
    def set_data(self, dataset):
        """
        Receive input data.
        """
        self.Warning.clear()
        self.Error.clear()

        if dataset and dataset.X.size:
            self.data = dataset
            self.determine_sweeps()
            self.info_type.setText('%d %s interferogram(s)' %
                                   (dataset.X.shape[0],
                                (["Single"] + 3*["Forward-Backward"])[self.sweeps]))
            self.info_pts.setText('%d points each' % dataset.X.shape[1])
            self.check_metadata()
        else:
            self.data = None
            self.spectra_table = None
            self.phases_table = None
            self.info_type.setText("No data on input.")
            self.info_pts.setText("")
            self.info_dx.setText("")
            self.enable_controls()
            self.Outputs.spectra.send(self.spectra_table)
            self.Outputs.phases.send(self.phases_table)

    @Inputs.stored_phase
    def set_stored_phase(self, dataset):
        """
        Receive stored phase data.
        """
        if dataset is not None:
            self.stored_phase = dataset
        else:
            self.stored_phase = None

    def handleNewSignals(self):
        self.commit.now()

    def setting_changed(self):
        self.commit.deferred()

    def out_limit_changed(self):
        values = [float(self.out_limit1), float(self.out_limit2)]
        minX, maxX = min(values), max(values)
        self.out_limit1 = minX
        self.out_limit2 = maxX
        self.commit.deferred()

    def sweeps_changed(self):
        self.controls.sweeps.setDisabled(self.auto_sweeps)
        self.determine_sweeps()
        if not self.peak_search_enable:
            self.controls.zpd2.setDisabled(self.sweeps == 0)
        self.commit.deferred()

    def dx_auto_changed(self):
        self.dx_edit.setDisabled(self.dx_auto)
        if self.dx_auto is True and self.data is not None:
            self.check_metadata()
        self.commit.deferred()

    def peak_search_changed(self):
        self.controls.peak_search.setEnabled(self.peak_search_enable)
        self.controls.zpd1.setDisabled(self.peak_search_enable)
        self.controls.zpd2.setDisabled(self.peak_search_enable or self.sweeps == 0)
        self.commit.deferred()

    def complex_fft_changed(self):
        self.configui_for_complex_fft()
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        if self.data is not None:
            if self.complexfft:
                self.calculate_complex_FFT()
            else:
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

        wavenumbers = None
        spectra = []
        phases = []

        zpd_fwd = []
        zpd_back = []

        # Reset info, error and warning dialogs
        self.Error.clear()
        self.Warning.clear()

        fft_single = irfft.IRFFT(dx=self.dx,
                                 apod_func=self.apod_func,
                                 zff=2**self.zff,
                                 phase_res=self.phase_resolution if self.phase_res_limit else None,
                                 phase_corr=self.phase_corr,
                                 peak_search=self.peak_search,
                                )

        ifg_data = self.data.X
        stored_phase = self.stored_phase
        stored_zpd_fwd, stored_zpd_back = None, None
        # Only use first row stored phase for now
        if stored_phase is not None:
            stored_phase = stored_phase[0]
            try:
                stored_zpd_fwd = int(stored_phase["zpd_fwd"].value)
            except ValueError:
                stored_zpd_fwd = None
            try:
                stored_zpd_back = int(stored_phase["zpd_back"].value)
            except ValueError:
                stored_zpd_back = None
            stored_phase = stored_phase.x # lowercase x for RowInstance
        # Use manual zpd value(s) if specified and enable batch processing
        elif not self.peak_search_enable:
            stored_zpd_fwd = self.zpd1
            stored_zpd_back = self.zpd2
            chunks = max(1, len(self.data) // CHUNK_SIZE)
            ifg_data = np.array_split(self.data.X, chunks, axis=0)
            fft_single = irfft.MultiIRFFT(
                dx=self.dx,
                apod_func=self.apod_func,
                zff=2**self.zff,
                phase_res=self.phase_resolution if self.phase_res_limit else None,
                phase_corr=self.phase_corr,
                peak_search=self.peak_search,
            )

        for row in ifg_data:
            if self.sweeps in [2, 3]:
                # split double-sweep for forward/backward
                # forward: 2-2 = 0 , backward: 3-2 = 1
                try:
                    row = np.hsplit(row, 2)[self.sweeps - 2]
                except ValueError as e:
                    self.Error.ifg_split_error(e)
                    return

            if self.sweeps in [0, 2, 3]:
                try:
                    spectrum_out, phase_out, wavenumbers = fft_single(
                        row, zpd=stored_zpd_fwd, phase=stored_phase)
                    zpd_fwd.append(fft_single.zpd)
                except ValueError as e:
                    self.Error.fft_error(e)
                    return
            elif self.sweeps == 1:
                # Double sweep interferogram is split, solved independently and the
                # two results are averaged.
                try:
                    data = np.hsplit(row, 2)
                except ValueError as e:
                    self.Error.ifg_split_error(e)
                    return

                fwd = data[0]
                # Reverse backward sweep to match fwd sweep
                back = data[1][::-1]

                # Calculate spectrum for both forward and backward sweeps
                try:
                    spectrum_fwd, phase_fwd, wavenumbers = fft_single(
                        fwd, zpd=stored_zpd_fwd, phase=stored_phase)
                    zpd_fwd.append(fft_single.zpd)
                    spectrum_back, phase_back, wavenumbers = fft_single(
                        back, zpd=stored_zpd_back, phase=stored_phase)
                    zpd_back.append(fft_single.zpd)
                except ValueError as e:
                    self.Error.fft_error(e)
                    return

                # Calculate the average of the forward and backward sweeps
                spectrum_out = np.mean(np.array([spectrum_fwd, spectrum_back]), axis=0)
                phase_out = np.mean(np.array([phase_fwd, phase_back]), axis=0)
            else:
                return

            spectra.append(spectrum_out)
            phases.append(phase_out)

        spectra = np.vstack(spectra)
        phases = np.vstack(phases)

        if None not in phases:
            self.phases_table = build_spec_table(wavenumbers, phases,
                                                additional_table=self.data)
            if not self.peak_search_enable:
                # All zpd values are equal by definition
                zpd_fwd = zpd_fwd[:1]
            self.phases_table = add_meta_to_table(self.phases_table,
                                                ContinuousVariable.make("zpd_fwd"),
                                                zpd_fwd)
            if zpd_back:
                if not self.peak_search_enable:
                    zpd_back = zpd_back[:1]
                self.phases_table = add_meta_to_table(self.phases_table,
                                                    ContinuousVariable.make("zpd_back"),
                                                    zpd_back)
        else:
            self.phases_table = None

        if self.limit_output is True:
            wavenumbers, spectra = self.limit_range(wavenumbers, spectra)

        self.spectra_table = build_spec_table(wavenumbers, spectra,
                                            additional_table=self.data)
        self.Outputs.spectra.send(self.spectra_table)
        self.Outputs.phases.send(self.phases_table)


    def calculate_complex_FFT(self):

        # Reset info, error and warning dialogs
        self.Error.clear()
        self.Warning.clear()

        if self.use_interleaved_data:
            amplitude_in = self.data.X[::2]
            phases_in = self.data.X[1::2]
        else:
            amplitude_in = self.data.X
            phases_in = self.stored_phase.X if self.stored_phase is not None else None

        if phases_in is None:
            phases_in = 0
            self.Warning.complex_data_phase_zero()

        ifg_data = amplitude_in * np.exp(phases_in * 1j)

        wavenumbers = None
        spectra = []
        phases = []

        fft_single = irfft.ComplexFFT(
            dx=self.dx,
            apod_func=self.apod_func,
            zff=2**self.zff,
            phase_res=self.phase_resolution if self.phase_res_limit else None,
            phase_corr=self.phase_corr,
            peak_search=self.peak_search,
        )
        for row in ifg_data:
            spectrum_out, phase_out, wavenumbers = fft_single(row, zpd=None)
            spectra.append(spectrum_out)
            phases.append(phase_out)

        spectra = np.vstack(spectra)
        phases = np.vstack(phases)

        if self.limit_output is True:
            wavenumbers, spectra = self.limit_range(wavenumbers, spectra)
            _, phases = self.limit_range(wavenumbers, phases)

        wavenumbers_domain = Domain(
            [ContinuousVariable.make(f"{w}") for w in wavenumbers],
            metas=self.data.domain.metas,
        )

        if self.use_interleaved_data:
            self.spectra_table = Table.from_numpy(
                wavenumbers_domain, X=spectra, metas=self.data.metas[::2]
            )
            phases_table = Table.from_numpy(
                wavenumbers_domain, X=phases, metas=self.data.metas[1::2]
            )
        else:
            self.spectra_table = Table.from_numpy(
            wavenumbers_domain, X=spectra, metas=self.data.metas
            )
            phases_table = Table.from_numpy(
                wavenumbers_domain, X=phases, metas=self.data.metas
            )

        self.Outputs.spectra.send(self.spectra_table)
        self.Outputs.phases.send(phases_table)

    def determine_sweeps(self):
        """
        Determine if input interferogram is single-sweep or
        double-sweep (Forward-Backward).

        Combines with auto_sweeps and custom sweeps setting.
        """
        # Just testing 1st row for now
        # assuming all in a group were collected the same way
        data = self.data.X[0]
        zpd = irfft.find_zpd(data, self.peak_search)
        middle = data.shape[0] // 2
        # Allow variation of +/- 0.4 % in zpd location
        var = middle // 250
        if zpd >= middle - var and zpd <= middle + var:
            # single, symmetric
            sweeps = 0
        else:
            try:
                data = np.hsplit(data, 2)
            except ValueError:
                # odd number of data points, probably single
                sweeps = 0
            else:
                zpd1 = irfft.find_zpd(data[0], self.peak_search)
                zpd2 = irfft.find_zpd(data[1][::-1], self.peak_search)
                # Forward / backward zpds never perfectly match
                if zpd1 >= zpd2 - var and zpd1 <= zpd2 + var:
                    # forward-backward, symmetric and asymmetric
                    sweeps = 1
                else:
                    # single, asymetric
                    sweeps = 0
        # Honour custom sweeps setting
        if self.auto_sweeps:
            self.sweeps = sweeps
        elif sweeps == 0:
            # Coerce setting to match input data (single)
            self.sweeps = sweeps
        elif sweeps == 1 and self.sweeps == 0:
            # Coerce setting to match input data (single -> forward)
            self.sweeps = 2

    def check_metadata(self):
        """Look for laser wavenumber and sampling interval metadata"""
        self.enable_controls()
        try:
            channel_data, detail = self.data.attributes["Channel Data Type"]
            if channel_data == "Polar":
                self.use_interleaved_data = True
                self.complexfft = True
                self.controls.complexfft.setDisabled(True)
                self.configui_for_complex_fft()
                self.info_type.setText(
                    f"{self.data.X.shape[0]} interleaved {channel_data} interferogram(s)\n"
                    f"({detail})"
                )
        except KeyError:
            self.use_interleaved_data = False
        try:
            domain_units, dx = self.data.attributes["Calculated Datapoint Spacing (Δx)"]
            if domain_units != "[cm]" or not dx:
                raise KeyError

            self.dx = dx
            self.zff = 2
            self.dx_auto = True
            self.dx_edit.setDisabled(self.dx_auto)
            self.controls.auto_sweeps.setDisabled(True)
            self.controls.sweeps.setDisabled(True)

            self.info_dx.setText("Using Calculated Datapoint Spacing (Δx) from metadata.")
            return
        except KeyError:
            pass

        try:
            udr = self.data.get_column("Under Sampling Ratio")
        except ValueError:
            udr = 1
        else:
            udr = udr[0] if (udr == udr[0]).all() else ValueError()
        try:
            lwn = self.data.get_column("Effective Laser Wavenumber")
        except ValueError:
            lwn = DEFAULT_HENE
            self.info_dx.setText("Dataspacing not found: auto HeNe\n")
        else:
            lwn = lwn[0] if (lwn == lwn[0]).all() else ValueError()
            self.info_dx.setText("")

        self.info_dx.setText(self.info_dx.text() + f"{lwn} cm⁻¹ laser \n{udr} sampling interval")
        if self.dx_auto:
            self.dx = (1 / lwn / 2 ) * udr

    def limit_range(self, wavenumbers, spectra):

        limits = np.searchsorted(wavenumbers,
                                [self.out_limit1, self.out_limit2])
        wavenumbers = wavenumbers[limits[0]:limits[1]]
        # Handle 1D array if necessary
        if spectra.ndim == 1:
            spectra = spectra[None, limits[0]:limits[1]]
        else:
            spectra = spectra[:, limits[0]:limits[1]]

        return wavenumbers, spectra

    def configui_for_complex_fft(self):
        """
        Disable Phase correction controls when calculating a complex FFT.
        """
        self.controls.phase_corr.setDisabled(self.complexfft)
        self.controls.phase_res_limit.setDisabled(self.complexfft)
        self.controls.phase_resolution.setDisabled(self.complexfft)

    def enable_controls(self):
        """
        Re-enable all controls that may be disabled by check_metadata().
        """
        self.controls.complexfft.setDisabled(False)
        self.controls.auto_sweeps.setDisabled(False)
        self.controls.sweeps.setDisabled(False)

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            if "dx_HeNe" in settings_:
                settings_["dx_auto"] = settings_["dx_HeNe"]


def load_test_gsf() -> Orange.data.Table:
    """Load a NeaSpec interferogram GSF file and return a Table object."""
    from orangecontrib.spectroscopy.io.neaspec import NeaReaderGSF
    from Orange.data.io import FileFormat
    from Orange.data import dataset_dirs

    fn = 'NeaReaderGSF_test/NeaReaderGSF_test O2A raw.gsf'
    absolute_filename = FileFormat.locate(fn, dataset_dirs)
    return NeaReaderGSF(absolute_filename).read()

if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=ungrouped-imports
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWFFT).run(load_test_gsf())
