import numpy as np
import pyqtgraph as pg

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout, QLabel

import Orange.data
from Orange.data.util import get_unique_names
from Orange.widgets import gui, settings
from Orange.widgets.settings import SettingProvider
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.gui import (
    MovableVline,
    connect_line,
    lineEditFloatRange,
)
from orangecontrib.spectroscopy.widgets.owspectra import CurvePlot, SELECTONE


RMS, PEAK_TO_PEAK = 0, 1
LINEAR, QUADRATIC = 1, 2

NOISE_COLOR = (225, 0, 0)
SIGNAL_COLOR = (0, 120, 225)

# standard noise regions (wavenumbers): name, (from, to)
NOISE_PRESETS = [
    ("FIR", (320, 280)),
    ("MIR", (2100, 1900)),
    ("NIR", (5500, 4500)),
]


def region_mask(x, a, b):
    """Boolean mask of x values within [a, b] (limits in any order)."""
    lo, hi = sorted((float(a), float(b)))
    return (x >= lo) & (x <= hi)


def _design_matrix(x, order):
    # center and scale x for a well-conditioned polynomial fit
    x = np.asarray(x, dtype=float)
    span = np.ptp(x) if len(x) else 0.0
    xs = (x - np.mean(x)) / (span if span else 1.0) if len(x) else x
    return np.vander(xs, order + 1)


def polynomial_fit(x, ys, order):
    """Least-squares polynomial fit of each row of ys over x.

    Rows with NaNs are fitted on their valid points only; rows with too few
    valid points (n <= order) return NaN fits.
    """
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    fits = np.full(ys.shape, np.nan)
    if ys.shape[1] == 0:
        return fits
    A = _design_matrix(x, order)
    finite = np.isfinite(ys)
    complete = np.all(finite, axis=1)
    if np.any(complete) and ys.shape[1] > order:
        coef, *_ = np.linalg.lstsq(A, ys[complete].T, rcond=None)
        fits[complete] = (A @ coef).T
    for i in np.flatnonzero(~complete):
        valid = finite[i]
        if np.count_nonzero(valid) > order:
            coef, *_ = np.linalg.lstsq(A[valid], ys[i, valid], rcond=None)
            fits[i] = A @ coef
    return fits


def noise_rms(x, ys, order):
    """RMS noise: sqrt(sum((y_i - y_i,fit)**2) / n) for each row of ys.

    Returns (rms, fits)."""
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    fits = polynomial_fit(x, ys, order)
    residuals = ys - fits
    n = np.count_nonzero(np.isfinite(residuals), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rms = np.sqrt(np.nansum(residuals**2, axis=1) / n)
    rms[n == 0] = np.nan
    return rms, fits


def _nan_reduce(func, ys):
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    out = np.full(len(ys), np.nan)
    valid = np.any(np.isfinite(ys), axis=1)
    if ys.shape[1] and np.any(valid):
        out[valid] = func(ys[valid], axis=1)
    return out


def noise_peak_to_peak(ys):
    """Peak-to-peak noise: max - min of each row of ys."""
    return _nan_reduce(np.nanmax, ys) - _nan_reduce(np.nanmin, ys)


def signal_max(ys):
    """Signal: maximum band value of each row of ys."""
    return _nan_reduce(np.nanmax, ys)


def compute_snr(x, X, noise_limits, signal_limits, method=RMS, degree=LINEAR):
    """Compute signal, noise and SNR for each spectrum (row of X).

    Returns (signal, noise, snr) arrays."""
    x = np.asarray(x, dtype=float)
    X = np.asarray(X, dtype=float)
    nmask = region_mask(x, *noise_limits)
    smask = region_mask(x, *signal_limits)
    if method == RMS:
        noise, _ = noise_rms(x[nmask], X[:, nmask], degree)
    else:
        noise = noise_peak_to_peak(X[:, nmask])
    signal = signal_max(X[:, smask])
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = signal / noise
    return signal, noise, snr


class OWSNRRegion(OWWidget):
    name = "SNR (Region)"
    description = (
        "Calculate the signal-to-noise ratio of each spectrum, with noise "
        "estimated as RMS or peak-to-peak within a selected region."
    )
    icon = "icons/snr.svg"
    keywords = ["signal", "noise", "snr", "rms", "peak to peak", "region"]

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        data = Output("Data", Orange.data.Table, default=True)

    METHODS = ["RMS", "Peak-to-peak"]
    FIT_ORDERS = ["Linear", "Quadratic"]  # polynomial degree is index + 1

    settingsHandler = settings.DomainContextHandler()
    curveplot = SettingProvider(CurvePlot)
    visual_settings = settings.Setting({}, schema_only=True)

    noise_method = settings.Setting(RMS)
    fit_order = settings.Setting(0)  # index into FIT_ORDERS
    noise_low = settings.Setting(None)
    noise_high = settings.Setting(None)
    signal_low = settings.Setting(None)
    signal_high = settings.Setting(None)
    lock_regions = settings.Setting(False)  # use the noise region for signal
    autocommit = settings.Setting(True)

    graph_name = "curveplot.plotview"

    class Warning(OWWidget.Warning):
        no_x = Msg("No continuous features in input data.")
        empty_noise = Msg("No data points in the noise region.")
        empty_signal = Msg("No data points in the signal region.")
        few_points = Msg("Too few points in the noise region for the selected fit.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.preview_markings = []

        box = gui.vBox(self.controlArea, "Noise")
        gui.radioButtons(
            box,
            self,
            "noise_method",
            self.METHODS,
            callback=self._method_changed,
        )
        self.fit_combo = gui.comboBox(
            gui.indentedBox(box),
            self,
            "fit_order",
            label="Fit:",
            orientation=Qt.Horizontal,
            items=self.FIT_ORDERS,
            callback=self.settings_changed,
        )

        self.curveplot = CurvePlot(self, select=SELECTONE)
        self.curveplot.select_at_least_1 = True
        self.curveplot.plot.vb.x_padding = 0.005  # so that lines are not hidden
        self.curveplot.selection_changed.connect(self.redraw_preview)
        self.mainArea.layout().addWidget(self.curveplot)

        self.lines = {}
        self.editors = {}
        form = QFormLayout()
        box.layout().addLayout(form)
        self.noise_preset = 0  # index into preset combo; 0 is custom
        self.preset_combo = gui.comboBox(
            None,
            self,
            "noise_preset",
            items=["Custom"] + [f"{n}: {a}-{b}" for n, (a, b) in NOISE_PRESETS],
            callback=self._preset_selected,
        )
        form.addRow("Preset:", self.preset_combo)
        self._region_editors(form, "noise", NOISE_COLOR)

        box = gui.vBox(self.controlArea, "Signal (maximum)")
        gui.checkBox(
            box,
            self,
            "lock_regions",
            "Same as noise region",
            callback=self._lock_changed,
        )
        form = QFormLayout()
        box.layout().addLayout(form)
        self._region_editors(form, "signal", SIGNAL_COLOR)

        box = gui.vBox(self.controlArea, "Selected spectrum")
        self.info_label = QLabel()
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.layout().addWidget(self.info_label)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")

        self._update_fit_enabled()
        self._update_signal_region_shown()
        VisualSettingsDialog(self, self.curveplot.parameter_setter.initial_settings)
        self.resize(900, 600)

    def _region_editors(self, form, name, color):
        for side, label in (("low", "From:"), ("high", "To:")):
            attr = f"{name}_{side}"
            le = lineEditFloatRange(self, self, attr, callback=self.settings_changed)
            form.addRow(label, le)
            self.editors[attr] = le
            line = MovableVline(label=name, color=color, report=self.curveplot)
            connect_line(line, self, attr)
            line.sigMoved.connect(self.redraw_preview)
            if name == "noise":
                line.sigMoved.connect(self._sync_preset)
            line.sigMoveFinished.connect(self.commit.deferred)
            self.curveplot.add_marking(line)
            self.lines[attr] = line

    def _preset_selected(self):
        if self.noise_preset > 0:
            a, b = NOISE_PRESETS[self.noise_preset - 1][1]
            self.noise_low, self.noise_high = min(a, b), max(a, b)
        self.settings_changed()

    def _sync_preset(self):
        """Show the preset matching the current noise region, else custom."""
        self.noise_preset = 0
        if self.noise_low is None or self.noise_high is None:
            return
        current = sorted((float(self.noise_low), float(self.noise_high)))
        for i, (_, limits) in enumerate(NOISE_PRESETS, start=1):
            if np.allclose(current, sorted(limits)):
                self.noise_preset = i

    def _lock_changed(self):
        self._update_signal_region_shown()
        self.settings_changed()

    def _update_signal_region_shown(self):
        for attr in ("signal_low", "signal_high"):
            self.editors[attr].setEnabled(not self.lock_regions)
            self.lines[attr].setVisible(not self.lock_regions)

    def _method_changed(self):
        self._update_fit_enabled()
        self.settings_changed()

    @property
    def degree(self):
        return self.fit_order + 1

    def _update_fit_enabled(self):
        self.fit_combo.setEnabled(self.noise_method == RMS)

    def settings_changed(self):
        self._sync_preset()
        self.redraw_preview()
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Warning.no_x.clear()
        self.data = data
        self.openContext(data)
        self.curveplot.set_data(data)
        if data is not None and not len(self.curveplot.data_x):
            self.Warning.no_x()
        self._init_regions()
        self._sync_preset()
        self.redraw_preview()
        self.commit.now()

    def _init_regions(self):
        x = self.curveplot.data_x
        if x is not None and len(x):
            minx, maxx = float(x[0]), float(x[-1])
        else:
            minx, maxx = 0.0, 1.0

        def valid(v):
            return v is not None and minx <= float(v) <= maxx

        # by default, noise is estimated on the first tenth of the spectrum
        if not (valid(self.noise_low) and valid(self.noise_high)):
            self.noise_low = minx
            self.noise_high = minx + (maxx - minx) / 10
        if not (valid(self.signal_low) and valid(self.signal_high)):
            self.signal_low = minx
            self.signal_high = maxx

    def _limits(self):
        noise = (float(self.noise_low), float(self.noise_high))
        if self.lock_regions:
            return noise, noise
        return noise, (float(self.signal_low), float(self.signal_high))

    def _check_regions(self, x):
        noise_limits, signal_limits = self._limits()
        n_noise = np.count_nonzero(region_mask(x, *noise_limits))
        self.Warning.empty_noise(shown=n_noise == 0)
        self.Warning.empty_signal(shown=not np.any(region_mask(x, *signal_limits)))
        self.Warning.few_points(
            shown=self.noise_method == RMS and 0 < n_noise <= self.degree
        )

    def _selected_index(self):
        if self.curveplot.data is None:
            return None
        selected = np.flatnonzero(self.curveplot.selection_group)
        return selected[0] if len(selected) else None

    def redraw_preview(self):
        for m in self.preview_markings:
            if self.curveplot.in_markings(m):
                self.curveplot.remove_marking(m)
        self.preview_markings = []
        self.info_label.setText("")

        ind = self._selected_index()
        if ind is None or self.data is None or not len(self.curveplot.data_x):
            return

        x = getx(self.curveplot.data)
        y = self.curveplot.data.X[ind]
        noise_limits, signal_limits = self._limits()
        nmask = region_mask(x, *noise_limits)
        smask = region_mask(x, *signal_limits)
        xn, yn = x[nmask], y[nmask]
        order = np.argsort(xn)
        xn, yn = xn[order], yn[order]

        def add(item):
            item.setZValue(10)
            self.curveplot.add_marking(item)
            self.preview_markings.append(item)

        noise_pen = pg.mkPen(color=NOISE_COLOR, width=2, style=Qt.DashLine)
        if self.noise_method == RMS:
            rms, fit = noise_rms(xn, yn, self.degree)
            noise = rms[0]
            if len(xn):
                add(pg.PlotCurveItem(x=xn, y=fit[0], pen=noise_pen))
        else:
            noise = noise_peak_to_peak(yn)[0]
            if len(xn) and np.isfinite(noise):
                for v in (np.nanmin(yn), np.nanmax(yn)):
                    add(pg.PlotCurveItem(x=xn[[0, -1]], y=[v, v], pen=noise_pen))

        signal = signal_max(y[smask])[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            snr = signal / noise
        self.info_label.setText(
            f"Signal: {signal:.6g}\n"
            f"Noise ({self.METHODS[self.noise_method]}): {noise:.6g}\n"
            f"SNR: {snr:.6g}"
        )

    @gui.deferred
    def commit(self):
        self.Warning.empty_noise.clear()
        self.Warning.empty_signal.clear()
        self.Warning.few_points.clear()
        if self.data is None or not len(self.data.domain.attributes):
            self.Outputs.data.send(None)
            return

        x = getx(self.data)
        self._check_regions(x)
        noise_limits, signal_limits = self._limits()
        signal, noise, snr = compute_snr(
            x,
            self.data.X,
            noise_limits,
            signal_limits,
            method=self.noise_method,
            degree=self.degree,
        )

        domain = self.data.domain
        names = get_unique_names(domain, ["Signal", "Noise", "SNR"])
        new_vars = [Orange.data.ContinuousVariable(n) for n in names]
        out_domain = Orange.data.Domain(
            domain.attributes, domain.class_vars, domain.metas + tuple(new_vars)
        )
        out = Orange.data.Table.from_numpy(
            out_domain,
            self.data.X,
            self.data.Y,
            np.hstack([self.data.metas, np.column_stack([signal, noise, snr])]).astype(
                object if domain.metas else float
            ),
            self.data.W,
            attributes=self.data.attributes,
            ids=self.data.ids,
        )
        self.Outputs.data.send(out)

    def set_visual_settings(self, key, value):
        self.curveplot.parameter_setter.set_parameter(key, value)
        self.visual_settings[key] = value

    def send_report(self):
        if self.data is None:
            return
        noise_limits, signal_limits = self._limits()
        items = [("Noise", self.METHODS[self.noise_method])]
        if self.noise_method == RMS:
            items.append(("Fit", self.FIT_ORDERS[self.fit_order]))
        items += [
            ("Noise region", "{:g} - {:g}".format(*noise_limits)),
            (
                "Signal region",
                "same as noise"
                if self.lock_regions
                else "{:g} - {:g}".format(*signal_limits),
            ),
        ]
        self.report_items(items)
        self.report_plot()

    def onDeleteWidget(self):
        self.curveplot.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSNRRegion).run(Orange.data.Table("collagen.csv"))
