import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QVBoxLayout, QLabel
from Orange.widgets import gui
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import \
    BaseEditorOrange, REFERENCE_DATA_PARAM
from orangecontrib.spectroscopy.preprocess.atm_corr import AtmCorr
from orangecontrib.spectroscopy.data import spectra_mean, getx


class AtmCorrEditor(BaseEditorOrange):
    """
       Atmospheric gas correction.
       Default ranges are two H2O regions (corrected) and one CO2 region (removed)
    """
    name = "Atmospheric gas (CO2/H2O) correction"
    categories = ["Correction"]
    qualname = "preprocessors.atm_corr"

    RANGES = [[1300, 2100, 1], [2190, 2480, 2],
              [3410, 3850, 1], ['', '', 0]]
    SMOOTH = True
    SMOOTH_WIN = 9
    BRIDGE_WIN = 9
    MEAN_REF = True

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None
        self.preview_data_max = 1

        self.smooth = self.SMOOTH
        self.smooth_win = self.SMOOTH_WIN
        self.bridge_win = self.BRIDGE_WIN
        self.mean_reference = self.MEAN_REF
        for an, v in self.get_range_defaults().items():
            setattr(self, an, v)

        self.controlArea.layout().addWidget(QLabel("Correction ranges", self))
        self.range_boxes = []
        for b in range(len(self.RANGES)):
            box = gui.hBox(self.controlArea)
            gui.comboBox(box, self, f"corrmode_{b}",
                         items=('No-op', 'Correct', 'Bridge'), callback=self.edited.emit)
            gui.lineEdit(box, self, f"low_{b}", label="from",
                          callback=self.edited.emit, orientation=Qt.Horizontal,
                          controlWidth=75)
            gui.lineEdit(box, self, f"high_{b}", label="to",
                          callback=self.edited.emit, orientation=Qt.Horizontal,
                          controlWidth=75)
            self.range_boxes.append(box)

        self.smooth_button = gui.checkBox(self.controlArea, self, "smooth",
                     "Smooth corrected regions", callback=self.edited.emit)
        self.smooth_win_spin = gui.spin(self.controlArea, self, "smooth_win",
                 label="Savitzky-Golay window size", minv=5, maxv=35,
                 step=2, controlWidth=60, callback=self.edited.emit)
        self.bridge_win_spin = gui.spin(self.controlArea, self, "bridge_win",
                 label="Bridge base window size", minv=3, maxv=35,
                 step=2, controlWidth=60, callback=self.edited.emit)
        gui.checkBox(self.controlArea, self, "mean_reference",
                      "Use mean of references", callback=self.edited.emit)
        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=1.5))
        self.reference_curve.setZValue(10)

        self.user_changed = False

    @classmethod
    def get_range_defaults(cls):
        defs = {}
        for b, r in enumerate(cls.RANGES):
            for i, a in enumerate(['low', 'high', 'corrmode']):
                defs[f'{a}_{b}'] = r[i]
        return defs

    def setParameters(self, params):
        if params:
            self.user_changed = True
        self.smooth = params.get("smooth", self.SMOOTH)
        self.smooth_win = params.get("smooth_win", self.SMOOTH_WIN)
        self.bridge_win = params.get("bridge_win", self.BRIDGE_WIN)
        self.update_reference_info()
        self.smooth_win_spin.setEnabled(self.smooth)
        for an, v in self.get_range_defaults().items():
            setattr(self, an, params.get(an, v))
        self.mean_reference = params.get("mean_reference", self.MEAN_REF)

    def parameters(self):
        parameters = super().parameters()
        return parameters

    def set_reference_data(self, data):
        self.reference = data
        self.update_reference_info()

    def set_preview_data(self, data):
        try:
            self.preview_data_max = np.nanmax(spectra_mean(data.X))
        except ValueError:  # if sequence is empty
            self.preview_data_max = 1
        self.update_reference_info()

    def update_reference_info(self):
        if self.reference_curve not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.reference_curve)
        if not self.reference:
            self.reference_curve.hide()
            self.reference_info.setText("Reference: missing!")
            self.reference_info.setStyleSheet("color: red")
        else:
            if len(self.reference) == 1:
                rinfo = "1 spectrum"
            elif self.mean_reference:
                rinfo = f"mean of {len(self.reference)} spectra"
            else:
                rinfo = f"{len(self.reference)} individual spectra"
            self.reference_info.setText("Reference: " + rinfo)
            self.reference_info.setStyleSheet("color: black")
            X_ref = spectra_mean(self.reference.X)
            X_ref = X_ref * (self.preview_data_max / X_ref.max())
            x = getx(self.reference)
            xsind = np.argsort(x)
            self.reference_curve.setData(x=x[xsind], y=X_ref[xsind])
            self.reference_curve.setVisible(True)

    @classmethod
    def createinstance(cls, params):
        cranges = []
        sranges = []
        rdefs = cls.get_range_defaults()
        for b in range(len(cls.RANGES)):
            r = [params.get(f'{a}_{b}', rdefs.get(f'{a}_{b}')) for a in ['low', 'high']]
            try:
                r = [float(v) for v in r]
            except (ValueError, TypeError):
                continue
            cm = params.get(f'corrmode_{b}', rdefs.get(f'corrmode_{b}'))
            if cm == 1:
                cranges.append(r)
            elif cm == 2:
                sranges.append(r)
        smooth_win = params.get("smooth_win", cls.SMOOTH_WIN) if \
            params.get("smooth", cls.SMOOTH) else 0
        bridge_win = params.get("bridge_win", cls.BRIDGE_WIN)
        mean_reference = params.get("mean_reference", cls.MEAN_REF)
        reference = params.get(REFERENCE_DATA_PARAM, None)

        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return AtmCorr(reference=reference, correct_ranges=cranges,
                           spline_ranges=sranges, smooth_win=smooth_win,
                           spline_base_win=bridge_win,
                           mean_reference=mean_reference)


preprocess_editors.register(AtmCorrEditor, 700)
