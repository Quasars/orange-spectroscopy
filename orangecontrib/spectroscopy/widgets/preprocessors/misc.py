import numpy as np
import pyqtgraph as pg

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel
)
from AnyQt.QtGui import QColor

from Orange.widgets import gui
from Orange.widgets.widget import Msg

from orangecontrib.spectroscopy.data import getx

from orangecontrib.spectroscopy.preprocess import (
    PCADenoising, GaussianSmoothing, Cut, SavitzkyGolayFiltering,
    Absorbance, Transmittance,
    ShiftAndScale, SpSubtract,
    MNFDenoising
)
from orangecontrib.spectroscopy.preprocess.transform import SpecTypes
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange, MovableVline, \
    connect_line, floatornone, round_virtual_pixels
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange, \
    REFERENCE_DATA_PARAM
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors


class GaussianSmoothingEditor(BaseEditorOrange):
    """
    Editor for GaussianSmoothing
    """

    name = "Gaussian smoothing"
    qualname = "orangecontrib.infrared.gaussian"

    DEFAULT_SD = 10.
    MINIMUM_SD = 10e-10

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QFormLayout()
        self.controlArea.setLayout(layout)
        self.sd = self.DEFAULT_SD

        # editing will always return a valid output (in the range)
        w = lineEditFloatRange(self, self, "sd", bottom=0., top=1000., default=self.DEFAULT_SD,
                               callback=self.edited.emit)
        layout.addRow("SD", w)

    def setParameters(self, params):
        self.sd = params.get("sd", self.DEFAULT_SD)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        sd = params.get("sd", cls.DEFAULT_SD)
        sd = float(sd)
        if sd < cls.MINIMUM_SD:
            sd = 0.0
        return GaussianSmoothing(sd=sd)


class CutEditor(BaseEditorOrange):
    """
    Editor for Cut
    """
    name = "Cut"
    qualname = "orangecontrib.spectroscopy.cut"
    replaces = ["orangecontrib.infrared.cut",
                "orangecontrib.infrared.cutinverse"]

    class Warning(BaseEditorOrange.Warning):
        out_of_range = Msg("Limits are out of range.")

    def __init__(self, parent=None, **kwargs):
        BaseEditorOrange.__init__(self, parent, **kwargs)

        self.lowlim = 0.
        self.highlim = 1.
        self.inverse = False

        layout = QFormLayout()
        self.controlArea.setLayout(layout)

        self._lowlime = lineEditFloatRange(self, self, "lowlim", callback=self.edited.emit)
        self._highlime = lineEditFloatRange(self, self, "highlim", callback=self.edited.emit)
        self._inverse = gui.radioButtons(self, self, "inverse",
                                         orientation=Qt.Horizontal, callback=self.edited.emit)

        gui.appendRadioButton(self._inverse, "Keep")
        gui.appendRadioButton(self._inverse, "Remove")

        layout.addRow("Low limit", self._lowlime)
        layout.addRow("High limit", self._highlime)
        layout.addRow(None, self._inverse)

        self._lowlime.focusIn.connect(self.activateOptions)
        self._highlime.focusIn.connect(self.activateOptions)
        self.focusIn = self.activateOptions

        self.line1 = MovableVline(label="Low limit")
        connect_line(self.line1, self, "lowlim")
        self.line1.sigMoveFinished.connect(self.edited)
        self.line2 = MovableVline(label="High limit")
        connect_line(self.line2, self, "highlim")
        self.line2.sigMoveFinished.connect(self.edited)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for line in [self.line1, self.line2]:
            line.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(line)

    def setParameters(self, params):
        if params: #parameters were manually set somewhere else
            self.user_changed = True
        self.lowlim = params.get("lowlim", 0.)
        self.highlim = params.get("highlim", 1.)
        self.inverse = params.get("inverse", False)

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        inverse = params.get("inverse", None)
        return Cut(lowlim=floatornone(lowlim), highlim=floatornone(highlim), inverse=inverse)

    def set_preview_data(self, data):
        self.Warning.out_of_range.clear()
        x = getx(data)
        if len(x):
            minx = np.min(x)
            maxx = np.max(x)
            range = maxx - minx

            init_lowlim = round_virtual_pixels(minx + 0.1 * range, range)
            init_highlim = round_virtual_pixels(maxx - 0.1 * range, range)

            self._lowlime.set_default(init_lowlim)
            self._highlime.set_default(init_highlim)

            if not self.user_changed:
                self.lowlim = init_lowlim
                self.highlim = init_highlim
                self.edited.emit()

            if (self.lowlim < minx and self.highlim < minx) \
                    or (self.lowlim > maxx and self.highlim > maxx):
                self.parent_widget.Warning.preprocessor()
                self.Warning.out_of_range()


class SpSubtractEditor(BaseEditorOrange):
    """
    Editor for preprocess.SpSubtract
    """
    name = "Spectrum subtraction"
    qualname = "orangecontrib.spectroscopy.sp_subtract"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.amount = 0.

        form = QFormLayout()
        amounte = lineEditFloatRange(self, self, "amount", callback=self.edited.emit)
        form.addRow("Reference multiplier", amounte)
        self.controlArea.setLayout(form)

        self.reference = None

        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=2.))
        self.reference_curve.setZValue(10)

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.reference_curve not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.reference_curve)

    def set_reference_data(self, reference):
        self.reference = reference
        self.update_reference_info()

    def setParameters(self, params):
        self.amount = params.get("amount", 0.)
        self.reference = params.get("reference", None)
        self.update_reference_info()

    def update_reference_info(self):
        if not self.reference:
            self.reference_info.setText("Reference: None")
            self.reference_curve.hide()
        else:
            rinfo = "{0:d} spectra".format(len(self.reference)) \
                if len(self.reference) > 1 else "1 spectrum"
            self.reference_info.setText("Reference: " + rinfo)
            X_ref = self.reference.X[0]
            x = getx(self.reference)
            xsind = np.argsort(x)
            self.reference_curve.setData(x=x[xsind], y=X_ref[xsind])
            self.reference_curve.show()

    @staticmethod
    def createinstance(params):
        params = dict(params)
        amount = float(params.get("amount", 0.))
        reference = params.get(REFERENCE_DATA_PARAM, None)
        return SpSubtract(reference, amount=amount)


class SavitzkyGolayFilteringEditor(BaseEditorOrange):
    """
    Editor for preprocess.savitzkygolayfiltering.
    """
    name = "Savitzky-Golay Filter"
    qualname = "orangecontrib.spectroscopy.savitzkygolay"

    DEFAULT_WINDOW = 5
    DEFAULT_POLYORDER = 2
    DEFAULT_DERIV = 0

    MAX_WINDOW = 99
    MIN_WINDOW = 3
    MAX_POLYORDER = MAX_WINDOW - 1
    MAX_DERIV = 3

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.window = self.DEFAULT_WINDOW
        self.polyorder = self.DEFAULT_POLYORDER
        self.deriv = self.DEFAULT_DERIV

        form = QFormLayout()

        self.wspin = gui.spin(self, self, "window", minv=self.MIN_WINDOW, maxv=self.MAX_WINDOW,
                              step=2, callback=self._window_edited)

        self.pspin = gui.spin(self, self, "polyorder", minv=0, maxv=self.MAX_POLYORDER,
                              step=1, callback=self._polyorder_edited)

        self.dspin = gui.spin(self, self, "deriv", minv=0, maxv=self.MAX_DERIV,
                              step=1, callback=self._deriv_edited)

        form.addRow("Window", self.wspin)
        form.addRow("Polynomial Order", self.pspin)
        form.addRow("Derivative Order", self.dspin)
        self.controlArea.setLayout(form)

    def _window_edited(self):
        # make window even on hand input
        if self.window % 2 == 0:
            self.window += 1
        # decrease other parameters if needed
        self.polyorder = min(self.polyorder, self.window - 1)
        self.deriv = min(self.polyorder, self.deriv)
        self.edited.emit()

    def _fix_window_for_polyorder(self):
        # next window will always exist as max polyorder is less than max window
        if self.polyorder >= self.window:
            self.window = self.polyorder + 1
            if self.window % 2 == 0:
                self.window += 1

    def _polyorder_edited(self):
        self._fix_window_for_polyorder()
        self.deriv = min(self.polyorder, self.deriv)
        self.edited.emit()

    def _deriv_edited(self):
        # valid polyorder will always exist as max deriv is less than max polyorder
        self.polyorder = max(self.polyorder, self.deriv)
        self._fix_window_for_polyorder()
        self.edited.emit()

    def setParameters(self, params):
        self.window = params.get("window", self.DEFAULT_WINDOW)
        self.polyorder = params.get("polyorder", self.DEFAULT_POLYORDER)
        self.deriv = params.get("deriv", self.DEFAULT_DERIV)

    @classmethod
    def createinstance(cls, params):
        window = params.get("window", cls.DEFAULT_WINDOW)
        polyorder = params.get("polyorder", cls.DEFAULT_POLYORDER)
        deriv = params.get("deriv", cls.DEFAULT_DERIV)
        return SavitzkyGolayFiltering(window=window, polyorder=polyorder, deriv=deriv)


class ShiftAndScaleEditor(BaseEditorOrange):
    """
    Editor for ShiftAndScale
    """
    # TODO: the layout changes when I click the area of the preprocessor
    #       EFFECT: the sidebar snaps in

    name = "Shift and scale"
    qualname = "orangecontrib.spectroscopy.shiftandscale"
    replaces = ["orangecontrib.infrared.curveshift"]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.offset = 0.
        self.scale = 1.

        form = QFormLayout()
        offset_input = lineEditFloatRange(self, self, "offset", callback=self.edited.emit)
        form.addRow("Vertical offset", offset_input)
        scale_input = lineEditFloatRange(self, self, "scale", callback=self.edited.emit)
        form.addRow("Vertical scaling", scale_input)
        self.controlArea.setLayout(form)

    def setParameters(self, params):
        self.amount = params.get("offset", 0.)
        self.scale = params.get("scale", 1.)

    @staticmethod
    def createinstance(params):
        params = dict(params)
        offset = float(params.get("offset", 0.))
        scale = float(params.get("scale", 1.))
        return ShiftAndScale(offset=offset, scale=scale)


class PCADenoisingEditor(BaseEditorOrange):
    name = "PCA denoising"
    qualname = "orangecontrib.infrared.pca_denoising"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.components = 5

        form = QFormLayout()
        self.controlArea.setLayout(form)

        compspin = gui.spin(self, self, "components", minv=1, maxv=100,
                            step=1, callback=self.edited.emit)

        endlabel = QLabel("components.")
        hboxLayout = QHBoxLayout()
        hboxLayout.addWidget(compspin)
        hboxLayout.addWidget(endlabel)

        form.addRow("Keep", hboxLayout)

    def setParameters(self, params):
        self.components = params.get("components", 5)

    @staticmethod
    def createinstance(params):
        params = dict(params)
        components = params.get("components", 5)
        return PCADenoising(components=components)


class MNFDenoisingEditor(BaseEditorOrange):
    """
    Editor for Minimum Noise Fraction denoising
    """

    name = "MNF denoising"
    qualname = "orangecontrib.spectroscopy.mnf_denoising"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.components = 5

        form = QFormLayout()
        self.controlArea.setLayout(form)

        compspin = gui.spin(self, self, "components", minv=1, maxv=100,
                            step=1, callback=self.edited.emit)

        endlabel = QLabel("components.")
        hboxLayout = QHBoxLayout()
        hboxLayout.addWidget(compspin)
        hboxLayout.addWidget(endlabel)

        form.addRow("Keep", hboxLayout)

    def setParameters(self, params):
        self.components = params.get("components", 5)

    @staticmethod
    def createinstance(params):
        params = dict(params)
        components = params.get("components", 5)
        return MNFDenoising(components=components)


class SpectralTransformEditor(BaseEditorOrange):
    name = "Spectral Transformations"
    qualname = "orangecontrib.spectroscopy.transforms"

    TRANSFORMS = [Absorbance,
                  Transmittance]

    transform_names = [a.__name__ for a in TRANSFORMS]
    from_names = [a.value for a in SpecTypes]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())

        form = QFormLayout()

        self.fromcb = QComboBox()
        self.fromcb.addItems(self.from_names)

        self.tocb = QComboBox()
        self.tocb.addItems(self.transform_names)

        form.addRow("Original", self.fromcb)
        form.addRow("Transformed", self.tocb)
        self.controlArea.layout().addLayout(form)

        self.fromcb.currentIndexChanged.connect(self.changed)
        self.fromcb.activated.connect(self.edited)
        self.tocb.currentIndexChanged.connect(self.changed)
        self.tocb.activated.connect(self.edited)

        self.reference = None

        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=2.))
        self.reference_curve.setZValue(10)

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.reference_curve not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.reference_curve)

    def setParameters(self, params):
        from_type = params.get("from_type", 0)
        to_type = params.get("to_type", 1)
        self.fromcb.setCurrentIndex(from_type)
        self.tocb.setCurrentIndex(to_type)
        self.update_reference_info()

    def parameters(self):
        return {"from_type": self.fromcb.currentIndex(),
                "to_type": self.tocb.currentIndex()}

    @staticmethod
    def createinstance(params):
        from_type = params.get("from_type", 0)
        to_type = params.get("to_type", 1)
        from_spec_type = SpecTypes(SpectralTransformEditor.from_names[from_type])
        transform = SpectralTransformEditor.TRANSFORMS[to_type]
        reference = params.get(REFERENCE_DATA_PARAM, None)
        if from_spec_type not in transform.from_types:
            return lambda data: data[:0]  # return an empty data table
        if reference:
            reference = reference[:1]
        return transform(reference=reference)

    def set_reference_data(self, reference):
        self.reference = reference
        self.update_reference_info()

    def update_reference_info(self):
        if not self.reference:
            self.reference_info.setText("Reference: None")
            self.reference_curve.hide()
        else:
            rinfo = "1st of {0:d} spectra".format(len(self.reference)) \
                if len(self.reference) > 1 else "1 spectrum"
            self.reference_info.setText("Reference: " + rinfo)
            X_ref = self.reference.X[0]
            x = getx(self.reference)
            xsind = np.argsort(x)
            self.reference_curve.setData(x=x[xsind], y=X_ref[xsind])
            self.reference_curve.show()



preprocess_editors.register(CutEditor, 25)
preprocess_editors.register(GaussianSmoothingEditor, 75)
preprocess_editors.register(SavitzkyGolayFilteringEditor, 100)
preprocess_editors.register(PCADenoisingEditor, 200)
preprocess_editors.register(SpectralTransformEditor, 225)
preprocess_editors.register(ShiftAndScaleEditor, 250)
preprocess_editors.register(SpSubtractEditor, 275)
preprocess_editors.register(MNFDenoisingEditor, 300)
