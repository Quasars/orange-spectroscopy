import sys

from AnyQt.QtWidgets import (
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QButtonGroup,
    QRadioButton,
)

from Orange.data import ContinuousVariable
from Orange.widgets import gui
from Orange.widgets.data.utils.preprocess import blocked
from Orange.widgets.data.owpreprocess import icon_path
from Orange.widgets.utils.itemmodels import DomainModel

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import (
    Normalize,
    Integrate,
    NormalizeReference,
    NormalizePhaseReference,
)
from orangecontrib.spectroscopy.widgets.gui import MovableVline
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.integrate import IntegrateEditor
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    BaseEditorOrange,
    SetXDoubleSpinBox,
    REFERENCE_DATA_PARAM,
)


NORMALIZE_BY_REFERENCE = 42
PHASE_REFERENCE = 13


class NormalizeEditor(BaseEditorOrange):
    """
    Normalize spectra.
    """

    name = "Normalize Spectra"
    qualname = "orangecontrib.infrared.normalize"
    icon = icon_path("Normalize.svg")

    # Normalization methods
    Normalizers = [
        ("Vector Normalization", Normalize.Vector),
        ("Min-Max Normalization", Normalize.MinMax),
        ("Area Normalization", Normalize.Area),
        ("Attribute Normalization", Normalize.Attribute),
        ("Standard Normal Variate (SNV)", Normalize.SNV),
        ("Normalize by Reference", NORMALIZE_BY_REFERENCE),
        ("Normalize by Reference (Complex Phase)", PHASE_REFERENCE),
    ]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        layout = QVBoxLayout()
        self.controlArea.setLayout(layout)

        self.__method = Normalize.Vector
        self.lower = 0
        self.upper = 4000
        self.int_method = 0
        self.attrs = DomainModel(
            DomainModel.METAS | DomainModel.CLASSES, valid_types=ContinuousVariable
        )
        self.attrform = QFormLayout()
        self.chosen_attr = None
        self.last_domain = None
        self.saved_attr = None
        self.attrcb = gui.comboBox(
            None, self, "chosen_attr", callback=self.edited.emit, model=self.attrs
        )
        self.attrform.addRow("Normalize to", self.attrcb)

        self.areaform = QFormLayout()
        self.int_method_cb = QComboBox(enabled=False)
        self.int_method_cb.addItems(IntegrateEditor.Integrators)
        minf, maxf = -sys.float_info.max, sys.float_info.max
        self.lspin = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5, value=self.lower, enabled=False
        )
        self.uspin = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5, value=self.upper, enabled=False
        )
        self.areaform.addRow("Normalize to", self.int_method_cb)
        self.areaform.addRow("Lower limit", self.lspin)
        self.areaform.addRow("Upper limit", self.uspin)

        self._group = group = QButtonGroup(self)

        for name, method in self.Normalizers:
            rb = QRadioButton(self, text=name, checked=self.__method == method)

            layout.addWidget(rb)
            if method is Normalize.Attribute:
                layout.addLayout(self.attrform)
            elif method is Normalize.Area:
                layout.addLayout(self.areaform)
            group.addButton(rb, method)

        group.buttonClicked.connect(self.__on_buttonClicked)

        self.lspin.focusIn = self.activateOptions
        self.uspin.focusIn = self.activateOptions
        self.focusIn = self.activateOptions

        self.lspin.valueChanged[float].connect(self.setL)
        self.lspin.editingFinished.connect(self.reorderLimits)
        self.uspin.valueChanged[float].connect(self.setU)
        self.uspin.editingFinished.connect(self.reorderLimits)
        self.int_method_cb.currentIndexChanged.connect(self.setinttype)
        self.int_method_cb.activated.connect(self.edited)

        def rounded(line):
            return float(line.rounded_value())

        self.lline = MovableVline(position=self.lower, label="Low limit")
        self.lline.sigMoved.connect(lambda _: self.setL(rounded(self.lline)))
        self.lline.sigMoveFinished.connect(self.reorderLimits)
        self.uline = MovableVline(position=self.upper, label="High limit")
        self.uline.sigMoved.connect(lambda _: self.setU(rounded(self.uline)))
        self.uline.sigMoveFinished.connect(self.reorderLimits)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.__method == Normalize.Area:
            if self.lline not in self.parent_widget.curveplot.markings:
                self.parent_widget.curveplot.add_marking(self.lline)
            if (
                self.uline not in self.parent_widget.curveplot.markings
                and IntegrateEditor.Integrators_classes[self.int_method]
                is not Integrate.PeakAt
            ):
                self.parent_widget.curveplot.add_marking(self.uline)

    def setParameters(self, params):
        if params:  # parameters were manually set somewhere else
            self.user_changed = True
        method = params.get("method", Normalize.Vector)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        int_method = params.get("int_method", 0)
        if method not in [method for name, method in self.Normalizers]:
            # handle old worksheets
            method = Normalize.Vector
        self.setMethod(method)
        self.int_method_cb.setCurrentIndex(int_method)
        self.setL(lower, user=False)
        self.setU(upper, user=False)
        self.saved_attr = params.get(
            "attr"
        )  # chosen_attr will be set when data are connected

    def parameters(self):
        return {
            "method": self.__method,
            "lower": self.lower,
            "upper": self.upper,
            "int_method": self.int_method,
            "attr": self.chosen_attr,
        }

    def setMethod(self, method):
        if self.__method != method:
            self.__method = method
            b = self._group.button(method)
            b.setChecked(True)
            for widget in [self.attrcb, self.int_method_cb, self.lspin, self.uspin]:
                widget.setEnabled(False)
            if method is Normalize.Attribute:
                self.attrcb.setEnabled(True)
            elif method is Normalize.Area:
                self.int_method_cb.setEnabled(True)
                self.lspin.setEnabled(True)
                self.uspin.setEnabled(True)
            self.activateOptions()
            self.changed.emit()

    def setL(self, lower, user=True):
        if user:
            self.user_changed = True
        if self.lower != lower:
            self.lower = lower
            with blocked(self.lspin):
                self.lspin.setValue(lower)
                self.lline.setValue(lower)
            self.changed.emit()

    def setU(self, upper, user=True):
        if user:
            self.user_changed = True
        if self.upper != upper:
            self.upper = upper
            with blocked(self.uspin):
                self.uspin.setValue(upper)
                self.uline.setValue(upper)
            self.changed.emit()

    def reorderLimits(self):
        if IntegrateEditor.Integrators_classes[self.int_method] is Integrate.PeakAt:
            self.upper = self.lower + 10
        limits = [self.lower, self.upper]
        self.lower, self.upper = min(limits), max(limits)
        self.lspin.setValue(self.lower)
        self.uspin.setValue(self.upper)
        self.lline.setValue(self.lower)
        self.uline.setValue(self.upper)
        self.edited.emit()

    def setinttype(self):
        if self.int_method != self.int_method_cb.currentIndex():
            self.int_method = self.int_method_cb.currentIndex()
            self.reorderLimits()
            self.activateOptions()
            self.changed.emit()

    def __on_buttonClicked(self):
        method = self._group.checkedId()
        if method != self.__method:
            self.setMethod(self._group.checkedId())
            self.edited.emit()

    @staticmethod
    def createinstance(params):
        method = params.get("method", Normalize.Vector)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        int_method_index = params.get("int_method", 0)
        int_method = IntegrateEditor.Integrators_classes[int_method_index]
        attr = params.get("attr", None)
        if method not in (NORMALIZE_BY_REFERENCE, PHASE_REFERENCE):
            return Normalize(
                method=method,
                lower=lower,
                upper=upper,
                int_method=int_method,
                attr=attr,
            )
        else:
            # avoids circular imports
            reference = params.get(REFERENCE_DATA_PARAM, None)
            if method == PHASE_REFERENCE:
                return NormalizePhaseReference(reference=reference)
            return NormalizeReference(reference=reference)

    def set_preview_data(self, data):
        edited = False
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.setL(min(x))
                self.setU(max(x))
                edited = True
        if data is not None and data.domain != self.last_domain:
            self.last_domain = data.domain
            self.attrs.set_domain(data.domain)
            try:  # try to load the feature
                self.chosen_attr = self.saved_attr
            except ValueError:  # could not load the chosen attr
                self.chosen_attr = self.attrs[0] if self.attrs else None
                self.saved_attr = self.chosen_attr
            edited = True
        if edited:
            self.edited.emit()


preprocess_editors.register(NormalizeEditor, 150)
