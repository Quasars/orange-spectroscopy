import sys

from AnyQt.QtCore import pyqtSignal as Signal, QObject
from AnyQt.QtWidgets import (
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QPushButton,
    QSizePolicy,
    QHBoxLayout,
    QLabel,
    QApplication,
    QStyle,
)

from Orange.widgets.data.utils.preprocess import blocked
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import Integrate
from orangecontrib.spectroscopy.widgets.gui import MovableVline
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    BaseEditor,
    SetXDoubleSpinBox,
)


class IntegrateEditor(BaseEditor):
    """
    Editor to integrate defined regions.
    """

    name = "Integrate"
    qualname = "orangecontrib.infrared.integrate"

    Integrators_classes = Integrate.INTEGRALS[:-1]  # without SeparateBaseline
    Integrators = [a.name for a in Integrators_classes]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._limits = []

        self.setLayout(QVBoxLayout())

        deprecation_info = QLabel(
            "This preprocessor is deprecated and will be removed in the future. "
            "Use the Integrate widget instead.",
            self,
        )
        deprecation_info.setWordWrap(True)
        deprecation_info.setStyleSheet("color: red")
        self.layout().addWidget(deprecation_info)

        self.form_set = QFormLayout()
        self.form_lim = QFormLayout()
        self.layout().addLayout(self.form_set)
        self.layout().addLayout(self.form_lim)

        self.methodcb = QComboBox()
        self.methodcb.addItems(self.Integrators)

        self.form_set.addRow("Integration method:", self.methodcb)
        self.methodcb.currentIndexChanged.connect(self.changed)
        self.methodcb.activated.connect(self.edited)

        self.focusIn = self.activateOptions

        self.add_limit()

        button = QPushButton("Add Region")
        self.layout().addWidget(button)
        button.clicked.connect(self.add_limit)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for row in range(self.form_lim.count()):
            limitbox = self.form_lim.itemAt(row, QFormLayout.FieldRole)
            if limitbox:
                self.parent_widget.curveplot.add_marking(limitbox.line1)
                self.parent_widget.curveplot.add_marking(limitbox.line2)

    def add_limit(self, *args, row=None):
        if row is None:
            row = len(self._limits)
            try:
                self._limits.append(self._limits[-1])
            except IndexError:
                self._limits.append([0.0, 1.0])
        label = "Region {0}".format(row + 1)
        limitbox = LimitsBox(limits=self._limits[row], label=label)
        if self.form_lim.rowCount() < row + 1:
            # new row
            self.form_lim.addRow(limitbox)
        else:
            # row already exists
            self.form_lim.setLayout(row, 2, limitbox)
        limitbox.focusIn = self.activateOptions
        limitbox.valueChanged.connect(self.set_limits)
        limitbox.editingFinished.connect(self.edited)
        limitbox.deleted.connect(self.remove_limit)
        self.edited.emit()
        return limitbox

    def remove_limit(self, limitbox):
        row, role = self.form_lim.getLayoutPosition(limitbox)
        for r in range(row, len(self._limits)):
            limitbox = self.form_lim.itemAt(r, QFormLayout.FieldRole)
            limitbox.removeLayout()
        self._limits.pop(row)
        self.set_all_limits(self._limits)

    def set_limits(self, limits, limitbox, user=True):
        if user:
            self.user_changed = True
        row, role = self.form_lim.getLayoutPosition(limitbox)
        if self._limits[row] != limits:
            self._limits[row] = limits
            with blocked(self.form_lim):
                limitbox.lowlime.setValue(limits[0])
                limitbox.highlime.setValue(limits[1])
            self.changed.emit()

    def set_all_limits(self, limits, user=True):
        if user:
            self.user_changed = True
        self._limits = limits
        for row in range(len(limits)):
            limitbox = self.form_lim.itemAt(row, QFormLayout.FieldRole)
            if limitbox is None:
                limitbox = self.add_limit(row=row)
            with blocked(limitbox):
                limitbox.lowlime.setValue(limits[row][0])
                limitbox.highlime.setValue(limits[row][1])
        self.changed.emit()

    def setParameters(self, params):
        if params:  # parameters were manually set somewhere else
            self.user_changed = True
        self.methodcb.setCurrentIndex(
            params.get("method", self.Integrators_classes.index(Integrate.Baseline))
        )
        self.set_all_limits(params.get("limits", [[0.0, 1.0]]), user=False)

    def parameters(self):
        return {"method": self.methodcb.currentIndex(), "limits": self._limits}

    @staticmethod
    def createinstance(params):
        methodindex = params.get(
            "method", IntegrateEditor.Integrators_classes.index(Integrate.Baseline)
        )
        method = IntegrateEditor.Integrators_classes[methodindex]
        limits = params.get("limits", None)
        return Integrate(methods=method, limits=limits)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_all_limits([[min(x), max(x)]])
                self.edited.emit()


class LimitsBox(QHBoxLayout):
    """
    Box with two limits and optional selection lines

    Args:
        limits (list): List containing low and high limit set
        label  (str) : Label widget
        delete (bool): Include self-deletion button
    """

    valueChanged = Signal(list, QObject)
    editingFinished = Signal(QObject)
    deleted = Signal(QObject)

    def __init__(self, parent=None, **kwargs):
        limits = kwargs.pop('limits', None)
        label = kwargs.pop('label', None)
        delete = kwargs.pop('delete', True)
        super().__init__(parent, **kwargs)

        minf, maxf = -sys.float_info.max, sys.float_info.max

        if label:
            self.addWidget(QLabel(label))

        self.lowlime = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5, value=limits[0], maximumWidth=75
        )
        self.highlime = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5, value=limits[1], maximumWidth=75
        )
        self.lowlime.setValue(limits[0])
        self.highlime.setValue(limits[1])
        self.addWidget(self.lowlime)
        self.addWidget(self.highlime)

        if delete:
            self.button = QPushButton(
                QApplication.style().standardIcon(QStyle.SP_DockWidgetCloseButton), ""
            )
            self.addWidget(self.button)
            self.button.clicked.connect(self.selfDelete)

        self.lowlime.valueChanged[float].connect(self.limitChanged)
        self.highlime.valueChanged[float].connect(self.limitChanged)
        self.lowlime.editingFinished.connect(self.editFinished)
        self.highlime.editingFinished.connect(self.editFinished)

        self.lowlime.focusIn = self.focusInChild
        self.highlime.focusIn = self.focusInChild

        self.line1 = MovableVline(position=limits[0], label=label + " - Low")
        self.line1.sigMoved.connect(self.lineLimitChanged)
        self.line2 = MovableVline(position=limits[1], label=label + " - High")
        self.line2.sigMoved.connect(self.lineLimitChanged)

        self.line1.sigMoveFinished.connect(self.editFinished)
        self.line2.sigMoveFinished.connect(self.editFinished)

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)

    def focusInChild(self):
        self.focusIn()

    def limitChanged(self):
        newlimits = [self.lowlime.value(), self.highlime.value()]
        self.line1.setValue(newlimits[0])
        self.line2.setValue(newlimits[1])
        self.valueChanged.emit(newlimits, self)

    def lineLimitChanged(self):
        newlimits = [self.line1.rounded_value(), self.line2.rounded_value()]
        self.lowlime.setValue(newlimits[0])
        self.highlime.setValue(newlimits[1])
        self.limitChanged()

    def editFinished(self):
        self.editingFinished.emit(self)

    def selfDelete(self):
        self.deleted.emit(self)
        self.removeLayout()

    def removeLayout(self):
        while self.count():
            self.takeAt(0).widget().setParent(None)
        self.setParent(None)


preprocess_editors.register(IntegrateEditor, 175)
