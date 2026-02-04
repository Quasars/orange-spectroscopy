import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QVBoxLayout, QFormLayout, QPushButton, QApplication, QStyle

from Orange.widgets import gui
from orangecontrib.spectroscopy.preprocess import LinearBaseline, RubberbandBaseline
from orangecontrib.spectroscopy.widgets.gui import XPosLineEdit
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    BaseEditorOrange,
    PreviewMinMaxMixin,
    layout_widgets,
)


class BaselineEditor(BaseEditorOrange, PreviewMinMaxMixin):
    """
    Baseline subtraction.
    """

    name = "Baseline Correction"
    qualname = "orangecontrib.infrared.baseline"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())

        form = QFormLayout()

        self.baseline_type = 0
        self.peak_dir = 0
        self.sub = 0

        self.baselinecb = gui.comboBox(
            None,
            self,
            "baseline_type",
            items=["Linear", "Rubber band"],
            callback=self.edited.emit,
        )
        self.peakcb = gui.comboBox(
            None,
            self,
            "peak_dir",
            items=["Positive", "Negative"],
            callback=self.edited.emit,
        )
        self.subcb = gui.comboBox(
            None,
            self,
            "sub",
            items=["Subtract", "Calculate"],
            callback=self.edited.emit,
        )

        form.addRow("Baseline Type", self.baselinecb)
        form.addRow("Peak Direction", self.peakcb)
        form.addRow("Background Action", self.subcb)

        self.controlArea.layout().addLayout(form)

        self.ranges_box = gui.vBox(self.controlArea)  # container for ranges

        self.range_button = QPushButton("", autoDefault=False)
        self.range_button.clicked.connect(self.add_point)
        self.controlArea.layout().addWidget(self.range_button)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=2.0))
        self.reference_curve.setZValue(10)

        self.preview_data = None

        self.user_changed = False

        self._adapt_ui()

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()

        for pair in self._range_widgets():
            for w in pair:
                if w.line not in self.parent_widget.curveplot.markings:
                    w.line.report = self.parent_widget.curveplot
                    self.parent_widget.curveplot.add_marking(w.line)

    def _set_button_text(self):
        self.range_button.setText(
            "Select point" if self.ranges_box.layout().count() == 0 else "Add point"
        )

    def _range_widgets(self):
        for b in layout_widgets(self.ranges_box):
            yield self._extract_pair(b)

    def add_point(self):
        pmin, pmax = self.preview_min_max()
        if (
            len(list(self._range_widgets())) == 0
        ):  # if empty, add two points at the same time
            lwmin = self.add_range_selection_ui()
            lwmax = self.add_range_selection_ui()
            self._extract_pair(lwmin)[0].position = pmin
            self._extract_pair(lwmax)[0].position = pmax
        else:
            lw = self.add_range_selection_ui()
            self._extract_pair(lw)[0].position = (pmin + pmax) / 2
        self.edited.emit()  # refresh output

    def _extract_pair(self, container):
        return list(layout_widgets(container))[:1]

    def add_range_selection_ui(self):
        linelayout = gui.hBox(self)
        pmin, pmax = self.preview_min_max()
        e = XPosLineEdit(label="")
        e.set_default((pmin + pmax) / 2)
        linelayout.layout().addWidget(e)
        e.edited.connect(self.edited)
        e.focusIn.connect(self.activateOptions)

        remove_button = QPushButton(
            QApplication.style().standardIcon(QStyle.SP_DockWidgetCloseButton),
            "",
            autoDefault=False,
        )
        remove_button.clicked.connect(lambda: self.delete_range(linelayout))
        linelayout.layout().addWidget(remove_button)

        self.ranges_box.layout().addWidget(linelayout)
        self._set_button_text()
        return linelayout

    def delete_range(self, box):
        removed = [box]
        self.ranges_box.layout().removeWidget(box)

        # if only 1 widget stayed that would be invalid
        if len(list(layout_widgets(self.ranges_box))) == 1:
            also_remove = next(layout_widgets(self.ranges_box))
            self.ranges_box.layout().removeWidget(also_remove)
            removed.append(also_remove)

        self._set_button_text()

        # remove selection lines
        curveplot = self.parent_widget.curveplot
        for r in removed:
            for w in self._extract_pair(r):
                if curveplot.in_markings(w.line):
                    curveplot.remove_marking(w.line)

        self.edited.emit()

    def setParameters(self, params):
        self.baseline_type = params.get("baseline_type", 0)
        self.peak_dir = params.get("peak_dir", 0)
        self.sub = params.get("sub", 0)
        self._adapt_ui()

    def _adapt_ui(self):
        # peak direction is only relevant for rubberband
        self.peakcb.setEnabled(self.baseline_type == 1)
        self._set_button_text()

    def parameters(self):
        parameters = super().parameters()
        zero_points = []
        for pair in self._range_widgets():
            zero_points.append(float(pair[0].position))
        parameters["zero_points"] = zero_points if zero_points else None
        return parameters

    @staticmethod
    def createinstance(params):
        baseline_type = params.get("baseline_type", 0)
        peak_dir = params.get("peak_dir", 0)
        sub = params.get("sub", 0)
        zero_points = params.get("zero_points", None)

        if baseline_type == 0:
            return LinearBaseline(peak_dir=peak_dir, sub=sub, zero_points=zero_points)
        elif baseline_type == 1:
            return RubberbandBaseline(peak_dir=peak_dir, sub=sub)
        else:
            raise Exception("unknown baseline type")

    def set_preview_data(self, data):
        self.preview_data = data


preprocess_editors.register(BaselineEditor, 125)
