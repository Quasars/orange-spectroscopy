import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QPushButton,
    QApplication,
    QStyle,
    QSizePolicy,
)

from Orange.widgets import gui
from orangecontrib.spectroscopy.data import spectra_mean, getx
from orangecontrib.spectroscopy.preprocess import EMSC
from orangecontrib.spectroscopy.preprocess.emsc import (
    SelectionFunction,
    SmoothedSelectionFunction,
)
from orangecontrib.spectroscopy.preprocess.npfunc import Sum
from orangecontrib.spectroscopy.widgets.gui import XPosLineEdit, lineEditFloatOrNone
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    BaseEditorOrange,
    PreviewMinMaxMixin,
    layout_widgets,
    REFERENCE_DATA_PARAM,
)


class EMSCEditor(BaseEditorOrange, PreviewMinMaxMixin):
    name = "EMSC"
    qualname = "orangecontrib.spectroscopy.preprocess.emsc"

    ORDER_DEFAULT = 2
    SCALING_DEFAULT = True
    OUTPUT_MODEL_DEFAULT = False

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.controlArea.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None

        self.order = self.ORDER_DEFAULT

        gui.spin(
            self.controlArea,
            self,
            "order",
            label="Polynomial order",
            minv=0,
            maxv=10,
            controlWidth=50,
            callback=self.edited.emit,
        )

        self.scaling = self.SCALING_DEFAULT
        gui.checkBox(
            self.controlArea, self, "scaling", "Scaling", callback=self.edited.emit
        )

        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.output_model = self.OUTPUT_MODEL_DEFAULT
        gui.checkBox(
            self.controlArea,
            self,
            "output_model",
            "Output EMSC model as metas",
            callback=self.edited.emit,
        )

        self._init_regions()
        self._init_reference_curve()

        self.user_changed = False

    def _init_regions(self):
        self.ranges_box = gui.vBox(self.controlArea)  # container for ranges

        self.range_button = QPushButton("Select Region", autoDefault=False)
        self.range_button.clicked.connect(self.add_range_selection)
        self.controlArea.layout().addWidget(self.range_button)

        self.weight_curve = pg.PlotCurveItem()
        self.weight_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=1.0))
        self.weight_curve.setZValue(11)

    def _init_reference_curve(self):
        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=2.0))
        self.reference_curve.setZValue(10)

    def _set_button_text(self):
        self.range_button.setText(
            "Select Region" if self.ranges_box.layout().count() == 0 else "Add Region"
        )

    def add_range_selection(self):
        pmin, pmax = self.preview_min_max()
        lw = self.add_range_selection_ui()
        pair = self._extract_all(lw)
        pair[0].position = pmin
        pair[1].position = pmax
        pair[2].position = None
        self.edited.emit()  # refresh output

    def add_range_selection_ui(self):
        linelayout = gui.hBox(self)
        pmin, pmax = self.preview_min_max()
        # TODO make the size appropriate so that the sidebar of the Preprocess Spectra
        # doesn't change when the region is added
        mine = XPosLineEdit(label="")
        maxe = XPosLineEdit(label="")
        smoothinge = XPosLineEdit(label="", element=lineEditFloatOrNone)
        smoothinge.edit.sizeHintFactor = 0.4
        mine.set_default(pmin)
        maxe.set_default(pmax)
        smoothinge.edit.setPlaceholderText("smoothing")
        for w in [mine, maxe, smoothinge]:
            linelayout.layout().addWidget(w)
            w.edited.connect(self.edited)
            w.focusIn.connect(self.activateOptions)

        remove_button = QPushButton(
            QApplication.style().standardIcon(QStyle.SP_DockWidgetCloseButton),
            "",
            autoDefault=False,
        )
        remove_button.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed))
        remove_button.clicked.connect(lambda: self.delete_range(linelayout))
        linelayout.layout().addWidget(remove_button)

        self.ranges_box.layout().addWidget(linelayout)
        self._set_button_text()
        return linelayout

    def delete_range(self, box):
        self.ranges_box.layout().removeWidget(box)
        self._set_button_text()

        # remove selection lines
        curveplot = self.parent_widget.curveplot
        for w in self._extract_all(box)[:2]:
            if curveplot.in_markings(w.line):
                curveplot.remove_marking(w.line)

        self.edited.emit()

    def _extract_all(self, container):
        return list(layout_widgets(container))

    def _range_widgets(self):
        for b in layout_widgets(self.ranges_box):
            yield self._extract_all(b)

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.reference_curve not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.reference_curve)

        for pair in self._range_widgets():
            for w in pair[:2]:
                if w.line not in self.parent_widget.curveplot.markings:
                    w.line.report = self.parent_widget.curveplot
                    self.parent_widget.curveplot.add_marking(w.line)

        # add the second viewbox for weight display with X connected but y free
        cp = self.parent_widget.curveplot
        p2 = pg.ViewBox()
        p2.setEnabled(False)  # disable mouse events
        cp.plot.scene().addItem(p2)
        p2.addItem(self.weight_curve)
        p2.setXLink(cp.plot)
        cp.add_connected_view(p2)

    def _set_range_parameters(self, params):
        ranges = params.get("ranges", [])
        rw = list(self._range_widgets())
        for i, (rmin, rhigh, _, smoothing) in enumerate(ranges):
            if i >= len(rw):
                lw = self.add_range_selection_ui()
                pair = self._extract_all(lw)
            else:
                pair = rw[i]
            pair[0].position = rmin
            pair[1].position = rhigh
            pair[2].position = smoothing

    def setParameters(self, params):
        if params:
            self.user_changed = True

        self.order = params.get("order", self.ORDER_DEFAULT)
        self.scaling = params.get("scaling", self.SCALING_DEFAULT)
        self.output_model = params.get("output_model", self.OUTPUT_MODEL_DEFAULT)
        self._set_range_parameters(params)

        self.update_reference_info()
        self.update_weight_curve(params)

    def parameters(self):
        parameters = super().parameters()
        parameters["ranges"] = []
        for pair in self._range_widgets():
            parameters["ranges"].append(
                [
                    pair[0].position,
                    pair[1].position,
                    1.0,  # for now weight is always 1.0
                    pair[2].position,
                ]
            )
        return parameters

    @classmethod
    def _compute_weights(cls, params):
        weights = None
        ranges = params.get("ranges", [])

        def sel(l, r, w, s):
            if s is None:
                s = 0
            l, r = float(min(l, r)), float(max(l, r))
            if s < 1e-20:
                return SelectionFunction(l, r, w)
            else:
                return SmoothedSelectionFunction(l, r, s, w)

        if ranges:
            weights = Sum(*[sel(l, r, w, s) for l, r, w, s in ranges])
        return weights

    @classmethod
    def createinstance(cls, params):
        order = params.get("order", cls.ORDER_DEFAULT)
        scaling = params.get("scaling", cls.SCALING_DEFAULT)
        output_model = params.get("output_model", cls.OUTPUT_MODEL_DEFAULT)

        weights = cls._compute_weights(params)

        reference = params.get(REFERENCE_DATA_PARAM, None)
        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return EMSC(
                reference=reference,
                weights=weights,
                order=order,
                scaling=scaling,
                output_model=output_model,
            )

    def set_reference_data(self, reference):
        self.reference = reference
        self.update_reference_info()

    def update_reference_info(self):
        if not self.reference:
            self.reference_curve.hide()
            self.reference_info.setText("Reference: missing!")
            self.reference_info.setStyleSheet("color: red")
        else:
            rinfo = (
                "mean of %d spectra" % len(self.reference)
                if len(self.reference) > 1
                else "1 spectrum"
            )
            self.reference_info.setText("Reference: " + rinfo)
            self.reference_info.setStyleSheet("color: black")
            X_ref = spectra_mean(self.reference.X)
            x = getx(self.reference)
            xsind = np.argsort(x)
            self.reference_curve.setData(x=x[xsind], y=X_ref[xsind])
            self.reference_curve.setVisible(self.scaling)

    def update_weight_curve(self, params):
        weights = self._compute_weights(params)
        if weights is None:
            self.weight_curve.hide()
        else:
            pmin, pmax = self.preview_min_max()
            dist = pmax - pmin
            xs = np.linspace(pmin - dist / 2, pmax + dist / 2, 10000)
            ys = weights(xs)
            self.weight_curve.setData(x=xs, y=ys)
            self.weight_curve.show()

    def set_preview_data(self, data):
        self.preview_data = data
        # set all minumum and maximum defaults
        pmin, pmax = self.preview_min_max()
        for pair in self._range_widgets():
            pair[0].set_default(pmin)
            pair[1].set_default(pmax)
        if not self.user_changed:
            for pair in self._range_widgets():
                pair[0] = pmin
                pair[1] = pmax
            self.edited.emit()


preprocess_editors.register(EMSCEditor, 300)
