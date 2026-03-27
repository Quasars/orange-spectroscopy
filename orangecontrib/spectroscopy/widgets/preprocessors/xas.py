from decimal import Decimal

import numpy as np

from AnyQt.QtWidgets import QFormLayout, QLabel, QGridLayout
from AnyQt.QtGui import QFont
from extranormal3 import curved_tools

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import XASnormalization, ExtractEXAFS
from orangecontrib.spectroscopy.widgets.gui import (
    ValueTransform,
    connect_settings,
    float_to_str_decimals,
    lineEditFloatRange,
    connect_line,
    MovableVline,
)
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange


def init_bounds_hform(
    prepro_widget,
    from_lim,
    to_lim,
    title='',
    from_line=None,
    to_line=None,
    from_val_name=None,
    to_val_name=None,
):
    bounds_form = QGridLayout()

    title_font = QFont()
    title_font.setBold(True)

    titlabel = QLabel()
    titlabel.setFont(title_font)
    if title != '':
        titlabel.setText(title)
        bounds_form.addWidget(titlabel, 1, 0)

    left_bound = QFormLayout()
    left_bound.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
    left_bound.addRow("from", from_lim)
    right_bound = QFormLayout()
    right_bound.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
    right_bound.addRow("to", to_lim)

    bounds_form.addLayout(left_bound, 2, 0)
    # bounds_form.setHorizontalSpacing(5)
    bounds_form.addLayout(right_bound, 2, 1)

    from_lim.focusIn.connect(prepro_widget.activateOptions)
    to_lim.focusIn.connect(prepro_widget.activateOptions)
    prepro_widget.focusIn = prepro_widget.activateOptions

    if (
        from_line is not None
        and to_line is not None
        and from_val_name is not None
        and to_val_name is not None
    ):
        connect_line(from_line, prepro_widget, from_val_name)
        from_line.sigMoveFinished.connect(prepro_widget.edited)

        connect_line(to_line, prepro_widget, to_val_name)
        to_line.sigMoveFinished.connect(prepro_widget.edited)

    return bounds_form


class XASnormalizationEditor(BaseEditorOrange):
    name = "XAS normalization"
    qualname = "orangecontrib.infrared.xasnormalization"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.controlArea.setLayout(QGridLayout())
        curr_row = 0

        self.edge = 0.0
        edge_form = QFormLayout()
        edge_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        edge_edit = lineEditFloatRange(self, self, "edge", callback=self.edited.emit)
        edge_form.addRow("Edge", edge_edit)
        dummylabel = QLabel()
        dummylabel.setText('   ')
        edge_form.addWidget(dummylabel)  # adding vertical space
        self.controlArea.layout().addLayout(edge_form, curr_row, 0, 1, 1)
        curr_row += 1

        # ---------------------------- pre-edge form ------------
        self.preedge_from = self.preedge_to = 0.0
        self._pre_from_lim = lineEditFloatRange(
            self, self, "preedge_from", callback=self.edited.emit
        )
        self._pre_to_lim = lineEditFloatRange(
            self, self, "preedge_to", callback=self.edited.emit
        )
        self.pre_from_line = MovableVline(label="Pre-edge start")
        self.pre_to_line = MovableVline(label="Pre-edge end")

        preedge_form = init_bounds_hform(
            self,
            self._pre_from_lim,
            self._pre_to_lim,
            "Pre-edge fit:",
            self.pre_from_line,
            self.pre_to_line,
            "preedge_from",
            "preedge_to",
        )
        self.controlArea.layout().addLayout(preedge_form, curr_row, 0, 1, 2)
        curr_row += 1

        self.preedge_deg = 1.0
        preedgedeg_form = QFormLayout()
        preedgedeg_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        preedgedeg_edit = lineEditFloatRange(
            self, self, "preedge_deg", callback=self.edited.emit
        )
        preedgedeg_form.addRow("poly degree", preedgedeg_edit)
        dummylabel2 = QLabel()
        dummylabel2.setText('   ')
        preedgedeg_form.addWidget(dummylabel2)  # adding vertical space
        self.controlArea.layout().addLayout(preedgedeg_form, curr_row, 0, 1, 1)
        curr_row += 1

        # ---------------------------- post-edge form ------------
        self.postedge_from = self.postedge_to = 0.0
        self._post_from_lim = lineEditFloatRange(
            self, self, "postedge_from", callback=self.edited.emit
        )
        self._post_to_lim = lineEditFloatRange(
            self, self, "postedge_to", callback=self.edited.emit
        )
        self.post_from_line = MovableVline(label="Post-edge start")
        self.post_to_line = MovableVline(label="Post-edge end:")

        postedge_form = init_bounds_hform(
            self,
            self._post_from_lim,
            self._post_to_lim,
            "Post-edge fit:",
            self.post_from_line,
            self.post_to_line,
            "postedge_from",
            "postedge_to",
        )
        self.controlArea.layout().addLayout(postedge_form, curr_row, 0, 1, 2)
        curr_row += 1

        self.postedge_deg = 2.0
        postedgedeg_form = QFormLayout()
        postedgedeg_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        postedgedeg_edit = lineEditFloatRange(
            self, self, "postedge_deg", callback=self.edited.emit
        )
        postedgedeg_form.addRow("poly degree", postedgedeg_edit)
        self.controlArea.layout().addLayout(postedgedeg_form, curr_row, 0, 1, 1)
        curr_row += 1

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for line in [
            self.pre_from_line,
            self.pre_to_line,
            self.post_from_line,
            self.post_to_line,
        ]:
            line.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(line)

    def setParameters(self, params):
        if params:  # parameters were manually set somewhere else
            self.user_changed = True

        self.edge = params.get("edge", 0.0)

        self.preedge_from = params.get("preedge_from", 0.0)
        self.preedge_to = params.get("preedge_to", 0.0)
        self.preedge_deg = params.get("preedge_deg", 1)

        self.postedge_from = params.get("postedge_from", 0.0)
        self.postedge_to = params.get("postedge_to", 0.0)
        self.postedge_deg = params.get("postedge_deg", 2)

    def set_preview_data(self, data):
        if data is None:
            return

        x = getx(data)

        if len(x):
            self._pre_from_lim.set_default(min(x))
            self._pre_to_lim.set_default(max(x))
            self._post_from_lim.set_default(min(x))
            self._post_to_lim.set_default(max(x))

            if not self.user_changed:
                if data:
                    y = data.X[0]
                    maxderiv_idx = np.argmax(
                        curved_tools.derivative_vals(np.array([x, y]))
                    )
                    self.edge = x[maxderiv_idx]
                else:
                    self.edge = (max(x) - min(x)) / 2
                self.preedge_from = min(x)

                self.preedge_to = self.edge - 50
                self.postedge_from = self.edge + 50

                self.postedge_to = max(x)

                self.edited.emit()

    @staticmethod
    def createinstance(params):
        params = dict(params)

        edge = float(params.get("edge", 0.0))

        preedge = {}
        preedge['from'] = float(params.get("preedge_from", 0.0))
        preedge['to'] = float(params.get("preedge_to", 0.0))
        preedge['deg'] = int(params.get("preedge_deg", 1))

        postedge = {}
        postedge['from'] = float(params.get("postedge_from", 0.0))
        postedge['to'] = float(params.get("postedge_to", 0.0))
        postedge['deg'] = int(params.get("postedge_deg", 2))

        return XASnormalization(edge=edge, preedge_dict=preedge, postedge_dict=postedge)


class E2K(ValueTransform):
    def __init__(self, xas_prepro_widget):
        self.xas_prepro_widget = xas_prepro_widget

    def transform(self, v):
        res = np.sqrt(0.2625 * (float(v) - float(self.xas_prepro_widget.edge)))
        return Decimal(float_to_str_decimals(res, 2))

    def inverse(self, v):
        res = (float(v) ** 2) / 0.2625 + float(self.xas_prepro_widget.edge)
        return Decimal(float_to_str_decimals(res, 2))


class ExtractEXAFSEditor(BaseEditorOrange):
    name = "Polynomial EXAFS extraction"
    qualname = "orangecontrib.infrared.extractexafs"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        title_font = QFont()
        title_font.setBold(True)
        titlabel = QLabel()
        titlabel.setFont(title_font)

        self.controlArea.setLayout(QGridLayout())
        curr_row = 0

        self.edge = 0.0
        edge_form = QFormLayout()
        edge_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        edge_edit = lineEditFloatRange(self, self, "edge", callback=self.edited.emit)
        edge_form.addRow("Edge", edge_edit)
        dummylabel = QLabel()
        dummylabel.setText('   ')
        edge_form.addWidget(dummylabel)  # adding vertical space
        self.controlArea.layout().addLayout(edge_form, curr_row, 0, 1, 1)
        curr_row += 1

        self.extra_from = self.extra_to = 0.0
        self._extrafrom_lim = lineEditFloatRange(
            self, self, "extra_from", callback=self.edited.emit
        )
        self._extrato_lim = lineEditFloatRange(
            self, self, "extra_to", callback=self.edited.emit
        )
        self.extrafrom_line = MovableVline(label="Extraction start")
        self.extrato_line = MovableVline(label="Extraction end")

        extrabounds_form = init_bounds_hform(
            self,
            self._extrafrom_lim,
            self._extrato_lim,
            "Energy bounds:",
            self.extrafrom_line,
            self.extrato_line,
            "extra_from",
            "extra_to",
        )
        self.controlArea.layout().addLayout(extrabounds_form, curr_row, 0, 1, 2)
        curr_row += 1

        self.extra_fromK = self.extra_toK = 0.0
        self._extrafromK_lim = lineEditFloatRange(
            self, self, "extra_fromK", callback=self.edited.emit
        )
        self._extratoK_lim = lineEditFloatRange(
            self, self, "extra_toK", callback=self.edited.emit
        )
        Kbounds_form = init_bounds_hform(
            self, self._extrafromK_lim, self._extratoK_lim, "K bounds:"
        )
        self.controlArea.layout().addLayout(Kbounds_form, curr_row, 0, 1, 2)
        curr_row += 1

        connect_settings(self, "extra_from", "extra_fromK", transform=E2K(self))
        connect_settings(self, "extra_to", "extra_toK", transform=E2K(self))

        # ---------------------------
        self.poly_deg = 0
        polydeg_form = QFormLayout()
        polydeg_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        polydeg_edit = lineEditFloatRange(
            self, self, "poly_deg", callback=self.edited.emit
        )
        titlabel.setText("Polynomial degree:")
        polydeg_form.addRow(titlabel, polydeg_edit)
        dummylabel2 = QLabel()
        dummylabel2.setText('   ')
        polydeg_form.addWidget(dummylabel2)
        self.controlArea.layout().addLayout(polydeg_form, curr_row, 0, 1, 1)
        curr_row += 1
        # ----------------------------
        self.kweight = 0
        kweight_form = QFormLayout()
        kweight_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        kweight_edit = lineEditFloatRange(
            self, self, "kweight", callback=self.edited.emit
        )
        kweight_form.addRow("Kweight (fit)", kweight_edit)
        self.controlArea.layout().addLayout(kweight_form, curr_row, 0, 1, 1)
        curr_row += 1
        # ----------------------------
        self.m = 3
        m_form = QFormLayout()
        m_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        m_edit = lineEditFloatRange(self, self, "m", callback=self.edited.emit)
        m_edit.setMinimumWidth(10)
        m_form.addRow("Kweight (plot)", m_edit)
        self.controlArea.layout().addLayout(m_form, curr_row, 0, 1, 1)
        curr_row += 1

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for line in [self.extrafrom_line, self.extrato_line]:
            line.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(line)

    def setParameters(self, params):
        if params:  # parameters were manually set somewhere else
            self.user_changed = True

        self.edge = params.get("edge", 0.0)

        self.extra_from = params.get("extra_from", 0.0)
        self.extra_to = params.get("extra_to", 0.0)

        self.poly_deg = params.get("poly_deg", 0)
        self.kweight = params.get("kweight", 0)
        self.m = params.get("m", 0)

    def set_preview_data(self, data):
        if data is None:
            return

        x = getx(data)

        if len(x):
            self._extrafrom_lim.set_default(min(x))
            self._extrato_lim.set_default(max(x))

            if not self.user_changed:
                if data:
                    y = data.X[0]
                    maxderiv_idx = np.argmax(
                        curved_tools.derivative_vals(np.array([x, y]))
                    )
                    self.edge = x[maxderiv_idx]
                else:
                    self.edge = (max(x) - min(x)) / 2

                self.extra_from = self.edge
                self.extra_to = max(x)

                # check at least the vals go in a right order

                self.edited.emit()

    @staticmethod
    def createinstance(params):
        params = dict(params)

        edge = float(params.get("edge", 0.0))

        extra_from = float(params.get("extra_from", 0.0))
        extra_to = float(params.get("extra_to", 0.0))

        poly_deg = int(params.get("poly_deg", 0))
        kweight = int(params.get("kweight", 0))
        m = int(params.get("m", 0))

        return ExtractEXAFS(
            edge=edge,
            extra_from=extra_from,
            extra_to=extra_to,
            poly_deg=poly_deg,
            kweight=kweight,
            m=m,
        )


preprocess_editors.register(XASnormalizationEditor, 900)
preprocess_editors.register(ExtractEXAFSEditor, 925)
