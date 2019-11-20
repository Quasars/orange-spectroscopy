import random
from collections import Iterable
from decimal import Decimal

from extranormal3 import curved_tools

import numpy as np
import pyqtgraph as pg

import Orange.data
from Orange import preprocess
from Orange.widgets import gui, settings
from Orange.widgets.settings import SettingsHandler
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.data.utils.preprocess import SequenceFlow, Controller, \
    StandardItemModel
from Orange.widgets.data.owpreprocess import (
    PreprocessAction, Description, icon_path, DescriptionRole, ParametersRole, blocked
)
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.overlay import OverlayWidget
from Orange.widgets.utils.colorpalette import DefaultColorBrewerPalette

from AnyQt.QtCore import (
    Qt, QEvent, QSize, QMimeData, QTimer
)
from AnyQt.QtWidgets import (
    QWidget, QComboBox, QSpinBox,
    QListView, QVBoxLayout, QFormLayout, QSizePolicy, QStyle,
    QPushButton, QLabel, QMenu, QApplication, QAction, QScrollArea, QGridLayout,
    QToolButton, QSplitter
)
from AnyQt.QtGui import (
    QIcon, QStandardItemModel, QStandardItem,
    QKeySequence, QFont, QColor
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from orangecontrib.spectroscopy.data import getx

from orangecontrib.spectroscopy.preprocess import (
    PCADenoising, GaussianSmoothing, Cut, SavitzkyGolayFiltering,
    Absorbance, Transmittance, XASnormalization, ExtractEXAFS,
    CurveShift
)
from orangecontrib.spectroscopy.preprocess.transform import SpecTypes
from orangecontrib.spectroscopy.preprocess.utils import PreprocessException
from orangecontrib.spectroscopy.widgets.owspectra import CurvePlot, NoSuchCurve
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange, MovableVline, connect_line, floatornone, round_virtual_pixels
from orangecontrib.spectroscopy.widgets.preprocessors.baseline import BaselineEditor
from orangecontrib.spectroscopy.widgets.preprocessors.emsc import EMSCEditor
from orangecontrib.spectroscopy.widgets.preprocessors.integrate import IntegrateEditor
from orangecontrib.spectroscopy.widgets.preprocessors.me_emsc import MeEMSCEditor
from orangecontrib.spectroscopy.widgets.preprocessors.normalize import NormalizeEditor
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditor, BaseEditorOrange, \
    REFERENCE_DATA_PARAM
from orangecontrib.spectroscopy.widgets.gui import ValueTransform, connect_settings, float_to_str_decimals

PREVIEW_COLORS = [QColor(*a).name() for a in DefaultColorBrewerPalette[8]]


class ViewController(Controller):

    def createWidgetFor(self, index):
        w = super().createWidgetFor(index)
        w.parent_widget = self.parent()
        return w

    # ensure that view on the right
    # and the model are sychronized when on_modelchanged is called

    def _dataChanged(self, topleft, bottomright):
        super()._dataChanged(topleft, bottomright)
        self.parent().on_modelchanged()

    def _rowsInserted(self, parent, start, end):
        super()._rowsInserted(parent, start, end)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def _rowsRemoved(self, parent, start, end):
        super()._rowsRemoved(parent, start, end)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def _widgetMoved(self, from_, to):
        super()._widgetMoved(from_, to)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def setModel(self, model):
        super().setModel(model)
        self.view.reset_preview_colors()


class FocusFrame(SequenceFlow.Frame):
    preview_changed = Signal()

    def __init__(self, parent=None, **kwargs):
        self.title_label = None
        super().__init__(parent=parent, **kwargs)
        self.preview = False
        self.color = "lightblue"
        self._build_tw()
        self.setTitleBarWidget(self.tw)

    def _build_tw(self):
        self.tw = tw = QWidget(self)
        tl = QGridLayout(tw)
        self.title_label = QLabel(self._title, tw)
        self.title_label.setMinimumWidth(100)
        tl.addWidget(self.title_label, 0, 1)
        close_button = QToolButton(self)
        ca = QAction("close", self, triggered=self.closeRequested,
                     icon=QIcon(self.style().standardPixmap(QStyle.SP_DockWidgetCloseButton)))
        close_button.setDefaultAction(ca)
        self.preview_button = QToolButton(self)
        pa = QAction("preview", self, triggered=self.toggle_preview, checkable=True,
                     icon=QIcon(self.style().standardPixmap(QStyle.SP_MediaPlay)),
                     shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_P))
        pa.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(pa)
        self.preview_button.setDefaultAction(pa)
        self.set_preview(self.preview)
        tl.addWidget(close_button, 0, 0)
        tl.addWidget(self.preview_button, 0, 2)
        tl.setColumnStretch(1, 1)
        tl.setSpacing(2)
        tl.setContentsMargins(0, 0, 0, 0)
        tw.setLayout(tl)

    def set_preview(self, p):
        self.preview = p
        self.update_status()

    def set_color(self, c):
        self.color = c
        self.update_status()

    def update_status(self):
        self.preview_button.setChecked(self.preview)
        self.tw.setStyleSheet("background:" + self.color + ";" if self.preview else "")

    def toggle_preview(self):
        self.set_preview(not self.preview)
        self.preview_changed.emit()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        try: #active selection on preview
            self.widget().activateOptions()
        except AttributeError:
            pass

    def setTitle(self, title):
        self._title = title
        super().setTitle(title)
        if self.title_label:
            self.title_label.setText(title)


class SequenceFlow(SequenceFlow):
    """
    FIXME Ugly hack: using the same name for access to private variables!
    """
    def __init__(self, *args, preview_callback=None, multiple_previews=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_callback = preview_callback
        self.multiple_previews = multiple_previews
        self.preview_colors = multiple_previews

    def preview_n(self):
        """How many preprocessors to apply for the preview?"""
        ppos = [i for i, item in enumerate(self.layout_iter(self.__flowlayout)) if item.widget().preview]
        # if any, show the chosen preview
        if not self.multiple_previews:
            return ppos[-1] if ppos else None
        else:
            return ppos

    def set_preview_n(self, n):
        """Set the preview position"""
        n = set(n if isinstance(n, Iterable) else [n])
        for i, item in enumerate(self.layout_iter(self.__flowlayout)):
            f = item.widget()
            f.set_preview(i in n)

    def preview_changed(self):
        if not self.multiple_previews:  # disable other previews
            sender = self.sender()
            for item in self.layout_iter(self.__flowlayout):
                f = item.widget()
                if sender != f:
                    f.set_preview(False)

        self.preview_callback(show_info=True)

    def insertWidget(self, index, widget, title):
        """ Mostly copied to get different kind of frame """
        frame = FocusFrame(widget=widget, title=title) #changed
        frame.closeRequested.connect(self.__closeRequested)
        frame.preview_changed.connect(self.preview_changed)

        layout = self.__flowlayout

        frames = [item.widget() for item in self.layout_iter(layout)
                  if item.widget()]

        if 0 < index < len(frames):
            # find the layout index of a widget occupying the current
            # index'th slot.
            insert_index = layout.indexOf(frames[index])
        elif index == 0:
            insert_index = 0
        elif index < 0 or index >= len(frames):
            insert_index = layout.count()
        else:
            assert False

        layout.insertWidget(insert_index, frame)
        frame.installEventFilter(self)

    def __closeRequested(self):
        self.sender().widget().parent_widget.curveplot.clear_markings()
        super().__closeRequested()

    def minimumSizeHint(self):
        """ Add space below so that dragging to bottom works """
        psh = super().minimumSizeHint()
        return QSize(psh.width(), psh.height() + 100)

    def reset_preview_colors(self):
        if self.preview_colors:
            for i, item in enumerate(self.layout_iter(self.__flowlayout)):
                item = item.widget()
                item.set_color(PREVIEW_COLORS[i % len(PREVIEW_COLORS)])

    def preview_color(self, i):
        """ Return preview color of a specific widget. """
        w = self.__flowlayout.itemAt(i).widget()
        return w.color


class GaussianSmoothingEditor(BaseEditorOrange):
    """
    Editor for GaussianSmoothing
    """

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

    class Warning(BaseEditorOrange.Warning):
        out_of_range = Msg("Limits are out of range.")

    def __init__(self, parent=None, **kwargs):
        BaseEditorOrange.__init__(self, parent, **kwargs)

        self.lowlim = 0.
        self.highlim = 1.

        layout = QFormLayout()
        self.controlArea.setLayout(layout)

        self._lowlime = lineEditFloatRange(self, self, "lowlim", callback=self.edited.emit)
        self._highlime = lineEditFloatRange(self, self, "highlim", callback=self.edited.emit)

        layout.addRow("Low limit", self._lowlime)
        layout.addRow("High limit", self._highlime)

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

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        return Cut(lowlim=floatornone(lowlim), highlim=floatornone(highlim))

    def execute_instance(self, instance: Cut, data):
        self.Warning.out_of_range.clear()
        xs = getx(data)
        if len(xs):
            minx = np.min(xs)
            maxx = np.max(xs)
            if (instance.lowlim < minx and instance.highlim < minx) \
                    or (instance.lowlim > maxx and instance.highlim > maxx):
                self.parent_widget.Warning.preprocessor()
                self.Warning.out_of_range()
        return instance(data)

    def set_preview_data(self, data):
        x = getx(data)
        if len(x):
            range = max(x) - min(x)

            init_lowlim = round_virtual_pixels(min(x) + 0.1 * range, range)
            init_highlim = round_virtual_pixels(max(x) - 0.1 * range, range)

            self._lowlime.set_default(init_lowlim)
            self._highlime.set_default(init_highlim)

            if not self.user_changed:
                self.lowlim = init_lowlim
                self.highlim = init_highlim
                self.edited.emit()


class CutEditorInverse(CutEditor):

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        return Cut(lowlim=floatornone(lowlim), highlim=floatornone(highlim), inverse=True)


class SavitzkyGolayFilteringEditor(BaseEditorOrange):
    """
    Editor for preprocess.savitzkygolayfiltering.
    """

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


class CurveShiftEditor(BaseEditorOrange):
    """
    Editor for CurveShift
    """
    # TODO: the layout changes when I click the area of the preprocessor
    #       EFFECT: the sidebar snaps in

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.amount = 0.

        form = QFormLayout()
        amounte = lineEditFloatRange(self, self, "amount", callback=self.edited.emit)
        form.addRow("Shift Amount", amounte)
        self.controlArea.setLayout(form)

    def setParameters(self, params):
        self.amount = params.get("amount", 0.)

    @staticmethod
    def createinstance(params):
        params = dict(params)
        amount = float(params.get("amount", 0.))
        return CurveShift(amount=amount)


class PCADenoisingEditor(BaseEditor):

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        self.__components = 5

        form = QFormLayout()

        self.__compspin = compspin = QSpinBox(
            minimum=1, maximum=100, value=self.__components)
        form.addRow("N components", compspin)

        self.setLayout(form)

        compspin.valueChanged[int].connect(self.setComponents)
        compspin.editingFinished.connect(self.edited)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def setComponents(self, components):
        if self.__components != components:
            self.__components = components
            with blocked(self.__compspin):
                self.__compspin.setValue(components)
            self.changed.emit()

    def sd(self):
        return self.__components

    def setParameters(self, params):
        self.setComponents(params.get("components", 5))

    def parameters(self):
        return {"components": self.__components}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        components = params.get("components", 5)
        return PCADenoising(components=components)


class SpectralTransformEditor(BaseEditorOrange):

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


def init_bounds_hform(prepro_widget,
                      from_lim, to_lim,
                      title='',
                      from_line=None, to_line=None,
                      from_val_name=None, to_val_name=None):

    bounds_form = QGridLayout()

    title_font = QFont()
    title_font.setBold(True)

    titlabel = QLabel()
    titlabel.setFont(title_font)
    if title != '':
        titlabel.setText(title)
        bounds_form.addWidget(titlabel, 1, 0)

    left_bound = QFormLayout()
    left_bound.setFieldGrowthPolicy(0)
    left_bound.addRow("from", from_lim)
    right_bound = QFormLayout()
    right_bound.setFieldGrowthPolicy(0)
    right_bound.addRow("to", to_lim)

    bounds_form.addLayout(left_bound, 2, 0)
    # bounds_form.setHorizontalSpacing(5)
    bounds_form.addLayout(right_bound, 2, 1)

    from_lim.focusIn.connect(prepro_widget.activateOptions)
    to_lim.focusIn.connect(prepro_widget.activateOptions)
    prepro_widget.focusIn = prepro_widget.activateOptions

    if from_line is not None and to_line is not None and \
            from_val_name is not None and to_val_name is not None:
        connect_line(from_line, prepro_widget, from_val_name)
        from_line.sigMoveFinished.connect(prepro_widget.edited)

        connect_line(to_line, prepro_widget, to_val_name)
        to_line.sigMoveFinished.connect(prepro_widget.edited)

    return bounds_form


class XASnormalizationEditor(BaseEditorOrange):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.controlArea.setLayout(QGridLayout())
        curr_row = 0

        self.edge = 0.
        edge_form = QFormLayout()
        edge_form.setFieldGrowthPolicy(0)
        edge_edit = lineEditFloatRange(self, self, "edge", callback=self.edited.emit)
        edge_form.addRow("Edge", edge_edit)
        dummylabel = QLabel()
        dummylabel.setText('   ')
        edge_form.addWidget(dummylabel)  # adding vertical space
        self.controlArea.layout().addLayout(edge_form, curr_row, 0, 1, 1)
        curr_row += 1

        # ---------------------------- pre-edge form ------------
        self.preedge_from = self.preedge_to = 0.
        self._pre_from_lim = lineEditFloatRange(self, self, "preedge_from",
                                                callback=self.edited.emit)
        self._pre_to_lim = lineEditFloatRange(self, self, "preedge_to",
                                              callback=self.edited.emit)
        self.pre_from_line = MovableVline(label="Pre-edge start")
        self.pre_to_line = MovableVline(label="Pre-edge end")

        preedge_form = init_bounds_hform(self,
                                         self._pre_from_lim, self._pre_to_lim,
                                         "Pre-edge fit:",
                                         self.pre_from_line, self.pre_to_line,
                                         "preedge_from", "preedge_to")
        self.controlArea.layout().addLayout(preedge_form, curr_row, 0, 1, 2)
        curr_row += 1

        self.preedge_deg = 1.
        preedgedeg_form = QFormLayout()
        preedgedeg_form.setFieldGrowthPolicy(0)
        preedgedeg_edit = lineEditFloatRange(self, self, "preedge_deg",
                                             callback=self.edited.emit)
        preedgedeg_form.addRow("poly degree", preedgedeg_edit)
        dummylabel2 = QLabel()
        dummylabel2.setText('   ')
        preedgedeg_form.addWidget(dummylabel2)  # adding vertical space
        self.controlArea.layout().addLayout(preedgedeg_form, curr_row, 0, 1, 1)
        curr_row += 1

        # ---------------------------- post-edge form ------------
        self.postedge_from = self.postedge_to = 0.
        self._post_from_lim = lineEditFloatRange(self, self, "postedge_from",
                                                 callback=self.edited.emit)
        self._post_to_lim = lineEditFloatRange(self, self, "postedge_to",
                                               callback=self.edited.emit)
        self.post_from_line = MovableVline(label="Post-edge start")
        self.post_to_line = MovableVline(label="Post-edge end:")

        postedge_form = init_bounds_hform(self,
                                          self._post_from_lim, self._post_to_lim,
                                          "Post-edge fit:",
                                          self.post_from_line, self.post_to_line,
                                          "postedge_from", "postedge_to")
        self.controlArea.layout().addLayout(postedge_form, curr_row, 0, 1, 2)
        curr_row += 1

        self.postedge_deg = 2.
        postedgedeg_form = QFormLayout()
        postedgedeg_form.setFieldGrowthPolicy(0)
        postedgedeg_edit = lineEditFloatRange(self, self, "postedge_deg",
                                              callback=self.edited.emit)
        postedgedeg_form.addRow("poly degree", postedgedeg_edit)
        self.controlArea.layout().addLayout(postedgedeg_form, curr_row, 0, 1, 1)
        curr_row += 1

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for line in [self.pre_from_line, self.pre_to_line, self.post_from_line, self.post_to_line]:
            line.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(line)

    def setParameters(self, params):

        if params:  # parameters were manually set somewhere else
            self.user_changed = True

        self.edge = params.get("edge", 0.)

        self.preedge_from = params.get("preedge_from", 0.)
        self.preedge_to = params.get("preedge_to", 0.)
        self.preedge_deg = params.get("preedge_deg", 1)

        self.postedge_from = params.get("postedge_from", 0.)
        self.postedge_to = params.get("postedge_to", 0.)
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
                    maxderiv_idx = np.argmax(curved_tools.derivative_vals(np.array([x, y])))
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

        edge = float(params.get("edge", 0.))

        preedge = {}
        preedge['from'] = float(params.get("preedge_from", 0.))
        preedge['to'] = float(params.get("preedge_to", 0.))
        preedge['deg'] = int(params.get("preedge_deg", 1))

        postedge = {}
        postedge['from'] = float(params.get("postedge_from", 0.))
        postedge['to'] = float(params.get("postedge_to", 0.))
        postedge['deg'] = int(params.get("postedge_deg", 2))

        return XASnormalization(edge=edge, preedge_dict=preedge, postedge_dict=postedge)


class E2K(ValueTransform):

    def __init__(self, xas_prepro_widget):
        self.xas_prepro_widget = xas_prepro_widget

    def transform(self, v):
        res = np.sqrt(0.2625 * (float(v)-float(self.xas_prepro_widget.edge)))
        return Decimal(float_to_str_decimals(res, 2))

    def inverse(self, v):
        res = (float(v)**2)/0.2625+float(self.xas_prepro_widget.edge)
        return Decimal(float_to_str_decimals(res, 2))


class ExtractEXAFSEditor(BaseEditorOrange):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        title_font = QFont()
        title_font.setBold(True)
        titlabel = QLabel()
        titlabel.setFont(title_font)

        self.controlArea.setLayout(QGridLayout())
        curr_row = 0

        self.edge = 0.
        edge_form = QFormLayout()
        edge_form.setFieldGrowthPolicy(0)
        edge_edit = lineEditFloatRange(self, self, "edge", callback=self.edited.emit)
        edge_form.addRow("Edge", edge_edit)
        dummylabel = QLabel()
        dummylabel.setText('   ')
        edge_form.addWidget(dummylabel) # adding vertical space
        self.controlArea.layout().addLayout(edge_form, curr_row, 0, 1, 1)
        curr_row += 1

        self.extra_from = self.extra_to = 0.
        self._extrafrom_lim = lineEditFloatRange(self, self, "extra_from",
                                                 callback=self.edited.emit)
        self._extrato_lim = lineEditFloatRange(self, self, "extra_to",
                                               callback=self.edited.emit)
        self.extrafrom_line = MovableVline(label="Extraction start")
        self.extrato_line = MovableVline(label="Extraction end")

        extrabounds_form = init_bounds_hform(self,
                                             self._extrafrom_lim, self._extrato_lim,
                                             "Energy bounds:",
                                             self.extrafrom_line, self.extrato_line,
                                             "extra_from", "extra_to")
        self.controlArea.layout().addLayout(extrabounds_form, curr_row, 0, 1, 2)
        curr_row += 1

        self.extra_fromK = self.extra_toK = 0.
        self._extrafromK_lim = lineEditFloatRange(self, self, "extra_fromK",
                                                  callback=self.edited.emit)
        self._extratoK_lim = lineEditFloatRange(self, self, "extra_toK",
                                                callback=self.edited.emit)
        Kbounds_form = init_bounds_hform(self,
                                         self._extrafromK_lim, self._extratoK_lim,
                                         "K bounds:")
        self.controlArea.layout().addLayout(Kbounds_form, curr_row, 0, 1, 2)
        curr_row += 1

        connect_settings(self, "extra_from", "extra_fromK", transform=E2K(self))
        connect_settings(self, "extra_to", "extra_toK", transform=E2K(self))

        # ---------------------------
        self.poly_deg = 0
        polydeg_form = QFormLayout()
        polydeg_form.setFieldGrowthPolicy(0)
        polydeg_edit = lineEditFloatRange(self, self, "poly_deg", callback=self.edited.emit)
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
        kweight_form.setFieldGrowthPolicy(0)
        kweight_edit = lineEditFloatRange(self, self, "kweight", callback=self.edited.emit)
        kweight_form.addRow("Kweight (fit)", kweight_edit)
        self.controlArea.layout().addLayout(kweight_form, curr_row, 0, 1, 1)
        curr_row += 1
        # ----------------------------
        self.m = 3
        m_form = QFormLayout()
        m_form.setFieldGrowthPolicy(0)
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

        self.edge = params.get("edge", 0.)

        self.extra_from = params.get("extra_from", 0.)
        self.extra_to = params.get("extra_to", 0.)

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
                    maxderiv_idx = np.argmax(curved_tools.derivative_vals(np.array([x, y])))
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

        edge = float(params.get("edge", 0.))

        extra_from = float(params.get("extra_from", 0.))
        extra_to = float(params.get("extra_to", 0.))

        poly_deg = int(params.get("poly_deg", 0))
        kweight = int(params.get("kweight", 0))
        m = int(params.get("m", 0))

        return ExtractEXAFS(edge=edge, extra_from=extra_from, extra_to=extra_to,
                            poly_deg=poly_deg, kweight=kweight, m=m)


PREPROCESSORS = [
    PreprocessAction(
        "Cut (keep)", "orangecontrib.infrared.cut", "Cut",
        Description("Cut (keep)",
                    icon_path("Discretize.svg")),
        CutEditor
    ),
    PreprocessAction(
        "Cut (remove)", "orangecontrib.infrared.cutinverse", "Cut",
        Description("Cut (remove)",
                    icon_path("Discretize.svg")),
        CutEditorInverse
    ),
    PreprocessAction(
        "Gaussian smoothing", "orangecontrib.infrared.gaussian", "Gaussian smoothing",
        Description("Gaussian smoothing",
                    icon_path("Discretize.svg")),
        GaussianSmoothingEditor
    ),
    PreprocessAction(
        "Savitzky-Golay Filter", "orangecontrib.spectroscopy.savitzkygolay", "Smoothing",
        Description("Savitzky-Golay Filter",
                    icon_path("Discretize.svg")),
        SavitzkyGolayFilteringEditor
    ),
    PreprocessAction(
        "Baseline Correction", "orangecontrib.infrared.baseline", "Baseline Correction",
        Description("Baseline Correction",
                    icon_path("Discretize.svg")),
        BaselineEditor
    ),
    PreprocessAction(
        "Normalize Spectra", "orangecontrib.infrared.normalize", "Normalize Spectra",
        Description("Normalize Spectra",
                    icon_path("Normalize.svg")),
        NormalizeEditor
    ),
    PreprocessAction(
        "Integrate", "orangecontrib.infrared.integrate", "Integrate",
        Description("Integrate",
                    icon_path("Discretize.svg")),
        IntegrateEditor
    ),
    PreprocessAction(
        "PCA denoising", "orangecontrib.infrared.pca_denoising", "PCA denoising",
        Description("PCA denoising",
                    icon_path("Discretize.svg")),
        PCADenoisingEditor
    ),
    PreprocessAction(
        "Spectral Transformations",
        "orangecontrib.spectroscopy.transforms",
        "Spectral Transformations",
        Description("Spectral Transformations",
                    icon_path("Discretize.svg")),
        SpectralTransformEditor
    ),
    PreprocessAction(
        "Shift Spectra", "orangecontrib.infrared.curveshift", "Shift Spectra",
        Description("Shift Spectra",
                    icon_path("Discretize.svg")),
        CurveShiftEditor
    ),
    PreprocessAction(
        "EMSC", "orangecontrib.spectroscopy.preprocess.emsc", "EMSC",
        Description("EMSC",
                    icon_path("Discretize.svg")),
        EMSCEditor
    ),
    PreprocessAction(
        "ME-EMSC", "orangecontrib.spectroscopy.preprocess.me_emsc.me_emsc", "ME-EMSC",
        Description("ME-EMSC",
                    icon_path("Discretize.svg")),
        MeEMSCEditor
    ),
    PreprocessAction(
        "XAS normalization", "orangecontrib.infrared.xasnormalization", "XAS normalization",
        Description("XAS normalization",
                    icon_path("Discretize.svg")),
        XASnormalizationEditor
    ),
    PreprocessAction(
        "EXAFS extraction", "orangecontrib.infrared.extractexafs", "EXAFS extraction",
        Description("Polynomial EXAFS extraction",
                    icon_path("Discretize.svg")),
        ExtractEXAFSEditor
    ),
    ]


def migrate_preprocessor(preprocessor, version):
    """ Migrate a preprocessor. A preprocessor should migrate into a list of preprocessors. """
    name, settings = preprocessor
    settings = settings.copy()
    if name == "orangecontrib.infrared.rubberband" and version < 2:
        name = "orangecontrib.infrared.baseline"
        settings["baseline_type"] = 1
        version = 2
    if name == "orangecontrib.infrared.savitzkygolay" and version < 4:
        name = "orangecontrib.spectroscopy.savitzkygolay"
        # make window, polyorder, deriv valid, even if they were saved differently
        SGE = SavitzkyGolayFilteringEditor
        # some old versions saved these as floats
        window = int(settings.get("window", SGE.DEFAULT_WINDOW))
        polyorder = int(settings.get("polyorder", SGE.DEFAULT_POLYORDER))
        deriv = int(settings.get("deriv", SGE.DEFAULT_DERIV))
        if window % 2 == 0:
            window = window + 1
        window = max(min(window, SGE.MAX_WINDOW), SGE.MIN_WINDOW)
        polyorder = max(min(polyorder, window - 1), 0)
        deriv = max(min(SGE.MAX_DERIV, deriv, polyorder), 0)
        settings["window"] = window
        settings["polyorder"] = polyorder
        settings["deriv"] = deriv
        version = 4
    if name == "orangecontrib.infrared.absorbance" and version < 5:
        name = "orangecontrib.spectroscopy.transforms"
        settings["from_type"] = 1
        settings["to_type"] = 0
        version = 5
    if name == "orangecontrib.infrared.transmittance" and version < 5:
        name = "orangecontrib.spectroscopy.transforms"
        settings["from_type"] = 0
        settings["to_type"] = 1
        version = 5
    return [((name, settings), version)]


def migrate_preprocessor_list(preprocessors):
    pl = []
    for p, v in preprocessors:
        tl = migrate_preprocessor(p, v)
        if tl != [(p, v)]:  # if changed, try another migration
            tl = migrate_preprocessor_list(tl)
        pl.extend(tl)
    return pl


class TimeoutLabel(QLabel):
    """ A label that disappears out after two seconds. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        self.hide()

    def setText(self, t):
        super().setText(t)
        self.show()
        self.timer.start(2000)

    def _timeout(self):
        self.hide()


def transfer_highlight(from_: CurvePlot, to: CurvePlot):
    """Highlight the curve that is highlighted on "from_" also on "to"."""
    if from_.data is None or to.data is None:
        return
    highlight = None
    from_index = from_.highlighted_index_in_data()
    if from_index is not None:
        index_with_same_id = np.flatnonzero(to.data.ids == from_.data.ids[from_index])
        if len(index_with_same_id):
            highlight = index_with_same_id[0]
    try:
        to.highlight_index_in_data(highlight, emit=False)  # do not emit to avoid recursion
    except NoSuchCurve:
        pass


class PrepareSavingSettingsHandler(SettingsHandler):
    """Calls storeSpecificSettings, which is currently not called from non-context handlers."""

    def pack_data(self, widget):
        widget.storeSpecificSettings()
        return super().pack_data(widget)


class SpectralPreprocess(OWWidget, openclass=True):

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        preprocessed_data = Output("Preprocessed Data", Orange.data.Table, default=True)
        preprocessor = Output("Preprocessor", preprocess.preprocess.Preprocess)

    settingsHandler = PrepareSavingSettingsHandler()

    storedsettings = settings.Setting({}, schema_only=True)
    autocommit = settings.Setting(False)
    preview_curves = settings.Setting(3)
    preview_n = settings.Setting(None, schema_only=True)

    # compatibility for old workflows when reference was not processed
    process_reference = settings.Setting(True, schema_only=True)

    curveplot = settings.SettingProvider(CurvePlot)
    curveplot_after = settings.SettingProvider(CurvePlot)

    # draw preview on top of current image
    preview_on_image = False

    _max_preview_spectra = 10

    class Error(OWWidget.Error):
        applying = Msg("Preprocessing error. {}")
        preview = Msg("Preview error. {}")
        preprocessor = Msg("Preprocessor error: see the widget for details.")

    class Warning(OWWidget.Warning):
        reference_compat = Msg("Reference is not processed for compatibility with the loaded "
                               "workflow. New instances of this widget will also process "
                               "the reference input.")
        preprocessor = Msg("Preprocessor warning: see the widget for details.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.reference_data = None
        self._invalidated = False

        # List of available preprocessors (DescriptionRole : Description)
        self.preprocessors = QStandardItemModel()

        def mimeData(indexlist):
            assert len(indexlist) == 1
            index = indexlist[0]
            qname = index.data(DescriptionRole).qualname
            m = QMimeData()
            m.setData("application/x-qwidget-ref", qname)
            return m
        # TODO: Fix this (subclass even if just to pass a function
        # for mimeData delegate)
        self.preprocessors.mimeData = mimeData

        self.button = QPushButton(self.BUTTON_ADD_LABEL, self)
        self.controlArea.layout().addWidget(self.button)
        self.preprocessor_menu = QMenu(self)
        self.button.setMenu(self.preprocessor_menu)
        self.button.setAutoDefault(False)

        self.preprocessorsView = view = QListView(
            selectionMode=QListView.SingleSelection,
            dragEnabled=True,
            dragDropMode=QListView.DragOnly
        )

        self._qname2ppdef = {ppdef.qualname: ppdef for ppdef in self.PREPROCESSORS}

        # List of 'selected' preprocessors and their parameters.
        self.preprocessormodel = None

        self.flow_view = SequenceFlow(preview_callback=self.show_preview,
                                      multiple_previews=self.preview_on_image)
        self.controler = ViewController(self.flow_view, parent=self)

        self.scroll_area = QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)

        self.flow_view.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.scroll_area.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        self.curveplot = CurvePlot(self)
        self.curveplot_after = CurvePlot(self)
        self.curveplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        self.curveplot_after.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden

        splitter.addWidget(self.curveplot)
        splitter.addWidget(self.curveplot_after)
        self.mainArea.layout().addWidget(splitter)

        def overlay(widget):
            o = OverlayWidget(self)
            o.setAttribute(Qt.WA_TransparentForMouseEvents)
            o.setWidget(widget)
            o.setLayout(QVBoxLayout(o))
            l = TimeoutLabel("", parent=o, wordWrap=True)
            l.setAlignment(Qt.AlignCenter)
            font = QFont()
            font.setPointSize(20)
            l.setFont(font)
            l.setStyleSheet("color: lightblue")
            o.layout().addWidget(l)
            return l

        self.curveplot_info = overlay(self.curveplot)
        self.curveplot_after_info = overlay(self.curveplot_after)

        self.curveplot.highlight_changed.connect(
            lambda: transfer_highlight(self.curveplot, self.curveplot_after))
        self.curveplot_after.highlight_changed.connect(
            lambda: transfer_highlight(self.curveplot_after, self.curveplot))

        if not self.preview_on_image:
            self.curveplot_after.show()
        else:
            self.curveplot_after.hide()

        self.controlArea.layout().addWidget(self.scroll_area)
        self.mainArea.layout().addWidget(splitter)

        self.flow_view.installEventFilter(self)

        box = gui.widgetBox(self.controlArea, "Preview")
        self.final_preview_toggle = False
        if not self.preview_on_image:
            self.final_preview = gui.button(
                box, self, "Final preview", self.flow_view.preview_changed,
                toggleButton=True, value="final_preview_toggle", autoDefault=False)
        gui.spin(box, self, "preview_curves", 1, self._max_preview_spectra, label="Show spectra",
                 callback=self._update_preview_number)

        self.output_box = gui.widgetBox(self.controlArea, "Output")
        b = gui.auto_commit(self.output_box, self, "autocommit", "Commit", box=False)
        b.button.setAutoDefault(False)

        self._initialize()

    def _update_preview_number(self):
        self.show_preview(show_info=False)

    def sample_data(self, data):
        if data is not None and len(data) > self.preview_curves:
            sampled_indices = random.Random(0).sample(range(len(data)), self.preview_curves)
            return data[sampled_indices]
        else:
            return self.data

    def _reference_compat_warning(self):
        self.Warning.reference_compat.clear()
        if not self.process_reference and self.reference_data is not None:
            self.Warning.reference_compat()

    def show_preview(self, show_info=False):
        """ Shows preview and also passes preview data to the widgets """
        self._reference_compat_warning()
        self.Warning.preprocessor.clear()
        self.Error.preprocessor.clear()
        self.Error.preview.clear()

        widgets = self.flow_view.widgets()
        for w in widgets:
            if getattr(w, "Error", None):  # only BaseEditorOrange supports errors
                w.Error.exception.clear()

        if self.data is not None:
            orig_data = data = self.sample_data(self.data)
            reference_data = self.reference_data
            preview_pos = self.flow_view.preview_n()
            n = self.preprocessormodel.rowCount()

            preview_data = None
            after_data = None

            for i in range(n):
                if preview_pos == i:
                    preview_data = data

                widgets[i].set_reference_data(reference_data)
                widgets[i].set_preview_data(data)
                item = self.preprocessormodel.item(i)
                try:
                    preproc = self._create_preprocessor(item, reference_data)
                    data = widgets[i].execute_instance(preproc, data)
                    if self.process_reference and reference_data is not None and i != n - 1:
                        reference_data = preproc(reference_data)
                except PreprocessException as e:
                    widgets[i].Error.exception(e.message())
                    self.Error.preview(e.message())
                    data = None
                    break

                if preview_pos == i:
                    after_data = data
                    if show_info:
                        current_name = item.data(DescriptionRole).description.title
                        self.curveplot_info.setText('Input to "' + current_name + '"')
                        self.curveplot_after_info.setText('Output of "' + current_name + '"')

            if preview_data is None:  # show final result
                preview_data = orig_data
                if not self.preview_on_image:
                    after_data = data
                    self.final_preview_toggle = True
                    self.curveplot_info.setText('Original data')
                    self.curveplot_after_info.setText('Preprocessed data')
            elif not self.preview_on_image:
                self.final_preview_toggle = False

            self.curveplot.set_data(preview_data)
            self.curveplot_after.set_data(after_data)
        else:
            self.curveplot.set_data(None)
            self.curveplot_after.set_data(None)

    def _initialize(self):
        for pp_def in self.PREPROCESSORS:
            description = pp_def.description
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item = QStandardItem(icon, description.title)
            item.setToolTip(description.summary or "")
            item.setData(pp_def, DescriptionRole)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                          Qt.ItemIsDragEnabled)
            self.preprocessors.appendRow([item])
            action = QAction(
                description.title, self, triggered=lambda x, p=pp_def: self.add_preprocessor(p)
            )
            action.setToolTip(description.summary or "")
            action.setIcon(icon)
            self.preprocessor_menu.addAction(action)

        try:
            model = self.load(self.storedsettings)
        except Exception:
            model = self.load({})

        self.set_model(model)
        self.flow_view.set_preview_n(self.preview_n)

        if not model.rowCount():
            # enforce default width constraint if no preprocessors
            # are instantiated (if the model is not empty the constraints
            # will be triggered by LayoutRequest event on the `flow_view`)
            self.__update_size_constraint()

        # call apply() to output the preprocessor even if there is no input data
        self.unconditional_commit()

    def load(self, saved):
        """Load a preprocessor list from a dict."""
        name = saved.get("name", "")
        preprocessors = saved.get("preprocessors", [])
        model = StandardItemModel()

        def dropMimeData(data, action, row, column, parent):
            if data.hasFormat("application/x-qwidget-ref") and \
                    action == Qt.CopyAction:
                qname = bytes(data.data("application/x-qwidget-ref")).decode()

                ppdef = self._qname2ppdef[qname]
                item = QStandardItem(ppdef.description.title)
                item.setData({}, ParametersRole)
                item.setData(ppdef.description.title, Qt.DisplayRole)
                item.setData(ppdef, DescriptionRole)
                self.preprocessormodel.insertRow(row, [item])
                return True
            else:
                return False

        model.dropMimeData = dropMimeData

        for qualname, params in preprocessors:
            pp_def = self._qname2ppdef[qualname]
            description = pp_def.description
            item = QStandardItem(description.title)
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item.setIcon(icon)
            item.setToolTip(description.summary)
            item.setData(pp_def, DescriptionRole)
            item.setData(params, ParametersRole)

            model.appendRow(item)
        return model

    def save(self, model):
        """Save the preprocessor list to a dict."""
        d = {"name": ""}
        preprocessors = []
        for i in range(model.rowCount()):
            item = model.item(i)
            pp_def = item.data(DescriptionRole)
            params = item.data(ParametersRole)
            preprocessors.append((pp_def.qualname, params))

        d["preprocessors"] = preprocessors
        return d

    def set_model(self, ppmodel):
        if self.preprocessormodel:
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = ppmodel
        self.controler.setModel(ppmodel)

    def on_modelchanged(self):
        self.show_preview()
        self.commit()

    @Inputs.data
    @check_sql_input
    def set_data(self, data=None):
        """Set the input data set."""
        self.data = data

    def handleNewSignals(self):
        self.show_preview(True)
        self.unconditional_commit()

    def add_preprocessor(self, action):
        item = QStandardItem()
        item.setData({}, ParametersRole)
        item.setData(action.description.title, Qt.DisplayRole)
        item.setData(action, DescriptionRole)
        self.preprocessormodel.appendRow([item])

    def _prepare_params(self, params, reference):
        if not isinstance(params, dict):
            params = {}
        # add optional reference data
        params[REFERENCE_DATA_PARAM] = reference
        return params

    def _create_preprocessor(self, item, reference):
        desc = item.data(DescriptionRole)
        params = item.data(ParametersRole)
        params = self._prepare_params(params, reference)
        create = desc.viewclass.createinstance
        return create(params)

    def create_outputs(self):
        raise NotImplementedError()

    def apply(self):
        self.show_preview()
        self.Error.applying.clear()
        try:
            data, preprocessor = self.create_outputs()
        except PreprocessException as e:
            self.Error.applying(e.message())
            data, preprocessor = None, None
        self.Outputs.preprocessor.send(preprocessor)
        self.Outputs.preprocessed_data.send(data)

    def commit(self):
        # Do not run() apply immediately: delegate it to the event loop.
        # Protects against running apply() in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Now, apply() will usually be only called once.
        if not self._invalidated:
            self._invalidated = True
            QApplication.postEvent(self, QEvent(QEvent.User))

    def customEvent(self, event):
        if event.type() == QEvent.User and self._invalidated:
            self._invalidated = False
            self.apply()

    def eventFilter(self, receiver, event):
        if receiver is self.flow_view and event.type() == QEvent.LayoutRequest:
            QTimer.singleShot(0, self.__update_size_constraint)

        return super().eventFilter(receiver, event)

    def storeSpecificSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        self.preview_n = self.flow_view.preview_n()
        super().storeSpecificSettings()

    def onDeleteWidget(self):
        self.data = None
        self.set_model(None)
        super().onDeleteWidget()

    @Slot()
    def __update_size_constraint(self):
        # Update minimum width constraint on the scroll area containing
        # the 'instantiated' preprocessor list (to avoid the horizontal
        # scroll bar).
        sh = self.flow_view.minimumSizeHint()
        self.scroll_area.setMinimumWidth(max(sh.width() + 50, 200))

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(sh.width(), 500))

    @classmethod
    def migrate_preprocessors(cls, preprocessors, version):
        input = list(zip(preprocessors, [version]*len(preprocessors)))
        migrated = migrate_preprocessor_list(input)
        return [p[0] for p in migrated], cls.settings_version

    @classmethod
    def migrate_settings(cls, settings_, version):
        # For backwards compatibility, set process_reference=False
        # but only if there were multiple preprocessors
        if "process_reference" not in settings_:
            settings_["process_reference"] = not(
                version <= 5
                and "storedsettings" in settings_
                and "preprocessors" in settings_["storedsettings"]
                and len(settings_["storedsettings"]["preprocessors"]) > 1
            )

        # migrate individual preprocessors
        if "storedsettings" in settings_ and "preprocessors" in settings_["storedsettings"]:
            settings_["storedsettings"]["preprocessors"], _ = \
                cls.migrate_preprocessors(settings_["storedsettings"]["preprocessors"], version)


class SpectralPreprocessReference(SpectralPreprocess, openclass=True):

    class Inputs(SpectralPreprocess.Inputs):
        reference = Input("Reference", Orange.data.Table)

    @Inputs.reference
    def set_reference(self, reference):
        self.reference_data = reference


class OWPreprocess(SpectralPreprocessReference):

    name = "Preprocess Spectra"
    description = "Construct a data preprocessing pipeline."
    icon = "icons/preprocess.svg"
    priority = 1000
    replaces = ["orangecontrib.infrared.widgets.owpreproc.OWPreprocess",
                "orangecontrib.infrared.widgets.owpreprocess.OWPreprocess"]

    settings_version = 6

    BUTTON_ADD_LABEL = "Add preprocessor..."
    PREPROCESSORS = PREPROCESSORS

    _max_preview_spectra = 100

    preview_curves = settings.Setting(25)

    # draw preview on top of current image
    preview_on_image = False

    def create_outputs(self):
        self._reference_compat_warning()
        plist = []
        data = self.data
        reference = self.reference_data
        n = self.preprocessormodel.rowCount()
        for i in range(n):
            item = self.preprocessormodel.item(i)
            pp = self._create_preprocessor(item, reference)
            plist.append(pp)
            if data is not None:
                data = pp(data)
            if self.process_reference and reference is not None and i != n - 1:
                reference = pp(reference)
        # output None if there are no preprocessors
        preprocessor = preprocess.preprocess.PreprocessorList(plist) if plist else None
        return data, preprocessor


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    data = Orange.data.Table("collagen.csv")
    WidgetPreview(OWPreprocess).run(set_data=data[:3], set_reference=data[10:11])
