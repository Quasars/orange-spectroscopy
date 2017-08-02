import random
import sys
from itertools import chain

import Orange.data
import Orange.widgets.data.owpreprocess as owpreprocess
import pyqtgraph as pg
from Orange import preprocess
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets.data.owpreprocess import (
    Controller, StandardItemModel,
    PreprocessAction, Description, icon_path, DescriptionRole, ParametersRole, BaseEditor, blocked
)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.sql import check_sql_input

from AnyQt.QtCore import (
    Qt, QObject, QEvent, QSize, QMimeData, QTimer
)
from AnyQt.QtWidgets import (
    QWidget, QButtonGroup, QRadioButton, QDoubleSpinBox, QComboBox, QSpinBox,
    QListView, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QStyle,
    QStylePainter, QStyleOptionFrame, QApplication, QPushButton, QLabel,
    QMenu, QApplication, QAction, QDockWidget, QScrollArea, QCheckBox
)
from AnyQt.QtGui import (
    QCursor, QIcon, QPainter, QPixmap, QStandardItemModel, QStandardItem,
    QDrag, QKeySequence
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.preprocess import PCADenoising, GaussianSmoothing, Cut, SavitzkyGolayFiltering, \
    RubberbandBaseline, Normalize, Integrate, Absorbance, Transmittance, Average
from orangecontrib.infrared.widgets.owcurves import CurvePlot


class ViewController(Controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._insertIndexAt = self.view.insertIndexAt

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
        self.parent().on_modelchanged()

    def _rowsRemoved(self, parent, start, end):
        super()._rowsRemoved(parent, start, end)
        self.parent().on_modelchanged()

    def _widgetMoved(self, from_, to):
        super()._widgetMoved(from_, to)
        self.parent().on_modelchanged()


class FocusFrame(owpreprocess.SequenceFlow.Frame):

    def focusInEvent(self, event):
        super().focusInEvent(event)
        try: #active selection on preview
            self.widget().activateOptions()
        except AttributeError:
            pass


class PreviewFrame(owpreprocess.SequenceFlow.Frame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFeatures(QDockWidget.DockWidgetMovable)
        self.setStyleSheet("""QDockWidget::title {
                              background:lightblue;
                              padding: 10px;
                              text-align: right;
                              }""");


class PreviewWidget(QWidget):
    pass


class SequenceFlow(owpreprocess.SequenceFlow):
    """
    FIXME Ugly hack: using the same name for access to private variables!
    """
    def __init__(self, *args, preview_callback=None, show_preview=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_callback = preview_callback
        self.show_preview = show_preview

    def clear(self):
        super().clear()
        self.__preview_widget = PreviewWidget()
        self.__preview_frame = PreviewFrame(widget=self.__preview_widget, title="Preview")
        self.__preview_frame.hide()
        self.__preview_frame.installEventFilter(self)

    def __preview_position(self):
        """ Return -1 if not fount """
        return self.__flowlayout.indexOf(self.__preview_frame)

    def preview_n(self):
        """How many preprocessors to apply for the preview?"""
        return max(self.__preview_position(), 0)

    def set_preview_n(self, n):
        """Set the preview position"""
        oldindex = self.__preview_position()
        if oldindex >= 0:
            layout = self.__flowlayout
            n = max(min(n, layout.count() - 1), 0)
            if n != oldindex:
                insertindex = n
                if n >= oldindex:
                    insertindex = n - 1
                item = layout.takeAt(oldindex)
                layout.insertWidget(insertindex, item.widget())

    def __initPreview(self):
        if self.__preview_position() == -1:
            index = len(self.widgets())
            self.__flowlayout.insertWidget(index, self.__preview_frame)
            if self.show_preview:
                self.__preview_frame.show()

    def insertWidget(self, index, widget, title):
        """ Mostly copied to get different kind of frame """
        self.__initPreview() #added
        if index > self.__preview_position(): #added
            index = index + 1 #added

        frame = FocusFrame(widget=widget, title=title) #changed
        frame.closeRequested.connect(self.__closeRequested)

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

    def removeWidget(self, widget):
        """ Remove preview when empty. """
        layout = self.__flowlayout
        super().removeWidget(widget)

        #remove preview if only preview is there
        if not self.widgets() and layout.count() == 1:
            w = layout.takeAt(0)
            w.widget().hide()

    def dropEvent(self, event):
        """ FIXME Possible without complete reimplementation? """
        layout = self.__flowlayout
        index = self.__insertIndexAt(self.mapFromGlobal(QCursor.pos()))

        if event.mimeData().hasFormat("application/x-internal-move") and \
                        event.source() is self:
            # Complete the internal move
            frame, oldindex, _ = self.__dragstart
            # Remove the drop indicator spacer item before re-inserting
            # the frame
            self.__setDropIndicatorAt(None)

            ppos = self.__preview_position()

            insertindex = index
            if index > oldindex:
                insertindex = index - 1

            if insertindex != oldindex:
                item = layout.takeAt(oldindex)
                assert item.widget() is frame
                layout.insertWidget(insertindex, frame)
                if oldindex != ppos:
                    movefrom = oldindex
                    moveto = index
                    if movefrom > ppos:
                        movefrom = movefrom - 1
                    if moveto > ppos:
                        moveto = moveto - 1
                    if moveto > movefrom:
                        moveto = moveto - 1
                    if movefrom != moveto:
                        self.widgetMoved.emit(movefrom, moveto)
                else:
                    #preview was moved
                    self.preview_callback()
                event.accept()

            self.__dragstart = None, None, None

    def widgets(self):
        widgets = super().widgets()
        return [w for w in widgets if w != self.__preview_widget]

    def eventFilter(self, obj, event):
        """Needed to modify because it used indexOf."""
        if isinstance(obj, SequenceFlow.Frame) and obj.parent() is self:
            etype = event.type()
            if etype == QEvent.MouseButtonPress and \
                            event.button() == Qt.LeftButton:
                # Is the mouse press on the dock title bar
                # (assume everything above obj.widget is a title bar)
                # TODO: Get the proper title bar geometry.
                if event.pos().y() < obj.widget().y():
                    #index = self.indexOf(obj.widget()) #remove indexOf usage
                    index = self.__flowlayout.indexOf(obj)
                    self.__dragstart = (obj, index, event.pos())
            elif etype == QEvent.MouseMove and \
                            event.buttons() & Qt.LeftButton and \
                            obj is self.__dragstart[0]:
                _, _, down = self.__dragstart
                if (down - event.pos()).manhattanLength() >= \
                        QApplication.startDragDistance():
                    self.__startInternalDrag(obj, event.pos())
                    self.__dragstart = None, None, None
                    return True
            elif etype == QEvent.MouseButtonRelease and \
                            event.button() == Qt.LeftButton and \
                            self.__dragstart[0] is obj:
                self.__dragstart = None, None, None

        return QWidget.eventFilter(self, obj, event)

    def insertIndexAt(self, pos):
        index = self.__insertIndexAt(pos)
        ppos = self.__preview_position()
        if ppos >= 0 and index > ppos:
            index = index - 1
        return index

    def __closeRequested(self):
        self.sender().widget().parent_widget.curveplot.clear_markings()
        super().__closeRequested()

    def minimumSizeHint(self):
        """ Add space below so that dragging to bottom works """
        psh = super().minimumSizeHint()
        return QSize(psh.width(), psh.height() + 100)


class GaussianSmoothingEditor(BaseEditor):
    """
    Editor for GausianSmoothing
    """

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        self.__sd = 10.

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.__sdspin = sdspin = QDoubleSpinBox(
           minimum=0.0, maximum=100.0, singleStep=0.5, value=self.__sd)
        layout.addWidget(sdspin)

        sdspin.valueChanged[float].connect(self.setSd)
        sdspin.editingFinished.connect(self.edited)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def setSd(self, sd):
        if self.__sd != sd:
            self.__sd = sd
            with blocked(self.__sdspin):
                self.__sdspin.setValue(sd)
            self.changed.emit()

    def sd(self):
        return self.__sd

    def intervals(self):
        return self.__nintervals

    def setParameters(self, params):
        self.setSd(params.get("sd", 10.))

    def parameters(self):
        return {"sd": self.__sd}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        sd = params.get("sd", 10.)
        return GaussianSmoothing(sd=sd)


class SetXDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)



class MovableVlineWD(pg.UIGraphicsItem):

    sigRegionChangeFinished = Signal(object)
    sigRegionChanged = Signal(object)
    Vertical = 0
    Horizontal = 1

    def __init__(self, position, label="", setvalfn=None, confirmfn=None,
                 color=(225, 0, 0), report=None):
        pg.UIGraphicsItem.__init__(self)
        self.moving = False
        self.mouseHovering = False
        self.report = report

        hp = pg.mkPen(color=color, width=3)
        np = pg.mkPen(color=color, width=2)
        self.line = pg.InfiniteLine(angle=90, movable=True, pen=np, hoverPen=hp)

        if position is not None:
            self.line.setValue(position)
        else:
            self.line.setValue(0)
            self.line.hide()
        self.line.setCursor(Qt.SizeHorCursor)

        self.line.setParentItem(self)
        self.line.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.line.sigPositionChanged.connect(self.lineMoved)

        self.label = pg.TextItem("", anchor=(0,0))
        self.label.setText(label, color=color)
        self.label.setParentItem(self)

        self.setvalfn = setvalfn
        self.confirmfn = confirmfn

        self.lastTransform = None

    def value(self):
        return self.line.value()

    def setValue(self, val):
        if self.line.isVisible() and val == self.line.value():
            # no effective change
            return
        if val is not None:
            self.line.setValue(val)
            self.line.show()
            self.lineMoveFinished()
        else:
            self.line.hide()

    def boundingRect(self):
        br = pg.UIGraphicsItem.boundingRect(self)
        val = self.value()
        br.setLeft(val)
        br.setRight(val)
        return br.normalized()

    def _move_label(self):
        if self.getViewBox():
            self.label.setPos(self.value(), self.getViewBox().viewRect().bottom())

    def lineMoved(self):
        self._move_label()
        if self.report:
            self.report.report(self, [("x", self.value())])
        if self.setvalfn:
            self.setvalfn(self.value())

    def lineMoveFinished(self):
        if self.report:
            self.report.report_finished(self)
        if self.setvalfn:
            self.setvalfn(self.value())
        if self.confirmfn:
            if hasattr(self.confirmfn, "emit"):
                self.confirmfn.emit()
            else:
                self.confirmfn()

    def paint(self, p, *args):
        tr = p.transform()
        if self.lastTransform != tr:
            self._move_label()
        self.lastTransform = tr
        super().paint(p, *args)


class CutEditor(BaseEditor):
    """
    Editor for Cut
    """

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        self.__lowlim = 0.
        self.__highlim = 1.

        layout = QFormLayout()

        self.setLayout(layout)

        minf,maxf = -sys.float_info.max, sys.float_info.max

        self.__lowlime = SetXDoubleSpinBox(decimals=4,
            minimum=minf, maximum=maxf, singleStep=0.5, value=self.__lowlim)
        self.__highlime = SetXDoubleSpinBox(decimals=4,
            minimum=minf, maximum=maxf, singleStep=0.5, value=self.__highlim)

        layout.addRow("Low limit", self.__lowlime)
        layout.addRow("High limit", self.__highlime)

        self.__lowlime.focusIn = self.activateOptions
        self.__highlime.focusIn = self.activateOptions
        self.focusIn = self.activateOptions

        self.__lowlime.valueChanged[float].connect(self.set_lowlim)
        self.__highlime.valueChanged[float].connect(self.set_highlim)
        self.__lowlime.editingFinished.connect(self.edited)
        self.__highlime.editingFinished.connect(self.edited)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.line1 = MovableVlineWD(position=self.__lowlim, label="Low limit", setvalfn=self.set_lowlim, confirmfn=self.edited)
        self.line2 = MovableVlineWD(position=self.__highlim, label="High limit", setvalfn=self.set_highlim, confirmfn=self.edited)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.line1 not in self.parent_widget.curveplot.markings:
            self.line1.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(self.line1)
        if self.line2 not in self.parent_widget.curveplot.markings:
            self.line2.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(self.line2)

    def set_lowlim(self, lowlim, user=True):
        if user:
            self.user_changed = True
        if self.__lowlim != lowlim:
            self.__lowlim = lowlim
            with blocked(self.__lowlime):
                self.__lowlime.setValue(lowlim)
                self.line1.setValue(lowlim)
            self.changed.emit()

    def left(self):
        return self.__lowlim

    def set_highlim(self, highlim, user=True):
        if user:
            self.user_changed = True
        if self.__highlim != highlim:
            self.__highlim = highlim
            with blocked(self.__highlime):
                self.__highlime.setValue(highlim)
                self.line2.setValue(highlim)
            self.changed.emit()

    def right(self):
        return self.__highlim

    def setParameters(self, params):
        if params: #parameters were manually set somewhere else
            self.user_changed = True
        self.set_lowlim(params.get("lowlim", 0.), user=False)
        self.set_highlim(params.get("highlim", 1.), user=False)

    def parameters(self):
        return {"lowlim": self.__lowlim, "highlim": self.__highlim}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        return Cut(lowlim=lowlim, highlim=highlim)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_lowlim(min(x))
                self.set_highlim(max(x))
                self.edited.emit()


class CutEditorInverse(CutEditor):

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        return Cut(lowlim=lowlim, highlim=highlim, inverse=True)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                avg = (min(x) + max(x))/2
                self.set_lowlim(avg)
                self.set_highlim(avg)
                self.edited.emit()


class SavitzkyGolayFilteringEditor(BaseEditor):
    """
    Editor for preprocess.savitzkygolayfiltering.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.window = 5
        self.polyorder = 2
        self.deriv = 0

        form = QFormLayout()

        self.wspin = QSpinBox(
            minimum=3, maximum=100, singleStep=2,
            value=self.window)
        self.wspin.valueChanged[int].connect(self.setW)
        self.wspin.editingFinished.connect(self.edited)

        self.pspin = QSpinBox(
            minimum=2, maximum=self.window, singleStep=1,
            value=self.polyorder)
        self.pspin.valueChanged[int].connect(self.setP)
        self.pspin.editingFinished.connect(self.edited)

        self.dspin = QSpinBox(
            minimum=0, maximum=3, singleStep=1,
            value=self.deriv)
        self.dspin.valueChanged[int].connect(self.setD)
        self.dspin.editingFinished.connect(self.edited)

        form.addRow("Window", self.wspin)
        form.addRow("Polynomial Order", self.pspin)
        form.addRow("Derivative Order", self.dspin)
        self.layout().addLayout(form)

    def setParameters(self, params):
        self.setW(params.get("window", 5))
        self.setP(params.get("polyorder", 2))
        self.setD(params.get("deriv", 0))

    def parameters(self):
        return {"window": self.window, "polyorder": self.polyorder, "deriv": self.deriv}

    def setW(self, window):
        if self.window != window:
            self.window = window
            self.wspin.setValue(window)
            self.changed.emit()

    def setP(self, polyorder):
        if self.polyorder != polyorder:
            self.polyorder = polyorder
            self.pspin.setValue(polyorder)
            self.changed.emit()

    def setD(self, deriv):
        if self.deriv != deriv:
            self.deriv = deriv
            self.dspin.setValue(deriv)
            self.changed.emit()

    @staticmethod
    def createinstance(params):
        window = params.get("window", 5)
        polyorder = params.get("polyorder",2)
        deriv = params.get("deriv", 0)
        # make window, polyorder, deriv valid, even if they were saved differently
        window, polyorder, deriv = int(window), int(polyorder), int(deriv)
        if window % 2 == 0:
            window = window + 1
        if polyorder >= window:
            polyorder = window - 1
        # FIXME notify changes
        return SavitzkyGolayFiltering(window=window, polyorder=polyorder, deriv=deriv)


class RubberbandBaselineEditor(BaseEditor):
    """
    Apply a rubberband baseline subtraction via convex hull calculation.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        form = QFormLayout()

        self.peakcb = QComboBox()
        self.peakcb.addItems(["Positive", "Negative"])

        self.subcb = QComboBox()
        self.subcb.addItems(["Subtract", "Calculate"])

        form.addRow("Peak Direction", self.peakcb)
        form.addRow("Background Action", self.subcb)
        self.layout().addLayout(form)
        self.peakcb.currentIndexChanged.connect(self.changed)
        self.peakcb.activated.connect(self.edited)
        self.subcb.currentIndexChanged.connect(self.changed)
        self.subcb.activated.connect(self.edited)

    def setParameters(self, params):
        peak_dir = params.get("peak_dir", 0)
        sub = params.get("sub", 0)
        self.peakcb.setCurrentIndex(peak_dir)
        self.subcb.setCurrentIndex(sub)

    def parameters(self):
        return {"peak_dir": self.peakcb.currentIndex(),
                "sub": self.subcb.currentIndex()}

    @staticmethod
    def createinstance(params):
        peak_dir = params.get("peak_dir", 0)
        sub = params.get("sub", 0)
        return RubberbandBaseline(peak_dir=peak_dir, sub=sub)


class NormalizeEditor(BaseEditor):
    """
    Normalize spectra.
    """
    # Normalization methods
    Normalizers = [
        ("Vector Normalization", Normalize.Vector),
        ("Area Normalization", Normalize.Area),
        ("Attribute Normalization", Normalize.Attribute)]


    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.__method = Normalize.Vector
        self.lower = 0
        self.upper = 4000
        self.int_method = 0
        self.attrs = ['None']

        model = VariableListModel()
        model.wrap(self.attrs)
        self.attrform = QFormLayout()
        self.attrcb = QComboBox(enabled=False)
        self.attrcb.setModel(model)
        self.attrform.addRow("Normalize to", self.attrcb)

        self.areaform = QFormLayout()
        self.int_method_cb = QComboBox(enabled=False)
        self.int_method_cb.addItems(IntegrateEditor.Integrators)
        minf,maxf = -sys.float_info.max, sys.float_info.max
        self.lspin = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=self.lower, enabled=False)
        self.uspin = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=self.upper, enabled=False)
        self.areaform.addRow("Normalize to", self.int_method_cb)
        self.areaform.addRow("Lower limit", self.lspin)
        self.areaform.addRow("Upper limit", self.uspin)

        self.__group = group = QButtonGroup(self)

        for name, method in self.Normalizers:
            rb = QRadioButton(
                        self, text=name,
                        checked=self.__method == method
                        )
            layout.addWidget(rb)
            if method is Normalize.Attribute:
                layout.addLayout(self.attrform)
            elif method is Normalize.Area:
                layout.addLayout(self.areaform)
            group.addButton(rb, method)

        group.buttonClicked.connect(self.__on_buttonClicked)
        self.attrcb.activated.connect(self.edited)

        self.lspin.focusIn = self.activateOptions
        self.uspin.focusIn = self.activateOptions
        self.focusIn = self.activateOptions

        self.lspin.valueChanged[float].connect(self.setL)
        self.lspin.editingFinished.connect(self.reorderLimits)
        self.uspin.valueChanged[float].connect(self.setU)
        self.uspin.editingFinished.connect(self.reorderLimits)
        self.int_method_cb.currentIndexChanged.connect(self.setinttype)
        self.int_method_cb.activated.connect(self.edited)

        self.lline = MovableVlineWD(position=self.lower, label="Low limit",
                                    setvalfn=self.setL, confirmfn=self.reorderLimits)
        self.uline = MovableVlineWD(position=self.upper, label="High limit",
                                    setvalfn=self.setU, confirmfn=self.reorderLimits)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.__method == Normalize.Area:
            if self.lline not in self.parent_widget.curveplot.markings:
                self.parent_widget.curveplot.add_marking(self.lline)
            if (self.uline not in self.parent_widget.curveplot.markings
                    and IntegrateEditor.Integrators_classes[self.int_method]
                        is not Integrate.PeakAt):
                self.parent_widget.curveplot.add_marking(self.uline)

    def setParameters(self, params):
        if params: #parameters were manually set somewhere else
            self.user_changed = True
        method = params.get("method", Normalize.Vector)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        int_method = params.get("int_method", 0)
        if method not in [method for name,method in self.Normalizers]:
            # handle old worksheets
            method = Normalize.Vector
        self.setMethod(method)
        self.int_method_cb.setCurrentIndex(int_method)
        self.setL(lower, user=False)
        self.setU(upper, user=False)

    def parameters(self):
        return {"method": self.__method, "lower": self.lower,
                "upper": self.upper, "int_method": self.int_method,
                "attr": self.attrcb.currentText()}

    def setMethod(self, method):
        if self.__method != method:
            self.__method = method
            b = self.__group.button(method)
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
        if (IntegrateEditor.Integrators_classes[self.int_method]
                is Integrate.PeakAt):
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
        method = self.__group.checkedId()
        if method != self.__method:
            self.setMethod(self.__group.checkedId())
            self.edited.emit()

    @staticmethod
    def createinstance(params):
        method = params.get("method", Normalize.Vector)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        int_method_index = params.get("int_method", 0)
        int_method = IntegrateEditor.Integrators_classes[int_method_index]
        attr = params.get("attr", None)
        return Normalize(method=method, lower=lower, upper=upper,
                         int_method=int_method, attr=attr)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.setL(min(x))
                self.setU(max(x))
                self.edited.emit()
        self.attrs[:] = [var for var in data.domain.metas
                         if var.is_continuous]


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

        minf,maxf = -sys.float_info.max, sys.float_info.max

        if label:
            self.addWidget(QLabel(label))

        self.lowlime = SetXDoubleSpinBox(decimals=2,
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=limits[0], maximumWidth=75)
        self.highlime = SetXDoubleSpinBox(decimals=2,
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=limits[1], maximumWidth=75)
        self.lowlime.setValue(limits[0])
        self.highlime.setValue(limits[1])
        self.addWidget(self.lowlime)
        self.addWidget(self.highlime)

        if delete:
            self.button = QPushButton(QApplication.style().standardIcon(QStyle.SP_DockWidgetCloseButton), "")
            self.addWidget(self.button)
            self.button.clicked.connect(self.selfDelete)

        self.lowlime.valueChanged[float].connect(self.limitChanged)
        self.highlime.valueChanged[float].connect(self.limitChanged)
        self.lowlime.editingFinished.connect(self.editFinished)
        self.highlime.editingFinished.connect(self.editFinished)

        self.lowlime.focusIn = self.focusInChild
        self.highlime.focusIn = self.focusInChild

        self.line1 = MovableVlineWD(position=limits[0], label=label + " - Low",
                        setvalfn=self.lineLimitChanged)
        self.line2 = MovableVlineWD(position=limits[1], label=label + " - High",
                        setvalfn=self.lineLimitChanged)

        self.line1.line.sigPositionChangeFinished.connect(self.editFinished)
        self.line2.line.sigPositionChangeFinished.connect(self.editFinished)

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

    def lineLimitChanged(self, value=None):
        newlimits = [self.line1.value(), self.line2.value()]
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

class IntegrateEditor(BaseEditor):
    """
    Editor to integrate defined regions.
    """

    Integrators_classes = [Integrate.Simple,
                           Integrate.Baseline,
                           Integrate.PeakMax,
                           Integrate.PeakBaseline,
                           Integrate.PeakAt]

    Integrators = [a.name for a in Integrators_classes]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._limits = []

        self.setLayout(QVBoxLayout())
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
            limitbox = self.form_lim.itemAt(row, 1)
            if limitbox:
                self.parent_widget.curveplot.add_marking(limitbox.line1)
                self.parent_widget.curveplot.add_marking(limitbox.line2)

    def add_limit(self, *args, row=None):
        if row is None:
            row = len(self._limits)
            try:
                self._limits.append(self._limits[-1])
            except IndexError:
                self._limits.append([0.,1.])
        label = "Region {0}".format(row+1)
        limitbox = LimitsBox(limits=self._limits[row], label=label)
        if self.form_lim.rowCount() < row+1:
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
            limitbox = self.form_lim.itemAt(r, 1)
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
            limitbox = self.form_lim.itemAt(row, 1)
            if limitbox is None:
                limitbox = self.add_limit(row=row)
            with blocked(limitbox):
                limitbox.lowlime.setValue(limits[row][0])
                limitbox.highlime.setValue(limits[row][1])
        self.changed.emit()

    def setParameters(self, params):
        if params: #parameters were manually set somewhere else
            self.user_changed = True
        self.methodcb.setCurrentIndex(params.get("method", self.Integrators_classes.index(Integrate.Baseline)))
        self.set_all_limits(params.get("limits", [[0.,1.]]), user=False)

    def parameters(self):
        return {"method": self.methodcb.currentIndex(),
                "limits": self._limits}

    @staticmethod
    def createinstance(params):
        methodindex = params.get("method", IntegrateEditor.Integrators_classes.index(Integrate.Baseline))
        method = IntegrateEditor.Integrators_classes[methodindex]
        limits = params.get("limits", None)
        return Integrate(methods=method, limits=limits)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_all_limits([[min(x),max(x)]])
                self.edited.emit()


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

class TransToAbsEditor(BaseEditor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setParameters(self, params):
        pass

    @staticmethod
    def createinstance(params):
        return Absorbance(ref=None)

class AbsToTransEditor(BaseEditor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setParameters(self, params):
        pass

    @staticmethod
    def createinstance(params):
        return Transmittance(ref=None)

class AverageEditor(BaseEditor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attrs = ["(All)"]

        model = VariableListModel()
        model.wrap(self.attrs)
        attrform = QFormLayout()
        self.attrcb = QComboBox()
        self.attrcb.setModel(model)
        attrform.addRow("Average:", self.attrcb)

        self.stdcheck = QCheckBox()
        attrform.addRow("Standard Deviation:", self.stdcheck)

        self.setLayout(attrform)

        self.attrcb.activated.connect(self.edited)
        self.stdcheck.stateChanged.connect(self.edited)

        self.user_changed = False

    def setParameters(self, params):
        if params:
            self.user_changed = True
        avar = params.get("avar", self.attrs[0])
        std = params.get("std", False)
        if self.attrs[self.attrcb.currentIndex()] != avar:
            try:
                self.attrcb.setCurrentIndex(self.attrs.index(avar))
            except ValueError:
                self.attrcb.setCurrentIndex(0)
        if self.stdcheck.isChecked() != std:
            self.stdcheck.setChecked(std)

    def parameters(self):
        return {"avar": self.attrs[self.attrcb.currentIndex()],
                "std": self.stdcheck.isChecked()}

    @staticmethod
    def createinstance(params):
        avar = params.get("avar", "(All)")
        if avar == "(All)":
            avar = None
        std = params.get("std", False)
        return Average(avar, std)

    def set_preview_data(self, data):
        self.attrs[:] = ["(All)"] + [
            var for var in chain(data.domain, data.domain.metas)
                if isinstance(var, str) or var.is_discrete]

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
        "Gaussian smoothing", "orangecontrib.infrared.gaussian", "Smoothing",
        Description("Smooth spectra (gaussian)",
        icon_path("Discretize.svg")),
        GaussianSmoothingEditor
    ),
    PreprocessAction(
        "Savitzky-Golay Filter", "orangecontrib.infrared.savitzkygolay", "Smoothing",
        Description("Savitzky-Golay Filter (smoothing and differentiation)",
        icon_path("Discretize.svg")),
        SavitzkyGolayFilteringEditor
    ),
    PreprocessAction(
        "Rubberband Baseline Subtraction", "orangecontrib.infrared.rubberband", "Baseline Subtraction",
        Description("Rubberband Baseline Subtraction (convex hull)",
        icon_path("Discretize.svg")),
        RubberbandBaselineEditor
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
        "Transmittance to Absorbance", "orangecontrib.infrared.absorbance", "Trasmittance to Absorbance",
        Description("Trasmittance to Absorbance",
                    icon_path("Discretize.svg")),
        TransToAbsEditor
    ),
    PreprocessAction(
        "Absorbance to Transmittance", "orangecontrib.infrared.transmittance", "Absorbance to Transmittance",
        Description("Absorbance to Transmittance",
                    icon_path("Discretize.svg")),
        AbsToTransEditor
    ),
    PreprocessAction(
        "Average", "orangecontrib.infrared.average", "Average",
        Description("Average",
                    icon_path("Discretize.svg")),
        AverageEditor
    ),
    ]


class OWPreprocess(OWWidget):
    name = "Preprocess Spectra"
    description = "Construct a data preprocessing pipeline."
    icon = "icons/preprocess.svg"
    priority = 2105
    id = "orangecontrib.infrared.widgets.owpreproc"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Preprocessed Data", Orange.data.Table),
               ("Preprocessor", preprocess.preprocess.Preprocess)]

    storedsettings = settings.Setting({})
    autocommit = settings.Setting(False)
    preview_curves = settings.Setting(3)
    preview_n = settings.Setting(0)

    curveplot = settings.SettingProvider(CurvePlot)

    BUTTON_ADD_LABEL = "Add preprocessor..."
    PREPROCESSORS = PREPROCESSORS

    preview_choose = True

    class Error(OWWidget.Error):
        applying = Msg("Error applying preprocessors.")

    def __init__(self):
        super().__init__()

        self.data = None
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

        self.flow_view = SequenceFlow(preview_callback=self.show_preview, show_preview=self.preview_choose)
        self.controler = ViewController(self.flow_view, parent=self)

        self.scroll_area = QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)

        self.flow_view.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.scroll_area.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)

        self.curveplot = CurvePlot(self)
        self.curveplot.plot.vb.x_padding = 0.005  # so that lines on the edges are visible

        self.topbox = gui.hBox(self)
        self.topbox.layout().addWidget(self.curveplot)
        self.controlArea.layout().addWidget(self.scroll_area)
        self.mainArea.layout().addWidget(self.topbox)
        self.flow_view.installEventFilter(self)

        box = gui.widgetBox(self.controlArea, "Preview")

        gui.spin(box, self, "preview_curves", 1, 10, label="Show curves", callback=self.show_preview)

        self.output_box = gui.widgetBox(self.controlArea, "Output")
        gui.auto_commit(self.output_box, self, "autocommit", "Commit", box=False)

        self._initialize()

    def show_preview(self):
        """ Shows preview and also passes preview data to the widgets """
        #self.storeSpecificSettings()

        if self.data is not None:
            data = self.data
            if len(data) > self.preview_curves: #sample data
                sampled_indices = sorted(random.Random(0).sample(range(len(data)), self.preview_curves))
                data = data[sampled_indices]

            widgets = self.flow_view.widgets()
            preview_pos = self.flow_view.preview_n()
            n = self.preprocessormodel.rowCount()

            preview_data = None

            for i in range(n):
                if preview_pos == i:
                    preview_data = data

                if hasattr(widgets[i], "set_preview_data"):
                    widgets[i].set_preview_data(data)

                item = self.preprocessormodel.item(i)
                desc = item.data(DescriptionRole)
                params = item.data(ParametersRole)

                if not isinstance(params, dict):
                    params = {}

                create = desc.viewclass.createinstance
                preproc = create(params)

                data = preproc(data)

            if preview_pos == len(widgets):
                preview_data = data

            self.curveplot.set_data(preview_data)
        else:
            self.curveplot.set_data(None)
        self.curveplot.update_view()

    def _initialize(self):
        for i,pp_def in enumerate(self.PREPROCESSORS):
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
                description.title, self, triggered=lambda x,id=i: self.add_preprocessor(id)
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

        self.apply()


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

    @check_sql_input
    def set_data(self, data=None):
        """Set the input data set."""
        self.data = data
        self.show_preview()

    def handleNewSignals(self):
        self.apply()

    def add_preprocessor(self, index):
        action = self.PREPROCESSORS[index]
        item = QStandardItem()
        item.setData({}, ParametersRole)
        item.setData(action.description.title, Qt.DisplayRole)
        item.setData(action, DescriptionRole)
        self.preprocessormodel.appendRow([item])

    def buildpreproc(self, limit=None):
        plist = []
        if limit == None:
            limit = self.preprocessormodel.rowCount()
        for i in range(limit):
            item = self.preprocessormodel.item(i)
            desc = item.data(DescriptionRole)
            params = item.data(ParametersRole)

            if not isinstance(params, dict):
                params = {}

            create = desc.viewclass.createinstance
            plist.append(create(params))

        if len(plist) == 1:
            return plist[0]
        else:
            return preprocess.preprocess.PreprocessorList(plist)

    def apply(self):
        # Sync the model into storedsettings on every apply.

        self.show_preview()

        self.storeSpecificSettings()
        preprocessor = self.buildpreproc()

        if self.data is not None:
            self.Error.applying.clear()
            try:
                data = preprocessor(self.data)
            except ValueError as e:
                self.Error.applying()
                return
        else:
            data = None

        self.send("Preprocessor", preprocessor)
        self.send(self.outputs[0].name, data)

    def commit(self):
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
        super().storeSpecificSettings()

    def saveSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        self.preview_n = self.flow_view.preview_n()
        super().saveSettings()

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


def test_main(argv=sys.argv):
    argv = list(argv)
    app = QApplication(argv)

    w = OWPreprocess()
    w.set_data(Orange.data.Table("collagen.csv"))
    w.show()
    w.raise_()
    r = app.exec_()
    w.set_data(None)
    w.saveSettings()
    w.onDeleteWidget()
    return r

if __name__ == "__main__":
    sys.exit(test_main())
