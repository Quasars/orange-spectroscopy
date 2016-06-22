import sys
import bisect
import contextlib
import warnings
import random
import math


import pyqtgraph as pg
from PyQt4 import QtCore
from PyQt4.QtGui import (
    QWidget, QButtonGroup, QGroupBox, QRadioButton, QSlider,
    QDoubleSpinBox, QComboBox, QSpinBox, QListView,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpacerItem, QSizePolicy,
    QCursor, QIcon,  QStandardItemModel, QStandardItem, QStyle,
    QStylePainter, QStyleOptionFrame, QPixmap,
    QApplication, QDrag
)
from PyQt4 import QtGui
from PyQt4.QtCore import (
    Qt, QObject, QEvent, QSize, QModelIndex, QMimeData, QTimer
)
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import Orange.data
from Orange import preprocess
from Orange.statistics import distribution
from Orange.preprocess import Continuize, ProjectPCA, \
    ProjectCUR, Randomize as Random
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.overlay import OverlayWidget
from Orange.widgets.utils.sql import check_sql_input

import Orange.widgets.data.owpreprocess as owpreprocess

from Orange.widgets.data.owpreprocess import (
    Controller, StandardItemModel,
    PreprocessAction, Description, icon_path, DiscretizeEditor,
    DescriptionRole, ParametersRole, BaseEditor, blocked
)

import numpy as np

from scipy.ndimage.filters import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.owcurves import CurvePlot


class ViewController(Controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._insertIndexAt = self.view.insertIndexAt

    def createWidgetFor(self, index):
        w = super().createWidgetFor(index)
        w.parent_widget = self.parent()
        return w


class FocusFrame(owpreprocess.SequenceFlow.Frame):

    def focusInEvent(self, event):
        super().focusInEvent(event)
        try: #active selection on preview
            self.widget().activateOptions()
        except AttributeError:
            pass

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        try:
            self.widget().deactivateOptions()
        except AttributeError:
            pass

class PreviewFrame(owpreprocess.SequenceFlow.Frame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFeatures(QtGui.QDockWidget.DockWidgetMovable)


class PreviewWidget(QWidget):
    pass


class SequenceFlow(owpreprocess.SequenceFlow):
    """
    FIXME Ugly hack: using the same name for access to private variables!
    """
    def __init__(self, *args, preview_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_callback = preview_callback

    def clear(self):
        super().clear()
        self.__preview_widget = PreviewWidget()
        self.__preview_frame = PreviewFrame(widget=self.__preview_widget, title="Preview")
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


class GaussianSmoothing():

    def __init__(self, sd=10.):
        #super().__init__(variable)
        self.sd = sd

    def __call__(self, data):
        #FIXME this filted does not do automatic domain conversions!
        #FIXME we need need data about frequencies:
        #what if frequencies are not sampled on equal intervals
        x = np.arange(len(data.domain.attributes))
        newd = gaussian_filter1d(data.X, sigma=self.sd, mode="nearest")
        data = data.copy()
        data.X = newd
        return data


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


class Cut():

    def __init__(self, lowlim=None, highlim=None):
        self.lowlim = lowlim
        self.highlim = highlim

    def __call__(self, data):
        x = getx(data)
        okattrs = [at for at, v in zip(data.domain.attributes, x) if (self.lowlim is None or self.lowlim <= v) and (self.highlim is None or v <= self.highlim)]
        domain = Orange.data.Domain(okattrs, data.domain.class_vars, metas=data.domain.metas)
        return data.from_table(domain, data)


class SetXDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)



class MovableVlineWD(pg.UIGraphicsItem):

    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    Vertical = 0
    Horizontal = 1

    def __init__(self, position, label="", setvalfn=None, confirmfn=None):
        pg.UIGraphicsItem.__init__(self)
        self.moving = False
        self.mouseHovering = False

        self.line = pg.InfiniteLine(angle=90, movable=True)
        self.line.setX(position)
        self.line.setCursor(Qt.SizeHorCursor)

        self.line.setParentItem(self)
        self.line.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.line.sigPositionChanged.connect(self.lineMoved)

        self.label = pg.TextItem("", anchor=(0,0))
        self.label.setText(label, color=(0, 0, 0))
        self.label.setParentItem(self)

        self.setvalfn = setvalfn
        self.confirmfn = confirmfn

        self.lastTransform = None

    def value(self):
        return self.line.value()

    def setValue(self, val):
        self.line.setValue(val)
        self._move_label()

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
        if self.setvalfn:
            self.setvalfn(self.value())

    def lineMoveFinished(self):
        if self.setvalfn:
            self.setvalfn(self.value())
        if self.confirmfn:
            self.confirmfn.emit()

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

    def activateOptions(self):
        if self.line1 not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.line1)
        if self.line2 not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.line2)

    def deactivateOptions(self):
        if self.line1 in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.remove_marking(self.line1)
        if self.line2 in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.remove_marking(self.line2)

    def set_lowlim(self, lowlim):
        if self.__lowlim != lowlim:
            self.__lowlim = lowlim
            with blocked(self.__lowlime):
                self.__lowlime.setValue(lowlim)
                self.line1.setValue(lowlim)
            self.changed.emit()

    def left(self):
        return self.__lowlim

    def set_highlim(self, highlim):
        if self.__highlim != highlim:
            self.__highlim = highlim
            with blocked(self.__highlime):
                self.__highlime.setValue(highlim)
                self.line2.setValue(highlim)
            self.changed.emit()

    def right(self):
        return self.__highlim

    def setParameters(self, params):
        self.set_lowlim(params.get("lowlim", 0.))
        self.set_highlim(params.get("highlim", 1.))

    def parameters(self):
        return {"lowlim": self.__lowlim, "highlim": self.__highlim}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        return Cut(lowlim=lowlim, highlim=highlim)


class SavitzkyGolayFiltering():
    """
    Apply a Savitzky-Golay[1] Filter to the data using SciPy Library.
    """
    def __init__(self, window=5,polyorder=2,deriv=0):
        #super().__init__(variable)
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def __call__(self, data):
        x = np.arange(len(data.domain.attributes))
        from scipy.signal import savgol_filter

        #savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        newd = savgol_filter(data.X, window_length=self.window, polyorder=self.polyorder, deriv=self.deriv, mode="nearest")

        data = data.copy()
        data.X = newd
        return data


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

        self.wspin = QDoubleSpinBox(
            minimum=3, maximum=100, singleStep=2,
            value=self.window)
        self.wspin.valueChanged[float].connect(self.setW)
        self.wspin.editingFinished.connect(self.edited)

        self.pspin = QDoubleSpinBox(
            minimum=2, maximum=self.window, singleStep=1,
            value=self.polyorder)
        self.pspin.valueChanged[float].connect(self.setP)
        self.pspin.editingFinished.connect(self.edited)

        self.dspin = QDoubleSpinBox(
            minimum=0, maximum=3, singleStep=1,
            value=self.deriv)
        self.dspin.valueChanged[float].connect(self.setD)
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
        return SavitzkyGolayFiltering(window=window,polyorder=polyorder,deriv=deriv)


class RubberbandBaseline():

    def __init__(self, peak_dir=0, sub=0):
        self.peak_dir = peak_dir
        self.sub = sub

    def __call__(self, data):
        x = getx(data)
        if self.sub == 0:
            newd = None
        elif self.sub == 1:
            newd = data.X
        for row in data.X:
            v = ConvexHull(np.column_stack((x, row))).vertices
            if self.peak_dir == 0:
                v = np.roll(v, -v.argmax())
                v = v[:v.argmin()+1]
            elif self.peak_dir == 1:
                v = np.roll(v, -v.argmin())
                v = v[:v.argmax()+1]
            baseline = interp1d(x[v], row[v])(x)
            if newd is not None and self.sub == 0:
                newd = np.vstack((newd, (row - baseline)))
            elif newd is not None and self.sub == 1:
                newd = np.vstack((newd, baseline))
            else:
                newd = row - baseline
                newd = newd[None,:]
        data = data.copy()
        data.X = newd
        return data


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


class Normalize():
    # Normalization methods
    MinMax, Vector, Offset, Attribute = 0, 1, 2, 3

    def __init__(self, method=MinMax, lower=float, upper=float, limits=0):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.limits = limits

    def __call__(self, data):
        x = getx(data)

        data = data.copy()

        if self.limits == 1:
            x_sorter = np.argsort(x)
            limits = np.searchsorted(x, [self.lower, self.upper], sorter=x_sorter)
            y_s = data.X[:,x_sorter][:,limits[0]:limits[1]]
        else:
            y_s = data.X

        if self.method == self.MinMax:
            data.X /= np.max(np.abs(y_s), axis=1, keepdims=True)
        elif self.method == self.Vector:
            # zero offset correction applies to entire spectrum, regardless of limits
            y_offsets = np.mean(data.X, axis=1, keepdims=True)
            data.X -= y_offsets
            y_s -= y_offsets
            rssq = np.sqrt(np.sum(y_s**2, axis=1, keepdims=True))
            data.X /= rssq
        elif self.method == self.Offset:
            data.X -= np.min(y_s, axis=1, keepdims=True)
        elif self.method == self.Attribute:
            # Not implemented
            pass

        return data

class NormalizeEditor(BaseEditor):
    """
    Normalize spectra.
    """
    # Normalization methods
    Normalizers = [
        ("Min-Max Scaling", Normalize.MinMax),
        ("Vector Normalization", Normalize.Vector),
        ("Offset Correction", Normalize.Offset),
        ("Attribute Normalization", Normalize.Attribute)]


    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.__method = Normalize.MinMax
        self.lower = 0
        self.upper = 4000
        self.limits = 0



        self.__group = group = QButtonGroup(self)

        for name, method in self.Normalizers:
            rb = QRadioButton(
                        self, text=name,
                        checked=self.__method == method
                        )
            layout.addWidget(rb)
            group.addButton(rb, method)

        group.buttonClicked.connect(self.__on_buttonClicked)

        form = QFormLayout()

        self.limitcb = QComboBox()
        self.limitcb.addItems(["Full Range", "Within Limits"])

        self.lspin = QDoubleSpinBox(
            minimum=0, maximum=16000, singleStep=50,
            value=self.lower, enabled=self.limits)
        self.lspin.valueChanged[float].connect(self.setL)
        self.lspin.editingFinished.connect(self.reorderLimits)

        self.uspin = QDoubleSpinBox(
            minimum=0, maximum=16000, singleStep=50,
            value=self.upper, enabled=self.limits)
        self.uspin.valueChanged[float].connect(self.setU)
        self.uspin.editingFinished.connect(self.reorderLimits)

        form.addRow("Normalize region", self.limitcb)
        form.addRow("Lower limit", self.lspin)
        form.addRow("Upper limit", self.uspin)
        self.layout().addLayout(form)
        self.limitcb.currentIndexChanged.connect(self.setlimittype)
        self.limitcb.activated.connect(self.edited)

    def setParameters(self, params):
        method = params.get("method", Normalize.MinMax)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        limits = params.get("limits", 0)
        self.setMethod(method)
        self.limitcb.setCurrentIndex(limits)
        self.setL(lower)
        self.setU(upper)

    def parameters(self):
        return {"method": self.__method, "lower": self.lower,
                "upper": self.upper, "limits": self.limits}

    def setMethod(self, method):
        if self.__method != method:
            self.__method = method
            b = self.__group.button(method)
            b.setChecked(True)
            self.changed.emit()

    def setL(self, lower):
        if self.lower != lower:
            self.lower = lower
            self.lspin.setValue(lower)
            self.changed.emit()

    def setU(self, upper):
        if self.upper != upper:
            self.upper = upper
            self.uspin.setValue(upper)
            self.changed.emit()

    def reorderLimits(self):
        limits = [self.lower, self.upper]
        self.lower, self.upper = min(limits), max(limits)
        self.lspin.setValue(self.lower)
        self.uspin.setValue(self.upper)
        self.edited.emit()

    def setlimittype(self):
        if self.limits != self.limitcb.currentIndex():
            self.limits = self.limitcb.currentIndex()
            self.lspin.setEnabled(self.limits)
            self.uspin.setEnabled(self.limits)
            self.changed.emit()

    def __on_buttonClicked(self):
        method = self.__group.checkedId()
        if method != self.__method:
            self.setMethod(self.__group.checkedId())
            self.edited.emit()

    @staticmethod
    def createinstance(params):
        method = params.get("method", Normalize.MinMax)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        limits = params.get("limits", 0)
        return Normalize(method=method,lower=lower,upper=upper,limits=limits)


PREPROCESSORS = [
    PreprocessAction(
        "Cut", "orangecontrib.infrared.cut", "Cut",
        Description("Cut",
                    icon_path("Discretize.svg")),
        CutEditor
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
        "Normalization", "orangecontrib.infrared.normalize", "Normalization",
        Description("Normalization",
        icon_path("Normalize.svg")),
        NormalizeEditor
    ),
]


class OWPreprocess(widget.OWWidget):
    name = "Preprocess"
    description = "Construct a data preprocessing pipeline."
    icon = "icons/preprocess.svg"
    priority = 2105

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Preprocessor", preprocess.preprocess.Preprocess),
               ("Preprocessed Data", Orange.data.Table)]

    storedsettings = settings.Setting({})
    autocommit = settings.Setting(False)
    preview_curves = settings.Setting(3)
    preview_n = settings.Setting(0)

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

        box = gui.widgetBox(self.controlArea, "Preprocessors")

        self.preprocessorsView = view = QListView(
            selectionMode=QListView.SingleSelection,
            dragEnabled=True,
            dragDropMode=QListView.DragOnly
        )
        view.setModel(self.preprocessors)
        view.activated.connect(self.__activated)

        box.layout().addWidget(view)

        ####
        self._qname2ppdef = {ppdef.qualname: ppdef for ppdef in PREPROCESSORS}

        # List of 'selected' preprocessors and their parameters.
        self.preprocessormodel = None

        self.flow_view = SequenceFlow(preview_callback=self.show_preview)
        self.controler = ViewController(self.flow_view, parent=self)

        self.overlay = OverlayWidget(self)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.overlay.setWidget(self.flow_view)
        self.overlay.setLayout(QVBoxLayout())
        self.overlay.layout().addWidget(
            QtGui.QLabel("Drag items from the list on the left",
                         wordWrap=True))

        self.scroll_area = QtGui.QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)

        self.curveplot = CurvePlot(self)

        self.scroll_area.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.topbox = gui.hBox(self)
        self.topbox.layout().addWidget(self.curveplot)
        self.topbox.layout().addWidget(self.scroll_area)

        self.mainArea.layout().addWidget(self.topbox)
        self.flow_view.installEventFilter(self)

        box = gui.widgetBox(self.controlArea, "Preview")

        gui.spin(box, self, "preview_curves", 1, 10, label="Show curves", callback=self.show_preview)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.auto_commit(box, self, "autocommit", "Commit", box=False)

        self._initialize()

    def show_preview(self):
        #self.storeSpecificSettings()
        preprocessor = self.buildpreproc(self.flow_view.preview_n())

        if self.data is not None:
            data = self.data
            if len(data) > self.preview_curves:
                sampled_indices = sorted(random.Random(0).sample(range(len(data)), self.preview_curves))
                data = data[sampled_indices]

            self.curveplot.set_data(preprocessor(data))

    def _initialize(self):
        for pp_def in PREPROCESSORS:
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
            self.preprocessormodel.dataChanged.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsInserted.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsRemoved.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsMoved.disconnect(self.__on_modelchanged)
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = ppmodel
        self.controler.setModel(ppmodel)
        if ppmodel is not None:
            self.preprocessormodel.dataChanged.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsInserted.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsRemoved.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsMoved.connect(self.__on_modelchanged)

        self.__update_overlay()

    def __update_overlay(self):
        if self.preprocessormodel is None or \
                self.preprocessormodel.rowCount() == 0:
            self.overlay.setWidget(self.flow_view)
            self.overlay.show()
        else:
            self.overlay.setWidget(None)
            self.overlay.hide()

    def __on_modelchanged(self):
        self.__update_overlay()
        self.show_preview()
        self.commit()

    @check_sql_input
    def set_data(self, data=None):
        """Set the input data set."""
        self.data = data
        self.show_preview()

    def handleNewSignals(self):
        self.apply()

    def __activated(self, index):
        item = self.preprocessors.itemFromIndex(index)
        action = item.data(DescriptionRole)
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
            self.error(0)
            try:
                data = preprocessor(self.data)
            except ValueError as e:
                self.error(0, str(e))
                return
        else:
            data = None

        self.send("Preprocessor", preprocessor)
        self.send("Preprocessed Data", data)

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
        scroll_width = self.scroll_area.verticalScrollBar().width()
        self.scroll_area.setMinimumWidth(
            min(max(sh.width() + scroll_width + 2, self.controlArea.width()),
                520))

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(sh.width(), 500))


def test_main(argv=sys.argv):
    argv = list(argv)
    app = QtGui.QApplication(argv)

    w = OWPreprocess()
    w.set_data(Orange.data.Table("iris"))
    w.show()
    w.raise_()
    r = app.exec_()
    w.set_data(None)
    w.saveSettings()
    w.onDeleteWidget()
    return r

if __name__ == "__main__":
    sys.exit(test_main())
