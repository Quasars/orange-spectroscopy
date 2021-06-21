import math
from decimal import Decimal
from abc import ABCMeta, abstractmethod

from AnyQt.QtCore import QLocale, Qt, QSize
from AnyQt.QtGui import QDoubleValidator, QIntValidator, QValidator, QColor
from AnyQt.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QSizePolicy
from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg
from pyqtgraph import Point, QtCore

from Orange.widgets.utils import getdeepattr
from Orange.widgets.widget import OWComponent
from Orange.widgets.data.owpreprocess import blocked

from orangewidget.gui import ValueCallback, ControlledCallFront, ControlledCallback


def pixels_to_decimals(n):
    try:
        return max(-int(math.floor(math.log10(n))) + 1, 0)
    except ValueError:
        return 10


def pixel_decimals(viewbox):
    """
    Decimals needed to accurately represent position on a viewbox.
    Return a tuple, decimals for x and y positions.
    """
    try:
        xpixel, ypixel = viewbox.viewPixelSize()
    except:
        xpixel, ypixel = 0, 0

    return pixels_to_decimals(xpixel), pixels_to_decimals(ypixel)


def float_to_str_decimals(f, decimals):
    return ("%0." + str(decimals) + "f") % f


def round_virtual_pixels(v, range, pixels=10000):
    """
    Round a value acording to some pixel precision as if the given range
    was displayed on a virtual screen with a given pixel size.
    :param v: input value
    :param range: virtual visible range
    :param pixels: virtual pixels covering the range
    :return: rounded value as a Decimal.Decimal
    """
    pixel_decimals = pixels_to_decimals(range / pixels)
    return Decimal(float_to_str_decimals(v, pixel_decimals))


class AnyOrEmptyValidator(QValidator):

    def __init__(self, parent, allow_empty, bottom, top, default_text):
        super().__init__(parent)
        self.allow_empty = allow_empty
        self.default_text = default_text
        self.dv.setLocale(QLocale.c())
        self.setBottom(bottom)
        self.setTop(top)

    def setDefault(self, s):
        self.default_text = s

    def setBottom(self, b):
        self.dv.setBottom(b)

    def setTop(self, t):
        self.dv.setTop(t)

    def fixup(self, p_str):
        # = only called at editingFinished so an Orange controlled value can still contain
        #   invalid, because they are synchronized and every change
        # = called before returnPressedHandler
        try:
            f = self.valid_type(p_str)
            if f > self.dv.top():
                return str(self.dv.top())
            if f < self.dv.bottom():
                return str(self.dv.bottom())
        except ValueError:
            return self.default_text

    def validate(self, s, pos):
        if self.allow_empty and len(s) == 0:
            return (QValidator.Acceptable, s, pos)
        if "," in s:
            return (QValidator.Invalid, s, pos)
        else:
            return self.dv.validate(s, pos)


class FloatOrEmptyValidator(AnyOrEmptyValidator):

    def __init__(self, parent, allow_empty=False, bottom=float("-inf"), top=float("inf"),
                 default_text=""):
        self.dv = QDoubleValidator(parent)
        self.valid_type = float
        super().__init__(parent, allow_empty=allow_empty, bottom=bottom, top=top,
                         default_text=default_text)


class IntOrEmptyValidator(AnyOrEmptyValidator):

    def __init__(self, parent, allow_empty=False, bottom=-2147483647, top=2147483647,
                 default_text=""):
        self.dv = QIntValidator(parent)
        self.valid_type = int
        super().__init__(parent, allow_empty=allow_empty, bottom=bottom, top=top,
                         default_text=default_text)


def floatornone(a):
    try:  # because also intermediate values are passed forward
        return float(a)
    except (ValueError, TypeError):
        return None


def decimalornone(a):
    try:  # because also intermediate values are passed forward
        return Decimal(a)
    except:
        return None


def intornone(a):
    if a == "":
        return None
    try:  # because also intermediate values are passed forward
        return int(a)
    except ValueError:
        return None


def str_or_empty(val):
    if val is None:
        return ""
    else:
        return str(val)


class CallFrontLineEditCustomConversion(ControlledCallFront):

    def __init__(self, control, valToStr):
        super().__init__(control)
        self.valToStr = valToStr

    def action(self, value):
        self.control.setText(self.valToStr(value))


class LineEditMarkFinished(QLineEdit):
    """QLineEdit that marks all text when pressed enter."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.returnPressed.connect(self.selectAll)


class LineEdit(LineEditMarkFinished):

    newInput = Signal(str)
    """ Emitted when the editing was finished and contents actually changed"""

    focusIn = Signal()

    def __init__(self, parent=None):
        LineEditMarkFinished.__init__(self, parent=parent)
        self.__changed = False
        self.editingFinished.connect(self.__signal_if_changed)
        self.textEdited.connect(self.__textEdited)
        self.sizeHintFactor = 1

    def setText(self, text):
        self.__changed = False
        super().setText(text)

    def __textEdited(self):
        self.__changed = True

    def __signal_if_changed(self):
        if self.__changed:
            self.__changed = False
            self.newInput.emit(self.text())

    def focusInEvent(self, *e):
        self.focusIn.emit()
        return QWidget.focusInEvent(self, *e)

    def sizeHint(self):
        sh = super().sizeHint()
        return QSize(int(sh.width()*self.sizeHintFactor), sh.height())


def connect_line_edit_finished(lineedit, master, value, valueToStr=str, valueType=str, callback=None):
    # callback is only for compatibility with the old code
    update_value = ValueCallback(master, value, valueType)  # save the value
    update_control = CallFrontLineEditCustomConversion(lineedit, valueToStr)  # update control
    update_value.opposite = update_control
    update_control(getdeepattr(master, value))  # set the first value
    master.connect_control(value, update_control)
    lineedit.newInput.connect(lambda x: (update_value(x),
                                         callback() if callback is not None else None))


def lineEditUpdateWhenFinished(parent, master, value, valueToStr=str, valueType=str, validator=None, callback=None):
    """ Line edit that only calls callback when update is finished """
    ledit = LineEdit(parent)
    ledit.setValidator(validator)
    if value:
        connect_line_edit_finished(ledit, master, value, valueToStr=valueToStr, valueType=valueType, callback=callback)
    return ledit


def lineEditValidator(widget, master, value, validator=None, valueType=None,
                      valueToStr=str, **kwargs):
    le = lineEditUpdateWhenFinished(widget, master, value, validator=validator, valueType=valueType,
                                    valueToStr=valueToStr, **kwargs)
    return le


def set_numeric_le(le):
    """ Smaller line edit that does not expand. """
    le.sizeHintFactor = 0.8
    le.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
    return le


def lineEditIntOrNone(widget, master, value, **kwargs):
    le = lineEditValidator(widget, master, value,
                           validator=IntOrEmptyValidator(master),
                           valueType=intornone,
                           valueToStr=str_or_empty,
                           **kwargs)
    return le


def lineEditFloatOrNone(widget, master, value, **kwargs):
    le = lineEditValidator(widget, master, value,
                           validator=FloatOrEmptyValidator(master, allow_empty=True),
                           valueType=floatornone,
                           valueToStr=str_or_empty,
                           **kwargs)
    set_numeric_le(le)
    return le


def lineEditFloatRange(widget, master, value, bottom=float("-inf"), top=float("inf"), default=0., **kwargs):
    le = lineEditValidator(widget, master, value,
                           validator=FloatOrEmptyValidator(master, allow_empty=False,
                                                           bottom=bottom, top=top, default_text=str(default)),
                           valueType=Decimal,  # every text need to be a valid float before saving setting
                           valueToStr=str,
                           **kwargs)
    le.set_default = lambda v: le.validator().setDefault(str(v))
    set_numeric_le(le)
    return le


def lineEditIntRange(widget, master, value, bottom=-2147483647, top=2147483647,
                     default=0, **kwargs):
    le = lineEditValidator(widget, master, value,
                           validator=IntOrEmptyValidator(master, allow_empty=False,
                                                         bottom=bottom, top=top,
                                                         default_text=str(default)),
                           valueType=int,  # every text need to be a valid before saving
                           valueToStr=str,
                           **kwargs)
    le.set_default = lambda v: le.validator().setDefault(str(v))
    set_numeric_le(le)
    return le


def lineEditDecimalOrNone(widget, master, value, bottom=float("-inf"), top=float("inf"), default=0., **kwargs):
    le = lineEditValidator(widget, master, value,
                           validator=FloatOrEmptyValidator(master, allow_empty=True,
                                                           bottom=bottom, top=top, default_text=str(default)),
                           valueType=decimalornone,  # every text need to be a valid float before saving setting
                           valueToStr=str_or_empty,
                           **kwargs)
    le.set_default = lambda v: le.validator().setDefault(str(v))
    set_numeric_le(le)
    return le


class MovableVline(pg.UIGraphicsItem):

    sigMoveFinished = Signal(object)
    sigMoved = Signal(object)

    def __init__(self, position=0., label="", color=(225, 0, 0), report=None):
        pg.UIGraphicsItem.__init__(self)
        self.moving = False
        self.mouseHovering = False
        self.report = report
        self.color = color
        self.isnone = False

        hp = pg.mkPen(color=color, width=3)
        np = pg.mkPen(color=color, width=2)

        self.line = pg.InfiniteLine(angle=90, movable=True, pen=np, hoverPen=hp)
        self.line.setParentItem(self)
        self.line.setCursor(Qt.SizeHorCursor)

        self.label = pg.TextItem("", anchor=(0, 1), angle=-90)
        self.label.setParentItem(self)

        self.setValue(position)
        self.setLabel(label)

        self.line.sigPositionChangeFinished.connect(self._moveFinished)
        self.line.sigPositionChanged.connect(lambda: (self._moved(), self.sigMoved.emit(self.value())))

        self._lastTransform = None

    def setLabel(self, l):
        # add space on top not to overlap with cursor coordinates
        # I can not think of a better way: the text
        # size is decided at rendering time. As it is always the same,
        # these spaces will scale with the cursor coordinates.
        top_space = " " * 7

        self.label.setText(top_space + l, color=self.color)

    def value(self):
        if self.isnone:
            return None
        return self.line.value()

    def rounded_value(self):
        """ Round the value according to current view on the graph.
        Return a decimal.Decimal object """
        v = self.value()
        dx, dy = pixel_decimals(self.getViewBox())
        if v is not None:
            v = Decimal(float_to_str_decimals(v, dx))
        return v

    def setValue(self, val):
        oldval = self.value()
        if oldval == val:
            return  # prevents recursion with None on input
        self.isnone = val is None
        if not self.isnone:
            rep = self.report  # temporarily disable report
            self.report = None
            with blocked(self.line):  # block sigPositionChanged by setValue
                self.line.setValue(val)
            self.report = rep
            self.line.show()
            self.label.show()
            self._move_label()
        else:
            self.line.hide()
            self.label.hide()
            self._moved()

    def boundingRect(self):
        br = pg.UIGraphicsItem.boundingRect(self)
        val = self.line.value()
        br.setLeft(val)
        br.setRight(val)
        return br.normalized()

    def _move_label(self):
        if self.value() is not None and self.getViewBox():
            self.label.setPos(self.value(), self.getViewBox().viewRect().bottom())

    def _moved(self):
        self._move_label()
        if self.report:
            if self.value() is not None:
                self.report.report(self, [("x", self.value())])
            else:
                self.report.report_finished(self)

    def _moveFinished(self):
        if self.report:
            self.report.report_finished(self)
        self.sigMoveFinished.emit(self.value())

    def paint(self, p, *args):
        tr = p.transform()
        if self._lastTransform != tr:
            self._move_label()
        self._lastTransform = tr
        super().paint(p, *args)


class LineCallFront(ControlledCallFront):

    def action(self, value):
        self.control.setValue(floatornone(value))


class ValueTransform(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, v):
        """Transform the argument"""

    @abstractmethod
    def inverse(self, v):
        """Inverse transform"""


class ValueCallbackTransform(ControlledCallback):

    def __call__(self, value):
        self.acyclic_setattr(value)


def connect_settings(master, value, value2, transform=None):
    """
    Connect two Orange settings, so that both will be mutually updated. One value
    can be a function of the other. This sets value of the value2 as a function of value.

    :param master: a settings controller
    :param value: name of the first value
    :param value2: name of the second value
    :param transform: a transform from value to value2 (an instance of ValueTransform)
    """
    update_value = ValueCallbackTransform(master, value,
                                          f=transform.inverse if transform is not None else None)
    update_value2 = ValueCallbackTransform(master, value2,
                                           f=transform.transform if transform is not None else None)
    update_value.opposite = update_value2
    update_value2.opposite = update_value
    update_value2(getdeepattr(master, value))
    master.connect_control(value, update_value2)
    master.connect_control(value2, update_value)


def connect_line(line, master, value):
    update_value = ValueCallback(master, value)  # save the value
    update_control = LineCallFront(line)  # update control
    update_value.opposite = update_control
    update_control(getdeepattr(master, value))  # set the first value
    master.connect_control(value, update_control)
    line.sigMoved.connect(lambda: update_value(line.rounded_value()))


class XPosLineEdit(QWidget, OWComponent):

    edited = Signal()
    focusIn = Signal()

    def __init__(self, parent=None, label="", element=lineEditFloatRange):
        QWidget.__init__(self, parent)
        OWComponent.__init__(self, None)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.position = 0

        self.edit = element(self, self, "position", callback=self.edited.emit)
        layout.addWidget(self.edit)
        self.edit.focusIn.connect(self.focusIn.emit)
        self.line = MovableVline(position=self.position, label=label)
        connect_line(self.line, self, "position")
        self.line.sigMoveFinished.connect(self.edited.emit)

    def set_default(self, v):
        self.edit.validator().setDefault(str(v))

    def focusInEvent(self, *e):
        self.focusIn.emit()
        return QWidget.focusInEvent(self, *e)


class VerticalPeakLine(pg.InfiniteLine):

    """pyqtgraph.InfiniteLine with adjustments for spectra analysis
    """

    def __init__(self, pos=None, angle=90, pen=None, movable=True,
                 bounds=None, label=None, span=None):

        super().__init__(pos, angle, pen, movable, bounds, span)

        if label is None:
            self.label = VerticalPeakLineLabel(self,
                                               text=str(self.getXPos()), position=(1))

        self.sigDragged.connect(self.updateLabel)
        self.selection = 0

    def _computeBoundingRect(self):
        vr = self.viewRect()
        if vr is None:
            return QtCore.QRectF()

        px = self.pixelLength(direction=Point(1, 0), ortho=True)
        if px is None:
            px = 0
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = max(4, 0 + pw) + 1
        w = w * px
        br = QtCore.QRectF(vr)
        br.setBottom(-w)
        br.setTop(w)
        left = self.span[0] #left = br.left() + length * self.span[0]
        right = self.span[1] #right = br.left() + length * self.span[1]
        #right and left changed to only go from min to max values of data
        br.setLeft(left)
        br.setRight(right)
        br = br.normalized()

        vs = self.getViewBox().size()
        if self.mouseDragEvent:
            left = self.span[0] - self.getYPos()

        if self._bounds != br or self._lastViewSize != vs:
            self._bounds = br
            self._lastViewSize = vs
            self.prepareGeometryChange()

        self._endPoints = (left, right)
        self._lastViewRect = vr

        return self._bounds

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and not self.moving and self.selection == 0:
            self.setPen(pg.mkPen(color=QColor(Qt.blue), width=2, style=Qt.DotLine))
            self.update()
            self.selection = 1
        else:
            self.setPen(pg.mkPen(color=QColor(Qt.black), width=2, style=Qt.DotLine))
            self.update()
            self.selection = 0

    def delete_line(self):
        if self.selection == 1:
            self.hide()
            self.update()

    def updateLabel(self):
        x = self.getXPos()
        length = len(str(round(x)))
        if length >= 4:
            length = 2
        elif 4 > length > 2:
            length = 3
        elif length <= 2:
            length = 4
        elif length == 0:
            length = len(x)
        self.label.setText(str(round(self.getXPos(), length)))
        self.update()

class VerticalPeakLineLabel(pg.InfLineLabel):
    def __init__(self, line, text="", movable=False, position=0.5, anchors=None, **kwds):
        super().__init__(line)
        self.line = line
