import itertools
import sys
from collections import defaultdict
import gc
import random
import warnings
from xml.sax.saxutils import escape
from AnyQt.QtWidgets import QWidget, QGraphicsItem, QPushButton, QMenu, \
    QGridLayout, QAction, QVBoxLayout, QApplication, QWidgetAction, QLabel, \
    QShortcut, QToolTip, QGraphicsRectItem, QGraphicsTextItem
from AnyQt.QtGui import QColor, QPixmapCache, QPen, QKeySequence
from AnyQt.QtCore import Qt, QRectF, QPointF, QObject
from AnyQt.QtCore import pyqtSignal
import numpy as np
import pyqtgraph as pg
from scipy.signal import find_peaks
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph import Point, GraphicsObject
import Orange.data
from Orange.data import DiscreteVariable
from Orange.widgets.widget import OWWidget, Msg, OWComponent, Input, Output
from Orange.widgets import gui
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator
from Orange.widgets.utils.plot import \
    SELECT, PANNING, ZOOMING
from Orange.widgets.utils import saveplot
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.visualize.owscatterplotgraph import LegendItem
from Orange.widgets.utils.concurrent import TaskState, ConcurrentMixin

try:
    # since Orange 3.17
    from Orange.widgets.visualize.utils.plotutils import HelpEventDelegate
except ImportError:
    from Orange.widgets.visualize.owscatterplotgraph import HelpEventDelegate

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.utils import apply_columns_numpy
from orangecontrib.spectroscopy.widgets.line_geometry import \
    distance_curves, intersect_curves_chunked
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatOrNone, pixel_decimals, \
    lineEditDecimalOrNone, VerticalPeakLine, \
    float_to_str_decimals as strdec
from orangecontrib.spectroscopy.widgets.utils import pack_selection, unpack_selection, \
    selections_to_length, groups_or_annotated_table

SELECT_SQUARE = 123
SELECT_POLYGON = 124

# view types
INDIVIDUAL = 0
AVERAGE = 1

# selections
SELECTNONE = 0
SELECTONE = 1
SELECTMANY = 2

MAX_INSTANCES_DRAWN = 1000
MAX_THICK_SELECTED = 10
NAN = float("nan")

# distance to the first point in pixels that finishes the polygon
SELECT_POLYGON_TOLERANCE = 10

COLORBREWER_SET1 = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0),
                    (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153)]



class SelectionGroupMixin:
    selection_group_saved = Setting(None, schema_only=True)

    def __init__(self):
        self.selection_group = np.array([], dtype=np.uint8)
        # Remember the saved state to restore with the first open file
        self._pending_selection_restore = self.selection_group_saved

    def restore_selection_settings(self):
        self.selection_group = unpack_selection(self._pending_selection_restore)
        self.selection_group = selections_to_length(self.selection_group, len(self.data))
        self._pending_selection_restore = None

    def prepare_settings_for_saving(self):
        self.selection_group_saved = pack_selection(self.selection_group)


def selection_modifiers():
    keys = QApplication.keyboardModifiers()
    add_to_group = bool(keys & Qt.ControlModifier and keys & Qt.ShiftModifier)
    add_group = bool(keys & Qt.ControlModifier or keys & Qt.ShiftModifier)
    remove = bool(keys & Qt.AltModifier)
    return add_to_group, add_group, remove


class MenuFocus(QMenu):  # menu that works well with subwidgets and focusing
    def focusNextPrevChild(self, next):
        return QWidget.focusNextPrevChild(self, next)


class FinitePlotCurveItem(pg.PlotCurveItem):
    """
    PlotCurveItem which filters non-finite values from set_data

    Required to work around Qt>=5.12 plotting bug
    See https://github.com/Quasars/orange-spectroscopy/issues/408
    and https://github.com/pyqtgraph/pyqtgraph/issues/1057
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def setData(self, *args, **kargs):
        """Filter non-finite values from x,y before passing on to updateData"""
        if len(args) == 1:
            y = args[0]
        elif len(args) == 2:
            x = args[0]
            y = args[1]
        else:
            x = kargs.get('x', None)
            y = kargs.get('y', None)

        if x is not None and y is not None:
            non_finite_values = np.logical_not(np.isfinite(x) & np.isfinite(y))
            kargs['x'] = x[~non_finite_values]
            kargs['y'] = y[~non_finite_values]
        elif y is not None:
            non_finite_values = np.logical_not(np.isfinite(y))
            kargs['y'] = y[~non_finite_values]
        else:
            kargs['x'] = x
            kargs['y'] = y

        # Don't pass args as (x,y) have been moved to kargs
        self.updateData(**kargs)


class PlotCurvesItem(GraphicsObject):
    """ Multiple curves on a single plot that can be cached together. """

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.clear()

    def clear(self):
        self.bounds = [NAN, NAN, NAN, NAN]
        self.default_bounds = 0, 0, 1, 1
        self.objs = []

    def paint(self, p, *args):
        for o in sorted(self.objs, key=lambda x: x.zValue()):
            o.paint(p, *args)

    def add_bounds(self, c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NaN warnings are expected
            try:
                cb = c.boundingRect()
            except ValueError:  # workaround for pyqtgraph 0.10 when there are infs
                cb = QRectF()
            # keep undefined elements NaN
            self.bounds[0] = np.nanmin([cb.left(), self.bounds[0]])
            self.bounds[1] = np.nanmin([cb.top(), self.bounds[1]])
            self.bounds[2] = np.nanmax([cb.right(), self.bounds[2]])
            self.bounds[3] = np.nanmax([cb.bottom(), self.bounds[3]])

    def add_curve(self, c, ignore_bounds=False):
        if not ignore_bounds:
            self.add_bounds(c)
        self.objs.append(c)

    def boundingRect(self):
        # replace undefined (NaN) elements with defaults
        bounds = [d if np.isnan(b) else b \
                  for b, d in zip(self.bounds, self.default_bounds)]
        return QRectF(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])


def closestindex(array, v, side="left"):
    """
    Return index of a 1d sorted array closest to value v.
    """
    fi = np.searchsorted(array, v, side=side)
    if fi == 0:
        return 0
    elif fi == len(array):
        return len(array) - 1
    else:
        return fi - 1 if v - array[fi - 1] < array[fi] - v else fi


def distancetocurves(array, x, y, xpixel, ypixel, r=5, cache=None):
    # xpixel, ypixel are sizes of pixels
    # r how many pixels do we look around
    # array - curves in both x and y
    # x,y position in curve coordinates
    if cache is not None and id(x) in cache:
        xmin, xmax = cache[id(x)]
    else:
        xmin = closestindex(array[0], x - r * xpixel)
        xmax = closestindex(array[0], x + r * xpixel, side="right")
        if cache is not None:
            cache[id(x)] = xmin, xmax
    xmin = max(0, xmin - 1)
    xp = array[0][xmin:xmax + 2]
    yp = array[1][:, xmin:xmax + 2]

    # convert to distances in pixels
    xp = ((xp - x) / xpixel)
    yp = ((yp - y) / ypixel)

    # add edge point so that distance_curves works if there is just one point
    xp = np.hstack((xp, float("nan")))
    yp = np.hstack((yp, np.zeros((yp.shape[0], 1)) * float("nan")))
    dc = distance_curves(xp, yp, (0, 0))
    return dc


class InterruptException(Exception):
    pass


class ShowAverage(QObject, ConcurrentMixin):
    average_shown = pyqtSignal()

    def __init__(self, master):
        super().__init__(parent=master)
        ConcurrentMixin.__init__(self)
        self.master = master

    def show_average(self):
        master = self.master
        master.clear_graph()  # calls cancel
        master.view_average_menu.setChecked(True)
        master.set_pen_colors()
        master.viewtype = AVERAGE
        if not master.data:
            self.average_shown.emit()
        else:
            color_var = master.feature_color
            self.start(self.compute_averages, master.data, color_var, master.subset_indices,
                       master.selection_group, master.data_xsind, master.selection_type)

    @staticmethod
    def compute_averages(data: Orange.data.Table, color_var, subset_indices,
                         selection_group, data_xsind, selection_type, state: TaskState):

        def progress_interrupt(i: float):
            if state.is_interruption_requested():
                raise InterruptException

        def _split_by_color_value(data, color_var):
            rd = {}
            if color_var is None:
                rd[None] = np.full((len(data.X),), True, dtype=bool)
            else:
                cvd = data.transform(Orange.data.Domain([color_var]))
                feature_values = cvd.X[:, 0]  # obtain 1D vector
                for v in range(len(color_var.values)):
                    v1 = np.in1d(feature_values, v)
                    if np.any(v1):
                        rd[color_var.values[v]] = v1
                nanind = np.isnan(feature_values)
                if np.any(nanind):
                    rd[None] = nanind
            return rd

        results = []

        dsplit = _split_by_color_value(data, color_var)
        for colorv, indices in dsplit.items():
            for part in [None, "subset", "selection"]:
                progress_interrupt(0)
                if part is None:
                    part_selection = indices
                elif part == "selection" and selection_type:
                    part_selection = indices & (selection_group > 0)
                elif part == "subset":
                    part_selection = indices & subset_indices
                if np.any(part_selection):
                    std = apply_columns_numpy(data.X,
                                              lambda x: np.nanstd(x, axis=0),
                                              part_selection,
                                              callback=progress_interrupt)
                    mean = apply_columns_numpy(data.X,
                                               lambda x: np.nanmean(x, axis=0),
                                               part_selection,
                                               callback=progress_interrupt)
                    std = std[data_xsind]
                    mean = mean[data_xsind]
                    results.append((colorv, part, mean, std, part_selection))
        progress_interrupt(0)
        return results

    def on_done(self, res):
        master = self.master

        ysall = []
        cinfo = []

        x = master.data_x

        for colorv, part, mean, std, part_selection in res:
            if part is None:
                pen = master.pen_normal if np.any(master.subset_indices) else master.pen_subset
            elif part == "selection" and master.selection_type:
                pen = master.pen_selected
            elif part == "subset":
                pen = master.pen_subset
            ysall.append(mean)
            penc = QPen(pen[colorv])
            penc.setWidth(3)
            master.add_curve(x, mean, pen=penc)
            master.add_fill_curve(x, mean + std, mean - std, pen=penc)
            cinfo.append((colorv, part, part_selection))

        master.curves.append((x, np.array(ysall)))
        master.multiple_curves_info = cinfo
        master.curves_cont.update()
        master.plot.vb.set_mode_panning()

        self.average_shown.emit()

    def on_partial_result(self, result):
        pass

    def on_exception(self, ex: Exception):
        if isinstance(ex, InterruptException):
            return

        raise ex


class InteractiveViewBox(ViewBox):
    def __init__(self, graph):
        ViewBox.__init__(self, enableMenu=False)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.zoomstartpoint = None
        self.current_selection = None
        self.action = PANNING
        self.y_padding = 0.02
        self.x_padding = 0

        # line for marking selection
        self.selection_line = pg.PlotCurveItem()
        self.selection_line.setPen(pg.mkPen(color=QColor(Qt.black), width=2, style=Qt.DotLine))
        self.selection_line.setZValue(1e9)
        self.selection_line.hide()
        self.addItem(self.selection_line, ignoreBounds=True)

        # yellow marker for ending the polygon
        self.selection_poly_marker = pg.ScatterPlotItem()
        self.selection_poly_marker.setPen(pg.mkPen(color=QColor(Qt.yellow), width=2))
        self.selection_poly_marker.setSize(SELECT_POLYGON_TOLERANCE * 2)
        self.selection_poly_marker.setBrush(None)
        self.selection_poly_marker.setZValue(1e9 + 1)
        self.selection_poly_marker.hide()
        self.selection_poly_marker.mouseClickEvent = lambda x: x  # ignore mouse clicks
        self.addItem(self.selection_poly_marker, ignoreBounds=True)

        self.sigRangeChanged.connect(self.resized)
        self.sigResized.connect(self.resized)

        self.tiptexts = None

    def resized(self):
        self.position_tooltip()

    def position_tooltip(self):
        if self.tiptexts:  # if initialized
            self.scene().select_tooltip.setPos(10, self.height())

    def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
        super().enableAutoRange(axis=axis, enable=False, x=x, y=y)

    def update_selection_tooltip(self, modifiers=Qt.NoModifier):
        if not self.tiptexts:
            self._create_select_tooltip()
        modifiers &= Qt.ShiftModifier + Qt.ControlModifier + Qt.AltModifier
        text = self.tiptexts.get(int(modifiers), self.tiptexts[0])
        self.tip_textitem.setHtml(text)
        if self.action in [SELECT, SELECT_SQUARE, SELECT_POLYGON]:
            self.scene().select_tooltip.show()
        else:
            self.scene().select_tooltip.hide()

    def _create_select_tooltip(self):
        scene = self.scene()
        tip_parts = [
            (Qt.ShiftModifier, "Shift: Add group"),
            (Qt.ShiftModifier + Qt.ControlModifier,
             "Shift-{}: Append to group".
             format("Cmd" if sys.platform == "darwin" else "Ctrl")),
            (Qt.AltModifier, "Alt: Remove")
        ]
        all_parts = ", ".join(part for _, part in tip_parts)
        self.tiptexts = {
            int(modifier): all_parts.replace(part, "<b>{}</b>".format(part))
            for modifier, part in tip_parts
        }
        self.tiptexts[0] = all_parts
        self.tip_textitem = text = QGraphicsTextItem()

        # Set to the longest text
        text.setHtml(self.tiptexts[Qt.ShiftModifier + Qt.ControlModifier])
        text.setPos(4, 2)
        r = text.boundingRect()
        rect = QGraphicsRectItem(0, 0, r.width() + 8, r.height() + 4)
        rect.setBrush(QColor(224, 224, 224, 212))
        rect.setPen(QPen(Qt.NoPen))
        scene.select_tooltip = scene.createItemGroup([rect, text])
        scene.select_tooltip.hide()
        self.position_tooltip()
        self.update_selection_tooltip(Qt.NoModifier)

    def safe_update_scale_box(self, buttonDownPos, currentPos):
        x, y = currentPos
        if buttonDownPos[0] == x:
            x += 1
        if buttonDownPos[1] == y:
            y += 1
        self.updateScaleBox(buttonDownPos, Point(x, y))

    # noinspection PyPep8Naming,PyMethodOverriding
    def mouseDragEvent(self, ev, axis=None):
        if ev.button() & Qt.RightButton:
            ev.accept()
        if self.action == ZOOMING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        elif self.action == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        elif self.action in [SELECT, SELECT_SQUARE, SELECT_POLYGON] \
                and ev.button() == Qt.LeftButton and self.graph.selection_type:
            pos = self.childGroup.mapFromParent(ev.pos())
            if self.current_selection is None:
                self.current_selection = [pos]
            elif ev.isFinish():
                self._add_selection_point(pos)
            ev.accept()
        else:
            ev.ignore()

    def suggestPadding(self, axis):
        return 0.

    def mouseMovedEvent(self, ev):  # not a Qt event!
        if self.action == ZOOMING and self.zoomstartpoint:
            pos = self.mapFromView(self.mapSceneToView(ev))
            self.updateScaleBox(self.zoomstartpoint, pos)
        if self.action in [SELECT, SELECT_SQUARE, SELECT_POLYGON] and self.current_selection:
            # ev is a position of the whole component (with axes)
            pos = self.childGroup.mapFromParent(self.mapFromView(self.mapSceneToView(ev)))
            if self.action == SELECT:
                self.updateSelectionLine(pos)
            elif self.action == SELECT_SQUARE:
                self.updateSelectionSquare(pos)
            elif self.action == SELECT_POLYGON:
                self.updateSelectionPolygon(pos)

    def updateSelectionLine(self, p2):
        p1 = self.current_selection[0]
        self.selection_line.setData(x=[p1.x(), p2.x()], y=[p1.y(), p2.y()])
        self.selection_line.show()

    def updateSelectionSquare(self, p2):
        p1 = self.current_selection[0]
        self.selection_line.setData(x=[p1.x(), p1.x(), p2.x(), p2.x(), p1.x()],
                                    y=[p1.y(), p2.y(), p2.y(), p1.y(), p1.y()])
        self.selection_line.show()

    def _distance_pixels(self, p1, p2):
        xpixel, ypixel = self.viewPixelSize()
        dx = (p1.x() - p2.x()) / xpixel
        dy = (p1.y() - p2.y()) / ypixel
        return (dx ** 2 + dy ** 2) ** 0.5

    def updateSelectionPolygon(self, p):
        first = self.current_selection[0]
        polygon = self.current_selection + [p]
        self.selection_line.setData(x=[e.x() for e in polygon],
                                    y=[e.y() for e in polygon])
        self.selection_line.show()
        if self._distance_pixels(first, p) < SELECT_POLYGON_TOLERANCE:
            self.selection_poly_marker.setData(x=[first.x()], y=[first.y()])
            self.selection_poly_marker.show()
        else:
            self.selection_poly_marker.hide()

    def keyPressEvent(self, ev):
        # cancel current selection process
        if self.action in [SELECT, SELECT_SQUARE, SELECT_POLYGON] and \
                ev.key() in [Qt.Key_Escape]:
            self.set_mode_panning()
            ev.accept()
        else:
            self.update_selection_tooltip(ev.modifiers())
            ev.ignore()

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.update_selection_tooltip(event.modifiers())

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton and \
                (self.action == ZOOMING or self.action in [SELECT, SELECT_SQUARE, SELECT_POLYGON]):
            ev.accept()
            self.set_mode_panning()
        elif ev.button() == Qt.RightButton:
            ev.accept()
            self.autoRange()
        if self.action != ZOOMING and self.action not in [SELECT, SELECT_SQUARE, SELECT_POLYGON] \
                and ev.button() == Qt.LeftButton and self.graph.selection_type:
            pos = self.childGroup.mapFromParent(ev.pos())
            self.graph.select_by_click(pos)
            ev.accept()
        if self.action == ZOOMING and ev.button() == Qt.LeftButton:
            if self.zoomstartpoint is None:
                self.zoomstartpoint = ev.pos()
            else:
                self.updateScaleBox(self.zoomstartpoint, ev.pos())
                self.rbScaleBox.hide()
                ax = QRectF(Point(self.zoomstartpoint), Point(ev.pos()))
                ax = self.childGroup.mapRectFromParent(ax)
                self.showAxRect(ax)
                self.axHistoryPointer += 1
                self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                self.set_mode_panning()
            ev.accept()
        if self.action in [SELECT, SELECT_SQUARE, SELECT_POLYGON] \
                and ev.button() == Qt.LeftButton and self.graph.selection_type:
            pos = self.childGroup.mapFromParent(ev.pos())
            if self.current_selection is None:
                self.current_selection = [pos]
            else:
                self._add_selection_point(pos)
            ev.accept()

    def _add_selection_point(self, pos):
        startp = self.current_selection[0]
        if self.action == SELECT:
            self.graph.select_line(startp, pos)
            self.set_mode_panning()
        elif self.action == SELECT_SQUARE:
            self.graph.select_square(startp, pos)
            self.set_mode_panning()
        elif self.action == SELECT_POLYGON:
            self.polygon_point_click(pos)

    def polygon_point_click(self, p):
        first = self.current_selection[0]
        if self._distance_pixels(first, p) < SELECT_POLYGON_TOLERANCE:
            self.current_selection.append(first)
            self.graph.select_polygon(self.current_selection)
            self.set_mode_panning()
        else:
            self.current_selection.append(p)

    def showAxRect(self, ax):
        super().showAxRect(ax)
        if self.action == ZOOMING:
            self.set_mode_panning()

    def pad_current_view_y(self):
        if self.y_padding:
            qrect = self.targetRect()
            self.setYRange(qrect.bottom(), qrect.top(), padding=self.y_padding)

    def pad_current_view_x(self):
        if self.x_padding:
            qrect = self.targetRect()
            self.setXRange(qrect.left(), qrect.right(), padding=self.x_padding)

    def autoRange(self):
        super().autoRange()
        self.pad_current_view_y()
        self.pad_current_view_x()

    def cancel_zoom(self):
        self.setMouseMode(self.PanMode)
        self.rbScaleBox.hide()
        self.zoomstartpoint = None
        self.action = PANNING
        self.unsetCursor()
        self.update_selection_tooltip()

    def set_mode_zooming(self):
        self.set_mode_panning()
        self.setMouseMode(self.RectMode)
        self.action = ZOOMING
        self.setCursor(Qt.CrossCursor)
        self.update_selection_tooltip()

    def set_mode_panning(self):
        self.cancel_zoom()
        self.cancel_select()

    def cancel_select(self):
        self.setMouseMode(self.PanMode)
        self.selection_line.hide()
        self.selection_poly_marker.hide()
        self.current_selection = None
        self.action = PANNING
        self.unsetCursor()
        self.update_selection_tooltip()

    def set_mode_select(self):
        self.set_mode_panning()
        self.setMouseMode(self.RectMode)
        self.action = SELECT
        self.setCursor(Qt.CrossCursor)
        self.update_selection_tooltip()

    def set_mode_select_square(self):
        self.set_mode_panning()
        self.setMouseMode(self.RectMode)
        self.action = SELECT_SQUARE
        self.setCursor(Qt.CrossCursor)
        self.update_selection_tooltip()

    def set_mode_select_polygon(self):
        self.set_mode_panning()
        self.setMouseMode(self.RectMode)
        self.action = SELECT_POLYGON
        self.setCursor(Qt.CrossCursor)
        self.update_selection_tooltip()


class InteractiveViewBoxC(InteractiveViewBox):

    def wheelEvent(self, ev, axis=None):
        # separate axis handling with modifier keys
        if axis is None:
            axis = 1 if ev.modifiers() & Qt.ControlModifier else 0
        super().wheelEvent(ev, axis=axis)


class SelectRegion(pg.LinearRegionItem):
    def __init__(self, *args, **kwargs):
        pg.LinearRegionItem.__init__(self, *args, **kwargs)
        for l in self.lines:
            l.setCursor(Qt.SizeHorCursor)
        self.setZValue(10)
        color = QColor(Qt.red)
        color.setAlphaF(0.05)
        self.setBrush(pg.mkBrush(color))


class NoSuchCurve(ValueError):
    pass


class CurvePlot(QWidget, OWComponent, SelectionGroupMixin):
    sample_seed = Setting(0, schema_only=True)
    label_title = Setting("")
    label_xaxis = Setting("")
    label_yaxis = Setting("")
    range_x1 = Setting(None)
    range_x2 = Setting(None)
    range_y1 = Setting(None)
    range_y2 = Setting(None)
    prominence = Setting(None, schema_only=True)
    line_overlap = Setting(None, schema_only=True)
    peak_min = Setting(None, schema_only=True)
    peak_max = Setting(None, schema_only=True)
    peak_locations = Setting([], schema_only=True)
    minimum_point = Setting(None, schema_only=True)
    maximum_point = Setting(None, schema_only=True)
    feature_color = ContextSetting(None)
    color_individual = Setting(False)  # color individual curves (in a cycle) if no feature_color
    invertX = Setting(False)
    viewtype = Setting(INDIVIDUAL)
    show_grid = Setting(False)

    selection_changed = pyqtSignal()
    new_sampling = pyqtSignal(object)
    highlight_changed = pyqtSignal()

    def __init__(self, parent: OWWidget, select=SELECTNONE):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)
        SelectionGroupMixin.__init__(self)

        self.show_average_thread = ShowAverage(self)
        self.show_average_thread.average_shown.connect(self.rescale)

        self.parent = parent

        self.selection_type = select
        self.select_at_least_1 = False
        self.saving_enabled = True
        self.clear_data()
        self.subset = None  # current subset input, an array of indices
        self.subset_indices = None  # boolean index array with indices in self.data
        self.peak_state = 0  # Value for determining what our peak clear should do
        self.minium_point = None
        self.maximum_point = None
        self.plotview = pg.PlotWidget(background="w", viewBox=InteractiveViewBoxC(self))
        self.plot = self.plotview.getPlotItem()
        self.plot.hideButtons()  # hide the autorange button
        self.plot.setDownsampling(auto=True, mode="peak")
        self.connected_views = []
        self.plot.vb.sigResized.connect(self._update_connected_views)

        for pos in ["top", "right"]:
            self.plot.showAxis(pos, True)
            self.plot.getAxis(pos).setStyle(showValues=False)

        self.markings = []
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.scene().sigMouseMoved.connect(self.mouse_moved_viewhelpers)
        self.plot.scene().sigMouseMoved.connect(self.plot.vb.mouseMovedEvent)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=20,
                                    slot=self.mouse_moved_closest, delay=0.1)
        self.plot.vb.sigRangeChanged.connect(self.resized)
        self.plot.vb.sigResized.connect(self.resized)
        self.pen_mouse = pg.mkPen(color=(0, 0, 255), width=2)
        pen_normal, pen_selected, pen_subset = self._generate_pens(QColor(60, 60, 60, 200),
                                                                   QColor(200, 200, 200, 127),
                                                                   QColor(0, 0, 0, 255))
        self._default_pen_selected = pen_selected
        self.pen_normal = defaultdict(lambda: pen_normal)
        self.pen_subset = defaultdict(lambda: pen_subset)
        self.pen_selected = defaultdict(lambda: pen_selected)
        self.label = pg.TextItem("", anchor=(1, 0))
        self.label.setText("", color=(0, 0, 0))
        self.discrete_palette = None
        QPixmapCache.setCacheLimit(max(QPixmapCache.cacheLimit(), 100 * 1024))
        self.curves_cont = PlotCurvesItem()
        self.important_decimals = 10, 10

        self.plot.scene().installEventFilter(
            HelpEventDelegate(self.help_event, self))

        # whether to rescale at next update
        self.rescale_next = True

        self.MOUSE_RADIUS = 20

        self.clear_graph()

        # interface settings
        self.location = True  # show current position
        self.markclosest = True  # mark
        self.crosshair = True
        self.crosshair_hidden = True

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)
        # prepare interface according to the new context
        self.parent.contextAboutToBeOpened.connect(lambda x: self.init_interface_data(x[0]))
        actions = []

        resample_curves = QAction(
            "Resample curves", self, shortcut=Qt.Key_R,
            triggered=lambda x: self.resample_curves(self.sample_seed + 1)
        )
        resample_curves.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(resample_curves)
        reset_curves = QAction(
            "Resampling reset", self, shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_R),
            triggered=lambda x: self.resample_curves(0)
        )
        reset_curves.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(reset_curves)

        zoom_in = QAction(
            "Zoom in", self, triggered=self.plot.vb.set_mode_zooming
        )
        zoom_in.setShortcuts([Qt.Key_Z, QKeySequence(QKeySequence.ZoomIn)])
        zoom_in.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(zoom_in)
        zoom_fit = QAction(
            "Zoom to fit", self,
            triggered=lambda x: (self.plot.vb.autoRange(), self.plot.vb.set_mode_panning())
        )
        zoom_fit.setShortcuts([Qt.Key_Backspace, QKeySequence(Qt.ControlModifier | Qt.Key_0)])
        zoom_fit.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(zoom_fit)
        rescale_y = QAction(
            "Rescale Y to fit", self, shortcut=Qt.Key_D,
            triggered=self.rescale_current_view_y
        )
        rescale_y.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(rescale_y)

        self.view_average_menu = QAction(
            "Show averages", self, shortcut=Qt.Key_A, checkable=True,
            triggered=lambda x: self.viewtype_changed()
        )
        self.view_average_menu.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(self.view_average_menu)

        self.show_grid_a = QAction(
            "Show grid", self, shortcut=Qt.Key_G, checkable=True,
            triggered=self.grid_changed
        )
        self.show_grid_a.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(self.show_grid_a)
        self.invertX_menu = QAction(
            "Invert X", self, shortcut=Qt.Key_X, checkable=True,
            triggered=self.invertX_changed
        )
        self.invertX_menu.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(self.invertX_menu)

        peaklabler_menu = MenuFocus("Peak Labeling Options", self)
        peak_action = QWidgetAction(self)
        layout = QGridLayout()
        prominence_box = gui.widgetBox(self, margin=5, orientation=layout)
        prominence_box.setFocusPolicy(Qt.TabFocus)
        self.single_peak = QAction(
            None, self, shortcut=Qt.Key_P, checkable=False,
            triggered=self.peak_apply
        )
        self.single_peak.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.single_peak = QPushButton("Add Single Line", self)
        self.reset_labels = QPushButton('Reset graph', self)
        self.single_peak.clicked.connect(self.find_peak_variables)
        self.single_peak.clicked.connect(self.peak_apply)
        self.reset_labels.clicked.connect(self.change_state)
        self.reset_labels.clicked.connect(self.clear_graph)
        self.reset_labels.clicked.connect(self.update_view)
        self.reset_labels.clicked.connect(self.change_state)
        self.prominence_box = lineEditDecimalOrNone(None, self, 'prominence', bottom=0)
        self.peak_height_min = lineEditDecimalOrNone(None, self, 'peak_min')
        self.peak_height_max = lineEditDecimalOrNone(None, self, 'peak_max')
        self.peak_label_distance = lineEditDecimalOrNone(None, self, "line_overlap", bottom=0)
        layout.addWidget(self.single_peak, 4, 0)
        layout.addWidget(QLabel("Prominence"), 0, 0, Qt.AlignRight)
        layout.addWidget(QLabel('Minimum Peak Height'), 1, 0, Qt.AlignRight)
        layout.addWidget(QLabel('Maximum Peak Height'), 2, 0, Qt.AlignRight)
        layout.addWidget(QLabel('Minimum Distance Between lines'), 3, 0)
        layout.addWidget(self.prominence_box, 0, 1)
        layout.addWidget(self.peak_height_min, 1, 1)
        layout.addWidget(self.peak_height_max, 2, 1)
        layout.addWidget(self.peak_label_distance, 3, 1)
        layout.addWidget(self.reset_labels, 5, 1)
        self.apply_button = QPushButton("Automatic label", self)
        self.apply_button.clicked.connect(self.find_peak_variables)
        self.apply_button.clicked.connect(self.set_auto_peaks)
        layout.addWidget(self.apply_button, 4, 1, Qt.AlignRight)
        peak_action.setDefaultWidget(prominence_box)
        peaklabler_menu.addAction(peak_action)

        if self.selection_type == SELECTMANY:
            select_curves = QAction(
                "Select (line)", self, triggered=self.line_select_start,
            )
            select_curves.setShortcuts([Qt.Key_S])
            select_curves.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            actions.append(select_curves)
        if self.saving_enabled:
            save_graph = QAction(
                "Save graph", self, triggered=self.save_graph,
            )
            save_graph.setShortcuts([QKeySequence(Qt.ControlModifier | Qt.Key_S)])
            save_graph.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            actions.append(save_graph)

        range_menu = MenuFocus("Define view range", self)
        range_action = QWidgetAction(self)
        layout = QGridLayout()
        range_box = gui.widgetBox(self, margin=5, orientation=layout)
        range_box.setFocusPolicy(Qt.TabFocus)
        self.range_e_x1 = lineEditFloatOrNone(None, self, "range_x1")
        range_box.setFocusProxy(self.range_e_x1)
        self.range_e_x2 = lineEditFloatOrNone(None, self, "range_x2")
        layout.addWidget(QLabel("X"), 0, 0, Qt.AlignRight)
        layout.addWidget(self.range_e_x1, 0, 1)
        layout.addWidget(QLabel("-"), 0, 2)
        layout.addWidget(self.range_e_x2, 0, 3)
        self.range_e_y1 = lineEditFloatOrNone(None, self, "range_y1")
        self.range_e_y2 = lineEditFloatOrNone(None, self, "range_y2")
        layout.addWidget(QLabel("Y"), 1, 0, Qt.AlignRight)
        layout.addWidget(self.range_e_y1, 1, 1)
        layout.addWidget(QLabel("-"), 1, 2)
        layout.addWidget(self.range_e_y2, 1, 3)
        b = gui.button(None, self, "Apply", callback=self.set_limits)
        layout.addWidget(b, 2, 3, Qt.AlignRight)
        range_action.setDefaultWidget(range_box)
        range_menu.addAction(range_action)

        layout = QGridLayout()
        self.plotview.setLayout(layout)
        self.button = QPushButton("Menu", self.plotview)
        self.button.setAutoDefault(False)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self.button, 0, 0)
        view_menu = MenuFocus(self)
        self.button.setMenu(view_menu)
        view_menu.addActions(actions)
        view_menu.addMenu(peaklabler_menu)
        view_menu.addMenu(range_menu)
        self.addActions(actions)

        self.color_individual_menu = QAction(
            "Color individual curves", self, shortcut=Qt.Key_I, checkable=True,
            triggered=lambda x: self.color_individual_changed()
        )
        self.color_individual_menu.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        view_menu.addAction(self.color_individual_menu)
        self.addAction(self.color_individual_menu)

        choose_color_action = QWidgetAction(self)
        choose_color_box = gui.hBox(self)
        choose_color_box.setFocusPolicy(Qt.TabFocus)
        label = gui.label(choose_color_box, self, "Color by")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.feature_color_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES,
                                               valid_types=(DiscreteVariable,), placeholder="None")
        self.feature_color_combo = gui.comboBox(
            choose_color_box, self, "feature_color",
            callback=self.update_view, model=self.feature_color_model)
        choose_color_box.setFocusProxy(self.feature_color_combo)
        choose_color_action.setDefaultWidget(choose_color_box)
        view_menu.addAction(choose_color_action)

        cycle_colors = QShortcut(Qt.Key_C, self, self.cycle_color_attr, context=Qt.WidgetWithChildrenShortcut)

        labels_action = QWidgetAction(self)
        layout = QGridLayout()
        labels_box = gui.widgetBox(self, margin=0, orientation=layout)
        t = gui.lineEdit(None, self, "label_title", label="Title:",
                         callback=self.labels_changed, callbackOnType=self.labels_changed)
        layout.addWidget(QLabel("Title:"), 0, 0, Qt.AlignRight)
        layout.addWidget(t, 0, 1)
        t = gui.lineEdit(None, self, "label_xaxis", label="X-axis:",
                         callback=self.labels_changed, callbackOnType=self.labels_changed)
        layout.addWidget(QLabel("X-axis:"), 1, 0, Qt.AlignRight)
        layout.addWidget(t, 1, 1)
        t = gui.lineEdit(None, self, "label_yaxis", label="Y-axis:",
                         callback=self.labels_changed, callbackOnType=self.labels_changed)
        layout.addWidget(QLabel("Y-axis:"), 2, 0, Qt.AlignRight)
        layout.addWidget(t, 2, 1)
        labels_action.setDefaultWidget(labels_box)
        view_menu.addAction(labels_action)
        self.labels_changed()  # apply saved labels

        self.grid_apply()
        self.invertX_apply()
        self.color_individual_apply()
        self._color_individual_cycle = COLORBREWER_SET1
        self.plot.vb.set_mode_panning()

        self.reports = {}  # current reports

        self.legend = self._create_legend()
        self.viewhelpers_show()

    def change_state(self):
        #A very ineffcient workaround to line 1201 and having our peaks reappear
        if self.peak_state == 0:
            self.peak_state = 1
        else:
            self.peak_state = 0

    def _update_connected_views(self):
        for w in self.connected_views:
            w.setGeometry(self.plot.vb.sceneBoundingRect())

    def _create_legend(self):
        legend = LegendItem()
        legend.setParentItem(self.plotview.getViewBox())
        legend.restoreAnchor(((1, 0), (1, 0)))
        return legend

    def init_interface_data(self, data):
        old_domain = self.data.domain if self.data else None
        domain = data.domain if data is not None else None
        self.feature_color_model.set_domain(domain)
        if old_domain is not None and domain != old_domain:  # do not reset feature_color
            self.feature_color = self.feature_color_model[0] if self.feature_color_model else None

    def line_select_start(self):
        if self.viewtype == INDIVIDUAL:
            self.plot.vb.set_mode_select()

    def help_event(self, ev):
        text = ""
        if self.highlighted is not None:
            if self.viewtype == INDIVIDUAL:
                index = self.sampled_indices[self.highlighted]
                variables = self.data.domain.metas + self.data.domain.class_vars
                text += "".join(
                    '{} = {}\n'.format(attr.name, self.data[index][attr])
                    for attr in variables)
            elif self.viewtype == AVERAGE:
                c = self.multiple_curves_info[self.highlighted]
                nc = sum(c[2])
                if c[0] is not None:
                    text += str(c[0]) + " "
                if c[1]:
                    text += "({})".format(c[1])
                if text:
                    text += "\n"
                text += "{} curves".format(nc)
        if text:
            text = text.rstrip()
            text = ('<span style="white-space:pre">{}</span>'
                    .format(escape(text)))
            QToolTip.showText(ev.screenPos(), text, widget=self.plotview)
            return True
        else:
            return False

    def report(self, reporter, contents):
        self.reports[id(reporter)] = contents

    def report_finished(self, reporter):
        try:
            self.reports.pop(id(reporter))
        except KeyError:
            pass  # ok if it was already removed
        if not self.reports:
            pass

    def cycle_color_attr(self):
        elements = [a for a in self.feature_color_model]
        currentind = elements.index(self.feature_color)
        next = (currentind + 1) % len(self.feature_color_model)
        self.feature_color = elements[next]
        self.update_view()

    def set_limits(self):
        vr = self.plot.vb.viewRect()
        x1 = self.range_x1 if self.range_x1 is not None else vr.left()
        x2 = self.range_x2 if self.range_x2 is not None else vr.right()
        y1 = self.range_y1 if self.range_y1 is not None else vr.top()
        y2 = self.range_y2 if self.range_y2 is not None else vr.bottom()
        self.plot.vb.setXRange(x1, x2)
        self.plot.vb.setYRange(y1, y2)

    def set_auto_peaks(self):
        prominence = self.prominence
        minHeight = self.peak_min
        maxHeight = self.peak_max
        line_overlap = self.line_overlap
        self.peak_apply_auto(prominence=prominence, minHeight=minHeight,
                             maxHeight=maxHeight, line_overlap=line_overlap)

    def labels_changed(self):
        self.plot.setTitle(self.label_title)
        if not self.label_title:
            self.plot.setTitle(None)
        self.plot.setLabels(bottom=self.label_xaxis)
        self.plot.showLabel("bottom", bool(self.label_xaxis))
        self.plot.getAxis("bottom").resizeEvent()  # align text
        self.plot.setLabels(left=self.label_yaxis)
        self.plot.showLabel("left", bool(self.label_yaxis))
        self.plot.getAxis("left").resizeEvent()  # align text

    def grid_changed(self):
        self.show_grid = not self.show_grid
        self.grid_apply()

    def grid_apply(self):
        self.plot.showGrid(self.show_grid, self.show_grid, alpha=0.3)
        self.show_grid_a.setChecked(self.show_grid)

    def find_peak_variables(self):
            self.maximum_point = np.nanmax(self.curves_plotted[0][1])
            self.minimum_point = np.nanmin(self.curves_plotted[0][1])
            self.start_point = np.median(getx(self.data))

    def peak_apply(self, position):
        if self.viewtype == INDIVIDUAL:
            Label_line = VerticalPeakLine()
            Label_line.setMovable(True)
            Label_line.setPen(pg.mkPen(color=QColor(Qt.black), width=2, style=Qt.DotLine))
            Label_line.setSpan(mn=self.minimum_point, mx=self.maximum_point)
            Label_line.label.setColor(color=QColor(Qt.black))
            Label_line.label.setPosition(1)
            Label_line.label.setMovable(True)
            if position:
                Label_line.setPos(position)
                Label_line.label.setText(str(round(position, 3)))
            else:
                Label_line.setPos(self.start_point)
                Label_line.label.setText(str(round(self.start_point, 3)))
            self.plotview.addItem(Label_line)

    def peak_apply_auto(self, prominence, minHeight, maxHeight, line_overlap):
        if self.viewtype == INDIVIDUAL:
            if line_overlap is None:
                line_overlap = 0
            x_axis = self.data_x
            data = self.curves_plotted[0][1]
            peak = []
            single_curve = None
            for i, val in enumerate(data):
                if not isinstance(val, (list, np.ndarray, tuple)) or single_curve == True :
                    single_curve = True
                    single_spectra = data
                else:
                    single_spectra = val
                    single_curve = False
                peaks, _ = find_peaks(single_spectra, height=([minHeight,
                                                                maxHeight]), prominence=prominence)
                peaks = x_axis[peaks]  # array with all locations of peaks
                for z, var in enumerate(peaks):
                    peak.append(var)
                if single_curve:
                    break
            peak_locations = []
            sorted_peaks = np.sort(peak)
            for i, val in enumerate(sorted_peaks):
                if val not in peak_locations and peak_locations != [] \
                        and abs(sorted_peaks[i-1]-val) >= line_overlap:
                    # First check for used peaks then check
                    # if its greater than than nearby values
                    peak_locations.append(val)
                elif not peak_locations:
                    peak_locations.append(val)
                    # used for first value computation as it isn't in the array
            self.peak_locations = peak_locations
            for k, val in enumerate(peak_locations):
                self.peak_apply(position=val)

    def invertX_changed(self):
        self.invertX = not self.invertX
        self.invertX_apply()

    def invertX_apply(self):
        self.plot.vb.invertX(self.invertX)
        self.resized()
        # force redraw of axes (to avoid a pyqtgraph bug)
        vr = self.plot.vb.viewRect()
        self.plot.vb.setRange(xRange=(0, 1), yRange=(0, 1))
        self.plot.vb.setRange(rect=vr)
        self.invertX_menu.setChecked(self.invertX)

    def color_individual_changed(self):
        self.color_individual = not self.color_individual
        self.color_individual_apply()
        self.update_view()

    def color_individual_apply(self):
        self.color_individual_menu.setChecked(self.color_individual)

    def save_graph(self):
        try:
            self.viewhelpers_hide()
            saveplot.save_plot(self.plotview, self.parent.graph_writers)
        finally:
            self.viewhelpers_show()

    def clear_data(self):
        self.data = None
        self.data_x = None  # already sorted x-axis
        self.data_xsind = None  # sorting indices for x-axis
        self.discrete_palette = None

    def clear_graph(self):
        self.show_average_thread.cancel()
        self.highlighted = None
        # reset caching. if not, it is not cleared when view changing when zoomed
        self.curves_cont.setCacheMode(QGraphicsItem.NoCache)
        self.curves_cont.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.plot.vb.disableAutoRange()
        self.curves_cont.clear()
        self.curves_cont.update()
        self.plotview.clear()
        self.multiple_curves_info = []
        self.curves_plotted = []  # currently plotted elements (for rescale)
        self.curves = []  # for finding closest curve
        self.plotview.addItem(self.label, ignoreBounds=True)
        self.highlighted_curve = FinitePlotCurveItem(pen=self.pen_mouse)
        self.highlighted_curve.setZValue(10)
        self.highlighted_curve.hide()
        self.plot.addItem(self.highlighted_curve, ignoreBounds=True)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.viewhelpers = True
        self.plot.addItem(self.curves_cont)

        self.sampled_indices = []
        self.sampled_indices_inverse = {}
        self.sampling = None
        self.new_sampling.emit(None)

        for m in self.markings:
            self.plot.addItem(m, ignoreBounds=True)

        if self.peak_state != 1:
            if self.minimum_point is not None and \
                    self.maximum_point is not None:
                #ensures we don't try and plot to a nonexistent bound
                for i in range(len(self.peak_locations)):
                    self.peak_apply(position=self.peak_locations[i])
                    #reloads our peaks when we reopen

    def resized(self):
        self.important_decimals = pixel_decimals(self.plot.vb)

        try:
            vr = self.plot.vb.viewRect()
        except:
            return

        if self.invertX:
            self.label.setPos(vr.bottomLeft())
        else:
            self.label.setPos(vr.bottomRight())
        xd, yd = self.important_decimals
        self.range_e_x1.setPlaceholderText(strdec(vr.left(), xd))
        self.range_e_x2.setPlaceholderText(strdec(vr.right(), xd))
        self.range_e_y1.setPlaceholderText(strdec(vr.top(), yd))
        self.range_e_y2.setPlaceholderText(strdec(vr.bottom(), yd))

    def make_selection(self, data_indices):
        add_to_group, add_group, remove = selection_modifiers()
        invd = self.sampled_indices_inverse
        data_indices_set = set(data_indices if data_indices is not None else set())
        redraw_curve_indices = set()

        def current_selection():
            return set(icurve for idata, icurve in invd.items() if self.selection_group[idata])

        old_sel_ci = current_selection()

        if add_to_group:  # both keys - need to test it before add_group
            selnum = np.max(self.selection_group)
        elif add_group:
            selnum = np.max(self.selection_group) + 1
        elif remove:
            selnum = 0
        else:
            # remove the current selection
            redraw_curve_indices.update(old_sel_ci)
            self.selection_group *= 0  # remove
            selnum = 1
        # add new
        if data_indices is not None:
            self.selection_group[data_indices] = selnum
            redraw_curve_indices.update(
                icurve for idata, icurve in invd.items() if idata in data_indices_set)

        fixes = self.make_selection_valid()
        redraw_curve_indices.update(
            icurve for idata, icurve in invd.items() if idata in fixes)

        new_sel_ci = current_selection()

        # redraw whole selection if it increased or decreased over the threshold
        if len(old_sel_ci) <= MAX_THICK_SELECTED < len(new_sel_ci) or \
                len(old_sel_ci) > MAX_THICK_SELECTED >= len(new_sel_ci):
            redraw_curve_indices.update(old_sel_ci)

        self.set_curve_pens(redraw_curve_indices)
        self.selection_changed_confirm()

    def make_selection_valid(self):
        """ Make the selection valid and return the changed positions. """
        if self.select_at_least_1 and not len(np.flatnonzero(self.selection_group)) \
                and len(self.data) > 0:  # no selection
            self.selection_group[0] = 1
            return set([0])
        return set()

    def selection_changed_confirm(self):
        # reset average view; individual was already handled in make_selection
        if self.viewtype == AVERAGE:
            self.show_average()
        self.prepare_settings_for_saving()
        self.selection_changed.emit()

    def viewhelpers_hide(self):
        self.label.hide()
        self.vLine.hide()
        self.hLine.hide()

    def viewhelpers_show(self):
        self.label.show()
        if self.crosshair and not self.crosshair_hidden:
            self.vLine.show()
            self.hLine.show()
        else:
            self.vLine.hide()
            self.hLine.hide()

    def mouse_moved_viewhelpers(self, pos):
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.vb.mapSceneToView(pos)
            posx, posy = mousePoint.x(), mousePoint.y()

            labels = []
            for a, vs in sorted(self.reports.items()):
                for v in vs:
                    if isinstance(v, tuple) and len(v) == 2:
                        if v[0] == "x":
                            labels.append(strdec(v[1], self.important_decimals[0]))
                            continue
                    labels.append(str(v))
            labels = " ".join(labels)
            self.crosshair_hidden = bool(labels)

            if self.location and not labels:
                labels = strdec(posx, self.important_decimals[0]) + " " + \
                         strdec(posy, self.important_decimals[1])
            self.label.setText(labels, color=(0, 0, 0))

            self.vLine.setPos(posx)
            self.hLine.setPos(posy)
            self.viewhelpers_show()
        else:
            self.viewhelpers_hide()

    def mouse_moved_closest(self, evt):
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos) and \
                self.curves and len(self.curves[0][0]):  # need non-zero x axis!
            mousePoint = self.plot.vb.mapSceneToView(pos)
            posx, posy = mousePoint.x(), mousePoint.y()

            cache = {}
            bd = None
            if self.markclosest and self.plot.vb.action != ZOOMING:
                xpixel, ypixel = self.plot.vb.viewPixelSize()
                distances = distancetocurves(self.curves[0], posx, posy, xpixel, ypixel,
                                             r=self.MOUSE_RADIUS, cache=cache)
                try:
                    mindi = np.nanargmin(distances)
                    if distances[mindi] < self.MOUSE_RADIUS:
                        bd = mindi
                except ValueError:  # if all distances are NaN
                    pass
            if self.highlighted != bd:
                QToolTip.hideText()
            self.highlight(bd)

    def highlight(self, index, emit=True):
        """
        Highlight shown curve with the given index (of the sampled curves).
        """
        old = self.highlighted
        if index is not None and not 0 <= index < len(self.curves[0][1]):
            raise NoSuchCurve()

        self.highlighted = index
        if self.highlighted is None:
            self.highlighted_curve.hide()
        else:
            if old != self.highlighted:
                x = self.curves[0][0]
                y = self.curves[0][1][self.highlighted]
                self.highlighted_curve.setData(x=x, y=y)
            self.highlighted_curve.show()
        if emit and old != self.highlighted:
            self.highlight_changed.emit()

    def highlighted_index_in_data(self):
        if self.highlighted is not None and self.viewtype == INDIVIDUAL:
            return self.sampled_indices[self.highlighted]
        return None

    def highlight_index_in_data(self, index, emit):
        if self.viewtype == AVERAGE:  # do not highlight average view
            index = None
        if index in self.sampled_indices_inverse:
            index = self.sampled_indices_inverse[index]
        self.highlight(index, emit)

    def _set_selection_pen_width(self):
        n_selected = np.count_nonzero(self.selection_group[self.sampled_indices])
        use_thick = n_selected <= MAX_THICK_SELECTED
        for v in itertools.chain(self.pen_selected.values(), [self._default_pen_selected]):
            v.setWidth(2 if use_thick else 1)

    def set_curve_pen(self, idc):
        idcdata = self.sampled_indices[idc]
        insubset = self.subset_indices[idcdata]
        inselected = self.selection_type and self.selection_group[idcdata]
        have_subset = np.any(self.subset_indices)
        thispen = self.pen_subset if insubset or not have_subset else self.pen_normal
        if inselected:
            thispen = self.pen_selected
        color_var = self.feature_color
        value = None
        if color_var is not None:
            value = str(self.data[idcdata][color_var])
        elif self.color_individual:
            value = idc % len(self._color_individual_cycle)
        self.curves_cont.objs[idc].setPen(thispen[value])
        # to always draw selected above subset, multiply with 2
        self.curves_cont.objs[idc].setZValue(int(insubset) + 2 * int(inselected))

    def set_curve_pens(self, curves=None):
        if self.viewtype == INDIVIDUAL and self.curves:
            self._set_selection_pen_width()
            curves = range(len(self.curves[0][1])) if curves is None else curves
            for i in curves:
                self.set_curve_pen(i)
            self.curves_cont.update()

    def add_marking(self, item):
        self.markings.append(item)
        self.plot.addItem(item, ignoreBounds=True)

    def in_markings(self, item):
        return item in self.markings

    def remove_marking(self, item):
        self.plot.removeItem(item)
        self.markings.remove(item)

    def clear_markings(self):
        self.clear_connect_views()
        for m in self.markings:
            self.plot.removeItem(m)
        self.markings = []

    def add_connected_view(self, vb):
        self.connected_views.append(vb)
        self._update_connected_views()

    def clear_connect_views(self):
        for v in self.connected_views:
            v.clear()  # if the viewbox is not clear the children would be deleted
            self.plot.scene().removeItem(v)
        self.connected_views = []

    def add_curves(self, x, ys, addc=True):
        """ Add multiple curves with the same x domain. """
        if len(ys) > MAX_INSTANCES_DRAWN:
            sample_selection = \
                random.Random(self.sample_seed).sample(range(len(ys)), MAX_INSTANCES_DRAWN)

            # with random selection also show at most MAX_INSTANCES_DRAW elements from the subset
            subset = set(np.where(self.subset_indices)[0])
            subset_to_show = subset - set(sample_selection)
            subset_additional = MAX_INSTANCES_DRAWN - (len(subset) - len(subset_to_show))
            if len(subset_to_show) > subset_additional:
                subset_to_show = \
                    random.Random(self.sample_seed).sample(subset_to_show, subset_additional)
            self.sampled_indices = sorted(sample_selection + list(subset_to_show))
            self.sampling = True
        else:
            self.sampled_indices = list(range(len(ys)))
        random.Random(self.sample_seed).shuffle(self.sampled_indices)  # for sequential classes#
        self.sampled_indices_inverse = {s: i for i, s in enumerate(self.sampled_indices)}
        ys = self.data.X[self.sampled_indices][:, self.data_xsind]
        ys[np.isinf(ys)] = np.nan  # remove infs that could ruin display
        self.new_sampling.emit(len(self.sampled_indices))
        self.curves.append((x, ys))

        # add curves efficiently
        for y in ys:
            self.add_curve(x, y, ignore_bounds=True)

        if x.size and ys.size:
            bounding_rect = QGraphicsRectItem(QRectF(QPointF(np.nanmin(x), np.nanmin(ys)),
                                                     QPointF(np.nanmax(x), np.nanmax(ys))))
            bounding_rect.setPen(QPen(Qt.NoPen))  # prevents border of 1
            self.curves_cont.add_bounds(bounding_rect)

        self.curves_plotted.append((x, ys))

    def add_curve(self, x, y, pen=None, ignore_bounds=False):
        c = FinitePlotCurveItem(x=x, y=y, pen=pen if pen else self.pen_normal[None])
        self.curves_cont.add_curve(c, ignore_bounds=ignore_bounds)
        # for rescale to work correctly
        if not ignore_bounds:
            self.curves_plotted.append((x, np.array([y])))

    def add_fill_curve(self, x, ylow, yhigh, pen):
        phigh = FinitePlotCurveItem(x, yhigh, pen=pen)
        plow = FinitePlotCurveItem(x, ylow, pen=pen)
        color = pen.color()
        color.setAlphaF(color.alphaF() * 0.3)
        cc = pg.mkBrush(color)
        pfill = pg.FillBetweenItem(plow, phigh, brush=cc)
        pfill.setZValue(10)
        self.curves_cont.add_bounds(phigh)
        self.curves_cont.add_bounds(plow)
        self.curves_cont.add_curve(pfill, ignore_bounds=True)
        # for zoom to work correctly
        self.curves_plotted.append((x, np.array([ylow, yhigh])))

    @staticmethod
    def _generate_pens(color, color_unselected=None, color_selected=None):
        pen_subset = pg.mkPen(color=color, width=1)
        if color_selected is None:
            color_selected = color.darker(135)
            color_selected.setAlphaF(1.0)  # only gains in a sparse space
        pen_selected = pg.mkPen(color=color_selected, width=2, style=Qt.DashLine)
        if color_unselected is None:
            color_unselected = color.lighter(160)
        pen_normal = pg.mkPen(color=color_unselected, width=1)
        return pen_normal, pen_selected, pen_subset

    def set_pen_colors(self):
        self.pen_normal.clear()
        self.pen_subset.clear()
        self.pen_selected.clear()
        color_var = self.feature_color
        self.legend.clear()
        palette, legend = False, False
        if color_var is not None:
            colors = color_var.colors
            discrete_palette = ColorPaletteGenerator(
                number_of_colors=len(colors), rgb_colors=colors)
            palette = [(v, discrete_palette[color_var.to_val(v)]) for v in color_var.values]
            legend = True
        elif self.color_individual:
            palette = [(i, QColor(*c)) for i, c in enumerate(self._color_individual_cycle)]
        if palette:
            for v, color in palette:
                color = QColor(color)  # copy color
                color.setAlphaF(0.9)
                self.pen_normal[v], self.pen_selected[v], self.pen_subset[v] = \
                    self._generate_pens(color)
                pen = pg.mkPen(color=color)
                brush = pg.mkBrush(color=color)
                if legend:
                    self.legend.addItem(
                        pg.ScatterPlotItem(pen=pen, brush=brush, size=10, symbol="o"), escape(v))
        self.legend.setVisible(bool(self.legend.items))

    def show_individual(self):
        self.clear_graph()
        self.view_average_menu.setChecked(False)
        self.set_pen_colors()
        self.viewtype = INDIVIDUAL
        if not self.data:
            return
        self.add_curves(self.data_x, self.data.X)
        self.set_curve_pens()
        self.curves_cont.update()
        self.plot.vb.set_mode_panning()

    def resample_curves(self, seed):
        self.sample_seed = seed
        self.update_view()

    def rescale_current_view_y(self):
        if self.curves_plotted:
            qrect = self.plot.vb.targetRect()
            bleft = qrect.left()
            bright = qrect.right()

            ymax = max(np.max(ys[:, np.searchsorted(x, bleft):
                                    np.searchsorted(x, bright, side="right")])
                       for x, ys in self.curves_plotted)
            ymin = min(np.min(ys[:, np.searchsorted(x, bleft):
                                    np.searchsorted(x, bright, side="right")])
                       for x, ys in self.curves_plotted)

            self.plot.vb.setYRange(ymin, ymax, padding=0.0)
            self.plot.vb.pad_current_view_y()

    def viewtype_changed(self):
        if self.viewtype == AVERAGE:
            self.viewtype = INDIVIDUAL
        else:
            self.viewtype = AVERAGE
        self.update_view()

    def show_average(self):
        self.show_average_thread.show_average()

    def update_view(self):
        if self.viewtype == INDIVIDUAL:
            self.show_individual()
            self.rescale()
        elif self.viewtype == AVERAGE:
            self.show_average()

    def rescale(self):
        if self.rescale_next:
            self.plot.vb.autoRange()

    def set_data(self, data, auto_update=True):
        if self.data is data:
            return
        if data is not None:
            if self.data:
                self.rescale_next = not data.domain == self.data.domain
            else:
                self.rescale_next = True

            self.data = data

            # new data implies that the graph is outdated
            self.clear_graph()

            self.restore_selection_settings()

            # get and sort input data
            x = getx(self.data)
            xsind = np.argsort(x)
            self.data_x = x[xsind]
            self.data_xsind = xsind
            self._set_subset_indices()  # refresh subset indices according to the current subset
            self.make_selection_valid()
        else:
            self.clear_data()
            self.clear_graph()
        if auto_update:
            self.update_view()

    def _set_subset_indices(self):
        ids = self.subset
        if ids is None:
            ids = []
        if self.data:
            self.subset_indices = np.in1d(self.data.ids, ids)

    def set_data_subset(self, ids, auto_update=True):
        self.subset = ids  # an array of indices
        self._set_subset_indices()
        if auto_update:
            self.update_view()

    def select_by_click(self, _):
        clicked_curve = self.highlighted
        if clicked_curve is not None:
            if self.viewtype == INDIVIDUAL:
                self.make_selection([self.sampled_indices[clicked_curve]])
            elif self.viewtype == AVERAGE:
                sel = np.where(self.multiple_curves_info[clicked_curve][2])[0]
                self.make_selection(sel)
        else:
            self.make_selection(None)

    def select_line(self, startp, endp):
        intersected = self.intersect_curves((startp.x(), startp.y()), (endp.x(), endp.y()))
        self.make_selection(intersected if len(intersected) else None)

    def intersect_curves(self, q1, q2):
        x, ys = self.data_x, self.data.X
        if len(x) < 2:
            return []
        x1, x2 = min(q1[0], q2[0]), max(q1[0], q2[0])
        xmin = closestindex(x, x1)
        xmax = closestindex(x, x2, side="right")
        xmin = max(0, xmin - 1)
        xmax = xmax + 2
        sel = np.flatnonzero(intersect_curves_chunked(x, ys, self.data_xsind, q1, q2, xmin, xmax))
        return sel

    def shutdown(self):
        self.show_average_thread.shutdown()

    @classmethod
    def migrate_settings_sub(cls, settings, version):
        # manually called from the parent
        if "selected_indices" in settings:
            # transform into list-of-tuples as we do not have data size
            if settings["selected_indices"]:
                settings["selection_group_saved"] = [(a, 1) for a in settings["selected_indices"]]

    @classmethod
    def migrate_context_sub_feature_color(cls, values, version):
        # convert strings to Variable
        name = "feature_color"
        var, vartype = values[name]
        if 0 <= vartype <= 100:
            values[name] = (var, 100 + vartype)


class OWSpectra(OWWidget):
    name = "Spectra"

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)
        data_subset = Input("Data subset", Orange.data.Table)

    class Outputs:
        selected_data = Output("Selection", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    icon = "icons/spectra.svg"

    priority = 10
    replaces = ["orangecontrib.infrared.widgets.owcurves.OWCurves",
                "orangecontrib.infrared.widgets.owspectra.OWSpectra"]
    keywords = ["curves", "lines", "spectrum"]

    settings_version = 2
    settingsHandler = DomainContextHandler()

    curveplot = SettingProvider(CurvePlot)

    graph_name = "curveplot.plotview"  # need to be defined for the save button to be shown

    class Information(OWWidget.Information):
        showing_sample = Msg("Showing {} of {} curves.")

    class Warning(OWWidget.Warning):
        no_x = Msg("No continuous features in input data.")

    def __init__(self):
        super().__init__()
        self.controlArea.hide()
        self.curveplot = CurvePlot(self, select=SELECTMANY)
        self.curveplot.selection_changed.connect(self.selection_changed)
        self.curveplot.new_sampling.connect(self._showing_sample_info)
        self.mainArea.layout().addWidget(self.curveplot)
        self.resize(900, 700)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()  # resets schema_only settings
        self._showing_sample_info(None)
        self.Warning.no_x.clear()
        self.openContext(data)
        self.curveplot.set_data(data, auto_update=False)
        if data is not None and not len(self.curveplot.data_x):
            self.Warning.no_x()
        self.selection_changed()

    @Inputs.data_subset
    def set_subset(self, data):
        self.curveplot.set_data_subset(data.ids if data else None, auto_update=False)

    def handleNewSignals(self):
        self.curveplot.update_view()

    def selection_changed(self):
        # selection table
        annotated_data = groups_or_annotated_table(self.curveplot.data,
                                                   self.curveplot.selection_group)
        self.Outputs.annotated_data.send(annotated_data)

        # selected elements
        selected = None
        if self.curveplot.data:
            selection_indices = np.flatnonzero(self.curveplot.selection_group)
            if len(selection_indices):
                selected = self.curveplot.data[selection_indices]
        self.Outputs.selected_data.send(selected)

    def _showing_sample_info(self, num):
        if num is not None and self.curveplot.data and num != len(self.curveplot.data):
            self.Information.showing_sample(num, len(self.curveplot.data))
        else:
            self.Information.showing_sample.clear()

    def save_graph(self):
        # directly call save_graph so it hides axes
        self.curveplot.save_graph()

    def onDeleteWidget(self):
        self.curveplot.shutdown()
        super().onDeleteWidget()

    @classmethod
    def migrate_settings(cls, settings, version):
        if "curveplot" in settings:
            CurvePlot.migrate_settings_sub(settings["curveplot"], version)

    @classmethod
    def migrate_context(cls, context, version):
        if version <= 1 and "curveplot" in context.values:
            CurvePlot.migrate_context_sub_feature_color(context.values["curveplot"], version)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    w = OWSpectra()
    w.show()
    from orangecontrib.spectroscopy.tests.bigdata import dust
    # data = Orange.data.Table(dust())
    data = Orange.data.Table("collagen.csv")
    w.set_data(data)
    w.set_subset(data[:40])
    w.handleNewSignals()
    rval = app.exec_()
    w.saveSettings()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval


if __name__ == "__main__":
    sys.exit(main())
