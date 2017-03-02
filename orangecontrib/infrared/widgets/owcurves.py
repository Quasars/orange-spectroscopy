from itertools import chain
import sys
from collections import defaultdict
import gc
import random
import warnings
import math

from AnyQt.QtWidgets import QWidget, QGraphicsItem, QPushButton, QMenu, \
    QGridLayout, QAction, QVBoxLayout, QApplication, QWidgetAction, QLabel
from AnyQt.QtGui import QColor, QPixmapCache, QPen, QKeySequence
from AnyQt.QtCore import Qt, QRectF

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph import Point, GraphicsObject

from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets.widget import OWWidget, Msg, OWComponent
from Orange.widgets import gui
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator
from Orange.widgets.utils.plot import \
    SELECT, PANNING, ZOOMING

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.line_geometry import \
    distance_curves, intersect_curves_chunked
from orangecontrib.infrared.widgets.gui import lineEditFloatOrNone

SELECT_SQUARE = 123

# view types
INDIVIDUAL = 0
AVERAGE = 1

# selections
SELECTNONE = 0
SELECTONE = 1
SELECTMANY = 2

MAX_INSTANCES_DRAWN = 100
NAN = float("nan")


class MenuFocus(QMenu):  # menu that works well with subwidgets and focusing
    def focusNextPrevChild(self, next):
        return QWidget.focusNextPrevChild(self, next)


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

    def add_curve(self, c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NaN warnings are expected
            cb = c.boundingRect()
            # keep undefined elements NaN
            self.bounds[0] = np.nanmin([cb.left(), self.bounds[0]])
            self.bounds[1] = np.nanmin([cb.top(), self.bounds[1]])
            self.bounds[2] = np.nanmax([cb.right(), self.bounds[2]])
            self.bounds[3] = np.nanmax([cb.bottom(), self.bounds[3]])
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


def searchsorted_cached(cache, arr, v, side="left"):
    key = (id(arr), v, side)
    if key not in cache:
        cache[key] = np.searchsorted(arr, v, side=side)
    return cache[key]


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

    distancepx = (xp ** 2 + yp ** 2) ** 0.5
    mini = np.argmin(distancepx)

    # add edge point so that distance_curves works if there is just one point
    xp = np.hstack((xp, float("nan")))
    yp = np.hstack((yp, np.zeros((yp.shape[0], 1)) * float("nan")))
    dc = distance_curves(xp, yp, (0, 0))
    return dc


class InteractiveViewBox(ViewBox):
    def __init__(self, graph):
        ViewBox.__init__(self, enableMenu=False)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.zoomstartpoint = None
        self.selection_start = None
        self.action = PANNING
        self.y_padding = 0.02
        self.x_padding = 0

        # line for line selection
        self.selection_line = pg.PlotCurveItem()
        self.selection_line.setPen(pg.mkPen(color=QColor(Qt.black), width=2, style=Qt.DotLine))
        self.selection_line.setZValue(1e9)
        self.selection_line.hide()
        self.addItem(self.selection_line, ignoreBounds=True)

        # square for square selection
        self.selection_square = pg.PlotCurveItem()
        self.selection_square.setPen(pg.mkPen(color=QColor(Qt.black), width=2, style=Qt.DotLine))
        self.selection_square.setZValue(1e9)
        self.selection_square.hide()
        self.addItem(self.selection_square, ignoreBounds=True)

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
        else:
            ev.ignore()

    def suggestPadding(self, axis):
        return 0.

    def mouseMovedEvent(self, ev):  # not a Qt event!
        if self.action == ZOOMING and self.zoomstartpoint:
            pos = self.mapFromView(self.mapSceneToView(ev))
            self.updateScaleBox(self.zoomstartpoint, pos)
        if (self.action == SELECT or self.action == SELECT_SQUARE) and self.selection_start:
            pos = self.mapFromView(self.mapSceneToView(ev))
            if self.action == SELECT:
                self.updateSelectionLine(self.selection_start, pos)
            elif self.action == SELECT_SQUARE:
                self.updateSelectionSquare(self.selection_start, pos)

    def updateSelectionLine(self, p1, p2):
        p1 = self.childGroup.mapFromParent(p1)
        p2 = self.childGroup.mapFromParent(p2)
        self.selection_line.setData(x=[p1.x(), p2.x()], y=[p1.y(), p2.y()])
        self.selection_line.show()

    def updateSelectionSquare(self, p1, p2):
        p1 = self.childGroup.mapFromParent(p1)
        p2 = self.childGroup.mapFromParent(p2)
        self.selection_square.setData(x=[p1.x(), p1.x(), p2.x(), p2.x(), p1.x()],
                                      y=[p1.y(), p2.y(), p2.y(), p1.y(), p1.y()])
        self.selection_square.show()

    def wheelEvent(self, ev, axis=None):
        ev.accept()  # ignore wheel zoom

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton and \
                (self.action == ZOOMING or self.action == SELECT):
            ev.accept()
            self.set_mode_panning()
        elif ev.button() == Qt.RightButton:
            ev.accept()
            self.autoRange()
        add = ev.modifiers() & Qt.ControlModifier and self.graph.selection_type == SELECTMANY
        if self.action != ZOOMING and self.action != SELECT and self.action != SELECT_SQUARE \
                and ev.button() == Qt.LeftButton and self.graph.selection_type \
                and self.graph.viewtype == INDIVIDUAL:
            pos = self.childGroup.mapFromParent(ev.pos())
            self.graph.select_by_click(pos, add)
            ev.accept()
        if self.action == ZOOMING and ev.button() == Qt.LeftButton:
            if self.zoomstartpoint == None:
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
        if (self.action == SELECT or self.action == SELECT_SQUARE) \
                and ev.button() == Qt.LeftButton and self.graph.selection_type:
            if self.selection_start is None:
                self.selection_start = ev.pos()
            else:
                startp = self.childGroup.mapFromParent(self.selection_start)
                endp = self.childGroup.mapFromParent(ev.pos())
                if self.action == SELECT:
                    self.graph.select_line(startp, endp, add)
                elif self.action == SELECT_SQUARE:
                    self.graph.select_square(startp, endp, add)
                self.set_mode_panning()
            ev.accept()

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

    def set_mode_zooming(self):
        self.set_mode_panning()
        self.setMouseMode(self.RectMode)
        self.action = ZOOMING
        self.setCursor(Qt.CrossCursor)

    def set_mode_panning(self):
        self.cancel_zoom()
        self.cancel_select()

    def cancel_select(self):
        self.setMouseMode(self.PanMode)
        self.selection_line.hide()
        self.selection_square.hide()
        self.selection_start = None
        self.action = PANNING
        self.unsetCursor()

    def set_mode_select(self):
        self.set_mode_panning()
        self.setMouseMode(self.RectMode)
        self.action = SELECT
        self.setCursor(Qt.CrossCursor)

    def set_mode_select_square(self):
        self.set_mode_select()
        self.action = SELECT_SQUARE

    def set_mode_select_square(self):
        self.set_mode_panning()
        self.setMouseMode(self.RectMode)
        self.action = SELECT_SQUARE
        self.setCursor(Qt.CrossCursor)


class SelectRegion(pg.LinearRegionItem):
    def __init__(self, *args, **kwargs):
        pg.LinearRegionItem.__init__(self, *args, **kwargs)
        for l in self.lines:
            l.setCursor(Qt.SizeHorCursor)
        self.setZValue(10)
        color = QColor(Qt.red)
        color.setAlphaF(0.05)
        self.setBrush(pg.mkBrush(color))


class CurvePlot(QWidget, OWComponent):
    label_title = Setting("")
    label_xaxis = Setting("")
    label_yaxis = Setting("")
    range_x1 = Setting(None)
    range_x2 = Setting(None)
    range_y1 = Setting(None)
    range_y2 = Setting(None)
    color_attr = ContextSetting(0)
    invertX = Setting(False)
    selected_indices = Setting(set())
    data_size = Setting(None)  # to invalidate selected_indices

    def __init__(self, parent=None, select=SELECTNONE):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)

        self.parent = parent

        self.selection_type = select
        self.saving_enabled = hasattr(self.parent, "save_graph")
        self.clear_data(init=True)

        self.plotview = pg.PlotWidget(background="w", viewBox=InteractiveViewBox(self))
        self.plot = self.plotview.getPlotItem()
        self.plot.setDownsampling(auto=True, mode="peak")

        self.markings = []
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=20, slot=self.mouseMoved, delay=0.1)
        self.plot.scene().sigMouseMoved.connect(self.plot.vb.mouseMovedEvent)
        self.plot.vb.sigRangeChanged.connect(self.resized)
        self.pen_mouse = pg.mkPen(color=(0, 0, 255), width=2)
        self.pen_normal = defaultdict(lambda: pg.mkPen(color=(200, 200, 200, 127), width=1))
        self.pen_subset = defaultdict(lambda: pg.mkPen(color=(0, 0, 0, 127), width=1))
        self.pen_selected = defaultdict(lambda: pg.mkPen(color=(0, 0, 0, 127), width=2, style=Qt.DotLine))
        self.label = pg.TextItem("", anchor=(1, 0))
        self.label.setText("", color=(0, 0, 0))
        self.discrete_palette = None
        QPixmapCache.setCacheLimit(max(QPixmapCache.cacheLimit(), 100 * 1024))
        self.curves_cont = PlotCurvesItem()
        self.important_decimals = 4, 4

        self.MOUSE_RADIUS = 20

        self.clear_graph()

        # interface settings
        self.location = True  # show current position
        self.markclosest = True  # mark
        self.crosshair = True
        self.crosshair_hidden = True
        self.viewtype = INDIVIDUAL

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)

        actions = []

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
        actions.append(rescale_y)
        view_individual = QAction(
            "Show individual", self, shortcut=Qt.Key_I,
            triggered=lambda x: self.show_individual()
        )
        actions.append(view_individual)
        view_average = QAction(
            "Show averages", self, shortcut=Qt.Key_A,
            triggered=lambda x: self.show_average()
        )
        actions.append(view_average)

        resample_curves = QAction(
            "Resample curves", self, shortcut=Qt.Key_R,
            triggered=lambda x: self.resample_curves()
        )
        actions.append(resample_curves)

        self.show_grid = False
        self.show_grid_a = QAction(
            "Show grid", self, shortcut=Qt.Key_G, checkable=True,
            triggered=self.grid_changed
        )
        actions.append(self.show_grid_a)
        self.invertX_menu = QAction(
            "Invert X", self, shortcut=Qt.Key_X, checkable=True,
            triggered=self.invertX_changed
        )
        actions.append(self.invertX_menu)
        if self.selection_type == SELECTMANY:
            select_curves = QAction(
                "Select (line)", self, triggered=self.plot.vb.set_mode_select,
            )
            select_curves.setShortcuts([Qt.Key_S])
            select_curves.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            actions.append(select_curves)
        if self.saving_enabled:
            save_graph = QAction(
                "Save graph", self, triggered=self.save_graph,
            )
            save_graph.setShortcuts([QKeySequence(Qt.ControlModifier | Qt.Key_S)])
            actions.append(save_graph)

        range_menu = MenuFocus("Define view range", self)
        range_action = QWidgetAction(self)
        layout = QGridLayout()
        range_box = gui.widgetBox(self, margin=5, orientation=layout)
        range_box.setFocusPolicy(Qt.TabFocus)
        self.range_e_x1 = lineEditFloatOrNone(None, self, "range_x1", label="e")
        range_box.setFocusProxy(self.range_e_x1)
        self.range_e_x2 = lineEditFloatOrNone(None, self, "range_x2", label="e")
        layout.addWidget(QLabel("X"), 0, 0, Qt.AlignRight)
        layout.addWidget(self.range_e_x1, 0, 1)
        layout.addWidget(QLabel("-"), 0, 2)
        layout.addWidget(self.range_e_x2, 0, 3)
        self.range_e_y1 = lineEditFloatOrNone(None, self, "range_y1", label="e")
        self.range_e_y2 = lineEditFloatOrNone(None, self, "range_y2", label="e")
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
        self.button = QPushButton("View", self.plotview)
        self.button.setAutoDefault(False)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self.button, 0, 0)
        view_menu = MenuFocus(self)
        self.button.setMenu(view_menu)
        view_menu.addActions(actions)
        view_menu.addMenu(range_menu)
        self.addActions(actions)

        choose_color_action = QWidgetAction(self)
        choose_color_box = gui.hBox(self)
        choose_color_box.setFocusPolicy(Qt.TabFocus)
        model = VariableListModel()
        self.attrs = []
        model.wrap(self.attrs)
        label = gui.label(choose_color_box, self, "Color by")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.attrCombo = gui.comboBox(
            choose_color_box, self, value="color_attr", contentsLength=12,
            callback=self.change_color_attr)
        self.attrCombo.setModel(model)
        choose_color_box.setFocusProxy(self.attrCombo)
        choose_color_action.setDefaultWidget(choose_color_box)
        view_menu.addAction(choose_color_action)

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

        self.invertX_apply()
        self.plot.vb.set_mode_panning()

        self.reports = {}  # current reports

        self.viewhelpers_show()

    def report(self, reporter, contents):
        self.reports[id(reporter)] = contents

    def report_finished(self, reporter):
        try:
            self.reports.pop(id(reporter))
        except KeyError:
            pass  # ok if it was already removed
        if not self.reports:
            pass

    def set_limits(self):
        vr = self.plot.vb.viewRect()
        x1 = self.range_x1 if self.range_x1 is not None else vr.left()
        x2 = self.range_x2 if self.range_x2 is not None else vr.right()
        y1 = self.range_y1 if self.range_y1 is not None else vr.top()
        y2 = self.range_y2 if self.range_y2 is not None else vr.bottom()
        self.plot.vb.setXRange(x1, x2)
        self.plot.vb.setYRange(y1, y2)

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

    def save_graph(self):
        self.viewhelpers_hide()
        self.plot.showAxis("top", True)
        self.plot.showAxis("right", True)
        self.parent.save_graph()
        self.plot.showAxis("top", False)
        self.plot.showAxis("right", False)
        self.viewhelpers_show()

    def clear_data(self, init=True):
        self.subset_ids = set()
        self.data = None
        self.data_x = None
        self.data_ys = None
        self.sampled_indices = []
        self.sampled_indices_inverse = {}
        self.sampling = None
        if not init:
            self.selection_changed()
        self.discrete_palette = None

    def clear_graph(self):
        # reset caching. if not, it is not cleared when view changing when zoomed
        self.highlighted = None
        self.curves_cont.setCacheMode(QGraphicsItem.NoCache)
        self.curves_cont.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.plot.vb.disableAutoRange()
        self.curves_cont.clear()
        self.curves_cont.update()
        self.plotview.clear()
        self.curves_plotted = []  # currently plotted elements (for rescale)
        self.curves = []  # for finding closest curve
        self.plotview.addItem(self.label, ignoreBounds=True)
        self.highlighted_curve = pg.PlotCurveItem(pen=self.pen_mouse)
        self.highlighted_curve.setZValue(10)
        self.highlighted_curve.hide()
        self.plot.addItem(self.highlighted_curve)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.viewhelpers = True
        self.plot.addItem(self.curves_cont)
        for m in self.markings:
            self.plot.addItem(m, ignoreBounds=True)

    def resized(self):
        vr = self.plot.vb.viewRect()
        xpixel, ypixel = self.plot.vb.viewPixelSize()

        def important_decimals(n):
            return max(-int(math.floor(math.log10(n))) + 1, 0)

        self.important_decimals = important_decimals(xpixel), important_decimals(ypixel)
        if self.invertX:
            self.label.setPos(vr.bottomLeft())
        else:
            self.label.setPos(vr.bottomRight())
        xd, yd = self.important_decimals
        self.range_e_x1.setPlaceholderText(("%0." + str(xd) + "f") % vr.left())
        self.range_e_x2.setPlaceholderText(("%0." + str(xd) + "f") % vr.right())
        self.range_e_y1.setPlaceholderText(("%0." + str(yd) + "f") % vr.top())
        self.range_e_y2.setPlaceholderText(("%0." + str(yd) + "f") % vr.bottom())

    def make_selection(self, data_indices, add=False):
        selected_indices = self.selected_indices
        oldids = selected_indices.copy()
        invd = self.sampled_indices_inverse
        if data_indices is None:
            if not add:
                selected_indices.clear()
                self.set_curve_pens([invd[a] for a in oldids if a in invd])
        else:
            if add:
                selected_indices.update(data_indices)
                self.set_curve_pens([invd[a] for a in data_indices if a in invd])
            else:
                selected_indices.clear()
                selected_indices.update(data_indices)
                self.set_curve_pens([invd[a] for a in (oldids | selected_indices) if a in invd])
        self.selection_changed()

    def selection_changed(self):
        if self.selection_type:
            self.parent.selection_changed()

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

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.vb.mapSceneToView(pos)
            posx, posy = mousePoint.x(), mousePoint.y()

            labels = []
            for a, vs in sorted(self.reports.items()):
                for v in vs:
                    if isinstance(v, tuple) and len(v) == 2:
                        if v[0] == "x":
                            labels.append(("%0." + str(self.important_decimals[0]) + "f") % v[1])
                            continue
                    labels.append(str(v))
            labels = " ".join(labels)
            self.crosshair_hidden = bool(labels)

            if self.location and not labels:
                fs = "%0." + str(self.important_decimals[0]) + "f %0." + str(self.important_decimals[1]) + "f"
                labels = fs % (posx, posy)
            self.label.setText(labels, color=(0, 0, 0))

            if self.curves and len(self.curves[0][0]):  # need non-zero x axis!
                cache = {}
                bd = None
                if self.markclosest and self.plot.vb.action != ZOOMING:
                    xpixel, ypixel = self.plot.vb.viewPixelSize()
                    distances = distancetocurves(self.curves[0], posx, posy, xpixel, ypixel, r=self.MOUSE_RADIUS,
                                                 cache=cache)
                    try:
                        bd = np.nanargmin(distances)
                        bd = (bd, distances[bd])
                    except ValueError:  # if all distances are NaN
                        pass
                if self.highlighted is not None:
                    self.highlighted = None
                    self.highlighted_curve.hide()
                if bd and bd[1] < self.MOUSE_RADIUS:
                    self.highlighted = bd[0]
                    x = self.curves[0][0]
                    y = self.curves[0][1][self.highlighted]
                    self.highlighted_curve.setData(x=x, y=y)
                    self.highlighted_curve.show()

            self.vLine.setPos(posx)
            self.hLine.setPos(posy)
            self.viewhelpers_show()
        else:
            self.viewhelpers_hide()

    def set_curve_pen(self, idc):
        idcdata = self.sampled_indices[idc]
        insubset = not self.subset_ids or self.data[idcdata].id in self.subset_ids
        inselected = self.selection_type and idcdata in self.selected_indices
        thispen = self.pen_subset if insubset else self.pen_normal
        if inselected:
            thispen = self.pen_selected
        color_var = self._current_color_var()
        value = None if isinstance(color_var, str) else str(self.data[idcdata][color_var])
        self.curves_cont.objs[idc].setPen(thispen[value])
        self.curves_cont.objs[idc].setZValue(int(insubset) + int(inselected))

    def set_curve_pens(self, curves=None):
        if self.viewtype == INDIVIDUAL and self.curves:
            curves = range(len(self.curves[0][1])) if curves is None else curves
            for i in curves:
                self.set_curve_pen(i)
            self.curves_cont.update()

    def add_marking(self, item):
        self.markings.append(item)
        self.plot.addItem(item, ignoreBounds=True)

    def remove_marking(self, item):
        self.plot.removeItem(item)
        self.markings.remove(item)

    def clear_markings(self):
        for m in self.markings:
            self.plot.removeItem(m)
        self.markings = []

    def add_curves(self, x, ys, addc=True):
        """ Add multiple curves with the same x domain. """
        if len(ys) > MAX_INSTANCES_DRAWN:
            self.sampled_indices = sorted(random.Random().sample(range(len(ys)), MAX_INSTANCES_DRAWN))
            self.sampling = True
        else:
            self.sampled_indices = list(range(len(ys)))
        random.Random().shuffle(self.sampled_indices)  # for sequential classes#
        self.sampled_indices_inverse = {s: i for i, s in enumerate(self.sampled_indices)}
        ys = ys[self.sampled_indices]
        self.curves.append((x, ys))
        for y in ys:
            c = pg.PlotCurveItem(x=x, y=y, pen=self.pen_normal[None])
            self.curves_cont.add_curve(c)
        self.curves_plotted = self.curves

    def add_curve(self, x, y, pen=None):
        c = pg.PlotCurveItem(x=x, y=y, pen=pen if pen else self.pen_normal[None])
        self.curves_cont.add_curve(c)
        # for rescale to work correctly
        self.curves_plotted.append((x, np.array([y])))

    def add_fill_curve(self, x, ylow, yhigh, pen):
        phigh = pg.PlotCurveItem(x, yhigh, pen=pen)
        plow = pg.PlotCurveItem(x, ylow, pen=pen)
        color = pen.color()
        color.setAlphaF(0.2)
        cc = pg.mkBrush(color)
        pfill = pg.FillBetweenItem(plow, phigh, brush=cc)
        pfill.setZValue(10)
        self.curves_cont.add_curve(pfill)
        # for zoom to work correctly
        self.curves_plotted.append((x, np.array([ylow, yhigh])))

    def _current_color_var(self):
        color_var = "(Same color)"
        try:
            color_var = self.attrs[self.color_attr]
        except IndexError:
            pass
        return color_var

    def change_color_attr(self):
        self.set_pen_colors()
        self.update_view()

    def set_pen_colors(self):
        self.pen_normal.clear()
        self.pen_subset.clear()
        self.pen_selected.clear()
        color_var = self._current_color_var()
        if color_var != "(Same color)":
            colors = color_var.colors
            discrete_palette = ColorPaletteGenerator(
                number_of_colors=len(colors), rgb_colors=colors)
            for v in color_var.values:
                basecolor = discrete_palette[color_var.to_val(v)]
                basecolor = QColor(basecolor)
                basecolor.setAlphaF(0.9)
                self.pen_subset[v] = pg.mkPen(color=basecolor, width=1)
                self.pen_selected[v] = pg.mkPen(color=basecolor, width=2, style=Qt.DotLine)
                notselcolor = basecolor.lighter(150)
                notselcolor.setAlphaF(0.5)
                self.pen_normal[v] = pg.mkPen(color=notselcolor, width=1)

    def show_individual(self):
        if not self.data:
            return
        self.viewtype = INDIVIDUAL
        self.clear_graph()
        self.add_curves(self.data_x, self.data_ys)
        self.set_curve_pens()
        self.curves_cont.update()

    def resample_curves(self):
        if not self.data:
            return
        if self.viewtype == INDIVIDUAL:
            self.show_individual()
        elif self.viewtype == AVERAGE:
            self.show_average()

    def rescale_current_view_y(self):
        if self.curves_plotted:
            cache = {}
            qrect = self.plot.vb.targetRect()
            bleft = qrect.left()
            bright = qrect.right()

            ymax = max(np.max(ys[:, searchsorted_cached(cache, x, bleft):
            searchsorted_cached(cache, x, bright, side="right")])
                       for x, ys in self.curves_plotted)
            ymin = min(np.min(ys[:, searchsorted_cached(cache, x, bleft):
            searchsorted_cached(cache, x, bright, side="right")])
                       for x, ys in self.curves_plotted)

            self.plot.vb.setYRange(ymin, ymax, padding=0.0)
            self.plot.vb.pad_current_view_y()

    def _split_by_color_value(self, data):
        color_var = self._current_color_var()
        rd = defaultdict(list)
        for i, inst in enumerate(data):
            value = None if isinstance(color_var, str) else str(inst[color_var])
            rd[value].append(i)
        return rd

    def show_average(self):
        if not self.data:
            return
        self.viewtype = AVERAGE
        self.clear_graph()
        x = self.data_x
        if self.data:
            ysall = []
            subset_indices = [i for i, id in enumerate(self.data.ids) if id in self.subset_ids]
            dsplit = self._split_by_color_value(self.data)
            for colorv, indices in dsplit.items():
                for part in ["everything", "subset", "selection"]:
                    if part == "everything":
                        ys = self.data_ys[indices]
                        pen = self.pen_normal if subset_indices else self.pen_subset
                    elif part == "selection" and self.selection_type:
                        current_selected = sorted(set(self.selected_indices) & set(indices))
                        if not current_selected:
                            continue
                        ys = self.data_ys[current_selected]
                        pen = self.pen_selected
                    elif part == "subset":
                        current_subset = sorted(set(subset_indices) & set(indices))
                        if not current_subset:
                            continue
                        ys = self.data_ys[current_subset]
                        pen = self.pen_subset
                    std = np.std(ys, axis=0)
                    mean = np.mean(ys, axis=0)
                    ysall.append(mean)
                    penc = QPen(pen[colorv])
                    penc.setWidth(3)
                    self.add_curve(x, mean, pen=penc)
                    self.add_fill_curve(x, mean + std, mean - std, pen=penc)
            self.curves.append((x, np.array(ysall)))
        self.curves_cont.update()

    def update_view(self):
        if self.viewtype == INDIVIDUAL:
            self.show_individual()
        elif self.viewtype == AVERAGE:
            self.show_average()

    def set_data(self, data, rescale="auto"):
        self.clear_graph()
        self.clear_data()
        self.attrs[:] = []
        if data is not None:
            self.attrs[:] = ["(Same color)"] + [
                var for var in chain(data.domain,
                                     data.domain.metas)
                if isinstance(var, str) or var.is_discrete]
            self.color_attr = 0
        self.set_pen_colors()
        if data is not None:
            if rescale == "auto":
                if self.data:
                    rescale = not data.domain == self.data.domain
                else:
                    rescale = True
            self.data = data
            # reset selection if dataset sizes do not match
            if self.selected_indices and \
                    (max(self.selected_indices) >= len(self.data) or self.data_size != len(self.data)):
                self.selected_indices.clear()
            self.data_size = len(self.data)
            # get and sort input data
            x = getx(self.data)
            xsind = np.argsort(x)
            self.data_x = x[xsind]
            self.data_ys = data.X[:, xsind]
            self.update_view()
            if rescale == True:
                self.plot.vb.autoRange()
        self.plot.vb.set_mode_panning()

    def update_display(self):
        self.curves_cont.update()

    def set_data_subset(self, ids):
        self.subset_ids = set(ids) if ids is not None else set()
        self.set_curve_pens()
        self.update_view()

    def select_by_click(self, pos, add):
        clicked_curve = self.highlighted
        if clicked_curve is not None:
            self.make_selection([self.sampled_indices[clicked_curve]], add)
        else:
            self.make_selection(None, add)

    def select_line(self, startp, endp, add):
        intersected = self.intersect_curves((startp.x(), startp.y()), (endp.x(), endp.y()))
        self.make_selection(intersected if len(intersected) else None, add)

    def intersect_curves(self, q1, q2):
        x, ys = self.data_x, self.data_ys
        if len(x) < 2:
            return []
        x1, x2 = min(q1[0], q2[0]), max(q1[0], q2[0])
        xmin = closestindex(x, x1)
        xmax = closestindex(x, x2, side="right")
        xmin = max(0, xmin - 1)
        xmax = xmax + 2
        x = x[xmin:xmax]
        ys = ys[:, xmin:xmax]
        sel = np.flatnonzero(intersect_curves_chunked(x, ys, q1, q2))
        return sel


class OWCurves(OWWidget):
    name = "Curves"
    inputs = [("Data", Orange.data.Table, 'set_data', Default),
              ("Data subset", Orange.data.Table, 'set_subset', Default)]
    outputs = [("Selection", Orange.data.Table)]
    icon = "icons/curves.svg"

    settingsHandler = DomainContextHandler()

    curveplot = SettingProvider(CurvePlot)

    class Information(OWWidget.Information):
        showing_sample = Msg("Showing {} of {} curves.")

    class Warning(OWWidget.Warning):
        no_x = Msg("No continuous features in input data.")

    def __init__(self):
        super().__init__()
        self.controlArea.hide()
        self.curveplot = CurvePlot(self, select=SELECTMANY)
        self.mainArea.layout().addWidget(self.curveplot)
        self.resize(900, 700)
        self.graph_name = "curveplot.plotview"

    def set_data(self, data):
        self.Information.showing_sample.clear()
        self.Warning.no_x.clear()
        self.closeContext()
        self.curveplot.set_data(data)
        if data is not None and not len(self.curveplot.data_x):
            self.Warning.no_x()
        if self.curveplot.sampled_indices \
                and len(self.curveplot.sampled_indices) != len(self.curveplot.data):
            self.Information.showing_sample(len(self.curveplot.sampled_indices), len(data))
        self.openContext(data)
        self.curveplot.set_pen_colors()
        self.curveplot.set_curve_pens()  # mark the selection
        self.selection_changed()

    def set_subset(self, data):
        self.curveplot.set_data_subset(data.ids if data else None)

    def selection_changed(self):
        if self.curveplot.selected_indices and self.curveplot.data:
            self.send("Selection", self.curveplot.data[sorted(self.curveplot.selected_indices)])
        else:
            self.send("Selection", None)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    w = OWCurves()
    w.show()
    data = Orange.data.Table("collagen.csv")
    # data = Orange.data.Table("/home/marko/dust/20160831_06_Paris_25x_highmag.hdr")
    w.set_data(data)
    w.set_subset(data[:40])
    # w.set_subset(None)
    w.handleNewSignals()
    region = SelectRegion()

    def update():
        minX, maxX = region.getRegion()
        print(minX, maxX)

    region.sigRegionChanged.connect(update)
    w.curveplot.add_marking(region)
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval


if __name__ == "__main__":
    sys.exit(main())