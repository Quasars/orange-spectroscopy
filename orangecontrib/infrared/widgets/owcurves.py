from itertools import chain
import sys
from collections import defaultdict
import gc
import random

import numpy as np
import pyqtgraph as pg
from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets import widget
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QWidget, QColor, QPixmapCache, QGraphicsItem
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph import Point, GraphicsObject
from PyQt4.QtCore import Qt, QObject, QEvent, QRectF, QPointF
from Orange.widgets.utils.plot import \
    SELECT, PANNING, ZOOMING
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets import gui
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator

from orangecontrib.infrared.data import getx

#view types
INDIVIDUAL = 0
AVERAGE = 1
MAXINST = 1000

class PlotCurvesItem(GraphicsObject):
    """ Multiple curves on a single plot that can be cached together. """

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.clear()

    def clear(self):
        self.bounds = QtCore.QRectF(0, 0, 1, 1)
        self.objs = []

    def paint(self, p, *args):
        for o in sorted(self.objs, key=lambda x: x.zValue()):
            o.paint(p, *args)

    def add_curve(self, c):
        if not self.objs:
            self.bounds = c.boundingRect()
        else:
            self.bounds = self.bounds.united(c.boundingRect())
        self.objs.append(c)

    def boundingRect(self):
        return self.bounds


def closestindex(array, v):
    """
    Return index of a 1d sorted array closest to value v.
    """
    fi = np.searchsorted(array, v)
    if fi == 0:
        return 0
    elif fi == len(array):
        return len(array) - 1
    else:
        return fi-1 if v - array[fi-1] < array[fi] - v else fi


def searchsorted_cached(cache, arr, v, side="left"):
    key = (id(arr),v,side)
    if key not in cache:
        cache[key] = np.searchsorted(arr, v, side=side)
    return cache[key]


def distancetocurve(array, x, y, xpixel, ypixel, r=5, cache=None):
    if cache is not None and id(x) in cache:
        xmin,xmax = cache[id(x)]
    else:
        xmin = closestindex(array[0], x-r*xpixel)
        xmax = closestindex(array[0], x+r*xpixel)
        if cache is not None:
            cache[id(x)] = xmin,xmax
    xp = array[0][xmin:xmax+1]
    yp = array[1][xmin:xmax+1]

    #convert to distances in pixels
    xp = ((xp-x)/xpixel)
    yp = ((yp-y)/ypixel)

    distancepx = (xp**2+yp**2)**0.5
    mini = np.argmin(distancepx)
    return distancepx[mini], xmin + mini


class InteractiveViewBox(ViewBox):

    def __init__(self, graph):
        ViewBox.__init__(self, enableMenu=False)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.zoomstartpoint = None

    def safe_update_scale_box(self, buttonDownPos, currentPos):
        x, y = currentPos
        if buttonDownPos[0] == x:
            x += 1
        if buttonDownPos[1] == y:
            y += 1
        self.updateScaleBox(buttonDownPos, Point(x, y))

    # noinspection PyPep8Naming,PyMethodOverriding
    def mouseDragEvent(self, ev, axis=None):
        if ev.button() & QtCore.Qt.RightButton:
            ev.accept()
        if self.graph.state == ZOOMING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        elif self.graph.state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()

    def suggestPadding(self, axis):
        return 0.

    def mouseMovedEvent(self, ev): #not a Qt event!
        if self.graph.state == ZOOMING and self.zoomstartpoint:
            pos = self.mapFromView(self.mapSceneToView(ev))
            self.updateScaleBox(self.zoomstartpoint, pos)

    def wheelEvent(self, ev, axis=None):
        ev.accept() #ignore wheel zoom

    def mouseClickEvent(self, ev):
        if ev.button() ==  Qt.RightButton:
            ev.accept()
            self.autoRange()
            self.graph.set_mode_panning()
        if self.graph.state != ZOOMING and ev.button() == Qt.LeftButton and self.graph.selection_enabled:
            add = True if ev.modifiers() & Qt.ControlModifier else False
            clicked_curve = self.graph.highlighted

            selected_indices = self.graph.parent.selected_indices
            if clicked_curve is not None:
                if add:
                    if clicked_curve not in selected_indices:
                        selected_indices.add(clicked_curve)
                    else:
                        selected_indices.remove(clicked_curve)
                    self.graph.set_curve_pen(clicked_curve)
                    self.graph.curves_cont.update()
                else:
                    oldids = selected_indices.copy()
                    selected_indices.clear()
                    selected_indices.add(clicked_curve)
                    self.graph.set_curve_pens(oldids | selected_indices)
            else:
                if not add:
                    oldids = selected_indices.copy()
                    selected_indices.clear()
                    self.graph.set_curve_pens(oldids)
            self.graph.selection_changed()
            ev.accept()
        if self.graph.state == ZOOMING and ev.button() == Qt.LeftButton:
            if self.zoomstartpoint == None:
                self.zoomstartpoint = ev.pos()
            else:
                self.updateScaleBox(self.zoomstartpoint, ev.pos())
                self.rbScaleBox.hide()
                ax = QtCore.QRectF(Point(self.zoomstartpoint), Point(ev.pos()))
                ax = self.childGroup.mapRectFromParent(ax)
                self.showAxRect(ax)
                self.axHistoryPointer += 1
                self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                self.zoomstartpoint = None
            ev.accept()

    def showAxRect(self, ax):
        super().showAxRect(ax)
        if self.graph.state == ZOOMING:
            self.graph.set_mode_panning()

    def pad_current_view_y(self):
        qrect = self.targetRect()
        self.setYRange(qrect.bottom(), qrect.top(), padding=0.02)

    def autoRange(self):
        super().autoRange()
        self.pad_current_view_y()

class SelectRegion(pg.LinearRegionItem):

    def __init__(self, *args, **kwargs):
        pg.LinearRegionItem.__init__(self, *args, **kwargs)
        for l in self.lines:
            l.setCursor(Qt.SizeHorCursor)
        self.setZValue(10)
        color = QColor(Qt.red)
        color.setAlphaF(0.05)
        self.setBrush(pg.mkBrush(color))

class CurvePlot(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.selection_enabled = hasattr(self.parent, "selected_indices")

        self.plotview = pg.PlotWidget(background="w", viewBox=InteractiveViewBox(self))
        self.plot = self.plotview.getPlotItem()
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.invertX(True)
        self.curves = [] #currently loaded curves
        self.curves_plotted = [] #currently plotted curves (different than loaded for averages)
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=20, slot=self.mouseMoved, delay=0.1)
        self.plot.scene().sigMouseMoved.connect(self.plot.vb.mouseMovedEvent)
        self.plot.vb.sigRangeChanged.connect(self.resized)
        self.pen_mouse = pg.mkPen(color=(0, 0, 255), width=2)
        self.pen_normal = defaultdict(lambda: pg.mkPen(color=(200, 200, 200, 127), width=1))
        self.pen_subset = defaultdict(lambda: pg.mkPen(color=(0, 0, 0, 127), width=1))
        self.pen_selected = defaultdict(lambda: pg.mkPen(color=(0, 0, 0, 127), width=2, style=QtCore.Qt.DotLine))
        self.label = pg.TextItem("", anchor=(1,0))
        self.label.setText("", color=(0,0,0))
        self.discrete_palette = None
        self.sampled_indices = []

        QPixmapCache.setCacheLimit(max(QPixmapCache.cacheLimit(), 100 * 1024))
        self.curves_cont = PlotCurvesItem()
        self.curves_cont.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

        #interface settings
        self.snap = True #snap to closest point on curve
        self.location = True #show current position
        self.markclosest = True #mark

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        self.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().addWidget(self.plotview)
        self.highlighted = None
        self.markings = []
        self.subset_ids = set()
        self.data = None

        self.viewtype = INDIVIDUAL

        #self.clear()

        zoom_in = QtGui.QAction(
            "Zoom in", self, triggered=self.set_mode_zooming
        )
        zoom_in.setShortcuts([QtGui.QKeySequence(QtGui.QKeySequence.ZoomIn), Qt.Key_Z])

        zoom_fit = QtGui.QAction(
            "Fit in view", self,
            triggered=lambda x: (self.plot.vb.autoRange(), self.set_mode_panning())
        )
        zoom_fit.setShortcuts([QtGui.QKeySequence(Qt.ControlModifier | Qt.Key_0), Qt.Key_Backspace])

        view_individual = QtGui.QAction(
            "Show individual", self, shortcut=Qt.Key_I,
            triggered=lambda x: self.show_individual()
        )

        development_key = QtGui.QAction(
            "Rescale Y", self, shortcut=Qt.Key_D,
            triggered=self.rescale_current_view_y
        )

        view_average = QtGui.QAction(
            "Show individual", self, shortcut=Qt.Key_A,
            triggered=lambda x: self.show_average()
        )

        self.set_mode_panning()
        self.addActions([zoom_in, zoom_fit, view_individual, view_average, development_key])

    def resized(self):
        self.label.setPos(self.plot.vb.viewRect().bottomLeft())

    def selection_changed(self):
        if self.parent and self.selection_enabled:
            self.parent.selection_changed()

    def mouseMoved(self, evt):
        pos = evt[0]

        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.vb.mapSceneToView(pos)
            posx, posy = mousePoint.x(), mousePoint.y()

            if self.location:
                self.label.setText("%g, %g" % (posx, posy), color=(0,0,0))
            else:
                self.label.setText("")

            if self.curves and len(self.curves[0][0]) and self.viewtype == INDIVIDUAL:
                cache = {}
                R = 20
                bd = None
                if self.markclosest and self.state != ZOOMING:
                    xpixel, ypixel = self.plot.vb.viewPixelSize()
                    distances = [ distancetocurve(c, posx, posy, xpixel, ypixel, r=R, cache=cache) for c in self.curves ]
                    bd = min(enumerate(distances), key= lambda x: x[1][0])
                if self.highlighted is not None:
                    self.highlighted = None
                    self.highlighted_curve.hide()
                if bd and bd[1][0] < R:
                    self.highlighted = bd[0]
                    x,y = self.curves[self.highlighted]
                    self.highlighted_curve.setData(x=x,y=y)
                    self.highlighted_curve.show()
                    if self.snap:
                        posx,posy = self.curves[bd[0]][0][bd[1][1]], self.curves[bd[0]][1][bd[1][1]]

            self.vLine.setPos(posx)
            self.hLine.setPos(posy)

    def set_curve_pen(self, idc):
        idcdata = idc if not self.sampled_indices else self.sampled_indices[idc]
        insubset = not self.subset_ids or self.data[idcdata].id in self.subset_ids
        inselected = self.selection_enabled and not self.sampled_indices and idc in self.parent.selected_indices
        thispen = self.pen_subset if insubset else self.pen_normal
        if inselected:
            thispen = self.pen_selected
        color_var = self._current_color_var()
        value = None if isinstance(color_var, str) else str(self.data[idcdata][color_var])
        self.curves_cont.objs[idc].setPen(thispen[value])
        self.curves_cont.objs[idc].setZValue(int(insubset) + int(inselected))

    def set_curve_pens(self, curves=None):
        if self.viewtype == INDIVIDUAL and self.curves:
            curves = range(len(self.curves)) if curves is None else curves
            for i in curves:
                self.set_curve_pen(i)
            self.curves_cont.update()

    def clear_data(self):
        self.subset_ids = set()
        if self.selection_enabled:
            self.parent.selected_indices.clear()
        self.curves = []
        self.sampled_indices = []
        self.selection_changed()
        self.discrete_palette = None

    def clear_graph(self):
        # reset caching. if not, it is not cleared when view changing when zoomed
        self.curves_cont.setCacheMode(QGraphicsItem.NoCache)
        self.curves_cont.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.plot.vb.disableAutoRange()
        self.curves_cont.clear()
        self.curves_cont.update()
        self.plotview.clear()
        self.curves_plotted = []
        self.plotview.addItem(self.label, ignoreBounds=True)
        self.highlighted_curve = pg.PlotCurveItem(pen=self.pen_mouse)
        self.highlighted_curve.setZValue(10)
        self.highlighted_curve.hide()
        self.plot.addItem(self.highlighted_curve)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.plot.addItem(self.curves_cont)
        for m in self.markings:
            self.plot.addItem(m, ignoreBounds=True)

    def add_marking(self, item):
        self.markings.append(item)
        self.plot.addItem(item, ignoreBounds=True)

    def remove_marking(self, item):
        self.markings.remove(item)
        self.plot.removeItem(item)

    def add_curves(self, x, data, addc=True):
        """ Add multiple curves with the same x domain. """
        xsind = np.argsort(x)
        x = x[xsind]
        if addc and len(data) > MAXINST:
            self.sampled_indices = sorted(random.Random(0).sample(range(len(data)), MAXINST))
        ys = data.X if not self.sampled_indices else data.X[self.sampled_indices]
        for y in ys:
            y = y[xsind]
            if addc:
                self.curves.append((x,y))
            c = pg.PlotCurveItem(x=x, y=y, pen=self.pen_normal[None])
            self.curves_cont.add_curve(c)
        self.curves_plotted = self.curves

    def add_curve(self, x, y, pen=None):
        xsind = np.argsort(x)
        x = x[xsind]
        c = pg.PlotCurveItem(x=x, y=y, pen=pen if pen else self.pen_normal[None])
        self.curves_cont.add_curve(c)
        self.curves_plotted.append((x, y))

    def _current_color_var(self):
        color_var = "(Same color)"
        if hasattr(self.parent, "attrs"):
            try:
                color_var = self.parent.attrs[self.parent.color_attr]
            except IndexError:
                pass
        return color_var

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
                self.pen_selected[v] = pg.mkPen(color=basecolor, width=2, style=QtCore.Qt.DotLine)
                notselcolor = basecolor.lighter(150)
                notselcolor.setAlphaF(0.5)
                self.pen_normal[v] = pg.mkPen(color=notselcolor, width=1)

    def show_individual(self):
        if not self.data:
            return
        self.viewtype = INDIVIDUAL
        self.clear_graph()
        x = getx(self.data)
        self.add_curves(x, self.data, addc=not self.curves)
        self.set_curve_pens()
        self.curves_cont.update()

    def rescale_current_view_y(self):
        if self.curves_plotted:
            cache = {}
            qrect = self.plot.vb.targetRect()
            bleft =  qrect.left()
            bright = qrect.right()

            ymax = max(np.max(y[searchsorted_cached(cache, x, bleft):
                                searchsorted_cached(cache, x, bright, side="right")])
                       for x,y in self.curves_plotted)
            ymin = min(np.min(y[searchsorted_cached(cache, x, bleft):
                                searchsorted_cached(cache, x, bright, side="right")])
                       for x,y in self.curves_plotted)

            self.plot.vb.setYRange(ymin, ymax, padding=0.0)
            self.plot.vb.pad_current_view_y()


    def _split_by_color_value(self, data):
        color_var = self._current_color_var()
        rd = defaultdict(list)
        for i,inst in enumerate(data):
            value = None if isinstance(color_var, str) else str(inst[color_var])
            rd[value].append(i)
        return rd

    def show_average(self):
        if not self.data:
            return
        self.viewtype = AVERAGE
        self.clear_graph()
        x = getx(self.data)
        xsind = np.argsort(x)
        x = x[xsind]
        if self.data:
            subset_indices = [i for i, id in enumerate(self.data.ids) if id in self.subset_ids]
            dsplit = self._split_by_color_value(self.data)
            for colorv, indices in dsplit.items():
                for part in ["everything", "subset", "selection"]:
                    if part == "everything":
                        ys = self.data.X[indices]
                        pen = self.pen_normal if subset_indices else self.pen_subset
                    elif self.selection_enabled and part == "selection" and not self.sampled_indices: #disable selection with sampled indices
                        current_selected = sorted(set(self.parent.selected_indices) & set(indices))
                        if not current_selected:
                            continue
                        ys = self.data.X[current_selected]
                        pen = self.pen_selected
                    elif part == "subset":
                        current_subset = sorted(set(subset_indices) & set(indices))
                        if not current_subset:
                            continue
                        ys = self.data.X[current_subset]
                        pen = self.pen_subset
                    std = np.std(ys, axis=0)
                    mean = np.mean(ys, axis=0)
                    std = std[xsind]
                    mean = mean[xsind]
                    self.add_curve(x, mean, pen=pen[colorv])
                    self.add_curve(x, mean+std, pen=pen[colorv])
                    self.add_curve(x, mean-std, pen=pen[colorv])
        self.curves_cont.update()

    def update_view(self):
        if self.viewtype == INDIVIDUAL:
            self.show_individual()
        elif self.viewtype == AVERAGE:
            self.show_average()

    def set_data(self, data, rescale="auto"):
        self.clear_graph()
        self.clear_data()
        self.set_pen_colors()
        if data is not None:
            if rescale == "auto":
                if self.data:
                    rescale = not data.domain == self.data.domain
                else:
                    rescale = True
            self.data = data
            self.update_view()
            if rescale == True:
                self.plot.vb.autoRange()

    def update_display(self):
        self.curves_cont.update()

    def set_data_subset(self, ids):
        self.subset_ids = set(ids) if ids is not None else set()
        self.set_curve_pens()
        self.update_view()

    def set_mode_zooming(self):
        self.plot.vb.setMouseMode(self.plot.vb.RectMode)
        self.state = ZOOMING
        self.plot.vb.zoomstartpoint = None
        self.setCursor(Qt.CrossCursor)

    def set_mode_panning(self):
        self.plot.vb.setMouseMode(self.plot.vb.PanMode)
        self.state = PANNING
        self.plot.vb.zoomstartpoint = None
        self.unsetCursor()


class OWCurves(widget.OWWidget):
    name = "Curves"
    inputs = [("Data", Orange.data.Table, 'set_data', Default),
              ("Data subset", Orange.data.Table, 'set_subset', Default)]
    outputs = [("Selection", Orange.data.Table)]
    icon = "icons/curves.svg"

    settingsHandler = DomainContextHandler(
        match_values=DomainContextHandler.MATCH_VALUES_ALL)
    selected_indices = ContextSetting(set())
    color_attr = ContextSetting(0)

    def __init__(self):
        super().__init__()
        self.controlArea.hide()
        self.plotview = CurvePlot(self)

        self.attrs = []
        self.topbox = gui.hBox(self)
        self.mainArea.layout().addWidget(self.topbox)
        model = VariableListModel()
        model.wrap(self.attrs)
        label = gui.label(self.topbox, self, "Color by")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.attrCombo = gui.comboBox(
            self.topbox, self, value="color_attr", contentsLength=12,
            callback=self.change_color_attr)
        self.attrCombo.setModel(model)

        self.mainArea.layout().addWidget(self.plotview)
        self.resize(900, 700)

    def change_color_attr(self):
        self.plotview.set_pen_colors()
        self.plotview.update_view()

    def set_data(self, data):
        self.information(0)
        self.closeContext()
        self.attrs[:] = []
        if data is not None:
            self.attrs[:] = ["(Same color)"] + [
                var for var in chain(data.domain,
                                     data.domain.metas)
                if isinstance(var, str) or var.is_discrete]
            self.color_attr = 0
        self.plotview.set_data(data)
        if self.plotview.sampled_indices:
            self.information(0, "Showing %d of %d curves." % (len(self.plotview.sampled_indices), len(data)))
        self.openContext(data)
        self.plotview.set_pen_colors()
        self.plotview.set_curve_pens() #mark the selection
        self.selection_changed()

    def set_subset(self, data):
        self.plotview.set_data_subset(data.ids if data else None)

    def selection_changed(self):
        if not self.plotview.sampled_indices \
                and self.selected_indices and self.plotview.data:
            # discard selected indices if they do not fit to data
            if any(a for a in self.selected_indices if a >= len(self.plotview.data)):
                self.selected_indices.clear()
            self.send("Selection", self.plotview.data[sorted(self.selected_indices)])
        else:
            self.send("Selection", None)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    w = OWCurves()
    w.show()
    import os.path
    data = Orange.data.Table("2012.11.09-11.45_Peach juice colorful spot.dpt")
    data = Orange.data.Table("iris.tab")
    w.set_data(data)
    w.set_subset(data[:40])
    #w.set_subset(None)
    w.handleNewSignals()
    region = SelectRegion()
    def update():
        minX, maxX = region.getRegion()
        print(minX, maxX)
    region.sigRegionChanged.connect(update)
    w.plotview.add_marking(region)
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
