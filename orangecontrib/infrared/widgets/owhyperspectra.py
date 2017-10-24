import sys
import gc
import collections
from xml.sax.saxutils import escape

from AnyQt.QtWidgets import QWidget, QPushButton, \
    QGridLayout, QFormLayout, QAction, QVBoxLayout, QApplication, QWidgetAction, QSplitter, \
    QToolTip
from AnyQt.QtGui import QColor, QKeySequence, QPainter, QBrush, QStandardItemModel, \
    QStandardItem, QLinearGradient, QPixmap, QIcon

from AnyQt.QtCore import Qt, QRectF, QPointF, QSize
from AnyQt.QtTest import QTest

import numpy as np
import pyqtgraph as pg
import colorcet

from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets.widget import OWWidget, Msg, OWComponent
from Orange.widgets import gui
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.data import DiscreteVariable

from orangecontrib.infrared.data import getx

from orangecontrib.infrared.preprocess import Integrate

from orangecontrib.infrared.widgets.owspectra import InteractiveViewBox, \
    MenuFocus, CurvePlot, SELECTONE, SELECTMANY, INDIVIDUAL, AVERAGE, \
    HelpEventDelegate

from orangecontrib.infrared.widgets.owpreprocess_spectra import MovableVlineWD
from orangecontrib.infrared.widgets.line_geometry import in_polygon

from Orange.widgets.utils.annotated_data import create_annotated_table


IMAGE_TOO_BIG = 1024*1024


def refresh_integral_markings(dis, markings_list, curveplot):
    for m in markings_list:
        if m in curveplot.markings:
            curveplot.remove_marking(m)
    markings_list.clear()

    def add_marking(a):
        markings_list.append(a)
        curveplot.add_marking(a)

    for di in dis:

        if di is None:
            continue  # nothing to draw

        color = QColor(di.get("color", "red"))

        for el in di["draw"]:

            if el[0] == "curve":
                bs_x, bs_ys, penargs = el[1]
                curve = pg.PlotCurveItem()
                curve.setPen(pg.mkPen(color=QColor(color), **penargs))
                curve.setZValue(10)
                curve.setData(x=bs_x, y=bs_ys[0])
                add_marking(curve)

            elif el[0] == "fill":
                (x1, ys1), (x2, ys2) = el[1]
                phigh = pg.PlotCurveItem(x1, ys1[0], pen=None)
                plow = pg.PlotCurveItem(x2, ys2[0], pen=None)
                color = QColor(color)
                color.setAlphaF(0.5)
                cc = pg.mkBrush(color)
                pfill = pg.FillBetweenItem(plow, phigh, brush=cc)
                pfill.setZValue(9)
                add_marking(pfill)

            elif el[0] == "line":
                (x1, y1), (x2, y2) = el[1]
                line = pg.PlotCurveItem()
                line.setPen(pg.mkPen(color=QColor(color), width=4))
                line.setZValue(10)
                line.setData(x=[x1[0], x2[0]], y=[y1[0], y2[0]])
                add_marking(line)

            elif el[0] == "dot":
                (x, ys) = el[1]
                dot = pg.ScatterPlotItem(x=x, y=ys[0])
                dot.setPen(pg.mkPen(color=QColor(color), width=5))
                dot.setZValue(10)
                add_marking(dot)


def values_to_linspace(vals):
    """Find a near maching linspace for the values given.
    The problem is that some values can be missing and
    that they are inexact. The minumum and maximum values
    are kept as limits."""
    vals = vals[~np.isnan(vals)]
    if len(vals):
        vals = np.unique(vals)
        if len(vals) == 1:
            return vals[0], vals[0], 1
        minabsdiff = (vals[-1] - vals[0])/(len(vals)*100)
        diffs = np.diff(vals)
        diffs = diffs[diffs > minabsdiff]
        first_valid = diffs[0]
        # allow for a percent mismatch
        diffs = diffs[diffs < first_valid*1.01]
        step = np.mean(diffs)
        size = int(round((vals[-1]-vals[0])/step) + 1)
        return vals[0], vals[-1], size
    return None


def location_values(vals, linspace):
    vals = np.asarray(vals)
    if linspace[2] == 1:  # everything is the same value
        width = 1
    else:
        width =  (linspace[1] - linspace[0]) / (linspace[2] - 1)
    return (vals - linspace[0]) / width


def index_values(vals, linspace):
    """ Remap values into index of array defined by linspace. """
    return np.round(location_values(vals, linspace)).astype(int)


def _shift(ls):
    if ls[2] == 1:
        return 0.5
    return (ls[1]-ls[0])/(2*(ls[2]-1))


def get_levels(img):
    """ Compute levels. Account for NaN values. """
    while img.size > 2 ** 16:
        img = img[::2, ::2]
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mn == mx:
        mn = 0
        mx = 255
    return [mn, mx]


class ImageItemNan(pg.ImageItem):
    """ Simplified ImageItem that can show NaN color. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = None

    def setSelection(self, selection):
        self.selection = selection
        self.updateImage()

    def render(self):
        # simplified pg.ImageITem

        if self.image is None or self.image.size == 0:
            return
        if isinstance(self.lut, collections.Callable):
            lut = self.lut(self.image)
        else:
            lut = self.lut

        image = self.image
        levels = self.levels

        if self.axisOrder == 'col-major':
            image = image.transpose((1, 0, 2)[:image.ndim])

        argb, alpha = pg.makeARGB(image, lut=lut, levels=levels)
        argb[np.isnan(image)] = (100, 100, 100, 255)  # replace unknown values with a color
        if np.any(self.selection):
            alpha = True
            argb[:, :, 3] = np.maximum(self.selection*255, 100)
        self.qimage = pg.makeQImage(argb, alpha, transpose=False)


def color_palette_table(colors, threshold_low=0.0, threshold_high=1.0,
                        underflow=None, overflow=None):
    N = len(colors)
    low, high = threshold_low * 255, threshold_high * 255
    points = np.linspace(low, high, N)
    space = np.linspace(0, 255, 256)

    if underflow is None:
        underflow = [None, None, None]

    if overflow is None:
        overflow = [None, None, None]

    r = np.interp(space, points, colors[:, 0],
                  left=underflow[0], right=overflow[0])
    g = np.interp(space, points, colors[:, 1],
                  left=underflow[1], right=overflow[1])
    b = np.interp(space, points, colors[:, 2],
                  left=underflow[2], right=overflow[2])

    return np.c_[r, g, b]


_color_palettes = [
    # linear
    ("bgy", {0: np.array(colorcet.linear_bgy_10_95_c74) * 255}),
    ("inferno", {0: np.array(colorcet.linear_bmy_10_95_c78) * 255}),
    ("dimgray", {0: np.array(colorcet.linear_grey_10_95_c0) * 255}),
    ("blues", {0: np.array(colorcet.linear_blue_95_50_c20) * 255}),
    ("fire", {0: np.array(colorcet.linear_kryw_0_100_c71) * 255}),

    # diverging - TODO set point
    ("bkr", {0: np.array(colorcet.diverging_bkr_55_10_c35) * 255}),
    ("bky", {0: np.array(colorcet.diverging_bky_60_10_c30) * 255}),
    ("coolwarm", {0: np.array(colorcet.diverging_bwr_40_95_c42) * 255}),
    ("bjy", {0: np.array(colorcet.diverging_linear_bjy_30_90_c45) * 255}),

    # misc
    ("rainbow", {0: np.array(colorcet.rainbow_bgyr_35_85_c73) * 255}),
    ("isolum", {0: np.array(colorcet.isoluminant_cgo_80_c38) * 255}),
]


def palette_gradient(colors):
    n = len(colors)
    stops = np.linspace(0.0, 1.0, n, endpoint=True)
    gradstops = [(float(stop), color) for stop, color in zip(stops, colors)]
    grad = QLinearGradient(QPointF(0, 0), QPointF(1, 0))
    grad.setStops(gradstops)
    return grad


def palette_pixmap(colors, size):
    img = QPixmap(size)
    img.fill(Qt.transparent)

    grad = palette_gradient(colors)
    grad.setCoordinateMode(QLinearGradient.ObjectBoundingMode)

    painter = QPainter(img)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(grad))
    painter.drawRect(0, 0, size.width(), size.height())
    painter.end()
    return img


def color_palette_model(palettes, iconsize=QSize(64, 16)):
    model = QStandardItemModel()
    for name, palette in palettes:
        _, colors = max(palette.items())
        colors = [QColor(*c) for c in colors]
        item = QStandardItem(name)
        item.setIcon(QIcon(palette_pixmap(colors, iconsize)))
        item.setData(palette, Qt.UserRole)
        model.appendRow([item])
    return model


class ImagePlot(QWidget, OWComponent):

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    gamma = Setting(0)
    threshold_low = Setting(0.0)
    threshold_high = Setting(1.0)
    palette_index = Setting(0)

    def __init__(self, parent, select_fn=None):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)

        self.parent = parent

        self.select_fn = select_fn

        self.selection_type = SELECTMANY
        self.saving_enabled = hasattr(self.parent, "save_graph")
        self.selection_enabled = True
        self.viewtype = INDIVIDUAL  # required bt InteractiveViewBox
        self.highlighted = None
        self.data_points = None
        self.data_values = None
        self.data_imagepixels = None
        self.selection = None

        self.plotview = pg.PlotWidget(background="w", viewBox=InteractiveViewBox(self))
        self.plot = self.plotview.getPlotItem()

        self.plot.scene().installEventFilter(
            HelpEventDelegate(self.help_event, self))

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)

        self.img = ImageItemNan()
        self.img.setOpts(axisOrder='row-major')
        self.plot.addItem(self.img)
        self.plot.vb.setAspectLocked()
        self.plot.scene().sigMouseMoved.connect(self.plot.vb.mouseMovedEvent)

        layout = QGridLayout()
        self.plotview.setLayout(layout)
        self.button = QPushButton("View", self.plotview)
        self.button.setAutoDefault(False)

        layout.setRowStretch(1, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self.button, 0, 0)
        view_menu = MenuFocus(self)
        self.button.setMenu(view_menu)

        # prepare interface according to the new context
        self.parent.contextAboutToBeOpened.connect(lambda x: self.init_interface_data(x[0]))

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
        select_square = QAction(
            "Select (square)", self, triggered=self.plot.vb.set_mode_select_square,
        )
        select_square.setShortcuts([Qt.Key_S])
        select_square.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(select_square)


        select_polygon = QAction(
            "Select (polygon)", self, triggered=self.plot.vb.set_mode_select_polygon,
        )
        select_polygon.setShortcuts([Qt.Key_P])
        select_polygon.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(select_polygon)

        if self.saving_enabled:
            save_graph = QAction(
                "Save graph", self, triggered=self.save_graph,
            )
            save_graph.setShortcuts([QKeySequence(Qt.ControlModifier | Qt.Key_I)])
            actions.append(save_graph)

        view_menu.addActions(actions)
        self.addActions(actions)

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
            valueType=str)

        choose_xy = QWidgetAction(self)
        box = gui.vBox(self)
        box.setFocusPolicy(Qt.TabFocus)
        self.xy_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES, valid_types=DomainModel.PRIMITIVE)
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        box.setFocusProxy(self.cb_attr_x)

        self.color_cb = gui.comboBox(box, self, "palette_index", label="Color:",
                                     labelWidth=50, orientation=Qt.Horizontal)
        self.color_cb.setIconSize(QSize(64, 16))
        palettes = _color_palettes

        self.palette_index = min(self.palette_index, len(palettes) - 1)

        model = color_palette_model(palettes, self.color_cb.iconSize())
        model.setParent(self)
        self.color_cb.setModel(model)
        self.color_cb.activated.connect(self.update_color_schema)

        self.color_cb.setCurrentIndex(self.palette_index)

        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )

        lowslider = gui.hSlider(
            box, self, "threshold_low", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_color_schema)
        highslider = gui.hSlider(
            box, self, "threshold_high", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_color_schema)

        form.addRow("Low:", lowslider)
        form.addRow("High:", highslider)

        box.layout().addLayout(form)

        choose_xy.setDefaultWidget(box)
        view_menu.addAction(choose_xy)

        self.markings_integral = []

        self.lsx = None  # info about the X axis
        self.lsy = None  # info about the Y axis

        self.data = None
        self.data_ids = {}

    def init_interface_data(self, data):
        same_domain = (self.data and data and
                       data.domain == self.data.domain)
        if not same_domain:
            self.init_attr_values(data)

    def help_event(self, ev):
        pos = self.plot.vb.mapSceneToView(ev.scenePos())
        sel = self._points_at_pos(pos)
        prepared = []
        if sel is not None:
            data, vals, points = self.data[sel], self.data_values[sel], self.data_points[sel]
            for d, v, p in zip(data, vals, points):
                basic = "({}, {}): {}".format(p[0], p[1], v)
                variables = [ v for v in self.data.domain.metas + self.data.domain.class_vars
                              if v not in [self.attr_x, self.attr_y]]
                features = ['{} = {}'.format(attr.name, d[attr]) for attr in variables]
                prepared.append("\n".join([basic] + features))
        text = "\n\n".join(prepared)
        if text:
            text = ('<span style="white-space:pre">{}</span>'
                    .format(escape(text)))
            QToolTip.showText(ev.screenPos(), text, widget=self.plotview)
            return True
        else:
            return False

    def update_color_schema(self):
        if not self.threshold_low < self.threshold_high:
            # TODO this belongs here, not in the parent
            self.parent.Warning.threshold_error()
            return
        else:
            self.parent.Warning.threshold_error.clear()
        data = self.color_cb.itemData(self.palette_index, role=Qt.UserRole)
        _, colors = max(data.items())
        cols = color_palette_table(
            colors, threshold_low=self.threshold_low,
            threshold_high=self.threshold_high)

        self.img.setLookupTable(cols)

        # use defined discrete palette
        if self.parent.value_type == 1:
            dat = self.data.domain[self.parent.attr_value]
            if isinstance(dat, DiscreteVariable):
                self.img.setLookupTable(dat.colors)

    def update_attr(self):
        self.update_view()

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

    def save_graph(self):
        self.parent.save_graph()

    def set_data(self, data):
        self.img.clear()
        self.data = data
        self.data_ids = {e: i for i, e in enumerate(data.ids)} if data else {}

    def refresh_markings(self, di):
        refresh_integral_markings([{"draw": di}], self.markings_integral, self.parent.curveplot)

    def update_view(self):
        self.img.clear()
        self.img.setSelection(None)
        self.lsx = None
        self.lsy = None
        self.data_points = None
        self.data_values = None
        self.data_imagepixels = None
        if self.data and self.attr_x and self.attr_y:
            xat = self.data.domain[self.attr_x]
            yat = self.data.domain[self.attr_y]

            ndom = Orange.data.Domain([xat, yat])
            datam = Orange.data.Table(ndom, self.data)
            coorx = datam.X[:, 0]
            coory = datam.X[:, 1]
            self.data_points = datam.X
            self.lsx = lsx = values_to_linspace(coorx)
            self.lsy = lsy = values_to_linspace(coory)
            if lsx[-1] * lsy[-1] > IMAGE_TOO_BIG:
                self.parent.Error.image_too_big(lsx[-1], lsy[-1])
                return
            else:
                self.parent.Error.image_too_big.clear()

            di = {}
            if self.parent.value_type == 0:  # integrals
                imethod = self.parent.integration_methods[self.parent.integration_method]

                l1, l2, l3 = self.parent.lowlim, self.parent.highlim, self.parent.choose

                gx = getx(self.data)

                if l1 is None:
                    l1 = min(gx) - 1
                if l2 is None:
                    l2 = max(gx) + 1

                l1, l2 = min(l1, l2), max(l1, l2)

                if l3 is None:
                    l3 = (l1 + l2)/2

                if imethod != Integrate.PeakAt:
                    datai = Integrate(methods=imethod, limits=[[l1, l2]])(self.data)
                else:
                    datai = Integrate(methods=imethod, limits=[[l3, l3]])(self.data)

                if self.parent.curveplot.selected_indices:
                    # curveplot can have a subset of curves on the input> match IDs
                    ind = list(self.parent.curveplot.selected_indices)[0]
                    dind = self.data_ids[self.parent.curveplot.data[ind].id]
                    di = datai.domain.attributes[0].compute_value.draw_info(self.data[dind:dind+1])
                d = datai.X[:, 0]
            else:
                dat = self.data.domain[self.parent.attr_value]
                ndom = Orange.data.Domain([dat])
                d = Orange.data.Table(ndom, self.data).X[:, 0]
            self.refresh_markings(di)

            # set data
            imdata = np.ones((lsy[2], lsx[2])) * float("nan")

            # if previous or saved selection is valid for this data set keep it
            if self.selection is None or len(self.selection) != len(self.data):
                self.selection = np.zeros(len(self.data), dtype="bool")

            xindex = index_values(coorx, lsx)
            yindex = index_values(coory, lsy)
            imdata[yindex, xindex] = d
            self.data_values = d
            self.data_imagepixels = np.vstack((yindex, xindex)).T

            levels = get_levels(imdata)
            self.update_color_schema()

            self.img.setImage(imdata, levels=levels)

            # shift centres of the pixels so that the axes are useful
            shiftx = _shift(lsx)
            shifty = _shift(lsy)
            left = lsx[0] - shiftx
            bottom = lsy[0] - shifty
            width = (lsx[1]-lsx[0]) + 2*shiftx
            height = (lsy[1]-lsy[0]) + 2*shifty
            self.img.setRect(QRectF(left, bottom, width, height))

            self.send_selection()
            self.refresh_img_selection()

    def refresh_img_selection(self):
        selected_px = np.zeros((self.lsy[2], self.lsx[2]), dtype=bool)
        selected_px_ind = self.data_imagepixels[self.selection]
        selected_px[selected_px_ind[:, 0], selected_px_ind[:, 1]] = 1
        self.img.setSelection(selected_px)

    def make_selection(self, selected, add):
        """Add selected indices to the selection."""
        if self.data and self.lsx and self.lsy:
            if selected is None and not add:
                self.selection *= False  # set all to False
            elif selected is not None:
                if add:
                    self.selection = np.logical_or(self.selection, selected)
                else:
                    self.selection = selected
            self.refresh_img_selection()
        self.send_selection()

    def send_selection(self):
        if self.data and self.selection is not None:
            selected = np.where(self.selection)[0]
        else:
            selected = []
        if self.select_fn:
            self.select_fn(selected)

    def select_square(self, p1, p2, add):
        """ Select elements within a square drawn by the user.
        A selection needs to contain whole pixels """
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        polygon = [QPointF(x1, y1), QPointF(x2, y1), QPointF(x2, y2), QPointF(x1, y2), QPointF(x1, y1)]
        self.select_polygon(polygon, add)

    def select_polygon(self, polygon, add):
        """ Select by a polygon which has to contain whole pixels. """
        if self.data and self.lsx and self.lsy:
            polygon = [(p.x(), p.y()) for p in polygon]
            # a polygon should contain all pixel
            shiftx = _shift(self.lsx)
            shifty = _shift(self.lsy)
            points_edges = [self.data_points + [[shiftx, shifty]],
                            self.data_points + [[-shiftx, shifty]],
                            self.data_points + [[shiftx, -shifty]],
                            self.data_points + [[-shiftx, -shifty]]]
            inp = in_polygon(points_edges[0], polygon)
            for p in points_edges[1:]:
                inp *= in_polygon(p, polygon)
            self.make_selection(inp, add)

    def _points_at_pos(self, pos):
        if self.data and self.lsx and self.lsy:
            x, y = pos.x(), pos.y()
            distance = np.abs(self.data_points - [[x, y]])
            sel = (distance[:, 0] < _shift(self.lsx)) * (distance[:, 1] < _shift(self.lsy))
            return sel

    def select_by_click(self, pos, add):
        sel = self._points_at_pos(pos)
        self.make_selection(sel, add)


class CurvePlotHyper(CurvePlot):
    viewtype = Setting(AVERAGE)  # average view by default


class OWHyper(OWWidget):
    name = "HyperSpectra"
    inputs = [("Data", Orange.data.Table, 'set_data', Default)]
    outputs = [("Selection", Orange.data.Table), ("Data", Orange.data.Table)]
    icon = "icons/hyper.svg"
    priority = 20

    settings_version = 2
    settingsHandler = DomainContextHandler(metas_in_res=True)

    imageplot = SettingProvider(ImagePlot)
    curveplot = SettingProvider(CurvePlotHyper)

    integration_method = Setting(0)
    integration_methods = Integrate.INTEGRALS
    value_type = Setting(0)
    attr_value = ContextSetting(None)

    lowlim = Setting(None)
    highlim = Setting(None)
    choose = Setting(None)

    class Warning(OWWidget.Warning):
        threshold_error = Msg("Low slider should be less than High")

    class Error(OWWidget.Warning):
        image_too_big = Msg("Image for chosen features is too big ({} x {}).")

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            # delete the saved attr_value to prevent crashes
            try:
                del settings_["context_settings"][0].values["attr_value"]
            except:
                pass

    def __init__(self):
        super().__init__()

        dbox = gui.widgetBox(self.controlArea, "Image values")

        rbox = gui.radioButtons(
            dbox, self, "value_type", callback=self._change_integration)

        gui.appendRadioButton(rbox, "From spectra")

        self.box_values_spectra = gui.indentedBox(rbox)

        gui.comboBox(
            self.box_values_spectra, self, "integration_method", valueType=int,
            items=(a.name for a in self.integration_methods),
            callback=self._change_integral_type)
        gui.rubber(self.controlArea)

        gui.appendRadioButton(rbox, "Use feature")

        self.box_values_feature = gui.indentedBox(rbox)

        self.feature_value_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES,
                                               valid_types=DomainModel.PRIMITIVE)
        self.feature_value = gui.comboBox(
            self.box_values_feature, self, "attr_value",
            callback=self.update_feature_value, model=self.feature_value_model,
            sendSelectedValue=True, valueType=str)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        self.imageplot = ImagePlot(self, select_fn=self.image_selection_changed)
        self.curveplot = CurvePlotHyper(self, select=SELECTONE)
        self.curveplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        splitter.addWidget(self.imageplot)
        splitter.addWidget(self.curveplot)
        self.mainArea.layout().addWidget(splitter)

        self.line1 = MovableVlineWD(position=self.lowlim, label="", setvalfn=self.set_lowlim,
                                    confirmfn=self.changed_integral_range, report=self.curveplot)
        self.line2 = MovableVlineWD(position=self.highlim, label="", setvalfn=self.set_highlim,
                                    confirmfn=self.changed_integral_range, report=self.curveplot)
        self.line3 = MovableVlineWD(position=self.choose, label="", setvalfn=self.set_choose,
                                    confirmfn=self.changed_integral_range, report=self.curveplot)
        self.curveplot.add_marking(self.line1)
        self.curveplot.add_marking(self.line2)
        self.curveplot.add_marking(self.line3)
        self.line1.hide()
        self.line2.hide()
        self.line3.hide()

        self.data = None
        self.disable_integral_range = False

        self.resize(900, 700)
        self.graph_name = "imageplot.plotview"
        self._update_integration_type()

        # prepare interface according to the new context
        self.contextAboutToBeOpened.connect(lambda x: self.init_interface_data(x[0]))

    def init_interface_data(self, data):
        same_domain = (self.data and data and
                       data.domain == self.data.domain)
        if not same_domain:
            self.init_attr_values(data)

    def image_selection_changed(self, indices):
        annotated = create_annotated_table(self.data, indices)
        self.send("Data", annotated)
        if self.data:
            selected = self.data[indices]
            self.send("Selection", selected if selected else None)
            if selected:
                self.curveplot.set_data(selected)
            else:
                self.curveplot.set_data(self.data)
        else:
            self.send("Selection", None)
            self.curveplot.set_data(None)

    def selection_changed(self):
        self.redraw_data()

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.feature_value_model.set_domain(domain)
        self.attr_value = self.feature_value_model[0] if self.feature_value_model else None

    def set_lowlim(self, v):
        self.lowlim = v

    def set_highlim(self, v):
        self.highlim = v

    def set_choose(self, v):
        self.choose = v

    def redraw_data(self):
        self.imageplot.update_view()

    def update_feature_value(self):
        self.redraw_data()

    def _update_integration_type(self):
        self.line1.hide()
        self.line2.hide()
        self.line3.hide()
        if self.value_type == 0:
            self.box_values_spectra.setDisabled(False)
            self.box_values_feature.setDisabled(True)
            if self.integration_methods[self.integration_method] != Integrate.PeakAt:
                self.line1.show()
                self.line2.show()
            else:
                self.line3.show()
        elif self.value_type == 1:
            self.box_values_spectra.setDisabled(True)
            self.box_values_feature.setDisabled(False)
        QTest.qWait(1)  # first update the interface

    def _change_integration(self):
        # change what to show on the image
        self._update_integration_type()
        self.redraw_data()

    def changed_integral_range(self):
        if self.disable_integral_range:
            return
        self.redraw_data()

    def _change_integral_type(self):
        self._change_integration()

    def set_data(self, data):
        self.closeContext()
        self.openContext(data)
        self.data = data
        self.imageplot.set_data(data)
        self.curveplot.set_data(data)
        self._init_integral_boundaries()
        self.imageplot.update_view()

    def _init_integral_boundaries(self):
        # requires data in curveplot
        self.disable_integral_range = True
        if self.curveplot.data_x is not None and len(self.curveplot.data_x):
            minx = self.curveplot.data_x[0]
            maxx = self.curveplot.data_x[-1]

            if self.lowlim is None or not minx <= self.lowlim <= maxx:
                self.lowlim = minx
            self.line1.setValue(self.lowlim)

            if self.highlim is None or not minx <= self.highlim <= maxx:
                self.highlim = maxx
            self.line2.setValue(self.highlim)

            if self.choose is None:
                self.choose = (minx + maxx)/2
            elif self.choose < minx:
                self.choose = minx
            elif self.choose > maxx:
                self.choose = maxx
            self.line3.setValue(self.choose)
        self.disable_integral_range = False

    # store selection as a list due to a bug in checking if numpy settings changed
    def storeSpecificSettings(self):
        selection = self.imageplot.selection
        if selection is not None:
            selection = list(selection)
        self.current_context.selection = selection

    def retrieveSpecificSettings(self):
        selection = getattr(self.current_context, "selection", None)
        if selection is not None:
            selection = np.array(selection, dtype="bool")
        self.imageplot.selection = selection


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    w = OWHyper()
    w.show()
    from orangecontrib.infrared.tests.bigdata import dust
    #data = Orange.data.Table("whitelight.gsf")
    data = Orange.data.Table(dust())
    #data = Orange.data.Table("iris.tab")
    w.set_data(data)
    w.handleNewSignals()
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
