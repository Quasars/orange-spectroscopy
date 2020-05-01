import collections.abc
from xml.sax.saxutils import escape

from AnyQt.QtWidgets import QWidget, QPushButton, \
    QGridLayout, QFormLayout, QAction, QVBoxLayout, QWidgetAction, QSplitter, \
    QToolTip, QGraphicsRectItem
from AnyQt.QtGui import QColor, QKeySequence, QPainter, QBrush, QStandardItemModel, \
    QStandardItem, QLinearGradient, QPixmap, QIcon

from AnyQt.QtCore import Qt, QRectF, QPointF, QSize
from AnyQt.QtTest import QTest

from AnyQt.QtCore import pyqtSignal as Signal

import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsWidget
import colorcet

import Orange.data
from Orange.data import Domain
from Orange.widgets.widget import OWWidget, Msg, OWComponent, Input, Output
from Orange.widgets import gui
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils import saveplot
from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.widgets.utils.concurrent import TaskState, ConcurrentMixin

from orangecontrib.spectroscopy.preprocess import Integrate
from orangecontrib.spectroscopy.utils import values_to_linspace, index_values_nan, split_to_size

from orangecontrib.spectroscopy.widgets.owspectra import InteractiveViewBox, \
    MenuFocus, CurvePlot, SELECTONE, SELECTMANY, INDIVIDUAL, AVERAGE, \
    HelpEventDelegate, SelectionGroupMixin, selection_modifiers

from orangecontrib.spectroscopy.widgets.gui import MovableVline, lineEditDecimalOrNone,\
    pixels_to_decimals, float_to_str_decimals
from orangecontrib.spectroscopy.widgets.line_geometry import in_polygon
from orangecontrib.spectroscopy.widgets.utils import groups_or_annotated_table

from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME


IMAGE_TOO_BIG = 1024*1024*100


class InterruptException(Exception):
    pass


class ImageTooBigException(Exception):
    pass


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
        if isinstance(self.lut, collections.abc.Callable):
            lut = self.lut(self.image)
        else:
            lut = self.lut

        image = self.image
        levels = self.levels

        if self.axisOrder == 'col-major':
            image = image.transpose((1, 0, 2)[:image.ndim])

        argb, alpha = pg.makeARGB(image, lut=lut, levels=levels)  # format is bgra
        argb[np.isnan(image)] = (100, 100, 100, 255)  # replace unknown values with a color
        w = 1
        if np.any(self.selection):
            max_sel = np.max(self.selection)
            colors = DiscreteVariable(name="colors", values=map(str, range(max_sel))).colors
            fargb = argb.astype(np.float32)
            for i, color in enumerate(colors):
                color = np.hstack((color[::-1], [255]))  # qt color
                sel = self.selection == i+1
                # average the current color with the selection color
                argb[sel] = (fargb[sel] + w*color) / (1+w)
            alpha = True
            argb[:, :, 3] = np.maximum((self.selection > 0)*255, 100)
        self.qimage = pg.makeQImage(argb, alpha, transpose=False)


def color_palette_table(colors, underflow=None, overflow=None):
    points = np.linspace(0, 255, len(colors))
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


class ImageColorSettingMixin:
    threshold_low = Setting(0.0, schema_only=True)
    threshold_high = Setting(1.0, schema_only=True)
    level_low = Setting(None, schema_only=True)
    level_high = Setting(None, schema_only=True)
    show_legend = Setting(True)
    palette_index = Setting(0)

    def color_settings_box(self):
        box = gui.vBox(self)
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

        gui.checkBox(box, self, "show_legend", label="Show legend",
                     callback=self.update_legend_visible)

        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )

        def limit_changed():
            self.update_levels()
            self.reset_thresholds()

        self._level_low_le = lineEditDecimalOrNone(self, self, "level_low", callback=limit_changed)
        self._level_low_le.validator().setDefault(0)
        form.addRow("Low limit:", self._level_low_le)

        self._level_high_le = lineEditDecimalOrNone(self, self, "level_high", callback=limit_changed)
        self._level_high_le.validator().setDefault(1)
        form.addRow("High limit:", self._level_high_le)

        lowslider = gui.hSlider(
            box, self, "threshold_low", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_levels)
        highslider = gui.hSlider(
            box, self, "threshold_high", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_levels)

        form.addRow("Low:", lowslider)
        form.addRow("High:", highslider)
        box.layout().addLayout(form)

        self.update_legend_visible()

        return box

    def update_legend_visible(self):
        self.legend.setVisible(self.show_legend)

    def update_levels(self):
        if not self.data:
            return

        if not self.threshold_low < self.threshold_high:
            # TODO this belongs here, not in the parent
            self.parent.Warning.threshold_error()
            return
        else:
            self.parent.Warning.threshold_error.clear()

        if self.img.image is not None:
            levels = get_levels(self.img.image)
        else:
            levels = [0, 255]

        prec = pixels_to_decimals((levels[1] - levels[0])/1000)

        rounded_levels = [float_to_str_decimals(levels[0], prec),
                          float_to_str_decimals(levels[1], prec)]

        self._level_low_le.validator().setDefault(rounded_levels[0])
        self._level_high_le.validator().setDefault(rounded_levels[1])

        self._level_low_le.setPlaceholderText(rounded_levels[0])
        self._level_high_le.setPlaceholderText(rounded_levels[1])

        ll = float(self.level_low) if self.level_low is not None else levels[0]
        lh = float(self.level_high) if self.level_high is not None else levels[1]

        ll_threshold = ll + (lh - ll) * self.threshold_low
        lh_threshold = ll + (lh - ll) * self.threshold_high

        self.img.setLevels([ll_threshold, lh_threshold])
        self.legend.set_range(ll_threshold, lh_threshold)

    def update_color_schema(self):
        if not self.data:
            return

        if self.parent.value_type == 1:
            dat = self.data.domain[self.parent.attr_value]
            if isinstance(dat, DiscreteVariable):
                # use a defined discrete palette
                self.img.setLookupTable(dat.colors)
                return

        # use a continuous palette
        data = self.color_cb.itemData(self.palette_index, role=Qt.UserRole)
        _, colors = max(data.items())
        cols = color_palette_table(colors)
        self.img.setLookupTable(cols)
        self.legend.set_colors(cols)

    def reset_thresholds(self):
        self.threshold_low = 0.
        self.threshold_high = 1.


class ImageZoomMixin:

    def add_zoom_actions(self, menu):
        zoom_in = QAction(
            "Zoom in", self, triggered=self.plot.vb.set_mode_zooming
        )
        zoom_in.setShortcuts([Qt.Key_Z, QKeySequence(QKeySequence.ZoomIn)])
        zoom_in.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(zoom_in)
        if menu:
            menu.addAction(zoom_in)
        zoom_fit = QAction(
            "Zoom to fit", self,
            triggered=lambda x: (self.plot.vb.autoRange(), self.plot.vb.set_mode_panning())
        )
        zoom_fit.setShortcuts([Qt.Key_Backspace, QKeySequence(Qt.ControlModifier | Qt.Key_0)])
        zoom_fit.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(zoom_fit)
        if menu:
            menu.addAction(zoom_fit)


class ImageColorLegend(GraphicsWidget):

    def __init__(self):
        GraphicsWidget.__init__(self)
        self.width_bar = 15
        self.colors = None
        self.gradient = QLinearGradient()
        self.setMaximumHeight(2**16)
        self.setMinimumWidth(self.width_bar)
        self.setMaximumWidth(self.width_bar)
        self.rect = QGraphicsRectItem(QRectF(0, 0, self.width_bar, 100), self)
        self.axis = pg.AxisItem('right', parent=self)
        self.axis.setX(self.width_bar)
        self.axis.geometryChanged.connect(self._update_width)
        self.adapt_to_size()
        self._initialized = True

    def _update_width(self):
        aw = self.axis.minimumWidth()
        self.setMinimumWidth(self.width_bar + aw)
        self.setMaximumWidth(self.width_bar + aw)

    def resizeEvent(self, ev):
        if hasattr(self, "_initialized"):
            self.adapt_to_size()

    def adapt_to_size(self):
        h = self.height()
        self.resetTransform()
        self.rect.setRect(0, 0, self.width_bar, h)
        self.axis.setHeight(h)
        self.gradient.setStart(QPointF(0, h))
        self.gradient.setFinalStop(QPointF(0, 0))
        self.update_rect()

    def set_colors(self, colors):
        # a Nx3 array containing colors
        self.colors = colors
        if self.colors is not None:
            positions = np.linspace(0, 1, len(self.colors))
            stops = []
            for p, c in zip(positions, self.colors):
                stops.append((p, QColor(*c)))
            self.gradient.setStops(stops)
        self.update_rect()

    def set_range(self, low, high):
        self.axis.setRange(low, high)

    def update_rect(self):
        if self.colors is None:
            self.rect.setBrush(QBrush(Qt.white))
        else:
            self.rect.setBrush(QBrush(self.gradient))


class ImagePlot(QWidget, OWComponent, SelectionGroupMixin,
                ImageColorSettingMixin, ImageZoomMixin, ConcurrentMixin):

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    gamma = Setting(0)

    selection_changed = Signal()
    image_updated = Signal()

    def __init__(self, parent):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)
        SelectionGroupMixin.__init__(self)
        ImageColorSettingMixin.__init__(self)
        ImageZoomMixin.__init__(self)
        ConcurrentMixin.__init__(self)
        self.parent = parent

        self.selection_type = SELECTMANY
        self.saving_enabled = True
        self.selection_enabled = True
        self.viewtype = INDIVIDUAL  # required bt InteractiveViewBox
        self.highlighted = None
        self.data_points = None
        self.data_values = None
        self.data_imagepixels = None
        self.data_valid_positions = None

        self.plotview = pg.GraphicsLayoutWidget()
        self.plot = pg.PlotItem(background="w", viewBox=InteractiveViewBox(self))
        self.plotview.addItem(self.plot)

        self.legend = ImageColorLegend()
        self.plotview.addItem(self.legend)

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
        self.button = QPushButton("Menu", self.plotview)
        self.button.setAutoDefault(False)

        layout.setRowStretch(1, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self.button, 0, 0)
        view_menu = MenuFocus(self)
        self.button.setMenu(view_menu)

        # prepare interface according to the new context
        self.parent.contextAboutToBeOpened.connect(lambda x: self.init_interface_data(x[0]))

        actions = []

        self.add_zoom_actions(view_menu)

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
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True)

        choose_xy = QWidgetAction(self)
        box = gui.vBox(self)
        box.setFocusPolicy(Qt.TabFocus)
        self.xy_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES,
                                    valid_types=DomainModel.PRIMITIVE)
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        box.setFocusProxy(self.cb_attr_x)

        box.layout().addWidget(self.color_settings_box())

        choose_xy.setDefaultWidget(box)
        view_menu.addAction(choose_xy)

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
                variables = [v for v in self.data.domain.metas + self.data.domain.class_vars
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

    def update_attr(self):
        self.update_view()

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

    def save_graph(self):
        saveplot.save_plot(self.plotview, self.parent.graph_writers)

    def set_data(self, data):
        if data:
            self.data = data
            self.data_ids = {e: i for i, e in enumerate(data.ids)}
            self.restore_selection_settings()
        else:
            self.data = None
            self.data_ids = {}

    def refresh_img_selection(self):
        selected_px = np.zeros((self.lsy[2], self.lsx[2]), dtype=np.uint8)
        selected_px[self.data_imagepixels[self.data_valid_positions, 0],
                    self.data_imagepixels[self.data_valid_positions, 1]] = \
            self.selection_group[self.data_valid_positions]
        self.img.setSelection(selected_px)

    def make_selection(self, selected):
        """Add selected indices to the selection."""
        add_to_group, add_group, remove = selection_modifiers()
        if self.data and self.lsx and self.lsy:
            if add_to_group:  # both keys - need to test it before add_group
                selnum = np.max(self.selection_group)
            elif add_group:
                selnum = np.max(self.selection_group) + 1
            elif remove:
                selnum = 0
            else:
                self.selection_group *= 0
                selnum = 1
            if selected is not None:
                self.selection_group[selected] = selnum
            self.refresh_img_selection()
        self.prepare_settings_for_saving()
        self.selection_changed.emit()

    def select_square(self, p1, p2):
        """ Select elements within a square drawn by the user.
        A selection needs to contain whole pixels """
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        polygon = [QPointF(x1, y1), QPointF(x2, y1), QPointF(x2, y2), QPointF(x1, y2), QPointF(x1, y1)]
        self.select_polygon(polygon)

    def select_polygon(self, polygon):
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
            self.make_selection(inp)

    def _points_at_pos(self, pos):
        if self.data and self.lsx and self.lsy:
            x, y = pos.x(), pos.y()
            distance = np.abs(self.data_points - [[x, y]])
            sel = (distance[:, 0] < _shift(self.lsx)) * (distance[:, 1] < _shift(self.lsy))
            return sel

    def select_by_click(self, pos):
        sel = self._points_at_pos(pos)
        self.make_selection(sel)

    def update_view(self):
        self.cancel()
        self.parent.Error.image_too_big.clear()
        self.parent.Information.not_shown.clear()
        self.img.clear()
        self.img.setSelection(None)
        self.legend.set_colors(None)
        self.lsx = None
        self.lsy = None
        self.data_points = None
        self.data_values = None
        self.data_imagepixels = None
        self.data_valid_positions = None

        if self.data and self.attr_x and self.attr_y:
            self.start(self.compute_image, self.data, self.attr_x, self.attr_y,
                       self.parent.integrate_fn())
        else:
            self.image_updated.emit()

    @staticmethod
    def compute_image(data: Orange.data.Table, attr_x, attr_y,
                      integrate_fn, state: TaskState):

        def progress_interrupt(i: float):
            if state.is_interruption_requested():
                raise InterruptException

        class Result():
            pass
        res = Result()

        xat = data.domain[attr_x]
        yat = data.domain[attr_y]

        ndom = Domain([xat, yat])
        datam = data.transform(ndom)
        progress_interrupt(0)
        res.coorx = datam.X[:, 0]
        res.coory = datam.X[:, 1]
        res.data_points = datam.X
        res.lsx = lsx = values_to_linspace(res.coorx)
        res.lsy = lsy = values_to_linspace(res.coory)
        progress_interrupt(0)

        if lsx[-1] * lsy[-1] > IMAGE_TOO_BIG:
            raise ImageTooBigException((lsx[-1], lsy[-1]))

        # the code bellow does this, but part-wise:
        # d = integrate_fn(data).X[:, 0]
        parts = []
        for slice in split_to_size(len(data), 10000):
            part = integrate_fn(data[slice]).X[:, 0]
            parts.append(part)
            progress_interrupt(0)
        d = np.concatenate(parts)

        res.d = d
        progress_interrupt(0)

        return res

    def on_done(self, res):

        self.lsx, self.lsy = res.lsx, res.lsy
        lsx, lsy = self.lsx, self.lsy

        d = res.d

        self.data_points = res.data_points

        xindex, xnan = index_values_nan(res.coorx, self.lsx)
        yindex, ynan = index_values_nan(res.coory, self.lsy)
        self.data_valid_positions = valid = np.logical_not(np.logical_or(xnan, ynan))
        invalid_positions = len(d) - np.sum(valid)
        if invalid_positions:
            self.parent.Information.not_shown(invalid_positions)

        imdata = np.ones((lsy[2], lsx[2])) * float("nan")
        imdata[yindex[valid], xindex[valid]] = d[valid]
        self.data_values = d
        self.data_imagepixels = np.vstack((yindex, xindex)).T

        self.img.setImage(imdata, autoLevels=False)
        self.update_levels()
        self.update_color_schema()

        # shift centres of the pixels so that the axes are useful
        shiftx = _shift(lsx)
        shifty = _shift(lsy)
        left = lsx[0] - shiftx
        bottom = lsy[0] - shifty
        width = (lsx[1]-lsx[0]) + 2*shiftx
        height = (lsy[1]-lsy[0]) + 2*shifty
        self.img.setRect(QRectF(left, bottom, width, height))

        self.refresh_img_selection()
        self.image_updated.emit()

    def on_partial_result(self, result):
        pass

    def on_exception(self, ex: Exception):
        if isinstance(ex, InterruptException):
            return

        if isinstance(ex, ImageTooBigException):
            self.parent.Error.image_too_big(ex.args[0][0], ex.args[0][1])
            self.image_updated.emit()
        else:
            raise ex


class CurvePlotHyper(CurvePlot):
    viewtype = Setting(AVERAGE)  # average view by default


class OWHyper(OWWidget):
    name = "HyperSpectra"

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        selected_data = Output("Selection", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    icon = "icons/hyper.svg"
    priority = 20
    replaces = ["orangecontrib.infrared.widgets.owhyper.OWHyper"]

    settings_version = 4
    settingsHandler = DomainContextHandler()

    imageplot = SettingProvider(ImagePlot)
    curveplot = SettingProvider(CurvePlotHyper)

    integration_method = Setting(0)
    integration_methods = Integrate.INTEGRALS
    value_type = Setting(0)
    attr_value = ContextSetting(None)

    lowlim = Setting(None)
    highlim = Setting(None)
    choose = Setting(None)

    graph_name = "imageplot.plotview"  # defined so that the save button is shown

    class Warning(OWWidget.Warning):
        threshold_error = Msg("Low slider should be less than High")

    class Error(OWWidget.Error):
        image_too_big = Msg("Image for chosen features is too big ({} x {}).")

    class Information(OWWidget.Information):
        not_shown = Msg("Undefined positions: {} data point(s) are not shown.")

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            # delete the saved attr_value to prevent crashes
            try:
                del settings_["context_settings"][0].values["attr_value"]
            except:  # pylint: disable=bare-except
                pass

        # migrate selection
        if version <= 2:
            try:
                current_context = settings_["context_settings"][0]
                selection = getattr(current_context, "selection", None)
                if selection is not None:
                    selection = [(i, 1) for i in np.flatnonzero(np.array(selection))]
                    settings_.setdefault("imageplot", {})["selection_group_saved"] = selection
            except:  # pylint: disable=bare-except
                pass

    @classmethod
    def migrate_context(cls, context, version):
        if version <= 3 and "curveplot" in context.values:
            CurvePlot.migrate_context_sub_feature_color(context.values["curveplot"], version)

    def __init__(self):
        super().__init__()

        dbox = gui.widgetBox(self.controlArea, "Image values")

        rbox = gui.radioButtons(
            dbox, self, "value_type", callback=self._change_integration)

        gui.appendRadioButton(rbox, "From spectra")

        self.box_values_spectra = gui.indentedBox(rbox)

        gui.comboBox(
            self.box_values_spectra, self, "integration_method",
            items=(a.name for a in self.integration_methods),
            callback=self._change_integral_type)
        gui.rubber(self.controlArea)

        gui.appendRadioButton(rbox, "Use feature")

        self.box_values_feature = gui.indentedBox(rbox)

        self.feature_value_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES,
                                               valid_types=DomainModel.PRIMITIVE)
        self.feature_value = gui.comboBox(
            self.box_values_feature, self, "attr_value",
            callback=self.update_feature_value, model=self.feature_value_model)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        self.imageplot = ImagePlot(self)
        self.imageplot.selection_changed.connect(self.output_image_selection)

        self.curveplot = CurvePlotHyper(self, select=SELECTONE)
        self.curveplot.selection_changed.connect(self.redraw_integral_info)
        self.curveplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        splitter.addWidget(self.imageplot)
        splitter.addWidget(self.curveplot)
        self.mainArea.layout().addWidget(splitter)

        self.line1 = MovableVline(position=self.lowlim, label="", report=self.curveplot)
        self.line1.sigMoved.connect(lambda v: setattr(self, "lowlim", v))
        self.line2 = MovableVline(position=self.highlim, label="", report=self.curveplot)
        self.line2.sigMoved.connect(lambda v: setattr(self, "highlim", v))
        self.line3 = MovableVline(position=self.choose, label="", report=self.curveplot)
        self.line3.sigMoved.connect(lambda v: setattr(self, "choose", v))
        for line in [self.line1, self.line2, self.line3]:
            line.sigMoveFinished.connect(self.changed_integral_range)
            self.curveplot.add_marking(line)
            line.hide()

        self.markings_integral = []

        self.data = None
        self.disable_integral_range = False

        self.resize(900, 700)
        self._update_integration_type()

        # prepare interface according to the new context
        self.contextAboutToBeOpened.connect(lambda x: self.init_interface_data(x[0]))

    def init_interface_data(self, data):
        same_domain = (self.data and data and
                       data.domain == self.data.domain)
        if not same_domain:
            self.init_attr_values(data)

    def output_image_selection(self):
        if not self.data:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            self.curveplot.set_data(None)
            return

        indices = np.flatnonzero(self.imageplot.selection_group)

        annotated_data = groups_or_annotated_table(self.data, self.imageplot.selection_group)
        self.Outputs.annotated_data.send(annotated_data)

        selected = self.data[indices]
        self.Outputs.selected_data.send(selected if selected else None)
        if selected:
            self.curveplot.set_data(selected)
        else:
            self.curveplot.set_data(self.data)

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.feature_value_model.set_domain(domain)
        self.attr_value = self.feature_value_model[0] if self.feature_value_model else None

    def redraw_integral_info(self):
        di = {}
        integrate = self.integrate_fn()
        if isinstance(integrate, Integrate) and np.any(self.curveplot.selection_group):
            # curveplot can have a subset of curves on the input> match IDs
            ind = np.flatnonzero(self.curveplot.selection_group)[0]
            dind = self.imageplot.data_ids[self.curveplot.data[ind].id]
            dshow = self.data[dind:dind+1]
            datai = integrate(dshow)
            draw_info = datai.domain.attributes[0].compute_value.draw_info
            di = draw_info(dshow)
        self.refresh_markings(di)

    def refresh_markings(self, di):
        refresh_integral_markings([{"draw": di}], self.markings_integral, self.curveplot)

    def integrate_fn(self):
        if self.value_type == 0:  # integrals
            imethod = self.integration_methods[self.integration_method]

            if imethod != Integrate.PeakAt:
                return Integrate(methods=imethod,
                                 limits=[[self.lowlim, self.highlim]])
            else:
                return Integrate(methods=imethod,
                                 limits=[[self.choose, self.choose]])
        else:
            return lambda data, attr=self.attr_value: \
                data.transform(Domain([data.domain[attr]]))

    def redraw_data(self):
        self.redraw_integral_info()
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

    @Inputs.data
    def set_data(self, data):
        self.closeContext()

        def valid_context(data):
            if data is None:
                return False
            annotation_features = [v for v in data.domain.metas + data.domain.class_vars
                                   if isinstance(v, (DiscreteVariable, ContinuousVariable))]
            return len(annotation_features) >= 1

        if valid_context(data):
            self.openContext(data)
        else:
            # to generate valid interface even if context was not loaded
            self.contextAboutToBeOpened.emit([data])
        self.data = data
        self.imageplot.set_data(data)
        self.curveplot.set_data(data)
        self._init_integral_boundaries()
        self.imageplot.update_view()
        self.output_image_selection()

    def _init_integral_boundaries(self):
        # requires data in curveplot
        self.disable_integral_range = True
        if self.curveplot.data_x is not None and len(self.curveplot.data_x):
            minx = self.curveplot.data_x[0]
            maxx = self.curveplot.data_x[-1]
        else:
            minx = 0.
            maxx = 1.

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

    def save_graph(self):
        self.imageplot.save_graph()

    def onDeleteWidget(self):
        self.curveplot.shutdown()
        self.imageplot.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWHyper).run(Orange.data.Table("iris.tab"))
