import collections.abc
import math
import warnings
from collections import OrderedDict
from xml.sax.saxutils import escape

from AnyQt.QtWidgets import QWidget, QPushButton, \
    QGridLayout, QFormLayout, QAction, QVBoxLayout, QWidgetAction, QSplitter, \
    QToolTip, QGraphicsRectItem, QLabel
from AnyQt.QtGui import QColor, QKeySequence, QPainter, QBrush, QStandardItemModel, \
    QStandardItem, QLinearGradient, QPixmap, QIcon, QPen

from AnyQt.QtCore import Qt, QRectF, QPointF, QSize
from AnyQt.QtTest import QTest

from AnyQt.QtCore import pyqtSignal as Signal

import bottleneck
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import GraphicsWidget
import colorcet
from PIL import Image

import Orange.data
from Orange.preprocess.transformation import Identity
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from Orange.util import OrangeDeprecationWarning
from Orange.widgets.visualize.utils.customizableplot import CommonParameterSetter
from Orange.widgets.widget import OWWidget, Msg, OWComponent, Input
from Orange.widgets import gui
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.widgets.utils import saveplot
from Orange.widgets.utils.concurrent import TaskState, ConcurrentMixin
from Orange.widgets.visualize.utils.plotutils import GraphicsView, PlotItem, AxisItem

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from orangecontrib.spectroscopy.preprocess import Integrate
from orangecontrib.spectroscopy.utils import values_to_linspace, index_values_nan, split_to_size

from orangecontrib.spectroscopy.widgets.owspectra import InteractiveViewBox, \
    MenuFocus, CurvePlot, SELECTONE, SELECTMANY, INDIVIDUAL, AVERAGE, \
    HelpEventDelegate, selection_modifiers, \
    ParameterSetter as SpectraParameterSetter

from orangecontrib.spectroscopy.io.util import VisibleImage, build_spec_table
from orangecontrib.spectroscopy.widgets.gui import MovableVline, lineEditDecimalOrNone,\
    pixels_to_decimals, float_to_str_decimals
from orangecontrib.spectroscopy.widgets.line_geometry import in_polygon, intersect_line_segments
from orangecontrib.spectroscopy.widgets.utils import \
    SelectionGroupMixin, SelectionOutputsMixin

IMAGE_TOO_BIG = 1024*1024*100


class InterruptException(Exception):
    pass


class ImageTooBigException(Exception):
    pass


class UndefinedImageException(Exception):
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
                bs_x, bs_ys = np.asarray(bs_x), np.asarray(bs_ys)
                curve = pg.PlotCurveItem()
                curve.setPen(pg.mkPen(color=QColor(color), **penargs))
                curve.setZValue(10)
                curve.setData(x=bs_x, y=bs_ys[0])
                add_marking(curve)

            elif el[0] == "fill":
                (x1, ys1), (x2, ys2) = el[1]
                bs_x, bs_ys = np.asarray(bs_x), np.asarray(bs_ys)
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
    mn, mx = bottleneck.nanmin(img), bottleneck.nanmax(img)
    if mn == mx or math.isnan(mx) or math.isnan(mn):
        mn = 0
        mx = 255
    return [mn, mx]


class VisibleImageListModel(PyListModel):

    def data(self, index, role=Qt.DisplayRole):
        if self._is_index_valid(index):
            img = self[index.row()]
            if role == Qt.DisplayRole:
                return img.name if isinstance(img, VisibleImage) else img["name"]
        return PyListModel.data(self, index, role)


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

        image = np.atleast_3d(self.image)

        if image.shape[2] == 3:
            # Direct RGB data
            lut = None
        elif isinstance(self.lut, collections.abc.Callable):
            lut = self.lut(self.image)
        else:
            lut = self.lut

        levels = self.levels

        if self.axisOrder == 'col-major':
            image = image.transpose((1, 0, 2)[:image.ndim])

        image_nans = np.isnan(image).all(axis=2)

        if image.shape[2] == 1:
            image = image[:, :, 0]

        argb, alpha = pg.makeARGB(image, lut=lut, levels=levels)  # format is bgra

        argb[image_nans] = (100, 100, 100, 255)  # replace unknown values with a color

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
    ("Jet", {0: pg.colormap.get("jet", source='matplotlib').getLookupTable(nPts=256)}),
    ("Viridis", {0: pg.colormap.get("viridis", source='matplotlib').getLookupTable(nPts=256)}),

    # cyclic
    ("HSV", {0: pg.colormap.get("hsv", source='matplotlib').getLookupTable(nPts=256)}),
]
#r, g, b, c, m, y, k, w
vector_color = [
    ("Black", {0: (0,0,0)}),
    ("White", {0: (255,255,255)}),
    ("Red", {0: (255,0,0)}),
    ("Green", {0: (0,255,0)}),
    ("Blue", {0: (0,0,255)}),
    ("Cyan", {0: (0,255,255)}),
    ("Magenta", {0: (255,0,255)}),
    ("Yellow", {0: (255,255,0)}),
    ("By Feature", {0: ('by feature')})
]

bins = [
    ("1 x 1", {0, (1)}),
    ("2 x 2", {0, (2)}),
    ("3 x 3", {0, (3)}),
    ("4 x 4", {0, (4)}),
    ("5 x 5", {0, (5)}),
    ("6 x 6", {0, (6)}),
    ("7 x 7", {0, (7)}),
    ("8 x 8", {0, (8)}),
    ("9 x 9", {0, (9)}),
    ("10 x 10", {0, (10)}),
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
        colors = np.round(colors).astype(int)
        colors = [QColor(*c) for c in colors]
        item = QStandardItem(name)
        item.setIcon(QIcon(palette_pixmap(colors, iconsize)))
        item.setData(palette, Qt.UserRole)
        model.appendRow([item])
    return model


def vector_color_model(colors):
    model = QStandardItemModel()
    for name, palette in colors:
        item = QStandardItem(name)
        item.setData(palette, Qt.UserRole)
        model.appendRow([item])
    return model


def circular_mean(degs):
    sin = np.nansum(np.sin(np.radians(degs*2)))
    cos = np.nansum(np.cos(np.radians(degs*2)))
    return np.arctan2(sin, cos)/2


class VectorSettingMixin:
    show_vector_plot = Setting(False, schema_only=True)
    vector_angle = ContextSetting(None)
    vector_magnitude = ContextSetting(None)
    vector_color_index = Setting(0)
    vcol_byval_index = Setting(0)
    vcol_byval_feat = ContextSetting(None)
    vector_scale = Setting(1)
    vector_width = Setting(1)
    vector_opacity = Setting(255)
    v_bin = Setting(0)

    def setup_vector_plot_controls(self):

        box = gui.vBox(self)

        self.cb_vector = gui.checkBox(box, self, "show_vector_plot",
                                      label="Show vector plot",
                                      callback=self.enable_vector)

        self.vectorbox = gui.widgetBox(box, box=False)

        self.vector_opts = DomainModel(DomainModel.SEPARATED,
                                       valid_types=DomainModel.PRIMITIVE, placeholder='None')

        self.vector_cbyf_opts = DomainModel(DomainModel.SEPARATED,
                                            valid_types=(ContinuousVariable,), placeholder='None')

        self.vector_col_opts = vector_color_model(vector_color)
        self.vector_pal_opts = color_palette_model(_color_palettes, (QSize(64, 16)))
        self.vector_bin_opts = vector_color_model(bins)

        self.vector_angle = None
        self.vector_magnitude = None
        self.vcol_byval_feat = None
        self.color_opts = vector_color

        gb = create_gridbox(self.vectorbox, box=False)

        v_angle_select = gui.comboBox(None, self, 'vector_angle', searchable=True,
                                      label="Vector Angle", model=self.vector_opts,
                                      contentsLength=10,
                                      callback=self._update_vector_params)
        grid_add_labelled_row(gb, "Angle: ", v_angle_select)

        v_mag_select = gui.comboBox(None, self, 'vector_magnitude', searchable=True,
                                    label="Vector Magnitude", model=self.vector_opts,
                                    contentsLength=10,
                                    callback=self._update_vector_params)
        grid_add_labelled_row(gb, "Magnitude: ", v_mag_select)

        v_bin_select = gui.comboBox(None, self, 'v_bin', label="Pixel Binning",
                                    model=self.vector_bin_opts,
                                    contentsLength=10,
                                    callback=self._update_binsize)
        grid_add_labelled_row(gb, "Binning: ", v_bin_select)

        v_color_select = gui.comboBox(None, self, 'vector_color_index',
                                       label="Vector color", model=self.vector_col_opts,
                                       contentsLength=10,
                                       callback=self._update_vector)
        grid_add_labelled_row(gb, "Color: ", v_color_select)

        v_color_byval = gui.comboBox(None, self, 'vcol_byval_feat',
                                      label="Vector color by Feature",
                                      model=self.vector_cbyf_opts,
                                      contentsLength=10,
                                      callback=self._update_cbyval)
        grid_add_labelled_row(gb, "Feature: ", v_color_byval)

        v_color_byval_select = gui.comboBox(None, self, 'vcol_byval_index',
                                             label="", model=self.vector_pal_opts,
                                             contentsLength=5,
                                             callback=self._update_cbyval)
        v_color_byval_select.setIconSize(QSize(64, 16))
        grid_add_labelled_row(gb, "Palette: ", v_color_byval_select)

        gb = create_gridbox(self.vectorbox, box=False)

        v_scale_slider = gui.hSlider(None, self, 'vector_scale', label="Scale",
                                     minValue=1, maxValue=1000, step=10, createLabel=False,
                                     callback=self._update_vector)
        grid_add_labelled_row(gb, "Scale: ", v_scale_slider)

        v_width_slider = gui.hSlider(None, self, 'vector_width', label="Width",
                                     minValue=1, maxValue=20, step=1, createLabel=False,
                                     callback=self._update_vector)
        grid_add_labelled_row(gb, "Width: ", v_width_slider)

        v_opacity_slider = gui.hSlider(None, self, 'vector_opacity', label="Opacity",
                                       minValue=0, maxValue=255, step=5, createLabel=False,
                                       callback=self._update_vector)
        grid_add_labelled_row(gb, "Opacity: ", v_opacity_slider)

        self.vectorbox.setVisible(self.show_vector_plot)
        self.update_vector_plot_interface()
        
        return box

    def update_vector_plot_interface(self):
        vector_params = ['vector_angle', 'vector_magnitude', 'vector_color_index',
                         'vector_scale', 'vector_width', 'vector_opacity', 'v_bin']
        for i in vector_params:
            getattr(self.controls, i).setEnabled(self.show_vector_plot)

        if self.vector_color_index == 8 and self.show_vector_plot:
            self.controls.vcol_byval_index.setEnabled(True)
            self.controls.vcol_byval_feat.setEnabled(True)
            if self.vcol_byval_feat is not None and self.show_vector_plot:
                self.vect_legend.setVisible(True)
                self.vect_legend.adapt_to_size()
            else:
                self.vect_legend.setVisible(False)
        else:
            self.controls.vcol_byval_index.setEnabled(False)
            self.controls.vcol_byval_feat.setEnabled(False)
            self.vect_legend.setVisible(False)

    def enable_vector(self):
        self.vectorbox.setVisible(self.show_vector_plot)
        self._update_vector()

    def _update_vector(self):
        self.update_vector_plot_interface()
        self.update_vectors()

    def _update_vector_params(self):
        self.update_binsize()
        self._update_vector()

    def _update_cbyval(self):
        self.cols = None
        self._update_vector()

    def _update_binsize(self):
        self.v_bin_change = 1
        self.cols = None
        self.update_binsize()

    def init_vector_plot(self, data):
        domain = data.domain if data is not None else None
        self.vector_opts.set_domain(domain)
        self.vector_cbyf_opts.set_domain(domain)

        # initialize values so that the combo boxes are not in invalid states
        if self.vector_opts:
            # TODO here we could instead set good default values if available
            self.vector_magnitude = self.vector_angle = self.vcol_byval_feat = None
        else:
            self.vector_magnitude = self.vector_angle = self.vcol_byval_feat = None


class VectorMixin:

    def __init__(self):
        self.a = None
        self.th = None
        self.cols = None
        self.new_xs = None
        self.new_ys = None
        self.v_bin_change = 0

        ci = self.plotview.centralWidget
        self.vector_plot = VectorPlot()
        self.vector_plot.hide()
        self.plot.addItem(self.vector_plot)
        self.vect_legend = ImageColorLegend()
        self.vect_legend.setVisible(False)
        ci.addItem(self.vect_legend)

    def update_vectors(self):
        v = self.get_vector_data()
        if self.lsx is None:  # image is not shown or is being computed
            v = None
        if v is None:
            self.vector_plot.hide()
        else:
            valid = self.data_valid_positions
            lsx, lsy = self.lsx, self.lsy
            xindex, yindex = self.xindex, self.yindex
            scale = self.vector_scale
            w = self.vector_width
            th = np.asarray(v[:,0], dtype=float)
            v_mag = np.asarray(v[:,1], dtype=float)
            wy = _shift(lsy)*2
            wx = _shift(lsx)*2
            if self.v_bin == 0:
                y = np.linspace(*lsy)[yindex[valid]]
                x = np.linspace(*lsx)[xindex[valid]]
                amp = v_mag / max(v_mag) * (scale/100)
                dispx = amp*wx/2*np.cos(np.radians(th))
                dispy = amp*wy/2*np.sin(np.radians(th))
                xcurve = np.empty((dispx.shape[0]*2))
                ycurve = np.empty((dispy.shape[0]*2))
                xcurve[0::2], xcurve[1::2] = x - dispx, x + dispx
                ycurve[0::2], ycurve[1::2] = y - dispy, y + dispy
                vcols = self.get_vector_color(v[:,2])
                v_params = [xcurve, ycurve, w, vcols]
                self.vector_plot.setData(v_params)
            else:
                if self.a is None:
                    self.update_binsize()
                amp = self.a / max(self.a) * (scale/100)
                dispx = amp*wx/2*np.cos(self.th)
                dispy = amp*wy/2*np.sin(self.th)
                xcurve = np.empty((dispx.shape[0]*2))
                ycurve = np.empty((dispy.shape[0]*2))
                xcurve[0::2], xcurve[1::2] = self.new_xs - dispx, self.new_xs + dispx
                ycurve[0::2], ycurve[1::2] = self.new_ys - dispy, self.new_ys + dispy
                vcols = self.get_vector_color(v[:,2])
                v_params = [xcurve, ycurve, w, vcols]
                self.vector_plot.setData(v_params)
            self.vector_plot.show()
            if self.vector_color_index == 8 and \
                self.vcol_byval_feat is not None:
                self.update_vect_legend()

    def update_vect_legend(self):#feat
        if self.v_bin != 0:
            feat = self.cols
        else:
            feat = self.data.get_column(self.vcol_byval_feat)
        fmin, fmax = np.min(feat), np.max(feat)
        self.vect_legend.set_range(fmin, fmax)
        self.vect_legend.set_colors(_color_palettes[self.vcol_byval_index][1][0])
        self.vect_legend.setVisible(True)
        self.vect_legend.adapt_to_size()

    def get_vector_data(self):
        if self.show_vector_plot is False or self.data is None:
            return None

        ang = self.vector_angle
        mag = self.vector_magnitude
        col = self.vcol_byval_feat
        angs = self.data.get_column(ang) if ang else np.full(len(self.data), 0)
        mags = self.data.get_column(mag) if mag else np.full(len(self.data), 1)
        cols = self.data.get_column(col) if col else np.full(len(self.data), None)

        return np.vstack([angs, mags, cols]).T

    def get_vector_color(self, feat):
        if self.vector_color_index == 8:
            if feat[0] is None: # a feat has not been selected yet
                return vector_color[0][1][0] + (self.vector_opacity,)
            else:
                if self.v_bin != 0:
                    if self.cols is None:
                        self.update_binsize()
                    feat = self.cols
                fmin, fmax = np.min(feat), np.max(feat)
                if fmin == fmax:
                    # put a warning here?
                    return vector_color[0][1][0] + (self.vector_opacity,)
                feat_idxs = np.asarray(((feat-fmin)/(fmax-fmin))*255, dtype=int)
                col_vals = np.asarray(_color_palettes[self.vcol_byval_index][1][0][feat_idxs],
                                      dtype=int)
                out = [np.hstack((np.expand_dims(feat_idxs, 1), col_vals)),
                       self.vector_opacity]
                return out
        else:
            return vector_color[self.vector_color_index][1][0] + (self.vector_opacity,)

    def update_binsize(self):
        self.parent.Warning.bin_size_error.clear()
        if self.v_bin == 0:
            self.v_bin_change = 0
            self.update_vectors()
        else:
            v = self.get_vector_data()
            valid = self.data_valid_positions
            lsx, lsy = self.lsx, self.lsy
            xindex, yindex = self.xindex, self.yindex
            if lsx is None:
                v = None
            if v is None:
                self.v_bin_change = 0
                self.vector_plot.hide()
            else:
                th = np.asarray(v[:,0], dtype=float)
                v_mag = np.asarray(v[:,1], dtype=float)
                col = np.asarray(v[:,2], dtype=float)
                y = np.linspace(*lsy)[yindex]
                x = np.linspace(*lsx)[xindex]
                df = pd.DataFrame(
                    [x, y, np.asarray([1 if i else 0 for i in valid]),v_mag, th, col],
                    index = ['x', 'y', 'valid', 'v_mag', 'th', 'cols']).T

                v_df = df.pivot_table(values = 'valid', columns = 'x', index = 'y', fill_value = 0)
                a_df = df.pivot_table(values = 'v_mag', columns = 'x', index = 'y')
                th_df = df.pivot_table(values = 'th', columns = 'x', index = 'y')
                col_df = df.pivot_table(values = 'cols', columns = 'x', index = 'y')
                bin_sz = self.v_bin+1
                if bin_sz > v_df.shape[0] or bin_sz > v_df.shape[1]:
                    bin_sz = v_df.shape[0] if bin_sz > v_df.shape[0] else v_df.shape[1]
                    self.parent.Warning.bin_size_error(bin_sz, bin_sz)
                x_mod, y_mod = v_df.shape[1] % bin_sz, v_df.shape[0] % bin_sz
                st_x_idx = int(np.floor(x_mod/2))
                st_y_idx = int(np.floor(y_mod/2))

                nvalid = np.zeros((int((v_df.shape[0]-y_mod)/bin_sz),
                                        int((v_df.shape[1]-x_mod)/bin_sz)))
                a = np.zeros((int((v_df.shape[0]-y_mod)/bin_sz),
                                        int((v_df.shape[1]-x_mod)/bin_sz)))
                th = np.zeros((int((v_df.shape[0]-y_mod)/bin_sz),
                                        int((v_df.shape[1]-x_mod)/bin_sz)))
                cols = np.zeros((int((v_df.shape[0]-y_mod)/bin_sz),
                                        int((v_df.shape[1]-x_mod)/bin_sz)))
                columns = v_df.columns
                rows = v_df.index
                new_xs, new_ys = [], []
                for i in range(st_y_idx, v_df.shape[0]-y_mod, bin_sz):
                    for j in range(st_x_idx, v_df.shape[1]-x_mod, bin_sz):
                        nvalid[int(i/bin_sz),int(j/bin_sz)] = \
                            np.nanmean(v_df.iloc[i:i+bin_sz,j:j+bin_sz].to_numpy())
                        a[int(i/bin_sz),int(j/bin_sz)] = \
                            np.nanmean(a_df.iloc[i:i+bin_sz,j:j+bin_sz].to_numpy())
                        th[int(i/bin_sz),int(j/bin_sz)] = \
                            circular_mean(th_df.iloc[i:i+bin_sz,j:j+bin_sz].to_numpy())
                        cols[int(i/bin_sz),int(j/bin_sz)] = \
                            np.nanmean(col_df.iloc[i:i+bin_sz,j:j+bin_sz].to_numpy())
                        new_xs.append(np.sum(columns[j:j+bin_sz])/bin_sz)
                        new_ys.append(np.sum(rows[i:i+bin_sz])/bin_sz)
                nvalid = nvalid.flatten() > 0 & ~np.isnan(nvalid.flatten())
                self.a = a.flatten()[nvalid]
                self.th = th.flatten()[nvalid]
                self.cols = cols.flatten()[nvalid]
                self.new_xs = np.asarray(new_xs)[nvalid]
                self.new_ys = np.asarray(new_ys)[nvalid]
                if self.v_bin_change == 1:
                    self.v_bin_change = 0
                    self.update_vectors()


class AxesSettingsMixin:

    def __init__(self):
        self.xy_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES,
                                    valid_types=DomainModel.PRIMITIVE)

    def setup_axes_settings_box(self):
        box = gui.vBox(self)

        common_options = {
            "labelWidth": 50,
            "orientation": Qt.Horizontal,
            "sendSelectedValue": True
        }

        cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        box.setFocusProxy(cb_attr_x)
        return box


class ImageColorSettingMixin:
    threshold_low = Setting(0.0, schema_only=True)
    threshold_high = Setting(1.0, schema_only=True)
    level_low = Setting(None, schema_only=True)
    level_high = Setting(None, schema_only=True)
    show_legend = Setting(True)
    palette_index = Setting(0)

    def __init__(self):
        self.fixed_levels = None  # fixed level settings for categoric data

    def setup_color_settings_box(self):
        box = gui.vBox(self)
        box.setContentsMargins(0, 0, 0, 5)
        self.color_cb = gui.comboBox(box, self, "palette_index", label="Color:",
                                     labelWidth=50, orientation=Qt.Horizontal,
                                     contentsLength=10)
        self.color_cb.setIconSize(QSize(64, 16))
        palettes = _color_palettes
        model = color_palette_model(palettes, self.color_cb.iconSize())
        model.setParent(self)
        self.color_cb.setModel(model)
        self.palette_index = min(self.palette_index, len(palettes) - 1)
        self.color_cb.activated.connect(self.update_color_schema)

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
        self._level_low_le.sizeHintFactor = 0.5
        form.addRow("Low limit:", self._level_low_le)

        self._level_high_le = lineEditDecimalOrNone(self, self, "level_high", callback=limit_changed)
        self._level_high_le.validator().setDefault(1)
        self._level_high_le.sizeHintFactor = 0.5
        form.addRow("High limit:", self._level_high_le)

        self._threshold_low_slider = lowslider = gui.hSlider(
            box, self, "threshold_low", minValue=0.0, maxValue=1.0,
            step=0.05, intOnly=False,
            createLabel=False, callback=self.update_levels)
        self._threshold_high_slider = highslider = gui.hSlider(
            box, self, "threshold_high", minValue=0.0, maxValue=1.0,
            step=0.05, intOnly=False,
            createLabel=False, callback=self.update_levels)

        form.addRow("Low:", lowslider)
        form.addRow("High:", highslider)
        box.layout().addLayout(form)

        self.update_legend_visible()

        return box

    def update_legend_visible(self):
        if self.fixed_levels is not None or self.parent.value_type == 2:
            self.legend.setVisible(False)
        else:
            self.legend.setVisible(self.show_legend)

    def update_levels(self):
        if not self.data:
            return

        if self.fixed_levels is not None:
            levels = list(self.fixed_levels)
        elif self.img.image is not None and self.img.image.ndim == 2:
            levels = get_levels(self.img.image)
        elif self.img.image is not None and self.img.image.shape[2] == 1:
            levels = get_levels(self.img.image[:, :, 0])
        elif self.img.image is not None and self.img.image.shape[2] == 3:
            return
        else:
            levels = [0, 255]

        prec = pixels_to_decimals((levels[1] - levels[0])/1000)

        rounded_levels = [float_to_str_decimals(levels[0], prec),
                          float_to_str_decimals(levels[1], prec)]

        self._level_low_le.validator().setDefault(rounded_levels[0])
        self._level_high_le.validator().setDefault(rounded_levels[1])

        self._level_low_le.setPlaceholderText(rounded_levels[0])
        self._level_high_le.setPlaceholderText(rounded_levels[1])

        enabled_level_settings = self.fixed_levels is None
        self._level_low_le.setEnabled(enabled_level_settings)
        self._level_high_le.setEnabled(enabled_level_settings)
        self._threshold_low_slider.setEnabled(enabled_level_settings)
        self._threshold_high_slider.setEnabled(enabled_level_settings)

        if self.fixed_levels is not None:
            self.img.setLevels(self.fixed_levels)
            return

        if not self.threshold_low < self.threshold_high:
            # TODO this belongs here, not in the parent
            self.parent.Warning.threshold_error()
            return
        else:
            self.parent.Warning.threshold_error.clear()

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
            dat = self.parent.data.domain[self.parent.attr_value]
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


class ImageRGBSettingMixin:
    red_level_low = Setting(None, schema_only=True)
    red_level_high = Setting(None, schema_only=True)
    green_level_low = Setting(None, schema_only=True)
    green_level_high = Setting(None, schema_only=True)
    blue_level_low = Setting(None, schema_only=True)
    blue_level_high = Setting(None, schema_only=True)

    def setup_rgb_settings_box(self):
        box = gui.vBox(self)

        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )

        self._red_level_low_le = lineEditDecimalOrNone(self, self, "red_level_low", callback=self.update_rgb_levels)
        self._red_level_low_le.validator().setDefault(0)
        form.addRow("Red Low limit:", self._red_level_low_le)

        self._red_level_high_le = lineEditDecimalOrNone(self, self, "red_level_high", callback=self.update_rgb_levels)
        self._red_level_high_le.validator().setDefault(1)
        form.addRow("Red High limit:", self._red_level_high_le)

        self._green_level_low_le = lineEditDecimalOrNone(self, self, "green_level_low", callback=self.update_rgb_levels)
        self._green_level_low_le.validator().setDefault(0)
        form.addRow("Green Low limit:", self._green_level_low_le)

        self._green_level_high_le = lineEditDecimalOrNone(self, self, "green_level_high", callback=self.update_rgb_levels)
        self._green_level_high_le.validator().setDefault(1)
        form.addRow("Green High limit:", self._green_level_high_le)

        self._blue_level_low_le = lineEditDecimalOrNone(self, self, "blue_level_low", callback=self.update_rgb_levels)
        self._blue_level_low_le.validator().setDefault(0)
        form.addRow("Blue Low limit:", self._blue_level_low_le)

        self._blue_level_high_le = lineEditDecimalOrNone(self, self, "blue_level_high", callback=self.update_rgb_levels)
        self._blue_level_high_le.validator().setDefault(1)
        form.addRow("Blue High limit:", self._blue_level_high_le)

        box.layout().addLayout(form)
        return box

    def update_rgb_levels(self):
        if not self.data:
            return

        if self.img.image is not None and self.img.image.shape[2] == 3:
            levels = [get_levels(self.img.image[:, :, i]) for i in range(self.img.image.shape[2])]
        else:
            return

        rgb_le = [
            [self._red_level_low_le, self._red_level_high_le],
            [self._green_level_low_le, self._green_level_high_le],
            [self._blue_level_low_le, self._blue_level_high_le]
        ]

        for i, (low_le, high_le) in enumerate(rgb_le):
            prec = pixels_to_decimals((levels[i][1] - levels[i][0]) / 1000)
            rounded_levels = [float_to_str_decimals(levels[i][0], prec),
                              float_to_str_decimals(levels[i][1], prec)]

            low_le.validator().setDefault(rounded_levels[0])
            high_le.validator().setDefault(rounded_levels[1])

            low_le.setPlaceholderText(rounded_levels[0])
            high_le.setPlaceholderText(rounded_levels[1])

        rll = float(self.red_level_low) if self.red_level_low is not None else levels[0][0]
        rlh = float(self.red_level_high) if self.red_level_high is not None else levels[0][1]
        gll = float(self.green_level_low) if self.green_level_low is not None else levels[1][0]
        glh = float(self.green_level_high) if self.green_level_high is not None else levels[1][1]
        bll = float(self.blue_level_low) if self.blue_level_low is not None else levels[2][0]
        blh = float(self.blue_level_high) if self.blue_level_high is not None else levels[2][1]

        new_levels = [[rll, rlh], [gll, glh], [bll, blh]]
        self.img.setLevels(new_levels)


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


class ImageSelectionMixin:

    def __init__(self):
        self.selection_distances = None

    def add_selection_actions(self, menu):

        select_square = QAction(
            "Select (square)", self, triggered=self.plot.vb.set_mode_select_square,
        )
        select_square.setShortcuts([Qt.Key_S])
        select_square.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(select_square)
        if menu:
            menu.addAction(select_square)

        select_polygon = QAction(
            "Select (polygon)", self, triggered=self.plot.vb.set_mode_select_polygon,
        )
        select_polygon.setShortcuts([Qt.Key_P])
        select_polygon.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(select_polygon)
        if menu:
            menu.addAction(select_polygon)

        line = QAction(
            "Trace line", self, triggered=self.plot.vb.set_mode_select,
        )
        line.setShortcuts([Qt.Key_L])
        line.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(line)
        if menu:
            menu.addAction(line)

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

    def select_line(self, p1, p2):
        # hijacking existing selection functions to do stuff
        p1x, p1y = p1.x(), p1.y()
        p2x, p2y = p2.x(), p2.y()
        if self.data and self.lsx and self.lsy:
            # Pixel selection works so that a pixel is selected
            # if the drawn line crosses any of its edges.
            # An alternative that would ensure that only one pixel in row/column
            # was selected is the Bresenham's line algorithm.
            shiftx = _shift(self.lsx)
            shifty = _shift(self.lsy)
            points_edges = [self.data_points + [[shiftx, shifty]],
                            self.data_points + [[-shiftx, shifty]],
                            self.data_points + [[-shiftx, -shifty]],
                            self.data_points + [[shiftx, -shifty]]]
            sel = None
            for i in range(4):
                res = intersect_line_segments(points_edges[i - 1][:, 0], points_edges[i - 1][:, 1],
                                              points_edges[i][:, 0], points_edges[i][:, 1],
                                              p1x, p1y, p2x, p2y)
                if sel is None:
                    sel = res
                else:
                    sel |= res

            distances = np.full(self.data_points.shape[0], np.nan)
            distances[sel] = np.linalg.norm(self.data_points[sel] - [[p1x, p1y]], axis=1)

            modifiers = (False, False, False)  # enforce a new selection
            self.make_selection(sel, distances=distances, modifiers=modifiers)


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
        self.axis = AxisItem('right', parent=self)
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
            self.colors = np.round(self.colors).astype(int)
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


class ImageParameterSetter(CommonParameterSetter):
    IMAGE_ANNOT_BOX = "Image annotations"
    BKG_CBAR = "Colorbar"
    VECT_CBAR = "Vector Colorbar"

    def update_cbar_label(self, **settings):
        self.colorbar.setLabel(settings[self.TITLE_LABEL])
        self.colorbar.resizeEvent(None)

    def update_vcbar_label(self, **settings):
        self.vcolorbar.setLabel(settings[self.TITLE_LABEL])
        self.vcolorbar.resizeEvent(None)

    def __init__(self, master):
        super().__init__()
        self.master = master

    def update_setters(self):
        self.initial_settings = {
            self.IMAGE_ANNOT_BOX: {
                self.TITLE_LABEL: {self.TITLE_LABEL: ("", "")},
                self.X_AXIS_LABEL: {self.TITLE_LABEL: ("", "")},
                self.Y_AXIS_LABEL: {self.TITLE_LABEL: ("", "")},
                self.BKG_CBAR: {self.TITLE_LABEL: ("", "")},
                self.VECT_CBAR: {self.TITLE_LABEL: ("", "")},
            },
        }

        self._setters[self.IMAGE_ANNOT_BOX] = self._setters[self.ANNOT_BOX]
        self._setters[self.IMAGE_ANNOT_BOX].update({
            self.BKG_CBAR: self.update_cbar_label,
            self.VECT_CBAR: self.update_vcbar_label,
        })

    @property
    def title_item(self):
        return self.master.plot.titleLabel

    @property
    def axis_items(self):
        return [value["item"] for value in self.master.plot.axes.values()] \
               + [self.master.legend.axis] + [self.master.vect_legend.axis]

    @property
    def getAxis(self):
        return self.master.plot.getAxis

    @property
    def legend_items(self):
        return []

    @property
    def colorbar(self):
        return self.master.legend.axis

    @property
    def vcolorbar(self):
        return self.master.vect_legend.axis


class VectorPlot(pg.GraphicsObject):

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.params = None

        self._maxSpotPxWidth = 0
        self._boundingRect = None

    def setData(self, params):
        self._maxSpotPxWidth = 0
        self._boundingRect = None

        self.params = params
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.update()

    def viewTransformChanged(self):
        self.prepareGeometryChange()

    def paint(self, p, option, widget):
        if self.params is not None:
            if isinstance(self.params[3], tuple):
                path = pg.arrayToQPath(self.params[0], self.params[1],
                                       connect = 'pairs', finiteCheck=False)
                pen = QPen(QBrush(QColor(*self.params[3])), self.params[2])
                pen.setCosmetic(True)
                p.setPen(pen)
                p.drawPath(path)
            elif isinstance(self.params[3], list):
                pen = QPen(QBrush(QColor()), self.params[2])
                pen.setCosmetic(True)
                unique_cols = np.unique(self.params[3][0], return_index=True, axis=0)
                irgbx2 = np.hstack((self.params[3][0],
                                    self.params[3][0])).reshape(self.params[3][0].shape[0]*2, 4)
                for i in unique_cols[0]:
                    path = pg.arrayToQPath(self.params[0][np.where(irgbx2[:,0] == i[0])],
                                        self.params[1][np.where(irgbx2[:,0] == i[0])],
                                        connect = 'pairs', finiteCheck=False)
                    pen.setColor(QColor(*i[1:], self.params[3][1]))
                    p.setPen(pen)
                    p.drawPath(path)

    # These functions are the same as pg.plotcurveitem with small adaptations
    def pixelPadding(self):
        self._maxSpotPxWidth = self.params[2]*0.7072
        return self._maxSpotPxWidth

    def boundingRect(self):
        if self.params is None:
            return QRectF()
        elif self._boundingRect is None and self.params is not None:
            (xmn, xmx) = (np.nanmin(self.params[0]), np.nanmax(self.params[0]))
            (ymn, ymx) = (np.nanmin(self.params[1]), np.nanmax(self.params[1]))
            if xmn is None or xmx is None:
                return QRectF()
            if ymn is None or ymx is None:
                return QRectF()

            px = py = 0.0
            pxPad = self.pixelPadding()
            if pxPad > 0:
                # determine length of pixel in local x, y directions
                px, py = self.pixelVectors()
                try:
                    px = 0 if px is None else px.length()
                except OverflowError:
                    px = 0
                try:
                    py = 0 if py is None else py.length()
                except OverflowError:
                    py = 0

                # return bounds expanded by pixel size
                px *= pxPad
                py *= pxPad
            #px += self._maxSpotWidth * 0.5
            #py += self._maxSpotWidth * 0.5
            self._boundingRect = QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)

        return self._boundingRect


class BasicImagePlot(QWidget, OWComponent, SelectionGroupMixin,
                    AxesSettingsMixin, ImageSelectionMixin,
                    ImageColorSettingMixin, ImageRGBSettingMixin,
                    ImageZoomMixin, ConcurrentMixin):

    gamma = Setting(0)

    selection_changed = Signal()
    image_updated = Signal()

    def __init__(self, parent):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)
        SelectionGroupMixin.__init__(self)
        AxesSettingsMixin.__init__(self)
        ImageSelectionMixin.__init__(self)
        ImageColorSettingMixin.__init__(self)
        ImageZoomMixin.__init__(self)
        ConcurrentMixin.__init__(self)
        self.parent = parent

        self.parameter_setter = ImageParameterSetter(self)

        self.selection_type = SELECTMANY
        self.saving_enabled = True
        self.selection_enabled = True
        self.viewtype = INDIVIDUAL  # required bt InteractiveViewBox
        self.highlighted = None
        self.data_points = None
        self.data_values = None
        self.data_imagepixels = None
        self.data_valid_positions = None
        self.xindex = None
        self.yindex = None

        self.plotview = GraphicsView()
        ci = pg.GraphicsLayout()
        self.plot = PlotItem(viewBox=InteractiveViewBox(self))
        self.plot.buttonsHidden = True
        self.plotview.setCentralItem(ci)
        ci.addItem(self.plot)

        self.legend = ImageColorLegend()
        ci.addItem(self.legend)

        self.plot.scene().installEventFilter(
            HelpEventDelegate(self.help_event, self))

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)

        self.img = ImageItemNan()
        self.img.setOpts(axisOrder='row-major')
        self.plot.addItem(self.img)
        self.vis_img = pg.ImageItem()
        self.vis_img.setOpts(axisOrder='row-major')
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

        self.add_zoom_actions(view_menu)
        self.add_selection_actions(view_menu)

        if self.saving_enabled:
            save_graph = QAction(
                "Save graph", self, triggered=self.save_graph,
            )
            save_graph.setShortcuts([QKeySequence(Qt.ControlModifier | Qt.Key_I)])
            self.addAction(save_graph)
            view_menu.addAction(save_graph)
            save_graph.setShortcutVisibleInContextMenu(True)

        choose_xy = QWidgetAction(self)
        box = gui.vBox(self)
        box.setContentsMargins(10, 0, 10, 0)
        box.setFocusPolicy(Qt.TabFocus)

        self.axes_settings_box = self.setup_axes_settings_box()
        self.color_settings_box = self.setup_color_settings_box()
        self.rgb_settings_box = self.setup_rgb_settings_box()

        box.layout().addWidget(self.axes_settings_box)
        box.layout().addWidget(self.color_settings_box)
        box.layout().addWidget(self.rgb_settings_box)

        choose_xy.setDefaultWidget(box)
        view_menu.addAction(choose_xy)

        self.lsx = None  # info about the X axis
        self.lsy = None  # info about the Y axis

        self.data = None
        self.data_ids = {}

    def init_interface_data(self, data):
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
        self.selection_distances = None

    def refresh_img_selection(self):
        if self.lsx is None or self.lsy is None:
            return
        selected_px = np.zeros((self.lsy[2], self.lsx[2]), dtype=np.uint8)
        selected_px[self.data_imagepixels[self.data_valid_positions, 0],
                    self.data_imagepixels[self.data_valid_positions, 1]] = \
            self.selection_group[self.data_valid_positions]
        self.img.setSelection(selected_px)

    def make_selection(self, selected, distances=None, modifiers=None):
        """Add selected indices to the selection."""
        add_to_group, add_group, remove = \
            selection_modifiers() if modifiers is None else modifiers
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
        self.selection_distances = distances  # these are not going to be saved for now
        self.prepare_settings_for_saving()
        self.selection_changed.emit()

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
        self.xindex = None
        self.yindex = None

        if isinstance(self, VectorMixin):
            self.update_binsize()
            self.update_vectors()  # clears the vector plot

        self.start(self.compute_image, self.data, self.attr_x, self.attr_y,
                    self.parent.image_values(),
                    self.parent.image_values_fixed_levels())

    def set_visible_image(self, img: np.ndarray, rect: QRectF):
        self.vis_img.setImage(img)
        self.vis_img.setRect(rect)

    def show_visible_image(self):
        if self.vis_img not in self.plot.items:
            self.plot.addItem(self.vis_img)

    def hide_visible_image(self):
        self.plot.removeItem(self.vis_img)

    def set_visible_image_opacity(self, opacity: int):
        """Opacity is an alpha channel intensity integer from 0 to 255"""
        self.vis_img.setOpacity(opacity / 255)

    def set_visible_image_comp_mode(self, comp_mode: QPainter.CompositionMode):
        self.vis_img.setCompositionMode(comp_mode)

    @staticmethod
    def compute_image(data: Orange.data.Table, attr_x, attr_y,
                      image_values, image_values_fixed_levels,
                      state: TaskState):

        if data is None or attr_x is None or attr_y is None:
            raise UndefinedImageException

        def progress_interrupt(i: float):
            if state.is_interruption_requested():
                raise InterruptException

        class Result():
            pass
        res = Result()

        progress_interrupt(0)

        res.coorx = data.get_column(attr_x.name)
        res.coory = data.get_column(attr_y.name)
        res.data_points = np.hstack([res.coorx.reshape(-1, 1), res.coory.reshape(-1, 1)])
        res.lsx = lsx = values_to_linspace(res.coorx)
        res.lsy = lsy = values_to_linspace(res.coory)
        res.image_values_fixed_levels = image_values_fixed_levels
        progress_interrupt(0)

        if lsx is not None and lsy is not None \
                and lsx[-1] * lsy[-1] > IMAGE_TOO_BIG:
            raise ImageTooBigException((lsx[-1], lsy[-1]))

        ims = image_values(data[:1]).X
        d = np.full((data.X.shape[0], ims.shape[1]), float("nan"))
        res.d = d

        step = 100000 if len(data) > 1e6 else 10000

        if lsx is not None and lsy is not None:
            # the code below does this, but part-wise:
            # d = image_values(data).X[:, 0]
            for slice in split_to_size(len(data), step):
                part = image_values(data[slice]).X
                d[slice, :] = part
                progress_interrupt(0)
                state.set_partial_result(res)

        progress_interrupt(0)

        return res

    def draw(self, res, finished=False):
        d = res.d
        lsx, lsy = res.lsx, res.lsy

        self.fixed_levels = res.image_values_fixed_levels
        if finished:
            self.lsx, self.lsy = lsx, lsy
            self.data_points = res.data_points

        xindex, xnan = index_values_nan(res.coorx, lsx)
        yindex, ynan = index_values_nan(res.coory, lsy)
        valid = np.logical_not(np.logical_or(xnan, ynan))
        invalid_positions = len(d) - np.sum(valid)

        if finished:
            self.data_valid_positions = valid
            if invalid_positions:
                self.parent.Information.not_shown(invalid_positions)

        if lsx is not None and lsy is not None:
            imdata = np.ones((lsy[2], lsx[2], d.shape[1])) * float("nan")
            imdata[yindex[valid], xindex[valid]] = d[valid]

            self.data_values = d
            self.data_imagepixels = np.vstack((yindex, xindex)).T
            self.img.setImage(imdata, autoLevels=False)
            self.update_levels()
            self.update_rgb_levels()
            self.update_color_schema()
            self.update_legend_visible()

            # indices need to be saved to quickly draw vectors
            self.yindex = yindex
            self.xindex = xindex

            if isinstance(self, VectorMixin):
                self.update_binsize()
                self.update_vectors()

            # shift centres of the pixels so that the axes are useful
            shiftx = _shift(lsx)
            shifty = _shift(lsy)
            left = lsx[0] - shiftx
            bottom = lsy[0] - shifty
            width = (lsx[1]-lsx[0]) + 2*shiftx
            height = (lsy[1]-lsy[0]) + 2*shifty
            self.img.setRect(QRectF(left, bottom, width, height))

        if finished:
            self.refresh_img_selection()
            self.image_updated.emit()

    def on_done(self, res):
        self.draw(res, finished=True)

    def on_partial_result(self, res):
        self.draw(res)

    def on_exception(self, ex: Exception):
        if isinstance(ex, InterruptException):
            return

        if isinstance(ex, ImageTooBigException):
            self.parent.Error.image_too_big(ex.args[0][0], ex.args[0][1])
            self.image_updated.emit()
        elif isinstance(ex, UndefinedImageException):
            self.image_updated.emit()
        else:
            raise ex


class ImagePlot(BasicImagePlot,
                VectorSettingMixin, VectorMixin):

    attr_x = ContextSetting(None, exclude_attributes=True)
    attr_y = ContextSetting(None, exclude_attributes=True)

    def __init__(self, parent):
        BasicImagePlot.__init__(self, parent)
        VectorSettingMixin.__init__(self)
        VectorMixin.__init__(self)


class CurvePlotHyper(CurvePlot):
    viewtype = Setting(AVERAGE)  # average view by default


def create_gridbox(widget, box=False, add_space=False):
    grid = QGridLayout()
    grid.setColumnMinimumWidth(0, 50)
    grid.setColumnStretch(1, 1)
    b = gui.widgetBox(widget, box=box, orientation=grid)
    if not box:
        if add_space:
            b.setContentsMargins(8, 4, 8, 4)
        else:
            b.setContentsMargins(0, 0, 0, 0)
    # This must come after calling widgetBox, since widgetBox overrides it
    grid.setVerticalSpacing(8)
    return b


def grid_add_labelled_row(grid, label, widget):
    if not isinstance(grid, QGridLayout):
        grid = grid.layout()
    row = grid.rowCount()
    grid.addWidget(QLabel(label), row, 0)
    grid.addWidget(widget, row, 1)


class OWHyper(OWWidget, SelectionOutputsMixin):
    name = "HyperSpectra"

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs(SelectionOutputsMixin.Outputs):
        pass

    icon = "icons/hyper.svg"
    priority = 20
    replaces = ["orangecontrib.infrared.widgets.owhyper.OWHyper"]
    keywords = ["image", "spectral", "chemical", "imaging"]

    settings_version = 8
    settingsHandler = DomainContextHandler()

    imageplot = SettingProvider(ImagePlot)
    curveplot = SettingProvider(CurvePlotHyper)

    integration_method = Setting(4)  # Closest value
    integration_methods = Integrate.INTEGRALS
    value_type = Setting(0)
    attr_value = ContextSetting(None)
    rgb_red_value = ContextSetting(None)
    rgb_green_value = ContextSetting(None)
    rgb_blue_value = ContextSetting(None)

    show_visible_image = Setting(False)
    visible_image_name = Setting(None)
    visible_image_composition = Setting('Normal')
    visible_image_opacity = Setting(120)

    lowlim = Setting(None)
    highlim = Setting(None)
    choose = Setting(None)
    lowlimb = Setting(None)
    highlimb = Setting(None)

    visual_settings = Setting({}, schema_only=True)

    graph_name = "imageplot.plotview"  # defined so that the save button is shown

    class Warning(OWWidget.Warning):
        threshold_error = Msg("Low slider should be less than High")
        bin_size_error = Msg("Selected bin size larger than image size, bin size {} x {} used")

    class Error(OWWidget.Error):
        image_too_big = Msg("Image for chosen features is too big ({} x {}).")

    class Information(SelectionOutputsMixin.Information):
        not_shown = Msg("Undefined positions: {} data point(s) are not shown.")
        view_locked = Msg("Axes are locked in the visual settings dialog.")

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

        if version < 6:
            settings_["compat_no_group"] = True

        if version < 7:
            from orangecontrib.spectroscopy.widgets.owspectra import OWSpectra
            OWSpectra.migrate_to_visual_settings(settings_)

    @classmethod
    def migrate_context(cls, context, version):
        if version <= 3 and "curveplot" in context.values:
            CurvePlot.migrate_context_sub_feature_color(context.values["curveplot"], version)

    def __init__(self):
        super().__init__()
        SelectionOutputsMixin.__init__(self)

        iabox = gui.widgetBox(self.controlArea, "Image axes")

        dbox = gui.widgetBox(self.controlArea, "Image values")

        icbox = gui.widgetBox(self.controlArea, "Image colors")
        
        ivbox = gui.widgetBox(self.controlArea, "Vector plot")

        rbox = gui.radioButtons(
            dbox, self, "value_type", callback=self._change_integration)

        gui.appendRadioButton(rbox, "From spectra")

        self.box_values_spectra = gui.indentedBox(rbox)

        gui.comboBox(
            self.box_values_spectra, self, "integration_method",
            items=(a.name for a in self.integration_methods),
            callback=self._change_integral_type)

        gui.appendRadioButton(rbox, "Use feature")

        self.box_values_feature = gui.indentedBox(rbox)

        self.feature_value_model = DomainModel(DomainModel.SEPARATED,
                                               valid_types=DomainModel.PRIMITIVE)
        self.feature_value = gui.comboBox(
            self.box_values_feature, self, "attr_value",
            contentsLength=12, searchable=True,
            callback=self.update_feature_value, model=self.feature_value_model)

        gui.appendRadioButton(rbox, "RGB")
        self.box_values_RGB_feature = gui.indentedBox(rbox)

        self.rgb_value_model = DomainModel(DomainModel.SEPARATED,
                                               valid_types=(ContinuousVariable,))

        self.red_feature_value = gui.comboBox(
            self.box_values_RGB_feature, self, "rgb_red_value",
            contentsLength=12, searchable=True,
            callback=self.update_rgb_value, model=self.rgb_value_model)

        self.green_feature_value = gui.comboBox(
            self.box_values_RGB_feature, self, "rgb_green_value",
            contentsLength=12, searchable=True,
            callback=self.update_rgb_value, model=self.rgb_value_model)

        self.blue_feature_value = gui.comboBox(
            self.box_values_RGB_feature, self, "rgb_blue_value",
            contentsLength=12, searchable=True,
            callback=self.update_rgb_value, model=self.rgb_value_model)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        self.imageplot = ImagePlot(self)
        self.imageplot.selection_changed.connect(self.output_image_selection)
        # add image settings to the main panne after ImagePlot.__init__
        iabox.layout().addWidget(self.imageplot.axes_settings_box)
        icbox.layout().addWidget(self.imageplot.color_settings_box)

        # add image settings to the main panne after ImagePlot.__init__
        iabox.layout().addWidget(self.imageplot.axes_settings_box)
        icbox.layout().addWidget(self.imageplot.color_settings_box)
        ivbox.layout().addWidget(self.imageplot.setup_vector_plot_controls())

        self.data = None

        # do not save visible image (a complex structure as a setting;
        # only save its name)
        self.visible_image = None
        self.setup_visible_image_controls()

        self.curveplot = CurvePlotHyper(self, select=SELECTONE)
        self.curveplot.selection_changed.connect(self.redraw_integral_info)
        self.curveplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        splitter.addWidget(self.imageplot)
        splitter.addWidget(self.curveplot)
        self.mainArea.layout().addWidget(splitter)
        self.curveplot.locked_axes_changed.connect(
            lambda locked: self.Information.view_locked(shown=locked))

        self.traceplot = CurvePlotHyper(self)
        self.traceplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        splitter.addWidget(self.traceplot)
        self.traceplot.button.hide()
        self.traceplot.hide()
        self.imageplot.selection_changed.connect(self.draw_trace)
        self.imageplot.image_updated.connect(self.draw_trace)

        self.line1 = MovableVline(position=self.lowlim, label="", report=self.curveplot)
        self.line1.sigMoved.connect(lambda v: setattr(self, "lowlim", v))
        self.line2 = MovableVline(position=self.highlim, label="", report=self.curveplot)
        self.line2.sigMoved.connect(lambda v: setattr(self, "highlim", v))
        self.line3 = MovableVline(position=self.choose, label="", report=self.curveplot)
        self.line3.sigMoved.connect(lambda v: setattr(self, "choose", v))
        self.line4 = MovableVline(position=self.choose, label="baseline", report=self.curveplot,
                                  color=(255, 140, 26))
        self.line4.sigMoved.connect(lambda v: setattr(self, "lowlimb", v))
        self.line5 = MovableVline(position=self.choose, label="baseline", report=self.curveplot,
                                  color=(255, 140, 26))
        self.line5.sigMoved.connect(lambda v: setattr(self, "highlimb", v))
        for line in [self.line1, self.line2, self.line3, self.line4, self.line5]:
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

        self._setup_plot_parameters()

        gui.rubber(self.controlArea)

    def _setup_plot_parameters(self):
        parts_from_spectra = [SpectraParameterSetter.ANNOT_BOX,
                              SpectraParameterSetter.LABELS_BOX,
                              SpectraParameterSetter.VIEW_RANGE_BOX]
        for cp in parts_from_spectra:
            self.imageplot.parameter_setter.initial_settings[cp] = \
                self.curveplot.parameter_setter.initial_settings[cp]

        VisualSettingsDialog(self, self.imageplot.parameter_setter.initial_settings)

    def setup_visible_image_controls(self):
        self.visbox = gui.widgetBox(self.controlArea, box="Visible image")

        gui.checkBox(
            self.visbox, self, 'show_visible_image',
            label='Show visible image',
            callback=lambda: (self.update_visible_image_interface(), self.update_visible_image()))

        self.visboxhide = gui.widgetBox(self.visbox, box=False)
        self.visboxhide.hide()

        self.visible_image_model = VisibleImageListModel()
        gui.comboBox(
            self.visboxhide, self, 'visible_image',
            model=self.visible_image_model,
            callback=self.update_visible_image)

        self.visual_image_composition_modes = OrderedDict([
            ('Normal', QPainter.CompositionMode_Source),
            ('Overlay', QPainter.CompositionMode_Overlay),
            ('Multiply', QPainter.CompositionMode_Multiply),
            ('Difference', QPainter.CompositionMode_Difference)
        ])
        gui.comboBox(
            self.visboxhide, self, 'visible_image_composition', label='Composition mode:',
            model=PyListModel(self.visual_image_composition_modes.keys()),
            callback=self.update_visible_image_composition_mode
        )

        gui.hSlider(
            self.visboxhide, self, 'visible_image_opacity', label='Opacity:',
            minValue=0, maxValue=255, step=10, createLabel=False,
            callback=self.update_visible_image_opacity
        )

        self.update_visible_image_interface()
        self.update_visible_image_composition_mode()
        self.update_visible_image_opacity()

    def update_visible_image_interface(self):
        controlled = ['visible_image', 'visible_image_composition', 'visible_image_opacity']
        self.visboxhide.setVisible(self.show_visible_image)
        for c in controlled:
            getattr(self.controls, c).setEnabled(self.show_visible_image)

    def update_visible_image_composition_mode(self):
        self.imageplot.set_visible_image_comp_mode(
            self.visual_image_composition_modes[self.visible_image_composition])

    def update_visible_image_opacity(self):
        self.imageplot.set_visible_image_opacity(self.visible_image_opacity)

    def init_interface_data(self, data):
        self.init_attr_values(data)
        self.init_visible_images(data)
        self.imageplot.init_vector_plot(data)

    def output_image_selection(self):
        _, selected = self.send_selection(self.data, self.imageplot.selection_group)
        self.curveplot.set_data(selected if selected else self.data)

    def draw_trace(self):
        distances = self.imageplot.selection_distances
        sel = self.imageplot.selection_group > 0
        self.traceplot.set_data(None)
        if distances is not None and self.imageplot.data \
            and self.imageplot.lsx and self.imageplot.lsy:
            distances = distances[sel]
            sortidx = np.argsort(distances)
            values = self.imageplot.data_values[sel]
            x = distances[sortidx]
            y = values[sortidx]

            # combine xs with the same value
            groups, pos, g_count = np.unique(x,
                                          return_index=True,
                                          return_counts=True)
            g_sum = np.add.reduceat(y, pos, axis=0)
            g_mean = g_sum / g_count[:, None]
            x, y = groups, g_mean

            traceplot_data = build_spec_table(x, y.T, None)
            self.traceplot.set_data(traceplot_data)
            self.traceplot.show()
        else:
            self.traceplot.hide()

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.feature_value_model.set_domain(domain)
        self.rgb_value_model.set_domain(domain)
        self.attr_value = self.feature_value_model[0] if self.feature_value_model else None
        if self.rgb_value_model:
            # Filter PyListModel.Separator objects
            rgb_attrs = [a for a in self.feature_value_model if isinstance(a, ContinuousVariable)]
            if len(rgb_attrs) <= 3:
                rgb_attrs = (rgb_attrs + rgb_attrs[-1:]*3)[:3]
            self.rgb_red_value, self.rgb_green_value, self.rgb_blue_value = rgb_attrs[:3]
        else:
            self.rgb_red_value = self.rgb_green_value = self.rgb_blue_value = None

    def init_visible_images(self, data):
        self.visible_image_model.clear()
        if data is not None and 'visible_images' in data.attributes:
            self.visbox.setEnabled(True)
            for img in data.attributes['visible_images']:
                if not isinstance(img, VisibleImage):
                    warnings.warn("Visible images need to subclass VisibleImage; "
                                  "Backward compatibility will be removed in the future.",
                                  OrangeDeprecationWarning)
                self.visible_image_model.append(img)
        else:
            self.visbox.setEnabled(False)
            self.show_visible_image = False
        self.update_visible_image_interface()
        self._choose_visible_image()
        self.update_visible_image()

    def _choose_visible_image(self):
        # choose an image according to visible_image_name setting
        if len(self.visible_image_model):
            for img in self.visible_image_model:
                name = img.name if isinstance(img, VisibleImage) else img["name"]
                if name == self.visible_image_name:
                    self.visible_image = img
                    break
            else:
                self.visible_image = self.visible_image_model[0]

    def redraw_integral_info(self):
        di = {}
        integrate = self.image_values()
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

    def image_values(self):
        if self.value_type == 0:  # integrals
            imethod = self.integration_methods[self.integration_method]

            if imethod == Integrate.Separate:
                return Integrate(methods=imethod,
                                 limits=[[self.lowlim, self.highlim,
                                          self.lowlimb, self.highlimb]])
            elif imethod != Integrate.PeakAt:
                return Integrate(methods=imethod,
                                 limits=[[self.lowlim, self.highlim]])
            else:
                return Integrate(methods=imethod,
                                 limits=[[self.choose, self.choose]])
        elif self.value_type == 1:  # feature
            return lambda data, attr=self.attr_value: \
                data.transform(Domain([data.domain[attr]]))
        elif self.value_type == 2:  # RGB
            red = ContinuousVariable("red", compute_value=Identity(self.rgb_red_value))
            green = ContinuousVariable("green", compute_value=Identity(self.rgb_green_value))
            blue = ContinuousVariable("blue", compute_value=Identity(self.rgb_blue_value))
            return lambda data: \
                    data.transform(Domain([red, green, blue]))

    def image_values_fixed_levels(self):
        if self.value_type == 1 and isinstance(self.attr_value, DiscreteVariable):
            return 0, len(self.attr_value.values) - 1
        return None

    def redraw_data(self):
        self.redraw_integral_info()
        self.imageplot.update_view()

    def update_feature_value(self):
        self.redraw_data()

    def update_rgb_value(self):
        self.redraw_data()

    def _update_integration_type(self):
        self.line1.hide()
        self.line2.hide()
        self.line3.hide()
        self.line4.hide()
        self.line5.hide()
        if self.value_type == 0:
            self.box_values_spectra.setDisabled(False)
            self.box_values_feature.setDisabled(True)
            self.box_values_RGB_feature.setDisabled(True)
            if self.integration_methods[self.integration_method] != Integrate.PeakAt:
                self.line1.show()
                self.line2.show()
            else:
                self.line3.show()
            if self.integration_methods[self.integration_method] == Integrate.Separate:
                self.line4.show()
                self.line5.show()
        elif self.value_type == 1:
            self.box_values_spectra.setDisabled(True)
            self.box_values_feature.setDisabled(False)
            self.box_values_RGB_feature.setDisabled(True)
        elif self.value_type == 2:
            self.box_values_spectra.setDisabled(True)
            self.box_values_feature.setDisabled(True)
            self.box_values_RGB_feature.setDisabled(False)
        # ImagePlot menu levels visibility
        rgb = self.value_type == 2
        self.imageplot.rgb_settings_box.setVisible(rgb)
        self.imageplot.color_settings_box.setVisible(not rgb)
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
        self.update_visible_image()

    def set_visual_settings(self, key, value):
        im_setter = self.imageplot.parameter_setter
        cv_setter = self.curveplot.parameter_setter
        skip_im_setter = [SpectraParameterSetter.ANNOT_BOX,
                          SpectraParameterSetter.VIEW_RANGE_BOX]
        if key[0] not in skip_im_setter and key[0] in im_setter.initial_settings:
            im_setter.set_parameter(key, value)
        if key[0] in cv_setter.initial_settings:
            cv_setter.set_parameter(key, value)
        self.visual_settings[key] = value

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

        if self.lowlimb is None or not minx <= self.lowlimb <= maxx:
            self.lowlimb = minx
        self.line4.setValue(self.lowlimb)

        if self.highlimb is None or not minx <= self.highlimb <= maxx:
            self.highlimb = maxx
        self.line5.setValue(self.highlimb)

        self.disable_integral_range = False

    def save_graph(self):
        self.imageplot.save_graph()

    def onDeleteWidget(self):
        self.curveplot.shutdown()
        self.imageplot.shutdown()
        super().onDeleteWidget()

    def update_visible_image(self):
        img_info = self.visible_image
        if self.show_visible_image and img_info is not None:
            if isinstance(img_info, VisibleImage):
                name = img_info.name
                img = np.array(img_info.image.convert('RGBA'))
                width = img_info.size_x
                height = img_info.size_y
                pos_x = img_info.pos_x
                pos_y = img_info.pos_y
            else:
                name = img_info["name"]
                img = np.array(Image.open(img_info['image_ref']).convert('RGBA'))
                width = img_info['img_size_x'] if 'img_size_x' in img_info \
                    else img.shape[1] * img_info['pixel_size_x']
                height = img_info['img_size_y'] if 'img_size_y' in img_info \
                    else img.shape[0] * img_info['pixel_size_y']
                pos_x = img_info['pos_x']
                pos_y = img_info['pos_y']
            self.visible_image_name = name  # save visual image name
            # image must be vertically flipped
            # https://github.com/pyqtgraph/pyqtgraph/issues/315#issuecomment-214042453
            # Behavior may change at pyqtgraph 1.0 version
            img = img[::-1]
            rect = QRectF(pos_x,
                          pos_y,
                          width,
                          height)
            self.imageplot.set_visible_image(img, rect)
            self.imageplot.show_visible_image()
        else:
            self.imageplot.hide_visible_image()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWHyper).run(Orange.data.Table("iris.tab"))
