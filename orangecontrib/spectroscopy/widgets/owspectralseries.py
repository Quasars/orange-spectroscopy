from xml.sax.saxutils import escape

from AnyQt.QtWidgets import (
    QWidget,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QWidgetAction,
    QToolTip,
)

from AnyQt.QtCore import Qt, QRectF

from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.widgets.widget import OWWidget, Msg, OWComponent, Input
from Orange.widgets import gui
from Orange.widgets.settings import (
    Setting,
    ContextSetting,
    DomainContextHandler,
    SettingProvider,
)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.utils.plotutils import GraphicsView, PlotItem

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.utils import values_to_linspace, index_values
from orangecontrib.spectroscopy.widgets.owhyper import (
    _shift,
    ImageColorSettingMixin,
    ImageZoomMixin,
    ImageItemNan,
    ImageColorLegend,
)
from orangecontrib.spectroscopy.widgets.owspectra import (
    HelpEventDelegate,
    selection_modifiers,
    SELECTMANY,
    INDIVIDUAL,
    InteractiveViewBox,
    MenuFocus,
)
from orangecontrib.spectroscopy.widgets.utils import (
    SelectionGroupMixin,
    SelectionOutputsMixin,
)


class LineScanPlot(
    QWidget, OWComponent, SelectionGroupMixin, ImageColorSettingMixin, ImageZoomMixin
):
    attr_x = ContextSetting(None, exclude_attributes=True)
    gamma = Setting(0)

    selection_changed = Signal()

    def __init__(self, parent):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)
        SelectionGroupMixin.__init__(self)
        ImageColorSettingMixin.__init__(self)

        self.parent = parent

        self.selection_type = SELECTMANY
        self.saving_enabled = True
        self.selection_enabled = True
        self.viewtype = INDIVIDUAL  # required bt InteractiveViewBox
        self.highlighted = None
        self.data_points = None
        self.data_imagepixels = None

        self.plotview = GraphicsView()
        ci = pg.GraphicsLayout()
        self.plot = PlotItem(viewBox=InteractiveViewBox(self))
        self.plot.buttonsHidden = True
        self.plotview.setCentralItem(ci)
        ci.addItem(self.plot)

        self.legend = ImageColorLegend()
        ci.addItem(self.legend)

        self.plot.scene().installEventFilter(HelpEventDelegate(self.help_event, self))

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)

        self.img = ImageItemNan()
        self.img.setOpts(axisOrder='row-major')
        self.plot.addItem(self.img)
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
        self.parent.contextAboutToBeOpened.connect(
            lambda x: self.init_interface_data(x[0])
        )

        self.add_zoom_actions(view_menu)

        common_options = dict(
            labelWidth=50,
            orientation=Qt.Horizontal,
            sendSelectedValue=True,
            valueType=str,
        )

        choose_xy = QWidgetAction(self)
        box = gui.vBox(self)
        box.setFocusPolicy(Qt.TabFocus)
        self.xy_model = DomainModel(
            DomainModel.METAS | DomainModel.CLASSES,
            valid_types=DomainModel.PRIMITIVE,
            placeholder="Position (index)",
        )
        self.cb_attr_x = gui.comboBox(
            box,
            self,
            "attr_x",
            label="Axis x:",
            callback=self.update_attr,
            model=self.xy_model,
            **common_options,
        )

        box.setFocusProxy(self.cb_attr_x)

        box.layout().addWidget(self.setup_color_settings_box())

        choose_xy.setDefaultWidget(box)
        view_menu.addAction(choose_xy)

        self.lsx = None  # info about the X axis
        self.lsy = None  # info about the Y axis

        self.data = None
        self.data_values = None
        self.data_ids = {}

    def init_interface_data(self, data):
        self.init_attr_values(data)

    def help_event(self, ev):
        pos = self.plot.vb.mapSceneToView(ev.scenePos())
        sel, wavenumber_ind = self._points_at_pos(pos)
        prepared = []
        if sel is not None:
            prepared.append(str(self.wavenumbers[wavenumber_ind]))
            for d in self.data[sel]:
                variables = [
                    v
                    for v in self.data.domain.metas + self.data.domain.class_vars
                    if v not in [self.attr_x]
                ]
                features = ['{} = {}'.format(attr.name, d[attr]) for attr in variables]
                features.append('value = {}'.format(d[wavenumber_ind]))
                prepared.append("\n".join(features))
        text = "\n\n".join(prepared)
        if text:
            text = '<span style="white-space:pre">{}</span>'.format(escape(text))
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

    def set_data(self, data):
        if data:
            self.data = data
            self.data_ids = {e: i for i, e in enumerate(data.ids)}
            self.restore_selection_settings()
        else:
            self.data = None
            self.data_ids = {}

    def update_view(self):
        self.img.clear()
        self.img.setSelection(None)
        self.legend.set_colors(None)
        self.lsx = None
        self.lsy = None
        self.data_values = None
        self.wavenumbers = None
        self.data_xs = None
        self.data_imagepixels = None
        if self.data and len(self.data.domain.attributes):
            if self.attr_x is not None:
                xat = self.data.domain[self.attr_x]
                ndom = Domain([xat])
                datam = self.data.transform(ndom)
                coorx = datam.X[:, 0]
            else:
                coorx = np.arange(len(self.data))
            self.lsx = lsx = values_to_linspace(coorx)
            self.data_xs = coorx

            self.wavenumbers = wavenumbers = getx(self.data)
            self.lsy = lsy = values_to_linspace(wavenumbers)

            self.data_values = self.data.X

            # set data
            imdata = np.ones((lsy[2], lsx[2])) * float("nan")
            xindex = index_values(coorx, lsx)
            yindex = index_values(wavenumbers, lsy)
            for xind, d in zip(xindex, self.data.X, strict=False):
                imdata[yindex, xind] = d

            self.data_imagepixels = xindex

            self.img.setImage(imdata, autoLevels=False)
            self.update_levels()
            self.update_color_schema()

            # shift centres of the pixels so that the axes are useful
            shiftx = _shift(lsx)
            shifty = _shift(lsy)
            left = lsx[0] - shiftx
            bottom = lsy[0] - shifty
            width = (lsx[1] - lsx[0]) + 2 * shiftx
            height = (lsy[1] - lsy[0]) + 2 * shifty
            self.img.setRect(QRectF(left, bottom, width, height))

            self.refresh_img_selection()

    def refresh_img_selection(self):
        selected_px = np.zeros((self.lsy[2], self.lsx[2]), dtype=np.uint8)
        selected_px[:, self.data_imagepixels] = self.selection_group
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

    def _points_at_pos(self, pos):
        if self.data and self.lsx and self.lsy:
            x, y = pos.x(), pos.y()
            x_distance = np.abs(self.data_xs - x)
            sel = x_distance < _shift(self.lsx)
            wavenumber_distance = np.abs(self.wavenumbers - y)
            wavenumber_ind = np.argmin(wavenumber_distance)
            return sel, wavenumber_ind
        return None, None

    def select_by_click(self, pos):
        sel, _ = self._points_at_pos(pos)
        self.make_selection(sel)


class OWSpectralSeries(OWWidget, SelectionOutputsMixin):
    name = "Spectral Series"

    class Inputs:
        data = Input("Data", Table, default=True)

    class Outputs(SelectionOutputsMixin.Outputs):
        pass

    class Information(SelectionOutputsMixin.Information):
        pass

    class Warning(OWWidget.Warning):
        threshold_error = Msg("Low slider should be less than High")

    icon = "icons/spectralseries.svg"
    priority = 100

    settings_version = 2
    settingsHandler = DomainContextHandler()

    imageplot = SettingProvider(LineScanPlot)

    want_control_area = False

    value_type = 0  # TODO: temporary because ImagePlot uses it

    graph_name = "imageplot.plotview"  # defined so that the save button is shown

    def __init__(self):
        super().__init__()
        SelectionOutputsMixin.__init__(self)

        self.imageplot = LineScanPlot(self)
        self.imageplot.selection_changed.connect(self.output_image_selection)
        self.mainArea.layout().addWidget(self.imageplot)

        self.data = None

        self.resize(900, 700)

    def output_image_selection(self):
        self.send_selection(self.data, self.imageplot.selection_group)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()

        def valid_context(data):
            if data is None:
                return False
            annotation_features = [
                v
                for v in data.domain.metas + data.domain.class_vars
                if isinstance(v, (DiscreteVariable, ContinuousVariable))
            ]
            return len(annotation_features) >= 1

        if valid_context(data):
            self.openContext(data)
        else:
            # to generate valid interface even if context was not loaded
            self.contextAboutToBeOpened.emit([data])
        self.data = data
        self.imageplot.set_data(data)
        self.imageplot.update_view()
        self.output_image_selection()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            settings["compat_no_group"] = True


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSpectralSeries).run(Table("collagen"))
