import numpy as np

from AnyQt.QtCore import Qt

from Orange.data import Table, ContinuousVariable
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings

from orangecontrib.spectroscopy.utils import InvalidAxisException
from orangecontrib.spectroscopy.utils.binning import bin_hyperspectra, InvalidBlockShape
from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange

MAX_DIMENSIONS = 5


class OWBin(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Bin"

    # Short widget description
    description = (
        "Bins a hyperspectral dataset by continous variable such as coordinates."
    )

    icon = "icons/bin.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Data", Table, default=True)

    class Outputs:
        bindata = Output("Binned Data", Table, default=True)

    class Error(OWWidget.Error):
        invalid_axis = Msg("Invalid axis: {}")
        invalid_block = Msg("Bin block size not compatible with dataset: {}")

    class Warning(OWWidget.Warning):
        nan_in_image = Msg("Unknown values within images: {} unknowns")

    autocommit = settings.Setting(True)

    want_main_area = False
    want_control_area = True
    resizing_enabled = False

    settingsHandler = DomainContextHandler()

    attrs = ContextSetting([None, None], exclude_attributes=True)
    bin_shape = settings.Setting((1, 1))
    square_bin = settings.Setting(True)

    def __init__(self):
        super().__init__()

        self.data = None

        for i in range(MAX_DIMENSIONS):
            setattr(self, f"bin_{i}", 1)
            setattr(self, f"attr_{i}", None)

        self._init_bins()
        self._init_ndim()
        self._init_attrs()

        box = gui.widgetBox(self.controlArea, "Parameters")

        gui.checkBox(
            box,
            self,
            "square_bin",
            label="Use square bin shape",
            callback=self._bin_changed,
        )

        gui.separator(box)

        gui.spin(
            box,
            self,
            "ndim",
            minv=1,
            maxv=MAX_DIMENSIONS,
            label="Number of axes to bin:",
            callback=self._dim_changed,
        )

        self.axes_box = gui.widgetBox(self.controlArea, "Axes")

        self.xy_model = DomainModel(
            DomainModel.METAS | DomainModel.CLASSES, valid_types=ContinuousVariable
        )

        self.contextAboutToBeOpened.connect(self._init_interface_data)

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True
        )
        for i in range(MAX_DIMENSIONS):
            hbox = gui.hBox(self.axes_box)
            gui.comboBox(
                hbox,
                self,
                f"attr_{i}",
                label=f"Axis {i}:",
                callback=self._attr_changed,
                model=self.xy_model,
                **common_options,
            )
            le = lineEditIntRange(
                hbox, self, f"bin_{i}", bottom=1, default=1, callback=self._bin_changed
            )
            le.setFixedWidth(40)
            gui.separator(hbox, width=40)
            gui.widgetLabel(hbox, label="Bin size:", labelWidth=50)
            hbox.layout().addWidget(le)

        self._update_cb_attr()

        box = gui.widgetBox(self.controlArea, "Info")
        gui.label(box, self, "Block shape:  %(bin_shape)s")

        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")

    def _sanitize_bin_value(self):
        pass  # TODO make sure bin value is compatible with dataset

    def _update_bins(self):
        if self.square_bin:
            self.bin_shape = tuple([self.bin_0] * len(self.bin_shape))
            self._init_bins()
            return
        new_shape = []
        for i, _ in enumerate(self.bin_shape):
            new_shape.append(getattr(self, f"bin_{i}"))
        self.bin_shape = tuple(new_shape)

    def _attr_changed(self):
        self._update_attrs()
        self.commit.deferred()

    def _bin_changed(self):
        self._update_bins()
        self._sanitize_bin_value()
        self.commit.deferred()

    def _dim_changed(self):
        while len(self.bin_shape) != self.ndim:
            if len(self.bin_shape) < self.ndim:
                self.bin_shape += (1,)
                self.attrs.append(None)
            elif len(self.bin_shape) > self.ndim:
                self.bin_shape = self.bin_shape[:-1]
                self.attrs = self.attrs[:-1]
        self._update_bins()
        self._update_attrs()
        self._update_cb_attr()
        self.commit.deferred()

    def _init_bins(self):
        for i, bin in enumerate(self.bin_shape):
            setattr(self, f"bin_{i}", bin)

    def _init_ndim(self):
        self.ndim = len(self.bin_shape)

    def _init_attrs(self):
        for i, attr in enumerate(self.attrs):
            setattr(self, f"attr_{i}", attr)

    def _init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.xy_model.set_domain(domain)
        attrs = []
        for i in range(self.ndim):
            try:
                attr = self.xy_model[i] if self.xy_model else None
            except IndexError:
                attr = None
            attrs.append(attr)
        self.attrs = attrs

    def _init_interface_data(self, args):
        data = args[0]
        self._init_attr_values(data)
        self._init_attrs()

    def _update_attrs(self):
        new_attrs = []
        for i, _ in enumerate(self.attrs):
            new_attrs.append(getattr(self, f"attr_{i}"))
        self.attrs = new_attrs

    def _update_cb_attr(self):
        for i in range(MAX_DIMENSIONS):
            w = self.axes_box.layout().itemAt(i).widget()
            if i < self.ndim:
                w.show()
            else:
                w.hide()

    @Inputs.data
    def set_data(self, dataset):
        self.closeContext()
        self.openContext(dataset)
        if dataset is not None:
            self.data = dataset
            self._sanitize_bin_value()
        else:
            self.data = None
        self.Warning.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.invalid_block.clear()
        self.commit.now()

    @gui.deferred
    def commit(self):
        bin_data = None

        self.Warning.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.invalid_block.clear()

        attrs = self.attrs

        if self.data and len(self.data.domain.attributes) and len(attrs):
            if np.any(np.isnan(self.data.X)):
                self.Warning.nan_in_image(np.sum(np.isnan(self.data.X)))
            try:
                bin_data = bin_hyperspectra(self.data, attrs, self.bin_shape)
            except InvalidAxisException as e:
                self.Error.invalid_axis(e.args[0])
            except InvalidBlockShape as e:
                self.Error.invalid_block(e.args[0])

        self.Outputs.bindata.send(bin_data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWBin).run(Table("agilent/5_mosaic_agg1024.dmt"))
