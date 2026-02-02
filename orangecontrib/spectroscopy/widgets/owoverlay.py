import time
from typing import Optional

import numpy as np

from AnyQt.QtCore import Qt, QBuffer

from Orange.data import Table, ContinuousVariable, Domain
from Orange.data.util import get_unique_names
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

from orangewidget.settings import SettingProvider, ContextSetting

import io
from orangecontrib.spectroscopy.io.util import ConstantBytesVisibleImage
from PIL import Image
from orangecontrib.spectroscopy.utils import get_ndim_hyperspec, InvalidAxisException

from orangecontrib.spectroscopy.widgets.owhyper import BasicImagePlot


# Copied from orangecontrib.snom.widgets.owpreprocessimage
class AImagePlot(BasicImagePlot):
    attr_x = None  # not settings, set from the parent class
    attr_y = None

    def __init__(self, parent):
        super().__init__(parent)
        self.axes_settings_box.hide()
        self.rgb_settings_box.hide()
        self.image_updated.connect(self.drawing_done)

    def update_view(self):
        self.drawn = False
        super().update_view()

    def drawing_done(self):
        self.drawn = True

    def add_selection_actions(self, _):
        pass

    def clear_markings(self):
        pass


class InterruptException(Exception):
    pass


class OWOverlay(OWWidget, ConcurrentWidgetMixin):
    name = "Add Image Overlay"
    description = "Add an image that can be displayed in Hyper Spectra to the dataset."
    icon = "icons/overlay.svg"

    settings_version = 2
    value_type = 1

    settingsHandler = DomainContextHandler()
    imageplot = SettingProvider(AImagePlot)

    autocommit = settings.Setting(True)

    attr_value = ContextSetting(None)
    attr_x = ContextSetting(None, exclude_attributes=True)
    attr_y = ContextSetting(None, exclude_attributes=True)

    # Define inputs and outputs
    class Inputs:
        maindata = Input("Data", Table)
        data = Input("Overlay Data", Table)

    class Outputs:
        outdata = Output("Decorated Data", Table, default=True)

    class Error(OWWidget.Error):
        invalid_axis = Msg("Invalid axis: {}")
        invalid_block = Msg("Bin block size not compatible with dataset: {}")
        image_too_big = Msg("Image for chosen features is too big ({} x {}).")

    class Warning(OWWidget.Warning):
        nan_in_image = Msg("Unknown values within images: {} unknowns")
        threshold_error = Msg("Low slider should be less than High")

    class Information(OWWidget.Information):
        not_shown = Msg("Undefined positions: {} data point(s) are not shown.")

    def image_values(self):
        attr_value = self.attr_value.name if self.attr_value else None
        return lambda data, attr=attr_value: data.transform(Domain([data.domain[attr]]))

    def image_values_fixed_levels(self):
        return None

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.maindata = None
        self.data = None

        # Imageplot for mainarea
        self.imageplot = AImagePlot(self)
        self.mainArea.layout().addWidget(self.imageplot)

        # Control area box
        self.imagebox = gui.widgetBox(self.controlArea, "Image")

        self.xy_model = DomainModel(
            DomainModel.METAS | DomainModel.CLASSES, valid_types=ContinuousVariable
        )

        self.feature_value_model = DomainModel(
            order=(
                DomainModel.ATTRIBUTES,
                DomainModel.Separator,
                DomainModel.CLASSES,
                DomainModel.Separator,
                DomainModel.METAS,
            ),
            valid_types=ContinuousVariable,
        )

        self.contextAboutToBeOpened.connect(lambda x: self._init_interface_data(x[0]))

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True
        )

        box = gui.vBox(self.imagebox)

        self.feature_value = gui.comboBox(
            box,
            self,
            "attr_value",
            label="Feature:",
            contentsLength=12,
            searchable=True,
            callback=self._update_feature_value,
            model=self.feature_value_model,
            **common_options,
        )

        gui.comboBox(
            box,
            self,
            "attr_x",
            label="Axis X:",
            callback=self._attr_changed,
            model=self.xy_model,
            **common_options,
        )
        gui.comboBox(
            box,
            self,
            "attr_y",
            label="Axis Y:",
            callback=self._attr_changed,
            model=self.xy_model,
            **common_options,
        )

        colorsbox = gui.widgetBox(self.controlArea, "Image colors")
        colorsbox.layout().addWidget(self.imageplot.color_settings_box)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")

        self.imageplot.color_cb.activated.connect(self.commit.deferred)
        self.imageplot._threshold_low_slider.sliderReleased.connect(
            self.commit.deferred
        )
        self.imageplot._threshold_high_slider.sliderReleased.connect(
            self.commit.deferred
        )
        self.imageplot._level_high_le.editingFinished.connect(self.commit.deferred)
        self.imageplot._level_low_le.editingFinished.connect(self.commit.deferred)
        self._init_attr_values(self.data)

    def _attr_changed(self):
        self._update_attrs()
        self.commit.deferred()

    def _init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.feature_value_model.set_domain(domain)
        self.attr_value = (
            self.feature_value_model[0] if self.feature_value_model else None
        )
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 else self.attr_x

    def _init_interface_data(self, data):
        self._init_attr_values(data)
        self.imageplot.init_interface_data(data)

    def _update_feature_value(self):
        self.redraw_data()

    def _update_attrs(self):
        self.imageplot.attr_x = self.attr_x
        self.imageplot.attr_y = self.attr_y
        self.redraw_data()

    def redraw_data(self):
        self.imageplot.update_view()
        self.commit.deferred()

    @staticmethod
    def with_overlay(imageplot, data, maindata, state):

        if data is None or maindata is None:
            return None

        def progress_interrupt():
            if state.is_interruption_requested():
                raise InterruptException

        # wait for the image to appear
        while True:
            time.sleep(0.010)
            progress_interrupt()
            if imageplot.drawn is True:
                break

        # the following could raise an InvalidAxisException
        hypercube, ls = get_ndim_hyperspec(
            data, (imageplot.attr_x, imageplot.attr_y)
        )

        # if drawing failed for some reason
        if imageplot.img.image is None:
            return None

        # Copy main data to avoid changing it in place
        newmaindata = maindata.copy()

        width = np.abs(ls[0][1] - ls[0][0])
        height = np.abs(ls[1][1] - ls[1][0])
        xres = width / hypercube.shape[:2][0]
        yres = height / hypercube.shape[:2][1]
        posx = ls[0][0]
        posy = ls[1][0]

        # Extract the image from imageplot so the coloring is done
        imageplot.img.render()  # ensures qimage is produced even when not displayed
        im = imageplot.img.qimage
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        im.save(buffer, "PNG")
        pil_im = Image.open(io.BytesIO(buffer.data()))
        pil_im = pil_im.transpose(Image.FLIP_TOP_BOTTOM)
        img_bytes = io.BytesIO()
        pil_im.save(img_bytes, format="PNG")
        # Update name
        allnames = []
        if "visible_images" in newmaindata.attributes:
            allnames = [
                im.name for im in newmaindata.attributes["visible_images"]
            ]
        basename = "Overlay Image"
        name = get_unique_names(names=allnames, proposed=basename)
        # Need to modify the position and the scale since visual imageplot
        # place the corner of the pixel to the given position
        vimage = ConstantBytesVisibleImage(
            name=name,
            pos_x=posx - xres / 2,
            pos_y=posy - yres / 2,
            size_x=width + xres,
            size_y=height + yres,
            image_bytes=img_bytes,
        )

        # # Assign it to the datatable attributes
        if newmaindata and vimage is not None:
            if "visible_images" in newmaindata.attributes:
                newmaindata.attributes["visible_images"].append(vimage)
            else:
                newmaindata.attributes["visible_images"] = [vimage]

        return newmaindata

    @Inputs.maindata
    def set_data(self, data):
        self.maindata = data

    @Inputs.data
    def set_overlaydata(self, data):
        self.closeContext()
        self.openContext(data)
        self.data = data

        self.Warning.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.invalid_block.clear()

        self.imageplot.set_data(data)
        self.imageplot.update_view()

    def handleNewSignals(self):
        self.commit.now()

    @gui.deferred
    def commit(self):
        self.cancel()
        self.Warning.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.invalid_block.clear()
        self.start(OWOverlay.with_overlay, self.imageplot, self.data, self.maindata)

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Optional[Table]):
        self.Outputs.outdata.send(result)

    def on_exception(self, ex):
        if isinstance(ex, InvalidAxisException):
            self.Error.invalid_axis(str(ex))
        self.Outputs.outdata.send(None)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWOverlay).run(
        set_data=Table("agilent/5_mosaic_agg1024.dmt"),
        set_overlaydata=Table("agilent/5_mosaic_agg1024.dmt"),
    )
