import numpy as np
import pyqtgraph as pg
import bottleneck as bn

from scipy.ndimage import sobel
from scipy.ndimage import shift

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel

from Orange.data import Table, ContinuousVariable
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings
from Orange.widgets.visualize.utils.plotutils import PlotWidget

from orangecontrib.spectroscopy.data import getx, build_spec_table
from orangecontrib.spectroscopy.io.util import _spectra_from_image
from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange
from orangecontrib.spectroscopy.utils import (
    NanInsideHypercube,
    InvalidAxisException,
    get_hypercube,
)
from orangecontrib.spectroscopy.preprocess.utils import WrongReferenceException
from orangecontrib.spectroscopy.utils.skimage._phase_cross_correlation import (
    phase_cross_correlation,
)
from orangecontrib.spectroscopy.widgets.owspectra import InteractiveViewBox


class RegisterTranslation:
    def __init__(self, upsample_factor=1):
        self.upsample_factor = upsample_factor

    def __call__(self, base, shifted):
        """Return the shift (in each axis) needed to align to the base.
        Shift down and right are positive. First coordinate belongs to
        the first axis (rows in numpy)."""
        s, _, _ = phase_cross_correlation(
            base, shifted, upsample_factor=self.upsample_factor
        )
        return s


def shift_fill(img, sh, fill=np.nan):
    """Shift and fill invalid positions"""
    aligned = shift(img, sh, mode='nearest')

    (u, v) = img.shape

    shifty = int(round(sh[0]))
    aligned[: max(0, shifty), :] = fill
    aligned[min(u, u + shifty) :, :] = fill

    shiftx = int(round(sh[1]))
    aligned[:, : max(0, shiftx)] = fill
    aligned[:, min(v, v + shiftx) :] = fill

    return aligned


def alignstack(raw, shiftfn, ref_frame_num=0, filterfn=lambda x: x):
    """Align to the first image"""
    shifts = calculate_stack_shifts(
        raw, shiftfn, ref_frame_num=ref_frame_num, filterfn=filterfn
    )
    aligned = alignstack_with_shifts(raw, shifts)

    return shifts, aligned


def calculate_stack_shifts(raw, shiftfn, ref_frame_num=0, filterfn=lambda x: x):
    """Calculate the shifts for each image in the stack"""
    base = filterfn(raw[ref_frame_num])
    shifts = []

    for i, image in enumerate(raw):
        if i != ref_frame_num:
            shifts.append(shiftfn(base, filterfn(image)))
        else:
            shifts.append((0, 0))
    shifts = np.array(shifts)

    return shifts


def alignstack_with_shifts(raw, shifts):
    """Aligns the stack using the provided shifts"""
    aligned = np.zeros((len(raw),) + raw[0].shape, dtype=raw[0].dtype)
    for k in range(len(raw)):
        aligned[k] = shift_fill(raw[k], shifts[k])

    return aligned


def process_stack(
    data, xat, yat, upsample_factor=100, use_sobel=False, ref_frame_num=0, refdata=None
):
    calculate_shift = RegisterTranslation(upsample_factor=upsample_factor)
    filterfn = sobel if use_sobel else lambda x: x

    hypercube, lsx, lsy = get_hypercube(data, xat, yat)
    if bn.anynan(hypercube):
        raise NanInsideHypercube(True)

    if refdata is None:
        shifts, aligned_stack = alignstack(
            hypercube.T,
            shiftfn=calculate_shift,
            ref_frame_num=ref_frame_num,
            filterfn=filterfn,
        )
    else:
        if refdata.X.shape[1] != data.X.shape[1]:
            raise WrongReferenceException(
                "Reference data must have the same number of frames as the input data."
            )

        hypercube_ref, _, _ = get_hypercube(refdata, xat, yat)
        if bn.anynan(hypercube_ref):
            raise NanInsideHypercube(True)
        shifts = calculate_stack_shifts(
            hypercube_ref.T,
            shiftfn=calculate_shift,
            ref_frame_num=ref_frame_num,
            filterfn=filterfn,
        )
        aligned_stack = alignstack_with_shifts(hypercube.T, shifts)

    xmin, ymin = shifts[:, 0].min(), shifts[:, 1].min()
    xmax, ymax = shifts[:, 0].max(), shifts[:, 1].max()
    xmin, xmax = int(round(xmin)), int(round(xmax))
    ymin, ymax = int(round(ymin)), int(round(ymax))

    shape = hypercube.shape
    slicex = slice(max(xmax, 0), min(shape[1], shape[1] + xmin))
    slicey = slice(max(ymax, 0), min(shape[0], shape[0] + ymin))
    cropped = np.array(aligned_stack).T[slicey, slicex]

    # transform numpy array back to Orange.data.Table
    return shifts, build_spec_table(
        *_spectra_from_image(
            cropped, getx(data), np.linspace(*lsx)[slicex], np.linspace(*lsy)[slicey]
        )
    )


class OWStackAlign(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Align Stack"

    # Short widget description
    description = "Aligns and crops a stack of images using various methods."

    icon = "icons/stackalign.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Stack of images", Table, default=True)
        refdata = Input("Reference images", Table, default=True)

    class Outputs:
        newstack = Output("Aligned image stack", Table, default=True)

    class Error(OWWidget.Error):
        nan_in_image = Msg("Unknown values within images: {} unknowns")
        invalid_axis = Msg("Invalid axis: {}")
        wrong_reference = Msg("Wrong reference: {}")

    class Warning(OWWidget.Warning):
        missing_reference = Msg("Missing reference: {}")

    autocommit = settings.Setting(True)

    want_main_area = True
    want_control_area = True
    resizing_enabled = False

    settingsHandler = DomainContextHandler()
    use_refinput = settings.Setting(False)

    sobel_filter = settings.Setting(False)
    attr_x = ContextSetting(None, exclude_attributes=True)
    attr_y = ContextSetting(None, exclude_attributes=True)
    upscale_factor = settings.Setting(1)
    ref_frame_num = settings.Setting(0)

    def __init__(self):
        super().__init__()

        # TODO: add input box for selecting which should be the reference frame
        box = gui.widgetBox(self.controlArea, "Axes")

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True
        )
        self.xy_model = DomainModel(
            DomainModel.METAS | DomainModel.CLASSES, valid_types=ContinuousVariable
        )
        self.cb_attr_x = gui.comboBox(
            box,
            self,
            "attr_x",
            label="Axis x:",
            callback=self._update_attr,
            model=self.xy_model,
            **common_options,
        )
        self.cb_attr_y = gui.comboBox(
            box,
            self,
            "attr_y",
            label="Axis y:",
            callback=self._update_attr,
            model=self.xy_model,
            **common_options,
        )

        self.contextAboutToBeOpened.connect(self._init_interface_data)

        refbox = gui.widgetBox(self.controlArea, "Tracking images")
        gui.checkBox(
            refbox,
            self,
            "use_refinput",
            label="Use reference images",
            callback=self._use_ref_changed,
        )

        box = gui.widgetBox(self.controlArea, "Parameters")

        gui.checkBox(
            box,
            self,
            "sobel_filter",
            label="Use sobel filter",
            callback=self._sobel_changed,
        )
        gui.separator(box)
        hbox1 = gui.hBox(box)
        self.le_upscale = lineEditIntRange(
            box, self, "upscale_factor", bottom=1, default=1, callback=self._update_attr
        )
        hbox1.layout().addWidget(QLabel("Upscale factor:", self))
        hbox1.layout().addWidget(self.le_upscale)

        self.le1 = lineEditIntRange(
            box,
            self,
            "ref_frame_num",
            bottom=1,
            default=1,
            callback=self._ref_frame_changed,
        )
        hbox2 = gui.hBox(box)
        hbox2.layout().addWidget(QLabel("Reference frame:", self))
        hbox2.layout().addWidget(self.le1)

        gui.rubber(self.controlArea)

        plot_box = gui.widgetBox(self.mainArea, "Shift curves")
        self.plotview = PlotWidget(viewBox=InteractiveViewBox(self))
        self.plotview.plotItem.buttonsHidden = True
        plot_box.layout().addWidget(self.plotview)
        # TODO:  resize widget to make it a bit smaller

        self.data = None
        self.refdata = None

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")

    def _sanitize_ref_frame(self):
        if self.refdata is None:
            if self.ref_frame_num > self.data.X.shape[1]:
                self.ref_frame_num = self.data.X.shape[1]
        else:
            if self.ref_frame_num > self.refdata.X.shape[1]:
                self.ref_frame_num = self.refdata.X.shape[1]

    def _ref_frame_changed(self):
        if self.refdata is not None or self.data is not None:
            self._sanitize_ref_frame()
        self.commit.deferred()

    def _use_ref_changed(self):
        self.Warning.missing_reference.clear()
        self._check_use_ref()
        self._ref_frame_changed()

    def _check_use_ref(self):
        if self.use_refinput and self.refdata is None:
            self.Warning.missing_reference(
                "Reference is not connected. Using data only."
            )

    def _sobel_changed(self):
        self.commit.deferred()

    def _init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 else self.attr_x

    def _init_interface_data(self, args):
        data = args[0]
        self._init_attr_values(data)

    def _update_attr(self):
        self.commit.deferred()

    @Inputs.data
    def set_data(self, dataset):
        self.data = dataset

    @Inputs.refdata
    def set_reference(self, refdataset):
        self.refdata = refdataset

    def handleNewSignals(self):
        self.Warning.missing_reference.clear()

        self.closeContext()
        self.openContext(self.data)

        if self.refdata is not None or self.data is not None:
            self._sanitize_ref_frame()

        self._check_use_ref()

        self.Error.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.wrong_reference.clear()
        self.commit.now()

    @gui.deferred
    def commit(self):
        new_stack = None

        self.Error.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.wrong_reference.clear()
        self.plotview.plotItem.clear()

        if not (
            self.data
            and len(self.data.domain.attributes)
            and self.attr_x
            and self.attr_y
        ):
            pass
        elif self.attr_x == self.attr_y:
            self.Error.invalid_axis("same values")
        else:
            try:
                refdata = self.refdata if self.use_refinput else None
                shifts, new_stack = process_stack(
                    self.data,
                    self.attr_x,
                    self.attr_y,
                    upsample_factor=self.upscale_factor,
                    use_sobel=self.sobel_filter,
                    ref_frame_num=self.ref_frame_num - 1,
                    refdata=refdata,
                )
            except NanInsideHypercube as e:
                self.Error.nan_in_image(e.args[0])
            except InvalidAxisException as e:
                self.Error.invalid_axis(e.args[0])
            except WrongReferenceException as e:
                self.Error.wrong_reference(e.args[0])
            else:
                frames = np.linspace(1, shifts.shape[0], shifts.shape[0])
                self.plotview.plotItem.plot(
                    frames,
                    shifts[:, 0],
                    pen=pg.mkPen(color=(255, 40, 0), width=3),
                    symbol='o',
                    symbolBrush=(255, 40, 0),
                    symbolPen='w',
                    symbolSize=7,
                )
                self.plotview.plotItem.plot(
                    frames,
                    shifts[:, 1],
                    pen=pg.mkPen(color=(0, 139, 139), width=3),
                    symbol='o',
                    symbolBrush=(0, 139, 139),
                    symbolPen='w',
                    symbolSize=7,
                )
                self.plotview.getPlotItem().setLabel('bottom', 'Frame number')
                self.plotview.getPlotItem().setLabel('left', 'Shift / pixel')
                self.plotview.getPlotItem().addLine(
                    self.ref_frame_num,
                    pen=pg.mkPen(
                        color=(150, 150, 150), width=3, style=Qt.DashDotDotLine
                    ),
                )

        self.Outputs.newstack.send(new_stack)

    def send_report(self):
        self.report_items((("Use sobel filter", str(self.sobel_filter)),))


if __name__ == "__main__":  # pragma: no cover
    from orangecontrib.spectroscopy.tests.test_owalignstack import stxm_diamond
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWStackAlign).run(set_data=stxm_diamond)
