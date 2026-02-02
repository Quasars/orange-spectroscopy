import sys
import time

from AnyQt.QtWidgets import QFormLayout, QSizePolicy

import numpy as np

import Orange.data
from Orange import preprocess
from Orange.widgets.data.owpreprocess import (
    PreprocessAction, Description, icon_path, DescriptionRole, ParametersRole, blocked,
)
from Orange.widgets import gui, settings
from Orange.widgets.widget import Output, Msg

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import Integrate

from orangecontrib.spectroscopy.widgets.owspectra import SELECTONE
from orangecontrib.spectroscopy.widgets.owhyper import refresh_integral_markings
from orangecontrib.spectroscopy.widgets.owpreprocess import (
    SpectralPreprocess, create_preprocessor, InterruptException
)
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange, \
    SetXDoubleSpinBox

from orangecontrib.spectroscopy.widgets.gui import MovableVline


class IntegrateOneEditor(BaseEditorOrange):

    class Warning(BaseEditorOrange.Warning):
        out_of_range = Msg("Limit out of range.")

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QFormLayout()
        self.controlArea.setLayout(layout)

        minf, maxf = -sys.float_info.max, sys.float_info.max

        self.__values = {}
        self.__editors = {}
        self.__lines = {}

        for name, _longname in self.integrator.parameters():
            v = 0.
            self.__values[name] = v

            e = SetXDoubleSpinBox(minimum=minf, maximum=maxf,
                                  singleStep=0.5, value=v)
            e.focusIn = self.activateOptions
            e.editingFinished.connect(self.edited)
            def cf(x, name=name):
                self.edited.emit()
                return self.set_value(name, x)
            e.valueChanged[float].connect(cf)
            self.__editors[name] = e
            layout.addRow(name, e)

            color = (225, 0, 0)
            if "baseline" in name:
                color = (255, 140, 26)

            l = MovableVline(position=v, label=name, color=color)
            def set_rounded(_, line=l, name=name):
                cf(float(line.rounded_value()), name)
            l.sigMoved.connect(set_rounded)
            self.__lines[name] = l

        self.focusIn = self.activateOptions
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        self.parent_widget.redraw_integral()
        for l in self.__lines.values():
            if l not in self.parent_widget.curveplot.markings:
                l.report = self.parent_widget.curveplot
                self.parent_widget.curveplot.add_marking(l)

    def set_value(self, name, v, user=True):
        if user:
            self.user_changed = True
        if self.__values[name] != v:
            self.__values[name] = v
            with blocked(self.__editors[name]):
                self.__editors[name].setValue(v)
                self.__lines[name].setValue(v)
            self.changed.emit()

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        for name, _ in self.integrator.parameters():
            self.set_value(name, params.get(name, 0.), user=False)

    def parameters(self):
        return self.__values

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        values = []
        for _ind, (name, _) in enumerate(cls.integrator.parameters()):
            values.append(params.get(name, 0.))
        return Integrate(methods=cls.integrator, limits=[values], metas=True)

    def set_preview_data(self, data):
        self.Warning.out_of_range.clear()
        if data:
            xs = getx(data)
            if len(xs):
                minx = np.min(xs)
                maxx = np.max(xs)
                limits = [self.__values.get(name, 0.)
                          for ind, (name, _) in enumerate(self.integrator.parameters())]
                for v in limits:
                    if v < minx or v > maxx:
                        self.parent_widget.Warning.preprocessor()
                        self.Warning.out_of_range()


class IntegrateSimpleEditor(IntegrateOneEditor):
    qualname = "orangecontrib.infrared.integrate.simple"
    integrator = Integrate.Simple

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_value("Low limit", min(x), user=False)
                self.set_value("High limit", max(x), user=False)
                self.edited.emit()
        super().set_preview_data(data)


class IntegrateBaselineEditor(IntegrateSimpleEditor):
    qualname = "orangecontrib.infrared.integrate.baseline"
    integrator = Integrate.Baseline


class IntegratePeakMaxEditor(IntegrateSimpleEditor):
    qualname = "orangecontrib.infrared.integrate.peak_max"
    integrator = Integrate.PeakMax


class IntegratePeakMaxBaselineEditor(IntegrateSimpleEditor):
    qualname = "orangecontrib.infrared.integrate.peak_max_baseline"
    integrator = Integrate.PeakBaseline


class IntegrateAtEditor(IntegrateOneEditor):
    qualname = "orangecontrib.infrared.integrate.closest"
    integrator = Integrate.PeakAt

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_value("Closest to", min(x), user=False)
                self.edited.emit()
        super().set_preview_data(data)


class IntegratePeakXEditor(IntegrateSimpleEditor):
    qualname = "orangecontrib.infrared.integrate.peakx"
    integrator = Integrate.PeakX


class IntegratePeakXBaselineEditor(IntegrateSimpleEditor):
    qualname = "orangecontrib.infrared.integrate.peakx_baseline"
    integrator = Integrate.PeakXBaseline


class IntegrateSeparateBaselineEditor(IntegrateSimpleEditor):
    qualname = "orangecontrib.infrared.integrate.baseline_separate"
    integrator = Integrate.Separate

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_value("Low limit (baseline)", min(x), user=False)
                self.set_value("High limit (baseline)", max(x), user=False)
                self.set_value("Low limit", min(x), user=False)
                self.set_value("High limit", max(x), user=False)
                self.edited.emit()
        super().set_preview_data(data)


PREPROCESSORS = [
    PreprocessAction(
        "Integrate", c.qualname, "Integration",
        Description(c.integrator.name, icon_path("Discretize.svg")),
        c
    ) for c in [
        IntegrateSimpleEditor,
        IntegrateBaselineEditor,
        IntegratePeakMaxEditor,
        IntegratePeakMaxBaselineEditor,
        IntegrateAtEditor,
        IntegratePeakXEditor,
        IntegratePeakXBaselineEditor,
        IntegrateSeparateBaselineEditor,
    ]
]


class OWIntegrate(SpectralPreprocess):
    name = "Integrate Spectra"
    id = "orangecontrib.spectroscopy.widgets.integrate"
    description = "Integrate spectra in various ways."
    icon = "icons/integrate.svg"
    priority = 1010
    replaces = ["orangecontrib.infrared.widgets.owintegrate.OWIntegrate"]

    settings_version = 2

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add integral..."

    class Outputs:
        preprocessed_data = Output("Integrated Data", Orange.data.Table, default=True)
        preprocessor = Output("Preprocessor", preprocess.preprocess.Preprocess)

    output_metas = settings.Setting(True)

    preview_on_image = True

    def __init__(self):
        self.markings_list = []
        super().__init__()
        cb = gui.checkBox(self.output_box, self, "output_metas", "Output as metas", callback=self.commit.deferred)
        self.output_box.layout().insertWidget(0, cb)  # move to top of the box
        self.curveplot.selection_type = SELECTONE
        self.curveplot.select_at_least_1 = True
        self.curveplot.selection_changed.connect(self.redraw_integral)
        self.preview_runner.preview_updated.connect(self.redraw_integral)

    def redraw_integral(self):
        dis = []
        if np.any(self.curveplot.selection_group) and self.curveplot.data:
            # select data
            ind = np.flatnonzero(self.curveplot.selection_group)[0]
            show = self.curveplot.data[ind:ind+1]

            previews = self.flow_view.preview_n()
            for i in range(self.preprocessormodel.rowCount()):
                if i in previews:
                    item = self.preprocessormodel.item(i)
                    desc = item.data(DescriptionRole)
                    params = item.data(ParametersRole)
                    if not isinstance(params, dict):
                        params = {}
                    preproc = desc.viewclass.createinstance(params)
                    preproc.metas = False
                    datai = preproc(show)
                    di = datai.domain.attributes[0].compute_value.draw_info(show)
                    color = self.flow_view.preview_color(i)
                    dis.append({"draw": di, "color": color})
        refresh_integral_markings(dis, self.markings_list, self.curveplot)

    def show_preview(self, show_info_anyway=False):
        # redraw integrals if number of preview curves was changed
        super().show_preview(False)

    def create_outputs(self):
        pp_def = [self.preprocessormodel.item(i) for i in range(self.preprocessormodel.rowCount())]
        self.start(self.run_task, self.data, pp_def, self.output_metas)

    @staticmethod
    def run_task(data: Orange.data.Table, pp_def, output_metas, state):

        def progress_interrupt(i: float):
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for _i in range(10):
            time.sleep(0.005)
            progress_interrupt(0)

        n = len(pp_def)
        plist = []
        for i in range(n):
            progress_interrupt(0)
            item = pp_def[i]
            pp = create_preprocessor(item, None)
            plist.append(pp)

        preprocessor = None
        if plist:
            preprocessor = PreprocessorListMoveMetas(not output_metas, preprocessors=plist)

        if data is not None and preprocessor is not None:
            data = preprocessor(data)

        progress_interrupt(100)

        return data, preprocessor


class PreprocessorListMoveMetas(preprocess.preprocess.PreprocessorList):
    """Move added meta variables to features if needed."""

    def __init__(self, move_metas, **kwargs):
        super().__init__(**kwargs)
        self.move_metas = move_metas

    def __call__(self, data):
        tdata = super().__call__(data)
        if self.move_metas:
            oldmetas = set(data.domain.metas)
            newmetas = [m for m in tdata.domain.metas if m not in oldmetas]
            domain = Orange.data.Domain(newmetas, data.domain.class_vars,
                                        metas=data.domain.metas)
            tdata = tdata.transform(domain)
        return tdata


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWIntegrate).run(Orange.data.Table("collagen.csv"))
