import time
from functools import reduce
import concurrent.futures
import multiprocessing
import os

from lmfit import Parameters
from lmfit.model import ModelResult
import numpy as np

import pebble

from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.data.owpreprocess import PreprocessAction, Description, icon_path
from Orange.widgets.data.utils.preprocess import DescriptionRole, ParametersRole
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import Cut
from orangecontrib.spectroscopy.preprocess.integrate import INTEGRATE_DRAW_CURVE_PENARGS, \
    INTEGRATE_DRAW_BASELINE_PENARGS
from orangecontrib.spectroscopy.widgets.owhyper import refresh_integral_markings
from orangecontrib.spectroscopy.widgets.owpreprocess import SpectralPreprocess, \
    InterruptException, PreviewRunner
from orangecontrib.spectroscopy.widgets.owspectra import SELECTONE
from orangecontrib.spectroscopy.widgets.peak_editors import GaussianModelEditor, \
    LorentzianModelEditor, SplitLorentzianModelEditor, VoigtModelEditor, PseudoVoigtModelEditor, \
    MoffatModelEditor, Pearson7ModelEditor, StudentsTModelEditor, BreitWignerModelEditor, \
    LognormalModelEditor, DampedOscillatorModelEditor, DampedHarmOscillatorModelEditor, \
    ExponentialGaussianModelEditor, SkewedGaussianModelEditor, SkewedVoigtModelEditor, \
    ThermalDistributionModelEditor, DoniachModelEditor, ConstantModelEditor, \
    LinearModelEditor, QuadraticModelEditor, PolynomialModelEditor, set_default_vary
from orangecontrib.spectroscopy.widgets.peakfit_compute import n_best_fit_parameters, \
    best_fit_results, pool_initializer, pool_fit, pool_fit2

# number of processes used for computation
# Use 2 or less unless overridden by QUASAR_N_PROCESSES
env_proc = os.getenv('QUASAR_N_PROCESSES')
N_PROCESSES = None if env_proc == "all" else int(env_proc) if env_proc else min(2, multiprocessing.cpu_count())


def fit_results_table(output, model_result, orig_data):
    """Return best fit parameters as Orange.data.Table"""
    out = model_result
    features = []
    for comp in out.model.components:
        prefix = comp.prefix.rstrip("_")
        features.append(ContinuousVariable(name=f"{prefix} area"))
        for param in [n for n in out.var_names if n.startswith(comp.prefix)]:
            features.append(ContinuousVariable(name=param.replace("_", " ")))
    features.append(ContinuousVariable(name="Reduced chi-square"))

    domain = Domain(features,
                    orig_data.domain.class_vars,
                    orig_data.domain.metas)
    out = orig_data.transform(domain)
    with out.unlocked_reference(out.X):
        out.X = output
    return out


def fit_peaks(data, model, params):
    """
    Calculate fits for all rows in a data table for a given model and parameters
    and return a table of best fit parameters.

    Args:
        data (Orange.data.Table): Table with data to be fit in features
        model (lmfit.model.Model): lmfit Model/CompositeModel to fit with
        params (lmfit.parameter.Parameters): Parameters for fit

    Returns:
        results_table (Orange.data.Table): Table with best fit parameters as features
    """
    output = []
    shape = n_best_fit_parameters(model, params)
    x = getx(data)
    for row in data:
        out = model.fit(row.x, params, x=x)
        output.append(best_fit_results(out, x, shape))

    output = np.vstack(output)

    return fit_results_table(output, out, data)


def pack_model_editor(editor):
    return PreprocessAction(
        name=editor.name,
        qualname=f"orangecontrib.spectroscopy.widgets.peak_editors.{editor.prefix_generic}",
        category=editor.category,
        description=Description(getattr(editor, 'description', editor.name),
                                icon_path(editor.icon)),
        viewclass=editor,
    )


PREPROCESSORS = [pack_model_editor(e) for e in [
    GaussianModelEditor,
    LorentzianModelEditor,
    SplitLorentzianModelEditor,
    VoigtModelEditor,
    PseudoVoigtModelEditor,
    MoffatModelEditor,
    Pearson7ModelEditor,
    StudentsTModelEditor,
    BreitWignerModelEditor,
    LognormalModelEditor,
    DampedOscillatorModelEditor,
    DampedHarmOscillatorModelEditor,
    ExponentialGaussianModelEditor,
    SkewedGaussianModelEditor,
    SkewedVoigtModelEditor,
    ThermalDistributionModelEditor,
    DoniachModelEditor,
    ConstantModelEditor,
    LinearModelEditor,
    QuadraticModelEditor,
    PolynomialModelEditor,
    ]
]


def unique_prefix(modelclass, rownum):
    return f"{modelclass.prefix_generic}{rownum}_"


def create_model(item, rownum):
    desc = item.data(DescriptionRole)
    create = desc.viewclass.createinstance
    prefix = unique_prefix(desc.viewclass, rownum)
    form = item.data(ParametersRole).get('form', None)
    return create(prefix=prefix, form=form)


def prepare_params(item, model):
    desc = item.data(DescriptionRole)
    translate_hints = desc.viewclass.translate_hints
    editor_params = item.data(ParametersRole)
    all_hints = editor_params
    all_hints = translate_hints(all_hints)
    for name, hints in all_hints.items():
        model.set_param_hint(name, **hints)
    params = model.make_params()
    return params


def create_composite_model(m_def):
    n = len(m_def)
    m_list = []
    parameters = Parameters()
    for i in range(n):
        item = m_def[i]
        m = create_model(item, i)
        p = prepare_params(item, m)
        m_list.append(m)
        parameters.update(p)

    model = None
    if m_list:
        model = reduce(lambda x, y: x + y, m_list)

    return model, parameters


class PeakPreviewRunner(PreviewRunner):

    def __init__(self, master):
        super().__init__(master=master)
        self.pool = pebble.ProcessPool(max_workers=1)
        self.preview_model_result = None

    def on_exception(self, ex: Exception):
        try:
            super().on_exception(ex)
        except ValueError:
            self.master.Error.preview(ex)

    def on_done(self, result):
        orig_data, after_data, model_result = result
        final_preview = self.preview_pos is None
        if final_preview:
            self.preview_data = orig_data
            self.after_data = after_data

        if self.preview_data is None:  # happens in OWIntegrate
            self.preview_data = orig_data

        self.preview_model_result = model_result

        self.master.curveplot.set_data(self.preview_data)
        self.master.curveplot_after.set_data(self.after_data)

        self.show_image_info(final_preview)

        self.preview_updated.emit()

    def show_preview(self, show_info_anyway=False):
        """ Shows preview and also passes preview data to the widgets """
        master = self.master
        self.preview_pos = master.flow_view.preview_n()
        self.last_partial = None
        self.show_info_anyway = show_info_anyway
        self.preview_data = None
        self.after_data = None
        pp_def = [master.preprocessormodel.item(i)
                  for i in range(master.preprocessormodel.rowCount())]
        if master.data is not None:
            # Clear markings to indicate preview is running
            refresh_integral_markings([], master.markings_list, master.curveplot)
            data = master.sample_data(master.data)
            # Pass preview data to widgets here as we don't use on_partial_result()
            for w in self.master.flow_view.widgets():
                w.set_preview_data(data)
            self.start(self.run_preview, data, pp_def, self.pool)
        else:
            master.curveplot.set_data(None)
            master.curveplot_after.set_data(None)

    def shutdown(self):
        super().shutdown()
        self.pool.stop()
        self.pool.join()

    @staticmethod
    def run_preview(data: Table,
                    m_def, pool, state: TaskState):

        def progress_interrupt(_: float):
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for _ in range(10):
            time.sleep(0.010)
            progress_interrupt(0)

        orig_data = data

        model, parameters = create_composite_model(m_def)


        model_result = {}
        x = getx(data)
        if data is not None and model is not None:

            for row in data:
                progress_interrupt(0)
                res = pool.schedule(pool_fit2, (row.x, model.dumps(), parameters, x))
                while not res.done():
                    try:
                        progress_interrupt(0)
                    except InterruptException:
                        # CANCEL
                        if multiprocessing.get_start_method() != "fork" and res.running():
                            # If slower start methods are used, give the current computation
                            # some time to exit gracefully; this avoids reloading processes
                            concurrent.futures.wait([res], 1.0)
                        if not res.done():
                            res.cancel()
                        raise
                    concurrent.futures.wait([res], 0.05)
                fits = res.result()
                model_result[row.id] = ModelResult(model, parameters).loads(fits)

        progress_interrupt(0)
        return orig_data, data, model_result


class OWPeakFit(SpectralPreprocess):
    name = "Peak Fit"
    description = "Fit peaks to spectral region"
    icon = "icons/peakfit.svg"
    priority = 1020
    settings_version = 2

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add model..."

    class Inputs(SpectralPreprocess.Inputs):
        data_subset = Input("Data Subset", Table)

    class Outputs:
        fit_params = Output("Fit Parameters", Table, default=True)
        fits = Output("Fits", Table)
        residuals = Output("Residuals", Table)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Warning(SpectralPreprocess.Warning):
        subset_not_subset = Msg(
            "Subset data contains some instances that do not appear in "
            "input data")
        subset_independent = Msg(
            "No subset data instances appear in input data")

    preview_on_image = True

    def __init__(self):
        self.markings_list = []
        super().__init__()
        self.subset_data = None
        self.subset_indices = None
        self._invalidated = False
        self.preview_runner = PeakPreviewRunner(self)
        self.curveplot.selection_type = SELECTONE
        self.curveplot.select_at_least_1 = True
        self.curveplot.view_average_menu.setEnabled(False)
        self.curveplot.selection_changed.connect(self.redraw_integral)
        self.preview_runner.preview_updated.connect(self.redraw_integral)
        # GUI
        # box = gui.widgetBox(self.controlArea, "Options")

    @Inputs.data
    @check_sql_input
    def set_data(self, data=None):
        self.data = data
        self._invalidated = True

    @Inputs.data_subset
    @check_sql_input
    def set_subset_data(self, subset):
        self.subset_data = subset

    def handleNewSignals(self):
        self._handle_subset_data()
        if self._invalidated:
            self._invalidated = False
            super().handleNewSignals()
        else:
            self.show_preview(show_info_anyway=True)

    # OWDataProjectionWidget
    def _handle_subset_data(self):
        self.Warning.subset_independent.clear()
        self.Warning.subset_not_subset.clear()
        if self.data is None or self.subset_data is None:
            self.subset_indices = set()
        else:
            self.subset_indices = set(self.subset_data.ids)
            ids = set(self.data.ids)
            if not self.subset_indices & ids:
                self.Warning.subset_independent()
            elif self.subset_indices - ids:
                self.Warning.subset_not_subset()

    def get_subset_mask(self):
        if not self.subset_indices:
            return None
        return np.fromiter((ex.id in self.subset_indices for ex in self.data),
                           dtype=bool, count=len(self.data))

    def sample_data(self, data):
        subset = self.get_subset_mask()
        if subset is not None:
            return data[subset][:self.preview_curves]
        return super().sample_data(data)

    def redraw_integral(self):
        dis = []
        if self.curveplot.data:
            x = np.sort(getx(self.curveplot.data))
            previews = self.flow_view.preview_n()
            for i in range(self.preprocessormodel.rowCount()):
                if i in previews:
                    item = self.preprocessormodel.item(i)
                    m = create_model(item, i)
                    p = prepare_params(item, m)
                    # Show initial fit values for now
                    init = np.atleast_2d(np.broadcast_to(m.eval(p, x=x), x.shape))
                    di = [("curve", (x, init, INTEGRATE_DRAW_BASELINE_PENARGS))]
                    color = self.flow_view.preview_color(i)
                    dis.append({"draw": di, "color": color})
        result = None
        if np.any(self.curveplot.selection_group) and self.curveplot.data \
                and self.preview_runner.preview_model_result:
            # select result
            ind = np.flatnonzero(self.curveplot.selection_group)[0]
            row_id = self.curveplot.data[ind].id
            result = self.preview_runner.preview_model_result.get(row_id, None)
        if result is not None:
            # show total fit
            eval = np.atleast_2d(np.broadcast_to(result.eval(x=x), x.shape))
            di = [("curve", (x, eval, INTEGRATE_DRAW_CURVE_PENARGS))]
            dis.append({"draw": di, "color": 'red'})
            # show components
            eval_comps = result.eval_components(x=x)
            for i in range(self.preprocessormodel.rowCount()):
                item = self.preprocessormodel.item(i)
                prefix = unique_prefix(item.data(DescriptionRole).viewclass, i)
                comp = eval_comps.get(prefix, None)
                if comp is not None:
                    comp = np.atleast_2d(np.broadcast_to(comp, x.shape))
                    di = [("curve", (x, comp, INTEGRATE_DRAW_CURVE_PENARGS))]
                    color = self.flow_view.preview_color(i)
                    dis.append({"draw": di, "color": color})

        refresh_integral_markings(dis, self.markings_list, self.curveplot)

    def create_outputs(self):
        m_def = [self.preprocessormodel.item(i) for i in range(self.preprocessormodel.rowCount())]
        self.start(self.run_task, self.data, m_def)

    @staticmethod
    def run_task(data: Table, m_def, state: TaskState):

        def progress_interrupt(i: float):
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for _ in range(10):
            time.sleep(0.010)
            progress_interrupt(0)

        model, parameters = create_composite_model(m_def)

        data_fits = data_anno = data_resid = None
        if data is not None and model is not None:
            orig_data = data
            output = []
            x = getx(data)
            n = len(data)
            fits = []
            residuals = []

            with multiprocessing.Pool(processes=N_PROCESSES,
                                      initializer=pool_initializer,
                                      initargs=(model.dumps(), parameters, x)) as p:
                res = p.map_async(pool_fit, data.X, chunksize=1)

                def done():
                    try:
                        return n - res._number_left * res._chunksize
                    except AttributeError:
                        return 0

                while not res.ready():
                    progress_interrupt(done() / n * 99)
                    res.wait(0.05)

                fitsr = res.get()

            progress_interrupt(99)

            for fit, bpar, fitted, resid in fitsr:
                out = ModelResult(model, parameters).loads(fit)
                output.append(bpar)
                fits.append(fitted)
                residuals.append(resid)
                progress_interrupt(99)
            data = fit_results_table(np.vstack(output), out, orig_data)
            data_fits = orig_data.from_table_rows(orig_data, ...)  # a shallow copy
            with data_fits.unlocked_reference(data_fits.X):
                data_fits.X = np.vstack(fits)
            data_resid = orig_data.from_table_rows(orig_data, ...)  # a shallow copy
            with data_resid.unlocked_reference(data_resid.X):
                data_resid.X = np.vstack(residuals)
            dom_anno = Domain(orig_data.domain.attributes,
                              orig_data.domain.class_vars,
                              orig_data.domain.metas + data.domain.attributes,
                              )
            data_anno = orig_data.transform(dom_anno)
            with data_anno.unlocked(data_anno.metas):
                data_anno.metas[:, len(orig_data.domain.metas):] = data.X

        progress_interrupt(100)

        return data, data_fits, data_resid, data_anno

    def on_done(self, results):
        fit_params, fits, residuals, annotated_data = results
        self.Outputs.fit_params.send(fit_params)
        self.Outputs.fits.send(fits)
        self.Outputs.residuals.send(residuals)
        self.Outputs.annotated_data.send(annotated_data)

    def on_exception(self, ex):
        try:
            super().on_exception(ex)
        except ValueError:
            self.Error.applying(ex)
            self.Outputs.fit_params.send(None)
            self.Outputs.fits.send(None)
            self.Outputs.residuals.send(None)
            self.Outputs.annotated_data.send(None)

    @classmethod
    def migrate_preprocessor(cls, preprocessor, version):
        name, settings = preprocessor
        settings = settings.copy()
        if version < 2:
            for n, h in settings.items():
                if isinstance(h, dict):
                    h = h.copy()
                    set_default_vary(h)
                    settings[n] = h
            version = 2
        return [((name, settings), version)]


if __name__ == "__main__":  # pragma: no cover
    data = Cut(lowlim=1360, highlim=1700)(Table("collagen")[0:3])
    WidgetPreview(OWPeakFit).run(data)
