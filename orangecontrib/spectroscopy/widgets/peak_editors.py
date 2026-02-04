import copy
import sys

import lmfit
import numpy as np
from AnyQt.QtCore import Signal
from Orange.widgets.data.utils.preprocess import blocked
from AnyQt.QtCore import QSize, QObject
from AnyQt.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QSizePolicy,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QLabel,
    QAbstractSpinBox,
)
from orangewidget.widget import Msg

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.gui import MovableVline
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    SetXDoubleSpinBox,
    BaseEditorOrange,
)


DEFAULT_DELTA = 1
DEFAULT_VALUE = 0
UNUSED_VALUE = float('-inf')


class CompactDoubleSpinBox(SetXDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, buttonSymbols=QAbstractSpinBox.NoButtons)

    def sizeHint(self) -> QSize:
        sh = super().sizeHint()
        sh.setWidth(int(sh.width() / 2))
        return sh

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class ParamHintBox(QWidget):
    """
    Box to interact with lmfit parameter hints

    Args:
        hints (dict): parameter hints (in internal format) for parameter given by 'name'
    """

    valueChanged = Signal()
    editingFinished = Signal(QObject)
    focus_in = None

    def __init__(self, hints, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        self.hints = hints
        assert 'vary' in self.hints

        minf, maxf = -sys.float_info.max, sys.float_info.max

        self._defaults = {
            'min': UNUSED_VALUE,
            'value': DEFAULT_VALUE,
            'max': UNUSED_VALUE,
            'delta': DEFAULT_DELTA,
            'expr': '',
        }

        self.min_e = CompactDoubleSpinBox(
            minimum=UNUSED_VALUE,
            maximum=maxf,
            singleStep=0.5,
            value=self.hints.get('min', self._defaults["min"]),
            specialValueText="None",
        )
        self.val_e = CompactDoubleSpinBox(
            minimum=minf,
            maximum=maxf,
            singleStep=0.5,
            value=self.hints.get('value', self._defaults["value"]),
        )
        self.max_e = CompactDoubleSpinBox(
            minimum=UNUSED_VALUE,
            maximum=maxf,
            singleStep=0.5,
            value=self.hints.get('max', self._defaults["min"]),
            specialValueText="None",
        )
        self.delta_e = CompactDoubleSpinBox(
            minimum=minf,
            maximum=maxf,
            singleStep=0.5,
            value=self.hints.get('delta', self._defaults["delta"]),
            prefix="±",
            visible=False,
        )
        self.vary_e = QComboBox()
        v_opt = (
            ('fixed', 'limits', 'delta', 'expr')
            if 'expr' in self.hints
            else ('fixed', 'limits', 'delta')
        )
        self.vary_e.insertItems(0, v_opt)
        with blocked(self.vary_e):
            self.vary_e.setCurrentText(self.hints['vary'])

        self.expr_e = QLineEdit(
            visible=False, enabled=False, text=self.hints.get('expr', "")
        )

        self.edits = [
            ("min", self.min_e),
            ("value", self.val_e),
            ("max", self.max_e),
            ("delta", self.delta_e),
            ("vary", self.vary_e),
            ("expr", self.expr_e),
        ]

        for name, widget in self.edits[:4]:  # float fields
            widget.valueChanged[float].connect(
                lambda x, name=name: self._changed_float(x, name)
            )
        self.vary_e.currentTextChanged.connect(self._changed_vary)
        self.expr_e.textChanged.connect(self._changed_expr)

        for _name, widget in self.edits:
            layout.addWidget(widget)
            widget.focusIn = self.focusIn

        self.setValues()

    def _change(self, v, name):
        if v != self.hints.get(name, None):
            self.hints[name] = v
            return True
        else:
            return False

    def _change_and_notify(self, v, name):
        changed = self._change(v, name)
        if changed:
            self.valueChanged.emit()

    def update_min_max_for_delta(self):
        vary = self.vary_e.currentText()
        if vary == "delta":
            v = self.hints.get("value", self._defaults["value"])
            self._change(v - self.hints.get("delta", self._defaults["delta"]), "min")
            self._change(v + self.hints.get("delta", self._defaults["delta"]), "max")
            self.setValues()  # update UI, no need for signal (min and max are unused in delta)

    def _changed_float(self, v, name):
        self._change_and_notify(v, name)
        if name in ["value", "delta"]:
            self.update_min_max_for_delta()

    def _changed_vary(self):
        v = self.vary_e.currentText()
        self.update_gui()
        self._change_and_notify(v, "vary")
        self.update_min_max_for_delta()

    def _changed_expr(self):
        v = self.expr_e.text()
        self._change_and_notify(v, "expr")

    def focusIn(self):
        """Call custom method on focus if present"""
        if self.focus_in is not None:
            self.focus_in()

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)

    def setValues(self):
        expr = self.hints.get('expr', self._defaults['expr'])
        vary = self.hints['vary']

        for name, widget in self.edits[:4]:  # floating point elements
            v = self.hints.get(name, self._defaults[name])
            with blocked(widget):
                widget.setValue(v)

        with blocked(self.vary_e):
            self.vary_e.setCurrentText(vary)

        with blocked(self.expr_e):
            self.expr_e.setText(expr)

        self.update_gui()

    def update_gui(self):
        vary = self.vary_e.currentText()
        self.min_e.setVisible(vary in ('limits', 'fixed', 'delta'))
        self.min_e.setEnabled(vary not in ('fixed', 'delta'))
        self.val_e.setVisible(vary != 'expr')
        self.max_e.setVisible(vary in ('limits', 'fixed'))
        self.max_e.setEnabled(vary != 'fixed')
        self.delta_e.setVisible(vary == 'delta')
        self.expr_e.setVisible(vary == 'expr')


def set_default_vary(h):
    # Set vary corresponding to the defaults:
    expr = h.get("expr", None)
    vary = h.get("vary", None)
    # If vary is not defined and expression is given, use it
    if expr is not None and expr != "" and vary is None:
        h["vary"] = "expr"
    elif vary is False:
        h["vary"] = "fixed"
    else:
        h["vary"] = "limits"
    return h


class ModelEditor(BaseEditorOrange):
    # Adapted from IntegrateOneEditor

    class Warning(BaseEditorOrange.Warning):
        out_of_range = Msg("{} out of range.")

    _defaults = None  # model defaults are stored here

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QGridLayout()
        layout.setSpacing(2)
        self.controlArea.setLayout(layout)

        self.__values = {}
        self.__editors = {}
        self.__lines = {}

        for row, name in enumerate(self.model_parameters()):
            h = copy.deepcopy(self.defaults().get(name, {}))
            set_default_vary(h)
            self.__values[name] = h

            e = ParamHintBox(h)
            e.focus_in = self.activateOptions

            def change_hint(name=name):
                self.edited.emit()
                self.changed_param_hints(name)

            e.valueChanged.connect(change_hint)
            self.__editors[name] = e
            layout.addWidget(QLabel(name), row, 0)
            layout.addWidget(e, row, 1)

            if name in self.model_lines():
                l = MovableVline(position=0.0, label=name)

                def change_value(_, line=l, name=name):
                    self.set_hint(name, "value", float(line.rounded_value()))
                    self.__editors[name].update_min_max_for_delta()
                    self.edited.emit()

                l.sigMoved.connect(change_value)
                self.__lines[name] = l

        self.focusIn = self.activateOptions
        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        self.parent_widget.redraw_integral()
        for l in self.__lines.values():
            if l not in self.parent_widget.curveplot.markings:
                l.report = self.parent_widget.curveplot
                self.parent_widget.curveplot.add_marking(l)

    def changed_param_hints(self, name, user=True):
        if user:
            self.user_changed = True
        self.__editors[name].setValues()
        l = self.__lines.get(name, None)
        h = self.__values[name]
        if l is not None and 'value' in h:
            l.setValue(h['value'])
        self.changed.emit()

    def set_hint(self, name, k, v):
        self.__values[name][k] = v
        self.changed_param_hints(name)

    def set_form(self, form):
        self.__values.update(form=form)
        self.edited.emit()

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        for name in self.model_parameters():
            # change contents within the same dictionary because the editor has the reference
            default = copy.deepcopy(self.defaults().get(name, {}))
            default = set_default_vary(default)
            nparams = copy.deepcopy(params.get(name, default))
            self.__values[name].clear()
            self.__values[name].update(nparams)
            self.changed_param_hints(name, user=False)

    def parameters(self):
        # need to copy.deepcopy to get on_modelchanged signal
        return copy.deepcopy(self.__values)

    @classmethod
    def createinstance(cls, prefix, form=None):
        if form is not None:
            return cls.model(prefix=prefix, form=form)
        return cls.model(prefix=prefix)

    def set_preview_data(self, data):
        self.Warning.out_of_range.clear()
        if data:
            xs = getx(data)
            if len(xs):
                minx = np.min(xs)
                maxx = np.max(xs)
                limits = [
                    (name, self.__values.get(name, {})) for name in self.model_lines()
                ]
                for name, h in limits:
                    v = h.get('value', None)
                    if v is not None and (v < minx or v > maxx):
                        self.parent_widget.Warning.preprocessor()
                        self.Warning.out_of_range(name)

    @classmethod
    def defaults(cls):
        if cls._defaults is None:
            m = cls.model()
            for name, value in m.def_vals.items():
                m.set_param_hint(name, value=value)
            cls._defaults = m.param_hints
        return cls._defaults

    @staticmethod
    def model_parameters():
        """
        Returns a tuple of Parameter names for the model which should be editable
        """
        raise NotImplementedError

    @staticmethod
    def model_lines():
        """
        Returns a tuple of model_parameter names that should have visualized selection lines
        """
        raise NotImplementedError

    @classmethod
    def translate(cls, name, hints):
        hints = hints.copy()
        defaults = cls.defaults()[name]

        vary = hints["vary"]
        delta = hints.get("delta", DEFAULT_DELTA)
        value = hints.get("value", DEFAULT_VALUE)

        # special delta case
        if vary == 'delta' and delta == 0:
            vary = 'fixed'

        if vary == 'delta':
            hints["min"] = value - delta
            hints["max"] = value + delta
        elif vary == 'limits':
            hints["value"] = value
        elif vary == 'expr':
            pass
        elif vary == 'fixed':
            pass
        else:
            raise Exception("Invalid vary")

        hints.pop('delta', None)

        # vary is implied False by 'expr' hint
        if vary == 'fixed':
            hints['vary'] = False
        else:
            hints.pop('vary', None)

        # Set expr to "" if default expr should be overridden
        if 'expr' in defaults and vary != 'expr':
            hints['expr'] = ""
        else:
            hints.pop('expr', None)

        # Avoid collecting unchanged hints, -inf corresponds to the special value
        if 'min' in hints and hints['min'] == defaults.get('min', UNUSED_VALUE):
            hints.pop('min', None)
        if 'max' in hints and hints['max'] == defaults.get('max', UNUSED_VALUE):
            hints.pop('max', None)

        # Only send default if expr selected, Parameter respects bounds even if expr is set
        if vary == 'expr':
            hints = defaults.copy()

        # Exclude 'expr' hints unless setting to "" to disable default
        #   Otherwise expression has variable references which are missing prefixes
        if hints.get('expr', "") != "":
            hints.pop('expr', None)

        return hints

    @classmethod
    def translate_hints(cls, all_hints):
        out = {}
        for name, hints in all_hints.items():
            # Exclude model init keyword 'form'
            if name != 'form':
                hints = cls.translate(name, hints)
                out[name] = hints
        return out


class PeakModelEditor(ModelEditor):
    category = "Peak"
    icon = "Normalize.svg"

    @staticmethod
    def model_parameters():
        return 'center', 'amplitude', 'sigma'

    @staticmethod
    def model_lines():
        return ('center',)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_hint('center', 'value', x[int(len(x) / 2)])
                self.edited.emit()
        super().set_preview_data(data)


class GaussianModelEditor(PeakModelEditor):
    name = "Gaussian"
    model = lmfit.models.GaussianModel
    prefix_generic = "g"


class LorentzianModelEditor(PeakModelEditor):
    name = "Lorentzian"
    model = lmfit.models.LorentzianModel
    prefix_generic = "l"


class SplitLorentzianModelEditor(PeakModelEditor):
    name = "Split Lorentzian"
    model = lmfit.models.SplitLorentzianModel
    prefix_generic = "sl"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('sigma_r',)


class VoigtModelEditor(PeakModelEditor):
    name = "Voigt"
    model = lmfit.models.VoigtModel
    prefix_generic = "v"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class PseudoVoigtModelEditor(PeakModelEditor):
    name = "pseudo-Voigt"
    model = lmfit.models.PseudoVoigtModel
    prefix_generic = "pv"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('fraction',)


class MoffatModelEditor(PeakModelEditor):
    name = "Moffat"
    model = lmfit.models.MoffatModel
    prefix_generic = "m"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('beta',)


class Pearson7ModelEditor(PeakModelEditor):
    name = "Pearson VII"
    model = lmfit.models.Pearson7Model
    prefix_generic = "ps"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('expon',)


class StudentsTModelEditor(PeakModelEditor):
    name = "Student's t"
    model = lmfit.models.StudentsTModel
    prefix_generic = "st"


class BreitWignerModelEditor(PeakModelEditor):
    name = "Breit-Wigner-Fano"
    model = lmfit.models.BreitWignerModel
    prefix_generic = "bwf"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('q',)


class LognormalModelEditor(PeakModelEditor):
    name = "Log-normal"
    model = lmfit.models.LognormalModel
    prefix_generic = "ln"


class DampedOscillatorModelEditor(PeakModelEditor):
    name = "Damped Harmonic Oscillator Amplitude"
    description = "Damped Harm. Osc. Amplitude"
    model = lmfit.models.DampedOscillatorModel
    prefix_generic = "do"


class DampedHarmOscillatorModelEditor(PeakModelEditor):
    name = "Damped Harmonic Oscillator (DAVE)"
    description = "Damped Harm. Osc. (DAVE)"
    model = lmfit.models.DampedHarmonicOscillatorModel
    prefix_generic = "dod"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class ExponentialGaussianModelEditor(PeakModelEditor):
    name = "Exponential Gaussian"
    model = lmfit.models.ExponentialGaussianModel
    prefix_generic = "eg"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class SkewedGaussianModelEditor(PeakModelEditor):
    name = "Skewed Gaussian"
    model = lmfit.models.SkewedGaussianModel
    prefix_generic = "sg"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class SkewedVoigtModelEditor(PeakModelEditor):
    name = "Skewed Voigt"
    model = lmfit.models.SkewedVoigtModel
    prefix_generic = "sv"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (
            'gamma',
            'skew',
        )


class ThermalDistributionModelEditor(PeakModelEditor):
    name = "Thermal Distribution"
    model = lmfit.models.ThermalDistributionModel
    prefix_generic = "td"

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(QLabel("Form: "))
        cb = QComboBox()
        cb.insertItems(0, self.model.valid_forms)
        cb.currentTextChanged.connect(self.set_form)
        layout.addWidget(cb)
        layout.addStretch()
        self.layout().insertLayout(0, layout)  # put at top of the editor

    @classmethod
    def model_parameters(cls):
        return super().model_parameters()[:2] + ('kt',)


class DoniachModelEditor(PeakModelEditor):
    name = "Doniach Sunjic"
    model = lmfit.models.DoniachModel
    prefix_generic = "d"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class BaselineModelEditor(ModelEditor):
    category = "Baseline"
    icon = "Continuize.svg"

    @staticmethod
    def model_parameters():
        """
        Returns a tuple of Parameter names for the model which should be editable
        """
        raise NotImplementedError

    @staticmethod
    def model_lines():
        return tuple()


class ConstantModelEditor(BaselineModelEditor):
    name = "Constant"
    model = lmfit.models.ConstantModel
    prefix_generic = "const"

    @staticmethod
    def model_parameters():
        return ('c',)


class LinearModelEditor(BaselineModelEditor):
    name = "Linear"
    model = lmfit.models.LinearModel
    prefix_generic = "lin"

    @staticmethod
    def model_parameters():
        return 'intercept', 'slope'


class QuadraticModelEditor(BaselineModelEditor):
    name = "Quadratic"
    model = lmfit.models.QuadraticModel
    prefix_generic = "quad"

    @staticmethod
    def model_parameters():
        return 'a', 'b', 'c'


class PolynomialModelEditor(BaselineModelEditor):
    name = "Polynomial"
    model = lmfit.models.PolynomialModel
    prefix_generic = "poly"

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(QLabel("Form: "))
        cb = QComboBox()
        cb.insertItems(0, tuple(str(vf) for vf in self.model.valid_forms))
        cb.currentTextChanged.connect(self.set_form)
        layout.addWidget(cb)
        layout.addStretch()
        self.layout().insertLayout(0, layout)  # put at top of the editor

    @classmethod
    def model_parameters(cls):
        return tuple(f"c{vf}" for vf in cls.model.valid_forms)
