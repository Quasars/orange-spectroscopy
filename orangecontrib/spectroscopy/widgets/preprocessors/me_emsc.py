import numpy as np

from AnyQt.QtWidgets import QVBoxLayout, QLabel, QFormLayout, QBoxLayout

from Orange.widgets import gui
from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    REFERENCE_DATA_PARAM,
    BaseEditorOrange,
)
from orangecontrib.spectroscopy.widgets.preprocessors.emsc import EMSCEditor


class MeEMSCEditor(EMSCEditor):
    name = "ME-EMSC"
    qualname = "orangecontrib.spectroscopy.preprocess.me_emsc.me_emsc"

    MAX_ITER_DEFAULT = 30
    FIXED_ITER_DEFAULT = False
    OUTPUT_MODEL_DEFAULT = False
    NCOMP_DEFAULT = 7
    AUTOSET_NCOMP_DEFAULT = True
    N0_LOW = 1.1  # real part of the refractive index
    N0_HIGH = 1.4
    A_LOW = 2  # radius of the spherical sample
    A_HIGH = 7.1

    def __init__(self, parent=None, **kwargs):
        BaseEditorOrange.__init__(self, parent, **kwargs)

        self.scaling = True  # FIXME: for the reference display to work. Why?

        self.controlArea.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None

        self.max_iter = self.MAX_ITER_DEFAULT
        self.fixed_iter = self.FIXED_ITER_DEFAULT
        self.ncomp = self.NCOMP_DEFAULT
        self.autoset_ncomp = self.AUTOSET_NCOMP_DEFAULT
        self.n0_low = self.N0_LOW
        self.n0_high = self.N0_HIGH
        self.a_low = self.A_LOW
        self.a_high = self.A_HIGH

        gui.spin(
            self.controlArea,
            self,
            "max_iter",
            label="Max iterations",
            minv=0,
            maxv=100,
            controlWidth=50,
            callback=self.edited.emit,
        )
        gui.checkBox(
            self.controlArea,
            self,
            "fixed_iter",
            label="Always perform max iterations",
            callback=self.edited.emit,
        )

        self.comp_spin = gui.spin(
            self.controlArea,
            self,
            "ncomp",
            label="Components",
            minv=3,
            maxv=15,
            controlWidth=50,
            callback=self.edited.emit,
        )
        gui.checkBox(
            self.controlArea,
            self,
            "autoset_ncomp",
            label="Automatically set components",
            callback=lambda: (self._auto_click(), self.edited.emit()),
        )

        form_set = QFormLayout()
        self.controlArea.layout().addLayout(form_set)

        bint = QBoxLayout(QBoxLayout.LeftToRight)
        low = lineEditFloatRange(
            self, self, "n0_low", bottom=1.1, top=3, callback=self.edited.emit
        )
        low.sizeHintFactor = 0.4
        bint.addWidget(low)
        high = lineEditFloatRange(
            self, self, "n0_high", bottom=1.1, top=3, callback=self.edited.emit
        )
        high.sizeHintFactor = 0.4
        bint.addWidget(high)
        form_set.addRow("Refractive index", bint)

        bint = QBoxLayout(QBoxLayout.LeftToRight)
        low = lineEditFloatRange(
            self, self, "a_low", bottom=2, top=50, callback=self.edited.emit
        )
        low.sizeHintFactor = 0.4
        bint.addWidget(low)
        high = lineEditFloatRange(
            self, self, "a_high", bottom=2, top=50, callback=self.edited.emit
        )
        high.sizeHintFactor = 0.4
        bint.addWidget(high)
        form_set.addRow("Spherical radius", bint)

        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.output_model = self.OUTPUT_MODEL_DEFAULT
        gui.checkBox(
            self.controlArea,
            self,
            "output_model",
            "Output EMSC model",
            callback=self.edited.emit,
        )

        self._auto_click()
        self._init_regions()
        self._init_reference_curve()

        self.user_changed = False

    def _auto_click(self):
        self.comp_spin.setEnabled(not self.autoset_ncomp)

    def setParameters(self, params):
        if params:
            self.user_changed = True

        self.max_iter = params.get("max_iter", self.MAX_ITER_DEFAULT)
        self.output_model = params.get("output_model", self.OUTPUT_MODEL_DEFAULT)
        self.fixed_iter = params.get("fixed_iter", self.FIXED_ITER_DEFAULT)
        self.ncomp = params.get("ncomp", self.NCOMP_DEFAULT)
        self.autoset_ncomp = params.get("autoset_ncomp", self.AUTOSET_NCOMP_DEFAULT)
        self.n0_low = params.get("n0_low", self.N0_LOW)
        self.n0_high = params.get("n0_high", self.N0_HIGH)
        self.a_low = params.get("a_low", self.A_LOW)
        self.a_high = params.get("a_high", self.A_HIGH)

        self._set_range_parameters(params)

        self._auto_click()
        self.update_reference_info()
        self.update_weight_curve(params)

    @classmethod
    def createinstance(cls, params):
        max_iter = params.get("max_iter", cls.MAX_ITER_DEFAULT)
        output_model = params.get("output_model", cls.OUTPUT_MODEL_DEFAULT)
        fixed_iter = params.get("fixed_iter", cls.FIXED_ITER_DEFAULT)
        ncomp = params.get("ncomp", cls.NCOMP_DEFAULT)
        autoset_ncomp = params.get("autoset_ncomp", cls.AUTOSET_NCOMP_DEFAULT)
        n0_low = float(params.get("n0_low", cls.N0_LOW))
        n0_high = float(params.get("n0_high", cls.N0_HIGH))
        a_low = float(params.get("a_low", cls.A_LOW))
        a_high = float(params.get("a_high", cls.A_HIGH))

        if fixed_iter:
            fixed_iter = max_iter
        ncomp = False if autoset_ncomp else ncomp
        n0 = np.linspace(n0_low, n0_high, 10)
        a = np.linspace(a_low, a_high, 10)

        weights = cls._compute_weights(params)

        reference = params.get(REFERENCE_DATA_PARAM, None)
        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return ME_EMSC(
                reference=reference,
                weights=weights,
                max_iter=max_iter,
                fixed_iter=fixed_iter,
                ncomp=ncomp,
                n0=n0,
                a=a,
                output_model=output_model,
            )


preprocess_editors.register(MeEMSCEditor, 325)
