from AnyQt.QtWidgets import QVBoxLayout, QFormLayout
from Orange.widgets import gui
from orangecontrib.spectroscopy.widgets.gui import lineEditDecimalOrNone
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.preprocess.als import ALSP, ARPLS, AIRPLS


class ALSEditor(BaseEditorOrange):
    """
    Asymmetric least squares subtraction.
    """

    name = "Asymmetric Least Squares Smoothing"
    qualname = "preprocessors.ALS"

    ALS_TYPE = 0
    LAM = 1e6
    ITERMAX = 10
    P = 0.1
    RATIO = 0.05
    PORDER = 1

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())
        form = QFormLayout()
        self.als_type = self.ALS_TYPE
        self.lam = self.LAM
        self.itermax = self.ITERMAX
        self.p = self.P
        self.ratio = self.RATIO
        self.porder = self.PORDER
        self.alst_combo = gui.comboBox(
            None,
            self,
            "als_type",
            items=[
                "Asymmetric",
                "Asymmetrically Reweighted",
                "Adaptive Iteratively Reweighted",
            ],
            callback=self.edited.emit,
        )
        self.controlArea.layout().addLayout(form)
        self.itermaxspin = gui.spin(
            None,
            self,
            "itermax",
            label="Max. iterations",
            minv=1,
            maxv=100000,
            controlWidth=100,
            callback=self.edited.emit,
        )
        self.lamspin = lineEditDecimalOrNone(
            None, self, value="lam", bottom=0, callback=self.edited.emit
        )
        form.addRow("ALS Type", self.alst_combo)
        form.addRow("Smoothing Constant", self.lamspin)
        form.addRow('Max. Iterations', self.itermaxspin)
        self.palsspin = lineEditDecimalOrNone(
            None, self, value="p", bottom=0.000000001, top=1, callback=self.edited.emit
        )
        self.palsspin.setToolTip(
            "0.5 = symmetric, <0.5: negative deviations are more strongly suppressed"
        )
        self.ratior = lineEditDecimalOrNone(
            None,
            self,
            value="ratio",
            bottom=0.000000001,
            top=1,
            callback=self.edited.emit,
        )
        self.ratior.setToolTip(
            "0 < ratio < 1, smaller values allow less negative values"
        )
        self.porderairplsspin = gui.spin(
            None,
            self,
            "porder",
            label="Order of the difference of penalties (Adaptive)",
            minv=1,
            maxv=100,
            controlWidth=100,
            step=1,
            callback=self.edited.emit,
        )
        form.addRow('Weighting Deviations', self.palsspin)
        form.addRow('Weighting Deviations', self.ratior)
        form.addRow('Penalties Order', self.porderairplsspin)

        self.user_changed = False

        self._adapt_ui()

    def setParameters(self, params):
        if params:
            self.user_changed = True
        self.als_type = params.get("als_type", self.ALS_TYPE)
        self.lam = params.get("lam", self.LAM)
        self.itermax = params.get("itermax", self.ITERMAX)
        self.p = params.get('p', self.P)
        self.porder = params.get('porder', self.PORDER)
        self.ratio = params.get('ratio', self.RATIO)
        self._adapt_ui()

    def _adapt_ui(self):
        self.palsspin.setEnabled(self.als_type == 0)
        self.ratior.setEnabled(self.als_type == 1)
        self.porderairplsspin.setEnabled(self.als_type == 2)

    def parameters(self):
        parameters = super().parameters()
        return parameters

    @classmethod
    def createinstance(cls, params):
        als_type = params.get("als_type", cls.ALS_TYPE)
        lam = float(params.get("lam", cls.LAM))
        itermax = params.get("itermax", cls.ITERMAX)
        pals = params.get('p', cls.P)
        ratioarpls = params.get('ratio', cls.RATIO)
        porderairpls = params.get('porder', cls.PORDER)

        if als_type == 0:
            return ALSP(lam=lam, itermax=itermax, p=float(pals))
        elif als_type == 1:
            return ARPLS(lam=lam, ratio=float(ratioarpls), itermax=itermax)
        elif als_type == 2:
            return AIRPLS(lam=lam, itermax=itermax, porder=porderairpls)
        else:
            raise Exception("unknown baseline type")


preprocess_editors.register(ALSEditor, 625)
