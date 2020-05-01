from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QSpinBox
import scipy.sparse as sp

from Orange.widgets import gui
from Orange.widgets.widget import Msg
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.spectroscopy.models.pls import PLSRegressionLearner


class OWPLS(OWBaseLearner):
    name = 'PLS'
    description = "Partial Least Squares Regression algorithm for multivariate data analysis"
    icon = "icons/SVM.svg"
    keywords = ["partial least squares"]

    LEARNER = PLSRegressionLearner

    class Outputs(OWBaseLearner.Outputs):
        coefsdata = Output("Coefficients", Table, explicit=True)

    class Warning(OWBaseLearner.Warning):
        sparse_data = Msg('Sparse input data: default preprocessing is to scale it.')

    #: number of components
    n_components = Setting(2)
    #: whether or not to limit number of iterations
    max_iter = Setting(500)

    def add_main_layout(self):

        self.optimization_box = gui.vBox(
            self.controlArea, "Optimization Parameters")
        self.ncomps_spin = gui.spin(
            self.optimization_box, self, "n_components", 1, 50, 1,
            label="Components: ",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed, spinType=QSpinBox)
        self.n_iters = gui.spin(
            self.optimization_box, self, "max_iter", 5, 1e6, 50,
            label="Iteration limit: ",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed,
            checkCallback=self.settings_changed, spinType=QSpinBox)

    def update_model(self):
        super().update_model()
        coef_table = None
        if self.model is not None:
            domain = Domain(
                [ContinuousVariable("coef")], metas=[StringVariable("name")])
            coefs = list(self.model.coefficients)
            waves = [[attr.name] for attr in self.model.domain.attributes]
            coef_table = Table.from_numpy(domain, X=list(coefs), metas=waves)
            coef_table.name = "coefficients"
        self.Outputs.coefsdata.send(coef_table)

    def set_data(self, data):
        self.Warning.sparse_data.clear()
        super().set_data(data)
        if self.data and sp.issparse(self.data.X):
            self.Warning.sparse_data()

    def handleNewSignals(self):
        self.apply()

    def create_learner(self):
        common_args = {'preprocessors': self.preprocessors}
        return PLSRegressionLearner(n_components=self.n_components,
                                    max_iter=self.max_iter,
                                    **common_args)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPLS).run(Table("housing"))
