from Orange.modelling import SklFitter
from orangecontrib.spectroscopy.models.pls import PLSRegressionLearner


class PLSRegressionLearner(SklFitter):
    __fits__ = {'regression': PLSRegressionLearner}
