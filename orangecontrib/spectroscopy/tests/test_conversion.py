import unittest

import numpy as np
import sklearn.model_selection as ms

import Orange
from Orange.classification import RandomForestLearner
from Orange.data import ContinuousVariable
from Orange.evaluation.testing import TestOnTestData
from Orange.evaluation.scoring import AUC
from Orange.data.table import DomainTransformationError

from orangecontrib.spectroscopy.preprocess import (
    Interpolate,
    Cut,
    SavitzkyGolayFiltering,
)
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.tests.util import smaller_data


learner = RandomForestLearner(random_state=42)

COLLAGEN = Orange.data.Table("collagen")
SMALL_COLLAGEN = smaller_data(COLLAGEN, 2, 2)


def separate_learn_test(data):
    sf = ms.ShuffleSplit(
        n_splits=1, test_size=0.2, random_state=np.random.RandomState(0)
    )
    ((traini, testi),) = sf.split(y=data.Y, X=data.X)
    return data[traini], data[testi]


def slightly_change_wavenumbers(data, change):
    natts = [ContinuousVariable(float(a.name) + change) for a in data.domain.attributes]
    ndomain = Orange.data.Domain(natts, data.domain.class_vars, metas=data.domain.metas)
    ndata = data.transform(ndomain)
    with ndata.unlocked():
        ndata.X = data.X
    return ndata


def odd_attr(data):
    natts = [a for i, a in enumerate(data.domain.attributes) if i % 2 == 0]
    ndomain = Orange.data.Domain(natts, data.domain.class_vars, metas=data.domain.metas)
    return data.transform(ndomain)


class TestConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collagen = SMALL_COLLAGEN

    def test_predict_same_domain(self):
        train, test = separate_learn_test(self.collagen)
        auc = AUC(TestOnTestData()(train, test, [learner]))
        self.assertGreater(auc, 0.9)  # easy dataset

    def test_predict_different_domain(self):
        train, test = separate_learn_test(self.collagen)
        test = Interpolate(points=getx(test) - 1)(test)  # other test domain
        with self.assertRaises(DomainTransformationError):
            learner(train)(test)

    def test_predict_different_domain_interpolation(self):
        train, test = separate_learn_test(self.collagen)
        aucorig = AUC(TestOnTestData()(train, test, [learner]))
        test = Interpolate(points=getx(test) - 1.0)(test)  # other test domain
        train = Interpolate(points=getx(train))(
            train
        )  # make train capable of interpolation
        aucshift = AUC(TestOnTestData()(train, test, [learner]))
        self.assertAlmostEqual(
            aucorig, aucshift, delta=0.01
        )  # shift can decrease AUC slightly
        test = Cut(1000, 1700)(test)
        auccut1 = AUC(TestOnTestData()(train, test, [learner]))
        test = Cut(1100, 1600)(test)
        auccut2 = AUC(TestOnTestData()(train, test, [learner]))
        test = Cut(1200, 1500)(test)
        auccut3 = AUC(TestOnTestData()(train, test, [learner]))
        # the more we cut the lower precision we get
        self.assertTrue(aucorig > auccut1 > auccut2 > auccut3)

    def test_predict_savgov_same_domain(self):
        data = SavitzkyGolayFiltering(window=9, polyorder=2, deriv=2)(self.collagen)
        train, test = separate_learn_test(data)
        auc = AUC(TestOnTestData()(train, test, [learner]))
        print(auc)
        self.assertGreater(auc, 0.85)

    def test_predict_savgol_another_interpolate(self):
        train, test = separate_learn_test(self.collagen)
        train = SavitzkyGolayFiltering(window=9, polyorder=2, deriv=2)(train)
        auc = AUC(TestOnTestData()(train, test, [learner]))
        train = Interpolate(points=getx(train))(train)
        aucai = AUC(TestOnTestData()(train, test, [learner]))
        self.assertAlmostEqual(auc, aucai, delta=0.02)
