import numpy as np

import Orange
import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.preprocess.utils import (
    SelectColumn,
    CommonDomainOrderUnknowns,
)
from orangecontrib.spectroscopy.preprocess.als.baseline import als, arpls, airpls


class ALSFeature(SelectColumn):
    InheritEq = True


class ALSCommon(CommonDomainOrderUnknowns):
    def __init__(self, lam, itermax, p, domain):
        super().__init__(domain)
        self.lam = lam
        self.itermax = itermax
        self.p = p

    def transformed(self, data, X):
        final = []
        data = np.array(data)
        if data.size > 0:
            for row in data:
                input_array = row
                final.append(
                    input_array
                    - als(input_array, lam=self.lam, p=self.p, itermax=self.itermax)
                )
            return np.array(final)
        else:
            return data


class ALSP(Preprocess):
    lam = 1e6
    itermax = 10
    p = 0.1

    def __init__(self, lam=lam, itermax=itermax, p=p):
        self.lam = lam
        self.itermax = itermax
        self.p = p

    def __call__(self, data):
        common = ALSCommon(self.lam, self.itermax, self.p, data.domain)
        atts = [
            a.copy(compute_value=ALSFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class ARPLSFeature(SelectColumn):
    InheritEq = True


class ARPLSCommon(CommonDomainOrderUnknowns):
    def __init__(self, lam, itermax, ratio, domain):
        super().__init__(domain)
        self.lam = lam
        self.itermax = itermax
        self.ratio = ratio

    def transformed(self, data, X):
        data = np.array(data)
        final = []
        if data.size > 0:
            for row in data:
                final.append(
                    row
                    - arpls(row, lam=self.lam, itermax=self.itermax, ratio=self.ratio)
                )
            return np.array(final)
        else:
            return data


class ARPLS(Preprocess):
    lam = 100e6
    itermax = 10
    ratio = 0.5

    def __init__(self, lam=lam, ratio=ratio, itermax=itermax):
        self.lam = lam
        self.itermax = itermax
        self.ratio = ratio

    def __call__(self, data):
        common = ARPLSCommon(self.lam, self.itermax, self.ratio, data.domain)
        atts = [
            a.copy(compute_value=ARPLSFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class AIRPLSFeature(SelectColumn):
    InheritEq = True


class AIRPLSCommon(CommonDomainOrderUnknowns):
    def __init__(self, lam, itermax, porder, domain):
        super().__init__(domain)
        self.lam = lam
        self.itermax = itermax
        self.porder = porder

    def transformed(self, data, X):
        data = np.array(data)
        final = []
        if data.size > 0:
            for row in data:
                final.append(
                    row
                    - airpls(
                        row, lam=self.lam, porder=self.porder, itermax=self.itermax
                    )
                )
            return np.array(final)

        else:
            return data


class AIRPLS(Preprocess):
    lam = 1e6
    itermax = 10
    porder = 1

    def __init__(self, lam=lam, itermax=itermax, porder=porder):
        self.lam = lam
        self.itermax = itermax
        self.porder = porder

    def __call__(self, data):
        common = AIRPLSCommon(self.lam, self.itermax, self.porder, data.domain)
        atts = [
            a.copy(compute_value=AIRPLSFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
