import numpy as np

import Orange
from Orange.data import Table
from Orange.preprocess.preprocess import Preprocess
from Orange.data.util import get_unique_names

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess.utils import (
    SelectColumn,
    CommonDomainOrderUnknowns,
    interp1d_with_unknowns_numpy,
    MissingReferenceException,
    interpolate_extend_to,
    CommonDomainRef,
    table_eq_x,
    subset_for_hash,
)
from orangecontrib.spectroscopy.preprocess.npfunc import Function, Segments


class SelectionFunction(Function):
    """
    Weighted selection function. Includes min and max.
    """

    def __init__(self, min_, max_, w):
        super().__init__(None)
        self.min_ = min_
        self.max_ = max_
        self.w = w

    def __call__(self, x):
        seg = Segments(
            (lambda x: True, lambda x: 0),
            (
                lambda x: np.logical_and(x >= self.min_, x <= self.max_),
                lambda x: self.w,
            ),
        )
        return seg(x)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.min_ == other.min_
            and self.max_ == other.max_
            and self.w == other.w
        )

    def __hash__(self):
        return hash((super().__hash__(), self.min_, self.max_, self.w))


class SmoothedSelectionFunction(SelectionFunction):
    """
    Weighted selection function. Min and max points are middle
    points of smoothing with hyperbolic tangent.
    """

    def __init__(self, min_, max_, s, w):
        super().__init__(min_, max_, w)
        self.s = s

    def __call__(self, x):
        middle = (self.min_ + self.max_) / 2
        seg = Segments(
            (
                lambda x: x < middle,
                lambda x: (np.tanh((x - self.min_) / self.s) + 1) / 2 * self.w,
            ),
            (
                lambda x: x >= middle,
                lambda x: (-np.tanh((x - self.max_) / self.s) + 1) / 2 * self.w,
            ),
        )
        return seg(x)

    def __eq__(self, other):
        return super().__eq__(other) and self.s == other.s

    def __hash__(self):
        return hash((super().__hash__(), self.s))


def weighted_wavenumbers(weights, wavenumbers):
    """
    Return weights for the given wavenumbers. If weights are a data table,
    the weights are interpolated. If they are a npfunc.Function, the function is
    computed on the given wavenumbers.
    """
    if isinstance(weights, Function):
        return weights(wavenumbers).reshape(1, -1)
    elif weights:
        # interpolate reference to the data
        w = interp1d_with_unknowns_numpy(getx(weights), weights.X, wavenumbers)
        # set whichever weights are undefined (usually at edges) to zero
        w[np.isnan(w)] = 0
        return w
    else:
        w = np.ones((1, len(wavenumbers)))
        return w


class EMSCFeature(SelectColumn):
    InheritEq = True


class EMSCModel(SelectColumn):
    InheritEq = True


class _EMSC(CommonDomainOrderUnknowns, CommonDomainRef):
    def __init__(self, reference, badspectra, weights, order, scaling, domain):
        CommonDomainOrderUnknowns.__init__(self, domain)
        CommonDomainRef.__init__(self, reference, domain)
        assert len(self.reference) == 1
        self.badspectra = badspectra
        self.weights = weights
        self.order = order
        self.scaling = scaling

    def transformed(self, X, wavenumbers):
        # wavenumber have to be input as sorted
        # about 85% of time in __call__ function is spent is lstsq
        ref_X = interpolate_extend_to(self.reference, wavenumbers)
        wei_X = weighted_wavenumbers(self.weights, wavenumbers)

        N = wavenumbers.shape[0]
        m0 = -2.0 / (wavenumbers[0] - wavenumbers[N - 1])
        c_coeff = 0.5 * (wavenumbers[0] + wavenumbers[N - 1])

        n_badspec = len(self.badspectra) if self.badspectra is not None else 0
        if self.badspectra:
            badspectra_X = interpolate_extend_to(self.badspectra, wavenumbers)

        M = []
        for x in range(0, self.order + 1):
            M.append((m0 * (wavenumbers - c_coeff)) ** x)
        for y in range(0, n_badspec):
            M.append(badspectra_X[y])
        M.append(ref_X)  # always add reference spectrum to the model
        n_add_model = len(M)
        M = np.vstack(
            M
        ).T  # M is for the correction, for par. estimation M_weighted is used

        M_weighted = M * wei_X.T

        newspectra = np.zeros((X.shape[0], X.shape[1] + n_add_model))
        for i, rawspectrum in enumerate(X):
            rawspectrumW = (rawspectrum * wei_X)[0]  # noqa: F841
            m = np.linalg.lstsq(M_weighted, rawspectrum, rcond=-1)[0]
            corrected = rawspectrum

            for x in range(0, self.order + 1 + n_badspec):
                corrected = corrected - (m[x] * M[:, x])
            if self.scaling:
                corrected = corrected / m[self.order + 1 + n_badspec]
            corrected[np.isinf(corrected)] = np.nan  # fix values caused by zero weights
            corrected = np.hstack((corrected, m))  # append the model parameters
            newspectra[i] = corrected

        return newspectra

    def __eq__(self, other):
        return (
            CommonDomainRef.__eq__(self, other)
            and table_eq_x(self.badspectra, other.badspectra)
            and self.order == other.order
            and self.scaling == other.scaling
            and (
                self.weights == other.weights
                if not isinstance(self.weights, Table)
                else table_eq_x(self.weights, other.weights)
            )
        )

    def __hash__(self):
        domain = self.badspectra.domain if self.badspectra is not None else None
        fv = subset_for_hash(self.badspectra.X) if self.badspectra is not None else None
        weights = (
            self.weights
            if not isinstance(self.weights, Table)
            else subset_for_hash(self.weights.X)
        )
        return hash(
            (
                CommonDomainRef.__hash__(self),
                domain,
                fv,
                weights,
                self.order,
                self.scaling,
            )
        )


def average_table_x(data):
    return Orange.data.Table.from_numpy(
        Orange.data.Domain(data.domain.attributes), X=data.X.mean(axis=0, keepdims=True)
    )


class EMSC(Preprocess):
    def __init__(
        self,
        reference=None,
        badspectra=None,
        weights=None,
        order=2,
        scaling=True,
        output_model=False,
        ranges=None,
    ):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        if len(self.reference) > 1:
            self.reference = average_table_x(self.reference)
        self.badspectra = badspectra
        self.weights = weights
        self.order = order
        self.scaling = scaling
        self.output_model = output_model

    def __call__(self, data):
        # creates function for transforming data
        common = _EMSC(
            self.reference,
            self.badspectra,
            self.weights,
            self.order,
            self.scaling,
            data.domain,
        )
        # takes care of domain column-wise, by above transformation function
        atts = [
            a.copy(compute_value=EMSCFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        model_metas = []
        n_badspec = len(self.badspectra) if self.badspectra is not None else 0
        used_names = set(
            [var.name for var in data.domain.variables + data.domain.metas]
        )
        if self.output_model:
            i = len(data.domain.attributes)
            for o in range(self.order + 1):
                n = get_unique_names(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(
                        name=n, compute_value=EMSCModel(i, common)
                    )
                )
                i += 1
            for o in range(n_badspec):
                n = get_unique_names(used_names, "EMSC parameter bad spec " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(
                        name=n, compute_value=EMSCModel(i, common)
                    )
                )
                i += 1
            n = get_unique_names(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(
                    name=n, compute_value=EMSCModel(i, common)
                )
            )
        domain = Orange.data.Domain(
            atts, data.domain.class_vars, data.domain.metas + tuple(model_metas)
        )
        return data.from_table(domain, data)
