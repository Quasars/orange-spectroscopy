import numpy as np
from numpy import nextafter

import Orange
from Orange.preprocess.preprocess import Preprocess

try:  # get_unique_names was introduced in Orange 3.20
    from Orange.widgets.utils.annotated_data import get_next_name as get_unique_names
except ImportError:
    from Orange.data.util import get_unique_names

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, \
    interp1d_with_unknowns_numpy, nan_extend_edges_and_interpolate, MissingReferenceException


def ranges_to_weight_table(ranges):
    """
    Create a table of weights from ranges. Include only edge points of ranges.
    Include each edge point twice: once as values within the range and zero
    value outside the range (with this output the weights can easily be interpolated).

    Weights of overlapping intervals are summed.

    Assumes 64-bit floats.

    :param ranges: list of triples (edge1, edge2, weight)
    :return: an Orange.data.Table
    """

    values = {}

    inf = float("inf")
    minf = float("-inf")

    def dict_to_numpy(d):
        x = []
        y = []
        for a, b in d.items():
            x.append(a)
            y.append(b)
        return np.array(x), np.array([y])

    for l, r, w in ranges:
        l, r = min(l, r), max(l, r)
        positions = [nextafter(l, minf), l, r, nextafter(r, inf)]
        weights = [0., float(w), float(w), 0.]

        all_positions = list(set(positions) | set(values))  # new and old positions

        # current values on all position
        x, y = dict_to_numpy(values)
        current = interp1d_with_unknowns_numpy(x, y, all_positions)[0]
        current[np.isnan(current)] = 0

        # new values on all positions
        new = interp1d_with_unknowns_numpy(np.array(positions), np.array([weights]),
                                           all_positions)[0]
        new[np.isnan(new)] = 0

        # update values
        for p, f in zip(all_positions, current + new):
            values[p] = f

    x, y = dict_to_numpy(values)
    dom = Orange.data.Domain([Orange.data.ContinuousVariable(name=str(float(a))) for a in x])
    data = Orange.data.Table.from_numpy(dom, y)
    return data


class EMSCFeature(SelectColumn):
    pass


class EMSCModel(SelectColumn):
    pass


class _EMSC(CommonDomainOrderUnknowns):

    def __init__(self, reference, badspectra, weights, order, scaling, domain):
        super().__init__(domain)
        self.reference = reference
        self.badspectra = badspectra
        self.weights = weights
        self.order = order
        self.scaling = scaling

    def transformed(self, X, wavenumbers):
        # wavenumber have to be input as sorted
        # about 85% of time in __call__ function is spent is lstsq
        # compute average spectrum from the reference
        ref_X = np.atleast_2d(spectra_mean(self.reference.X))

        def interpolate_to_data(other_xs, other_data):
            # all input data needs to be interpolated (and NaNs removed)
            interpolated = interp1d_with_unknowns_numpy(other_xs, other_data, wavenumbers)
            # we know that X is not NaN. same handling of reference as of X
            interpolated, _ = nan_extend_edges_and_interpolate(wavenumbers, interpolated)
            return interpolated

        ref_X = interpolate_to_data(getx(self.reference), ref_X)

        if self.weights:
            # interpolate reference to the data
            wei_X = interp1d_with_unknowns_numpy(getx(self.weights), self.weights.X, wavenumbers)
            # set whichever weights are undefined (usually at edges) to zero
            wei_X[np.isnan(wei_X)] = 0
        else:
            wei_X = np.ones((1, len(wavenumbers)))

        N = wavenumbers.shape[0]
        m0 = - 2.0 / (wavenumbers[0] - wavenumbers[N - 1])
        c_coeff = 0.5 * (wavenumbers[0] + wavenumbers[N - 1])

        n_badspec = len(self.badspectra) if self.badspectra is not None else 0
        if self.badspectra:
            badspectra_X = interpolate_to_data(getx(self.badspectra), self.badspectra.X)

        M = []
        for x in range(0, self.order+1):
            M.append((m0 * (wavenumbers - c_coeff)) ** x)
        for y in range(0, n_badspec):
            M.append(badspectra_X[y])
        M.append(ref_X)  # always add reference spectrum to the model
        n_add_model = len(M)
        M = np.vstack(M).T  # M is for the correction, for par. estimation M_weighted is used

        M_weighted = M*wei_X.T

        newspectra = np.zeros((X.shape[0], X.shape[1] + n_add_model))
        for i, rawspectrum in enumerate(X):
            if np.any(np.isnan(rawspectrum)):
                # lstsq can not handle NaNs
                newspectra[i] = np.nan
                continue
            rawspectrumW = (rawspectrum*wei_X)[0]
            m = np.linalg.lstsq(M_weighted, rawspectrum, rcond=-1)[0]
            corrected = rawspectrum

            for x in range(0, self.order+1+n_badspec):
                corrected = (corrected - (m[x] * M[:, x]))
            if self.scaling:
                corrected = corrected/m[self.order+1+n_badspec]
            corrected[np.isinf(corrected)] = np.nan  # fix values caused by zero weights
            corrected = np.hstack((corrected, m))  # append the model parameters
            newspectra[i] = corrected

        return newspectra


class EMSC(Preprocess):

    def __init__(self, reference=None, badspectra=None, weights=None, order=2, scaling=True,
                 output_model=False, ranges=None):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.badspectra = badspectra
        self.weights = weights
        self.order = order
        self.scaling = scaling
        self.output_model = output_model

    def __call__(self, data):
        # creates function for transforming data
        common = _EMSC(self.reference, self.badspectra, self.weights, self.order,
                       self.scaling, data.domain)
        # takes care of domain column-wise, by above transformation function
        atts = [a.copy(compute_value=EMSCFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        model_metas = []
        n_badspec = len(self.badspectra) if self.badspectra is not None else 0
        used_names = set([var.name for var in data.domain.variables + data.domain.metas])
        if self.output_model:
            i = len(data.domain.attributes)
            for o in range(self.order+1):
                n = get_unique_names(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=EMSCModel(i, common)))
                i += 1
            for o in range(n_badspec):
                n = get_unique_names(used_names, "EMSC parameter bad spec " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=EMSCModel(i, common)))
                i += 1
            n = get_unique_names(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(name=n,
                                               compute_value=EMSCModel(i, common)))
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas + tuple(model_metas))
        return data.from_table(domain, data)
