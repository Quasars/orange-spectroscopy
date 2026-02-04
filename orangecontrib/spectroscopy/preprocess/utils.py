import math
from typing import Optional

import numpy as np
from Orange.data import Table, Domain
from Orange.data.util import SharedComputeValue
from scipy.interpolate import interp1d

from orangecontrib.spectroscopy.data import getx


try:
    import dask
    import dask.array
except ImportError:
    dask = False


def is_increasing(a):
    return np.all(np.diff(a) >= 0)


def full_like_type(orig, shape, val):
    if isinstance(orig, np.ndarray):
        return np.full(shape, val)
    elif dask and isinstance(orig, dask.array.Array):
        return dask.array.full(shape, val)
    else:
        raise RuntimeError("Unknown matrix txpe")


class PreprocessException(Exception):
    def message(self):
        if self.args:
            return self.args[0]
        else:
            return self.__class__.__name__


class MissingReferenceException(PreprocessException):
    pass


class WrongReferenceException(PreprocessException):
    pass


class SelectColumn(SharedComputeValue):
    def __init__(self, feature, commonfn):
        super().__init__(commonfn)
        self.feature = feature

    def compute(self, data, common):
        return common[:, self.feature]

    def __eq__(self, other):
        return super().__eq__(other) and self.feature == other.feature

    def __hash__(self):
        return hash((super().__hash__(), self.feature))


class CommonDomain:
    """A utility class that helps constructing common transformation for
    SharedComputeValue features. It does the domain transformation
    (input domain needs to be the same as it was with training data).
    """

    def __init__(self, domain: Domain):
        self.domain = domain

    def __call__(self, data):
        data = self.transform_domain(data)
        return self.transformed(data)

    def transform_domain(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        return data

    def transformed(self, data):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self.domain == other.domain

    def __hash__(self):
        return hash((type(self), self.domain))


class CommonDomainRef(CommonDomain):
    """CommonDomain which also ensures reference domain transformation"""

    def __init__(self, reference: Table, domain: Domain):
        super().__init__(domain)
        self.reference = reference

    def interpolate_extend_to(self, interpolate: Table, wavenumbers):
        return interpolate_extend_to(interpolate, wavenumbers)

    def __eq__(self, other):
        return super().__eq__(other) and table_eq_x(self.reference, other.reference)

    def __hash__(self):
        domain = self.reference.domain if self.reference is not None else None
        fv = subset_for_hash(self.reference.X) if self.reference is not None else None
        return hash((super().__hash__(), domain, fv))


class CommonDomainOrder(CommonDomain):
    """CommonDomain + it also handles wavenumber order."""

    def __call__(self, data):
        data = self.transform_domain(data)

        # order X by wavenumbers
        xs, xsind, mon, X = transform_to_sorted_features(data)
        xc = X.shape[1]

        # do the transformation
        X = self.transformed(X, xs[xsind])

        # restore order
        return self._restore_order(X, mon, xsind, xc)

    def _restore_order(self, X, mon, xsind, xc):
        # restore order and leave additional columns as they are
        restored = transform_back_to_features(xsind, mon, X[:, :xc])
        return np.hstack((restored, X[:, xc:]))

    def transformed(self, X, wavenumbers):
        raise NotImplementedError

    def __eq__(self, other):
        # pylint: disable=useless-parent-delegation
        return super().__eq__(other)

    def __hash__(self):
        # pylint: disable=useless-parent-delegation
        return super().__hash__()


class CommonDomainOrderUnknowns(CommonDomainOrder):
    """CommonDomainOrder + it also handles unknown values: it interpolates
    values before computation and afterwards sets them back to unknown.
    """

    def __call__(self, data):
        data = self.transform_domain(data)

        # order X by wavenumbers
        xs, xsind, mon, X = transform_to_sorted_features(data)
        xc = X.shape[1]

        # interpolates unknowns
        X, nans = nan_extend_edges_and_interpolate(xs[xsind], X)

        # Replace remaining NaNs (where whole rows were NaN) with
        # with some values so that the function does not crash.
        # Results are going to be discarded later.
        remaining_nans = np.isnan(X)
        if np.any(
            remaining_nans
        ):  # if there were no nans X is a view, so do not modify
            X[remaining_nans] = 1.0

        # do the transformation
        X = self.transformed(X, xs[xsind])

        # set NaNs where there were NaNs in the original array
        if nans is not None:
            # transformed can have additional columns
            addc = X.shape[1] - xc
            if addc:
                nans = np.hstack((nans, np.zeros((X.shape[0], addc), dtype=bool)))
            X[nans] = np.nan

        # restore order
        return self._restore_order(X, mon, xsind, xc)

    def __eq__(self, other):
        # pylint: disable=useless-parent-delegation
        return super().__eq__(other)

    def __hash__(self):
        # pylint: disable=useless-parent-delegation
        return super().__hash__()


def table_eq_x(first: Optional[Table], second: Optional[Table]):
    if first is second:
        return True
    elif first is None or second is None:
        return False
    else:
        return first.domain.attributes == second.domain.attributes and np.array_equal(
            first.X, second.X
        )


def subset_for_hash(array, size=10):
    if array is None:
        return tuple()
    else:
        vals = list(array.ravel()[:size])
        return tuple(v if not math.isnan(v) else None for v in vals)


def nan_extend_edges_and_interpolate(xs, X):
    """
    Handle NaNs at the edges are handled as with savgol_filter mode nearest:
    the edge values are interpolated. NaNs in the middle are interpolated
    so that they do not propagate.
    """
    nans = None
    if np.any(np.isnan(X)):
        nans = np.isnan(X)
        xs, xsind, mon, X = transform_to_sorted_wavenumbers(xs, X)
        X = interp1d_with_unknowns_numpy(xs[xsind], X, xs[xsind], sides=None)
        X = transform_back_to_features(xsind, mon, X)
    return X, nans


def transform_to_sorted_features(data):
    xs = getx(data)
    return transform_to_sorted_wavenumbers(xs, data.X)


def transform_to_sorted_wavenumbers(xs, X):
    xsind = np.argsort(xs)
    mon = is_increasing(xsind)
    X = X if mon else X[:, xsind]
    return xs, xsind, mon, X


def transform_back_to_features(xsind, mon, X):
    return X if mon else X[:, np.argsort(xsind)]


def fill_edges_1d(l):
    """Replace (inplace!) NaN at sides with the closest value"""
    loc = np.where(~np.isnan(l))[0]
    try:
        fi, li = np.array(loc[[0, -1]])
    except IndexError:
        # nothing to do, no valid value
        return l
    else:
        l[:fi] = l[fi]
        l[li + 1 :] = l[li]
        return l


def fill_edges(mat):
    """Replace (inplace!) NaN at sides with the closest value"""
    for i, l in enumerate(mat):
        if dask and isinstance(mat, dask.array.Array):
            l = fill_edges_1d(l)
            mat[i] = l
        else:
            fill_edges_1d(l)


def remove_whole_nan_ys(x, ys):
    """Remove whole NaN columns of ys with corresponding x coordinates."""
    whole_nan_columns = np.isnan(ys).all(axis=0)
    if np.any(whole_nan_columns):
        x = x[~whole_nan_columns]
        ys = ys[:, ~whole_nan_columns]
    return x, ys


def interp1d_with_unknowns_numpy(x, ys, points, kind="linear", sides=np.nan):
    if kind != "linear":
        raise NotImplementedError
    out = full_like_type(ys, (len(ys), len(points)), np.nan)
    sorti = np.argsort(x)
    x = x[sorti]
    for i, y in enumerate(ys):
        # the next line ensures numpy arrays
        # for Dask, it would be much more efficient to work with larger sections
        y = np.array(y[sorti])
        nan = np.isnan(y)
        xt = x[~nan]
        yt = y[~nan]
        # do not interpolate unknowns at the edges
        if len(xt):  # check if all values are removed
            out[i] = np.interp(points, xt, yt, left=sides, right=sides)
    return out


def interp1d_with_unknowns_scipy(x, ys, points, kind="linear"):
    out = full_like_type(ys, (len(ys), len(points)), np.nan)
    sorti = np.argsort(x)
    x = x[sorti]
    for i, y in enumerate(ys):
        y = y[sorti]
        nan = np.isnan(y)
        xt = x[~nan]
        yt = y[~nan]
        if len(xt):  # check if all values are removed
            out[i] = interp1d(
                xt,
                yt,
                fill_value=np.nan,
                assume_sorted=True,
                bounds_error=False,
                kind=kind,
                copy=False,
            )(points)
    return out


def interp1d_wo_unknowns_scipy(x, ys, points, kind="linear"):
    return interp1d(x, ys, fill_value=np.nan, kind=kind, bounds_error=False)(points)


def edge_baseline(x, y):
    """Baseline from edges. Assumes data without NaNs"""
    return linear_baseline(x, y, zero_points=[x[0], x[-1]]) if len(x) else 0


def linear_baseline(x, y, zero_points=None):
    if len(x) == 0:
        return 0
    values_zero_points = interp1d(x, y, axis=1, fill_value="extrapolate")(zero_points)
    return interp1d(zero_points, values_zero_points, axis=1, fill_value="extrapolate")(
        x
    )


def replace_infs(array):
    """Replaces inf and -inf with nan.
    This should be used anywhere a divide-by-zero can happen (/, np.log10, etc)"""
    array[np.isinf(array)] = np.nan
    return array


def replacex(data: Table, replacement: list):
    assert len(data.domain.attributes) == len(replacement)
    natts = [
        at.renamed(str(n))
        for n, at in zip(replacement, data.domain.attributes, strict=True)
    ]
    ndom = Domain(natts, data.domain.class_vars, data.domain.metas)
    return data.transform(ndom)


def interpolate_extend_to(interpolate: Table, wavenumbers):
    """
    Interpolate data to given wavenumbers and extend the possibly
    nan-edges with the nearest values.
    """
    # interpolate reference to the given wavenumbers
    X = interp1d_with_unknowns_numpy(getx(interpolate), interpolate.X, wavenumbers)
    # we know that X is not NaN. same handling of reference as of X
    X, _ = nan_extend_edges_and_interpolate(wavenumbers, X)
    return X
