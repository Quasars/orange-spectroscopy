from collections import Iterable

import Orange
import Orange.data
import numpy as np
from Orange.data.util import SharedComputeValue
from Orange.preprocess.preprocess import Preprocess
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.qhull import ConvexHull, QhullError
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize as sknormalize
from AnyQt.QtCore import Qt

from orangecontrib.infrared.data import getx
from Orange.widgets.utils.annotated_data import get_next_name


def is_increasing(a):
    return np.all(np.diff(a) >= 0)


class SelectColumn(SharedComputeValue):

    def __init__(self, feature, commonfn):
        super().__init__(commonfn)
        self.feature = feature

    def compute(self, data, common):
        return common[:, self.feature]


class _PCAReconstructCommon:
    """Computation common for all PCA variables."""

    def __init__(self, pca, components=None):
        self.pca = pca
        self.components = components

    def __call__(self, data):
        if data.domain != self.pca.pre_domain:
            data = data.from_table(self.pca.pre_domain, data)
        pca_space = self.pca.transform(data.X)
        if self.components is not None:
            #set unused components to zero
            remove = np.ones(pca_space.shape[1])
            remove[self.components] = 0
            remove = np.extract(remove, np.arange(pca_space.shape[1]))
            pca_space[:,remove] = 0
        return self.pca.proj.inverse_transform(pca_space)


class PCADenoising(Preprocess):

    def __init__(self, components=None):
        self.components = components

    def __call__(self, data):
        if data and len(data.domain.attributes):
            maxpca = min(len(data.domain.attributes), len(data))
            pca = Orange.projection.PCA(n_components=min(maxpca, self.components))(data)
            commonfn = _PCAReconstructCommon(pca)

            nats = []
            for i, at in enumerate(data.domain.attributes):
                at = at.copy(compute_value=Orange.projection.pca.Projector(self, i, commonfn))
                nats.append(at)
        else:
            # FIXME we should have a warning here
            nats = [ at.copy() for at in data.domain.attributes ]  # unknown values

        domain = Orange.data.Domain(nats, data.domain.class_vars,
                                    data.domain.metas)

        return data.from_table(domain, data)


class GaussianFeature(SelectColumn):
    pass


def _nan_extend_edges_and_interpolate(xs, X):
    """
    Handle NaNs at the edges are handled as with savgol_filter mode nearest:
    the edge values are interpolated. NaNs in the middle are interpolated
    so that they do not propagate.
    """
    nans = None
    if np.any(np.isnan(X)):
        nans = np.isnan(X)
        X = X.copy()
        _fill_edges(X)
        X = interp1d_with_unknowns_numpy(xs, X, xs)
    return X, nans


class _GaussianCommon:

    def __init__(self, sd, domain):
        self.sd = sd
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        xs, xsind, mon, X = _transform_to_sorted_features(data)
        X, nans = _nan_extend_edges_and_interpolate(xs[xsind], X)
        X = gaussian_filter1d(X, sigma=self.sd, mode="nearest")
        if nans is not None:
            X[nans] = np.nan
        return _transform_back_to_features(xsind, mon, X)


class GaussianSmoothing(Preprocess):

    def __init__(self, sd=10.):
        self.sd = sd

    def __call__(self, data):
        common = _GaussianCommon(self.sd, data.domain)
        atts = [a.copy(compute_value=GaussianFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class Cut(Preprocess):

    def __init__(self, lowlim=None, highlim=None, inverse=False):
        self.lowlim = lowlim
        self.highlim = highlim
        self.inverse = inverse

    def __call__(self, data):
        x = getx(data)
        if not self.inverse:
            okattrs = [at for at, v in zip(data.domain.attributes, x)
                       if (self.lowlim is None or self.lowlim <= v) and
                          (self.highlim is None or v <= self.highlim)]
        else:
            okattrs = [at for at, v in zip(data.domain.attributes, x)
                       if (self.lowlim is not None and v <= self.lowlim) or
                          (self.highlim is not None and self.highlim <= v)]
        domain = Orange.data.Domain(okattrs, data.domain.class_vars, metas=data.domain.metas)
        return data.from_table(domain, data)


class SavitzkyGolayFeature(SelectColumn):
    pass


def _transform_to_sorted_features(data):
    xs = getx(data)
    xsind = np.argsort(xs)
    mon = is_increasing(xsind)
    X = data.X
    X = X if mon else X[:, xsind]
    return xs, xsind, mon, X


def _transform_back_to_features(xsind, mon, X):
    return X if mon else X[:, np.argsort(xsind)]


def _fill_edges(mat):
    """Replace (inplace!) NaN at sides with the closest value"""
    for l in mat:
        loc = np.where(~np.isnan(l))[0]
        if len(loc):
            fi, li = loc[[0, -1]]
            l[:fi] = l[fi]
            l[li + 1:] = l[li]


class _SavitzkyGolayCommon:

    def __init__(self, window, polyorder, deriv, domain):
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        xs, xsind, mon, X = _transform_to_sorted_features(data)
        X, nans = _nan_extend_edges_and_interpolate(xs[xsind], X)
        X = savgol_filter(X, window_length=self.window,
                             polyorder=self.polyorder,
                             deriv=self.deriv, mode="nearest")
        # set NaNs where there were NaNs in the original array
        if nans is not None:
            X[nans] = np.nan
        return _transform_back_to_features(xsind, mon, X)


class SavitzkyGolayFiltering(Preprocess):
    """
    Apply a Savitzky-Golay[1] Filter to the data using SciPy Library.
    """
    def __init__(self, window=5, polyorder=2, deriv=0):
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def __call__(self, data):
        common = _SavitzkyGolayCommon(self.window, self.polyorder,
                                      self.deriv, data.domain)
        atts = [ a.copy(compute_value=SavitzkyGolayFeature(i, common))
                        for i,a in enumerate(data.domain.attributes) ]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class RubberbandBaselineFeature(SelectColumn):
    pass


class _RubberbandBaselineCommon:

    def __init__(self, peak_dir, sub, domain):
        self.peak_dir = peak_dir
        self.sub = sub
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        xs, xsind, mon, X = _transform_to_sorted_features(data)
        x = xs[xsind]
        newd = np.zeros_like(data.X)
        for rowi, row in enumerate(X):
            # remove NaNs which ConvexHull can not handle
            source = np.column_stack((x, row))
            source = source[~np.isnan(source).any(axis=1)]
            try:
                v = ConvexHull(source).vertices
            except QhullError:
                # FIXME notify user
                baseline = np.zeros_like(row)
            else:
                if self.peak_dir == RubberbandBaseline.PeakPositive:
                    v = np.roll(v, -v.argmin())
                    v = v[:v.argmax() + 1]
                elif self.peak_dir == RubberbandBaseline.PeakNegative:
                    v = np.roll(v, -v.argmax())
                    v = v[:v.argmin() + 1]
                # If there are NaN values at the edges of data then convex hull
                # does not include the endpoints. Because the same values are also
                # NaN in the current row, we can fill them with NaN (bounds_error
                # achieves this).
                baseline = interp1d(source[v, 0], source[v, 1], bounds_error=False)(x)
            finally:
                if self.sub == 0:
                    newd[rowi] = row - baseline
                else:
                    newd[rowi] = baseline
        return _transform_back_to_features(xsind, mon, newd)


class RubberbandBaseline(Preprocess):

    PeakPositive, PeakNegative = 0, 1
    Subtract, View = 0, 1

    def __init__(self, peak_dir=PeakPositive, sub=Subtract):
        """
        :param peak_dir: PeakPositive or PeakNegative
        :param sub: Subtract (baseline is subtracted) or View
        """
        self.peak_dir = peak_dir
        self.sub = sub

    def __call__(self, data):
        common = _RubberbandBaselineCommon(self.peak_dir, self.sub,
                                           data.domain)
        atts = [a.copy(compute_value=RubberbandBaselineFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class LinearBaselineFeature(SelectColumn):
    pass


class _LinearBaselineCommon:

    def __init__(self, peak_dir, sub, domain):
        self.peak_dir = peak_dir
        self.sub = sub
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        xs, xsind, mon, y = _transform_to_sorted_features(data)
        x = xs[xsind]

        if np.any(np.isnan(y)):
            y, _ = _nan_extend_edges_and_interpolate(x, y)

        if self.sub == 0:
            newd = y - _edge_baseline(x, y)
        else:
            newd = _edge_baseline(x, y)

        return _transform_back_to_features(xsind, mon, newd)


class LinearBaseline(Preprocess):

    PeakPositive, PeakNegative = 0, 1
    Subtract, View = 0, 1

    def __init__(self, peak_dir=PeakPositive, sub=Subtract):
        """
        :param peak_dir: PeakPositive or PeakNegative
        :param sub: Subtract (baseline is subtracted) or View
        """
        self.peak_dir = peak_dir
        self.sub = sub

    def __call__(self, data):
        common = _LinearBaselineCommon(self.peak_dir, self.sub,
                                           data.domain)
        atts = [a.copy(compute_value=LinearBaselineFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class NormalizeFeature(SelectColumn):
    pass


class _NormalizeCommon:

    def __init__(self, method, lower, upper, int_method, attr, domain):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.int_method = int_method
        self.attr = attr
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)

        if data.X.shape[0] == 0:
            return data.X
        data = data.copy()

        if self.method == Normalize.Vector:
            nans = np.isnan(data.X)
            nan_num = nans.sum(axis=1, keepdims=True)
            ys = data.X
            if np.any(nan_num > 0):
                # interpolate nan elements for normalization
                x = getx(data)
                ys = interp1d_with_unknowns_numpy(x, ys, x)
                ys = np.nan_to_num(ys)  # edge elements can still be zero
            data.X = sknormalize(ys, norm='l2', axis=1, copy=False)
            if np.any(nan_num > 0):
                # keep nans where they were
                data.X[nans] = float("nan")
        elif self.method == Normalize.Area:
            norm_data = Integrate(methods=self.int_method,
                                  limits=[[self.lower, self.upper]])(data)
            data.X /= norm_data.X
        elif self.method == Normalize.Attribute:
            # attr normalization applies to entire spectrum, regardless of limits
            # meta indices are -ve and start at -1
            if self.attr not in (None, "None", ""):
                attr_index = -1 - data.domain.index(self.attr)
                factors = data.metas[:, attr_index].astype(float)
                data.X /= factors[:, None]
        return data.X


class Normalize(Preprocess):
    # Normalization methods
    Vector, Area, Attribute = 0, 1, 2

    def __init__(self, method=Vector, lower=float, upper=float, int_method=0, attr=None):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.int_method = int_method
        self.attr = attr

    def __call__(self, data):
        common = _NormalizeCommon(self.method, self.lower, self.upper,
                                           self.int_method, self.attr, data.domain)
        atts = [a.copy(compute_value=NormalizeFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


INTEGRATE_DRAW_CURVE_WIDTH = 2
INTEGRATE_DRAW_EDGE_WIDTH = 1
INTEGRATE_DRAW_BASELINE_PENARGS = {"width": INTEGRATE_DRAW_CURVE_WIDTH, "style": Qt.DotLine}
INTEGRATE_DRAW_CURVE_PENARGS = {"width": INTEGRATE_DRAW_CURVE_WIDTH}
INTEGRATE_DRAW_EDGE_PENARGS = {"width": INTEGRATE_DRAW_EDGE_WIDTH}


class IntegrateFeature(SharedComputeValue):

    def __init__(self, limits, commonfn):
        self.limits = limits
        super().__init__(commonfn)

    def baseline(self, data, common=None):
        if common is None:
            common = self.compute_shared(data)
        x_s, y_s = self.extract_data(data, common)
        return x_s, self.compute_baseline(x_s, y_s)

    def draw_info(self, data, common=None):
        if common is None:
            common = self.compute_shared(data)
        x_s, y_s = self.extract_data(data, common)
        return self.compute_draw_info(x_s, y_s)

    def extract_data(self, data, common):
        data, x, x_sorter = common
        # find limiting indices (inclusive left, exclusive right)
        lim_min, lim_max = min(self.limits), max(self.limits)
        lim_min = np.searchsorted(x, lim_min, sorter=x_sorter, side="left")
        lim_max = np.searchsorted(x, lim_max, sorter=x_sorter, side="right")
        x_s = x[x_sorter][lim_min:lim_max]
        y_s = data.X[:, x_sorter][:, lim_min:lim_max]
        return x_s, y_s

    def compute_draw_info(self, x_s, y_s):
        return {}

    @staticmethod
    def parameters():
        """ Return parameters for this type of integral """
        raise NotImplementedError

    def compute_baseline(self, x_s, y_s):
        raise NotImplementedError

    def compute_integral(self, x_s, y_s):
        raise NotImplementedError

    def compute(self, data, common):
        x_s, y_s = self.extract_data(data, common)
        return self.compute_integral(x_s, y_s)


class IntegrateFeatureEdgeBaseline(IntegrateFeature):
    """ A linear edge-to-edge baseline subtraction. """

    name = "Integral from baseline"

    @staticmethod
    def parameters():
        return (("Low limit", "Low limit for integration (inclusive)"),
                ("High limit", "High limit for integration (inclusive)"),
            )

    def compute_baseline(self, x, y):
        if np.any(np.isnan(y)):
            y, _ = _nan_extend_edges_and_interpolate(x, y)
        return _edge_baseline(x, y)

    def compute_integral(self, x, y_s):
        y_s = y_s - self.compute_baseline(x, y_s)
        if np.any(np.isnan(y_s)):
            # interpolate unknowns as trapz can not handle them
            y_s, _ = _nan_extend_edges_and_interpolate(x, y_s)
        return np.trapz(y_s, x, axis=1)

    def compute_draw_info(self, x, ys):
        return [("curve", (x, self.compute_baseline(x, ys), INTEGRATE_DRAW_BASELINE_PENARGS)),
                ("curve", (x, ys, INTEGRATE_DRAW_BASELINE_PENARGS)),
                ("fill", ((x, self.compute_baseline(x, ys)), (x, ys)))]


class IntegrateFeatureSimple(IntegrateFeatureEdgeBaseline):
    """ A simple y=0 integration on the provided data window. """

    name = "Integral from 0"

    def compute_baseline(self, x_s, y_s):
        return np.zeros(y_s.shape)


class IntegrateFeaturePeakEdgeBaseline(IntegrateFeature):
    """ The maximum baseline-subtracted peak height in the provided window. """

    name = "Peak from baseline"

    @staticmethod
    def parameters():
        return (("Low limit", "Low limit for integration (inclusive)"),
                ("High limit", "High limit for integration (inclusive)"),
            )

    def compute_baseline(self, x, y):
        return _edge_baseline(x, y)

    def compute_integral(self, x_s, y_s):
        y_s = y_s - self.compute_baseline(x_s, y_s)
        if len(x_s) == 0:
            return np.zeros((y_s.shape[0],)) * np.nan
        return np.nanmax(y_s, axis=1)

    def compute_draw_info(self, x, ys):
        bs = self.compute_baseline(x, ys)
        im = np.nanargmax(ys-bs, axis=1)
        lines = (x[im], bs[np.arange(bs.shape[0]), im]), (x[im], ys[np.arange(ys.shape[0]), im])
        return [("curve", (x, self.compute_baseline(x, ys), INTEGRATE_DRAW_BASELINE_PENARGS)),
                ("curve", (x, ys, INTEGRATE_DRAW_BASELINE_PENARGS)),
                ("line", lines)]


class IntegrateFeaturePeakSimple(IntegrateFeaturePeakEdgeBaseline):
    """ The maximum peak height in the provided data window. """

    name = "Peak from 0"

    def compute_baseline(self, x_s, y_s):
        return np.zeros(y_s.shape)


class IntegrateFeaturePeakXEdgeBaseline(IntegrateFeature):
    """ The X-value of the maximum baseline-subtracted peak height in the provided window. """

    name = "X-value of maximum from baseline"

    @staticmethod
    def parameters():
        return (("Low limit", "Low limit for integration (inclusive)"),
                ("High limit", "High limit for integration (inclusive)"),
            )

    def compute_baseline(self, x, y):
        return _edge_baseline(x, y)

    def compute_integral(self, x_s, y_s):
        y_s = y_s - self.compute_baseline(x_s, y_s)
        if len(x_s) == 0:
            return np.zeros((y_s.shape[0],)) * np.nan
        # avoid whole nan rows
        whole_nan_rows = np.isnan(y_s).all(axis=1)
        y_s[whole_nan_rows] = 0
        # select positions
        pos = x_s[np.nanargmax(y_s, axis=1)]
        # set unknown results
        pos[whole_nan_rows] = np.nan
        return pos

    def compute_draw_info(self, x, ys):
        bs = self.compute_baseline(x, ys)
        im = np.nanargmax(ys-bs, axis=1)
        lines = (x[im], bs[np.arange(bs.shape[0]), im]), (x[im], ys[np.arange(ys.shape[0]), im])
        return [("curve", (x, self.compute_baseline(x, ys), INTEGRATE_DRAW_BASELINE_PENARGS)),
                ("curve", (x, ys, INTEGRATE_DRAW_BASELINE_PENARGS)),
                ("line", lines)]


class IntegrateFeaturePeakXSimple(IntegrateFeaturePeakXEdgeBaseline):
    """ The X-value of the maximum peak height in the provided data window. """

    name = "X-value of maximum from 0"

    def compute_baseline(self, x_s, y_s):
        return np.zeros(y_s.shape)


class IntegrateFeatureAtPeak(IntegrateFeature):
    """ Find the closest x and return the value there. """

    name = "Closest value"

    @staticmethod
    def parameters():
        return (("Closest to", "Nearest value"),
            )

    def extract_data(self, data, common):
        data, x, x_sorter = common
        return x, data.X

    def compute_baseline(self, x, y):
        return np.zeros(y.shape)

    def compute_integral(self, x_s, y_s):
        if len(x_s) == 0:
            return np.zeros((y_s.shape[0],)) * np.nan
        closer = np.nanargmin(abs(x_s - self.limits[0]))
        return y_s[:, closer]

    def compute_draw_info(self, x, ys):
        bs = self.compute_baseline(x, ys)
        im = np.array([np.nanargmin(abs(x - self.limits[0]))])
        dx = [self.limits[0], self.limits[0]]
        dys = np.hstack((bs[:, im], ys[:, im]))
        return [("curve", (dx, dys, INTEGRATE_DRAW_EDGE_PENARGS)),  # line to value
                ("dot", (x[im], ys[:, im]))]


def _edge_baseline(x, y):
    i = np.array([0, -1])
    return interp1d(x[i], y[:, i], axis=1)(x) if len(x) else 0


class _IntegrateCommon:

    def __init__(self, domain):
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        x = getx(data)
        x_sorter = np.argsort(x)
        return data, x, x_sorter


class Integrate(Preprocess):

    INTEGRALS = [IntegrateFeatureSimple,
                 IntegrateFeatureEdgeBaseline,
                 IntegrateFeaturePeakSimple,
                 IntegrateFeaturePeakEdgeBaseline,
                 IntegrateFeatureAtPeak,
                 IntegrateFeaturePeakXSimple,
                 IntegrateFeaturePeakXEdgeBaseline]

    # Integration methods
    Simple, Baseline, PeakMax, PeakBaseline, PeakAt, PeakX, PeakXBaseline = INTEGRALS

    def __init__(self, methods=Baseline, limits=None, names=None, metas=False):
        self.methods = methods
        self.limits = limits
        self.names = names
        self.metas = metas

    def __call__(self, data):
        common = _IntegrateCommon(data.domain)
        atts = []
        if self.limits:
            methods = self.methods
            if not isinstance(methods, Iterable):
                methods = [methods] * len(self.limits)
            names = self.names
            if not names:
                names = [" - ".join("{0}".format(e) for e in l) for l in self.limits]
            # no names in data should be repeated
            used_names = [var.name for var in data.domain.variables + data.domain.metas]
            for i, n in enumerate(names):
                n = get_next_name(used_names, n)
                names[i] = n
                used_names.append(n)
            for limits, method, name in zip(self.limits, methods, names):
                atts.append(Orange.data.ContinuousVariable(
                    name=name,
                    compute_value=method(limits, common)))
        if not self.metas:
            domain = Orange.data.Domain(atts, data.domain.class_vars,
                                        metas=data.domain.metas)
        else:
            domain = Orange.data.Domain(data.domain.attributes, data.domain.class_vars,
                                        metas=data.domain.metas + tuple(atts))
        return data.from_table(domain, data)


def features_with_interpolation(points, kind="linear", domain=None, handle_nans=True, interpfn=None):
    common = _InterpolateCommon(points, kind, domain, handle_nans=handle_nans, interpfn=interpfn)
    atts = []
    for i, p in enumerate(points):
        atts.append(
            Orange.data.ContinuousVariable(name=str(p),
                                           compute_value=InterpolatedFeature(i, common)))
    return atts


class InterpolatedFeature(SelectColumn):
    pass


def remove_whole_nan_ys(x, ys):
    """Remove whole NaN columns of ys with corresponding x coordinates."""
    whole_nan_columns = np.isnan(ys).all(axis=0)
    if np.any(whole_nan_columns):
        x = x[~whole_nan_columns]
        ys = ys[:, ~whole_nan_columns]
    return x, ys


def interp1d_with_unknowns_numpy(x, ys, points, kind="linear"):
    if kind != "linear":
        raise NotImplementedError
    out = np.zeros((len(ys), len(points)))*np.nan
    sorti = np.argsort(x)
    x = x[sorti]
    for i, y in enumerate(ys):
        y = y[sorti]
        nan = np.isnan(y)
        xt = x[~nan]
        yt = y[~nan]
        # do not interpolate unknowns at the edges
        if len(xt):  # check if all values are removed
            out[i] = np.interp(points, xt, yt, left=np.nan, right=np.nan)
    return out


def interp1d_with_unknowns_scipy(x, ys, points, kind="linear"):
    out = np.zeros((len(ys), len(points)))*np.nan
    sorti = np.argsort(x)
    x = x[sorti]
    for i, y in enumerate(ys):
        y = y[sorti]
        nan = np.isnan(y)
        xt = x[~nan]
        yt = y[~nan]
        if len(xt):  # check if all values are removed
            out[i] = interp1d(xt, yt, fill_value=np.nan, assume_sorted=True,
                              bounds_error=False, kind=kind, copy=False)(points)
    return out


def interp1d_wo_unknowns_scipy(x, ys, points, kind="linear"):
    return interp1d(x, ys, fill_value=np.nan, kind=kind, bounds_error=False)(points)


class _InterpolateCommon:

    def __init__(self, points, kind, domain, handle_nans=True, interpfn=None):
        self.points = points
        self.kind = kind
        self.domain = domain
        self.handle_nans = handle_nans
        self.interpfn = interpfn

    def __call__(self, data):
        # convert to data domain if any conversion is possible,
        # otherwise we use the interpolator directly to make domains compatible
        if self.domain and data.domain != self.domain \
                and any(at.compute_value for at in self.domain.attributes):
            data = data.from_table(self.domain, data)
        x = getx(data)
        # removing whole NaN columns from the data will effectively replace
        # NaNs that are not on the edges with interpolated values
        ys = data.X
        if self.handle_nans:
            x, ys = remove_whole_nan_ys(x, ys)  # relatively fast
        if len(x) == 0:
            return np.ones((len(data), len(self.points)))*np.nan
        interpfn = self.interpfn
        if interpfn is None:
            if self.handle_nans and np.isnan(ys).any():
                if self.kind == "linear":
                    interpfn = interp1d_with_unknowns_numpy
                else:
                    interpfn = interp1d_with_unknowns_scipy
            else:
                interpfn = interp1d_wo_unknowns_scipy
        return interpfn(x, ys, self.points, kind=self.kind)


class Interpolate(Preprocess):
    """
    Linear interpolation of the domain.

    Parameters
    ----------
    points : interpolation points (numpy array)
    kind   : type of interpolation (linear by default, passed to
             scipy.interpolate.interp1d)
    """

    def __init__(self, points, kind="linear", handle_nans=True):
        self.points = np.asarray(points)
        self.kind = kind
        self.handle_nans = handle_nans
        self.interpfn = None

    def __call__(self, data):
        atts = features_with_interpolation(self.points, self.kind, data.domain,
                                           self.handle_nans, interpfn=self.interpfn)
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class NotAllContinuousException(Exception):
    pass


class InterpolateToDomain(Preprocess):
    """
    Linear interpolation of the domain.  Attributes are exactly the same
    as in target.

    It differs to Interpolate because the modified data can
    not transform other domains into the interpolated domain. This is
    necessary so that attributes are the same and are thus
    compatible for prediction.
    """

    def __init__(self, target, kind="linear", handle_nans=True):
        self.target = target
        if not all(isinstance(a, Orange.data.ContinuousVariable) for a in self.target.domain.attributes):
            raise NotAllContinuousException()
        self.points = getx(self.target)
        self.kind = kind
        self.handle_nans = handle_nans
        self.interpfn = None

    def __call__(self, data):
        X = _InterpolateCommon(self.points, self.kind, None, handle_nans=self.handle_nans,
                                    interpfn=self.interpfn)(data)
        domain = Orange.data.Domain(self.target.domain.attributes, data.domain.class_vars,
                                    data.domain.metas)
        data = data.transform(domain)
        data.X = X
        return data


class AbsorbanceFeature(SelectColumn):
    pass


class _AbsorbanceCommon:

    def __init__(self, ref, domain):
        self.ref = ref
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        if self.ref:
            # Calculate from single-channel data
            absd = self.ref.X / data.X
            np.log10(absd, absd)
        else:
            # Calculate from transmittance data
            absd = np.log10(data.X)
            absd *= -1
        return absd


class Absorbance(Preprocess):
    """
    Convert data to absorbance.

    Set ref to calculate from single-channel spectra, otherwise convert from transmittance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    def __init__(self, ref=None):
        self.ref = ref

    def __call__(self, data):
        common = _AbsorbanceCommon(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
                    name=var.name, compute_value=AbsorbanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
                    newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class TransmittanceFeature(SelectColumn):
    pass


class _TransmittanceCommon:

    def __init__(self, ref, domain):
        self.ref = ref
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        if self.ref:
            # Calculate from single-channel data
            transd = data.X / self.ref.X
        else:
            # Calculate from absorbance data
            transd = data.X.copy()
            transd *= -1
            np.power(10, transd, transd)
        return transd


class Transmittance(Preprocess):
    """
    Convert data to transmittance.

    Set ref to calculate from single-channel spectra, otherwise convert from absorbance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    def __init__(self, ref=None):
        self.ref = ref

    def __call__(self, data):
        common = _TransmittanceCommon(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
                    name=var.name, compute_value=TransmittanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
                    newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
