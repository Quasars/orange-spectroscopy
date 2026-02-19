from typing import Union, Sequence

import bottleneck
import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull, QhullError
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize as sknormalize

from extranormal3 import normal_xas, extra_exafs

import Orange
import Orange.data
from Orange.data import ContinuousVariable
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.unsupervised.owpca import MAX_COMPONENTS

from orangecontrib.spectroscopy.data import getx

from orangecontrib.spectroscopy.preprocess.integrate import Integrate
from orangecontrib.spectroscopy.preprocess.emsc import EMSC
from orangecontrib.spectroscopy.preprocess.transform import Absorbance, Transmittance, \
    CommonDomainRef
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomain, \
    CommonDomainOrder, CommonDomainOrderUnknowns, nan_extend_edges_and_interpolate, \
    remove_whole_nan_ys, interp1d_with_unknowns_numpy, interp1d_with_unknowns_scipy, \
    interp1d_wo_unknowns_scipy, edge_baseline, MissingReferenceException, \
    WrongReferenceException, replace_infs, transform_to_sorted_features, PreprocessException, \
    linear_baseline


class MNFDenoising(Preprocess):

    def __init__(self, components=None):
        self.components = components

    def __call__(self, data):
        if data and len(data.domain.attributes):
            # maxsvd = min(len(data.domain.attributes), len(data))
            commonfn = _MNFCommon(data.domain, self.components)

            nats = [at.copy(compute_value=MNFDenoisingFeature(i, commonfn))
                    for i, at in enumerate(data.domain.attributes)]
        else:
            # FIXME we should have a warning here
            nats = [at.copy(compute_value=lambda d: np.full((len(d), 1), np.nan))
                    for at in data.domain.attributes]

        domain = Orange.data.Domain(nats, data.domain.class_vars,
                                    data.domain.metas)

        return data.from_table(domain, data)


class MNFDenoisingFeature(SelectColumn):
    InheritEq = True


class _MNFCommon(CommonDomainOrderUnknowns):

    def __init__(self, domain, components):
        super().__init__(domain)
        self.domain = domain
        self.components = components

    def transformed(self, X, _):
        """
        Minimum Noise Fraction calculation

        This version is from the paper of R. Bhargava
            https://doi.org/10.1371/journal.pone.0205219

        TODO: implement the enhanced version using consecutive measurements
            https://doi.org/10.1016/j.chemolab.2023.105042

        :param data: data table
        :param components: number of components to retain
        :return: denoised data cube
        """

        m = X.shape

        diffs = -np.diff(X, axis=0)
        N = diffs

        # noise whitening
        V1, S, _ = np.linalg.svd(np.dot(N.T, N))
        # s1 = s1 ** -0.5
        S = np.linalg.inv(np.sqrt(np.diag(S)))
        W = np.dot(np.dot(X, V1), S)

        # directions maximizing signal variance
        G1, _, _ = np.linalg.svd(np.dot(W.T, W))

        # Final MNF projection vectors calculation
        P = np.dot(np.dot(V1, S), G1)

        # X data on the MNF directions space
        M = np.dot(X, P)

        # choice of top components
        R = np.eye(m[1], m[1])
        R[self.components:, self.components:] = 0

        # inverse MNF transformation
        D = np.dot(np.dot(M, R), np.linalg.inv(P))

        return D


class PCADenoisingFeature(SelectColumn):
    InheritEq = True


class _PCAReconstructCommon(CommonDomain):
    """Computation common for all PCA variables."""

    def __init__(self, pca, components=None):
        super().__init__(pca.pre_domain)
        self.pca = pca
        self.components = components

    def transformed(self, data):
        pca_space = self.pca.transform(data.X)
        if self.components is not None:
            # set unused components to zero
            remove = np.ones(pca_space.shape[1])
            if isinstance(self.components, int):
                remove[:self.components] = 0
            else:
                remove[self.components] = 0
            remove = np.extract(remove, np.arange(pca_space.shape[1]))
            pca_space[:, remove] = 0
        return self.pca.proj.inverse_transform(pca_space)


class PCADenoising(Preprocess):

    def __init__(self, components: Union[None, int, Sequence[int]]=None,
                 random_state=0, svd_solver="randomized"):
        self.components = components
        self.random_state = random_state
        self.svd_solver = svd_solver

    def __call__(self, data):
        if data and len(data.domain.attributes):
            maxpca = min(len(data.domain.attributes), len(data), MAX_COMPONENTS)
            pca = Orange.projection.PCA(n_components=maxpca,
                                        random_state=self.random_state,
                                        svd_solver=self.svd_solver)(data)
            commonfn = _PCAReconstructCommon(pca, self.components)
            nats = [at.copy(compute_value=PCADenoisingFeature(i, commonfn))
                    for i, at in enumerate(data.domain.attributes)]
        else:
            # FIXME we should have a warning here
            nats = [at.copy(compute_value=lambda d: np.full((len(d), 1), np.nan))
                    for at in data.domain.attributes]

        domain = Orange.data.Domain(nats, data.domain.class_vars,
                                    data.domain.metas)

        return data.from_table(domain, data)


class GaussianFeature(SelectColumn):
    InheritEq = True


class _GaussianCommon(CommonDomainOrderUnknowns):

    def __init__(self, sd, domain):
        super().__init__(domain)
        self.sd = sd

    def transformed(self, X, wavenumbers):
        if self.sd > 0:
            return gaussian_filter1d(X, sigma=self.sd, mode="nearest")
        else:
            return X


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
    InheritEq = True


class _SavitzkyGolayCommon(CommonDomainOrderUnknowns):

    def __init__(self, window, polyorder, deriv, domain):
        super().__init__(domain)
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def transformed(self, X, wavenumbers):
        return savgol_filter(X, window_length=self.window,
                             polyorder=self.polyorder,
                             deriv=self.deriv, mode="nearest")

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.window == other.window \
               and self.polyorder == other.polyorder \
               and self.deriv == other.deriv

    def __hash__(self):
        return hash((super().__hash__(), self.window, self.polyorder, self.deriv))


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
        atts = [a.copy(compute_value=SavitzkyGolayFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class RubberbandBaselineFeature(SelectColumn):
    InheritEq = True


class _RubberbandBaselineCommon(CommonDomainOrder):

    def __init__(self, peak_dir, sub, domain):
        super().__init__(domain)
        self.peak_dir = peak_dir
        self.sub = sub

    def transformed(self, X, x):
        newd = np.zeros_like(X)
        for rowi, row in enumerate(X):
            # remove NaNs which ConvexHull can not handle
            source = np.column_stack((x, row))
            source = source[~bottleneck.anynan(source, axis=1)]
            try:
                v = ConvexHull(source).vertices
            except (QhullError, ValueError):
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
        return newd


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
    InheritEq = True


class _LinearBaselineCommon(CommonDomainOrderUnknowns):

    def __init__(self, peak_dir, sub, zero_points, domain):
        super().__init__(domain)
        self.peak_dir = peak_dir
        self.sub = sub
        self.zero_points = zero_points

    def transformed(self, y, x):
        if self.zero_points:
            baseline = linear_baseline(x, y, self.zero_points)
        else:
            baseline = edge_baseline(x, y)
        return y - baseline if self.sub == 0 else baseline


class LinearBaseline(Preprocess):
    PeakPositive, PeakNegative = 0, 1
    Subtract, View = 0, 1

    def __init__(self, peak_dir=PeakPositive, sub=Subtract, zero_points=None):
        """
        :param peak_dir: PeakPositive or PeakNegative
        :param sub: Subtract (baseline is subtracted) or View
        """
        self.peak_dir = peak_dir
        self.sub = sub
        self.zero_points = zero_points

    def __call__(self, data):
        common = _LinearBaselineCommon(self.peak_dir, self.sub, self.zero_points,
                                       data.domain)
        atts = [a.copy(compute_value=LinearBaselineFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class NormalizeFeature(SelectColumn):
    InheritEq = True


class _NormalizeCommon(CommonDomain):

    def __init__(self, method, lower, upper, int_method, attr, domain):
        super().__init__(domain)
        self.method = method
        self.lower = lower
        self.upper = upper
        self.int_method = int_method
        self.attr = attr

    def transformed(self, data):
        if data.X.shape[0] == 0:
            return data.X
        data = data.copy()

        with data.unlocked():
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
                replace_infs(data.X)
            elif self.method == Normalize.SNV:
                data.X = (data.X - bottleneck.nanmean(data.X, axis=1).reshape(-1, 1)) / \
                         bottleneck.nanstd(data.X, axis=1).reshape(-1, 1)
                replace_infs(data.X)
            elif self.method == Normalize.Attribute:
                if self.attr in data.domain and isinstance(data.domain[self.attr], Orange.data.ContinuousVariable):
                    ndom = Orange.data.Domain([data.domain[self.attr]])
                    factors = data.transform(ndom)
                    data.X /= factors.X
                    replace_infs(data.X)
                    nd = data.domain[self.attr]
                else:  # invalid attribute for normalization
                    data.X *= float("nan")
            elif self.method == Normalize.MinMax:
                min = bottleneck.nanmin(data.X, axis=1).reshape(-1, 1)
                max = bottleneck.nanmax(data.X, axis=1).reshape(-1, 1)
                data.X = data.X / (max - min)
                replace_infs(data.X)
        return data.X

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.method == other.method \
               and self.lower == other.lower \
               and self.upper == other.upper \
               and self.int_method == other.int_method \
               and self.attr == other.attr

    def __hash__(self):
        return hash((super().__hash__(), self.method, self.lower,
                     self.upper, self.int_method, self.attr))


class Normalize(Preprocess):
    # Normalization methods
    Vector, Area, Attribute, MinMax, SNV = 0, 1, 2, 3, 4

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


class _NormalizeReferenceCommon(CommonDomainRef):

    def transformed(self, data):
        ref_X = self.interpolate_extend_to(self.reference, getx(data))
        return replace_infs(data.X / ref_X)


class NormalizeReference(Preprocess):

    def __init__(self, reference):
        if reference is None:
            raise MissingReferenceException()
        elif len(reference) != 1:
            raise WrongReferenceException("Reference data should have length 1")
        self.reference = reference

    def __call__(self, data):
        common = _NormalizeReferenceCommon(self.reference, data.domain)
        atts = [a.copy(compute_value=NormalizeFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.transform(domain)


class _NormalizePhaseReferenceCommon(CommonDomainRef):

    def transformed(self, data):
        ref_X = self.interpolate_extend_to(self.reference, getx(data))
        return replace_infs(np.angle(np.exp(data.X * 1j) / np.exp(ref_X * 1j)))



class NormalizePhaseReference(NormalizeReference):

    def __call__(self, data):
        common = _NormalizePhaseReferenceCommon(self.reference, data.domain)
        atts = [a.copy(compute_value=NormalizeFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.transform(domain)


def features_with_interpolation(points, kind="linear", domain=None, handle_nans=True, interpfn=None):
    common = _InterpolateCommon(points, kind, domain, handle_nans=handle_nans, interpfn=interpfn)
    atts = []
    for i, p in enumerate(points):
        atts.append(
            Orange.data.ContinuousVariable(name=str(p),
                                           compute_value=InterpolatedFeature(i, common)))
    return atts


class InterpolatedFeature(SelectColumn):
    InheritEq = True


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
        if self.domain is not None and data.domain != self.domain \
                and any(at.compute_value for at in self.domain.attributes):
            data = data.from_table(self.domain, data)
        x = getx(data)
        # removing whole NaN columns from the data will effectively replace
        # NaNs that are not on the edges with interpolated values
        ys = data.X
        if self.handle_nans:
            x, ys = remove_whole_nan_ys(x, ys)  # relatively fast
        if len(x) == 0:
            return np.ones((len(data), len(self.points))) * np.nan
        interpfn = self.interpfn
        if interpfn is None:
            if self.handle_nans and np.any(np.isnan(ys)):
                if self.kind == "linear":
                    interpfn = interp1d_with_unknowns_numpy
                else:
                    interpfn = interp1d_with_unknowns_scipy
            else:
                interpfn = interp1d_wo_unknowns_scipy
        return interpfn(x, ys, self.points, kind=self.kind)

    def __eq__(self, other):
        return type(self) is type(other) \
               and np.all(self.points == other.points) \
               and self.kind == other.kind \
               and self.domain == other.domain \
               and self.handle_nans == other.handle_nans \
               and self.interpfn == other.interpfn

    def __hash__(self):
        return hash((type(self), tuple(self.points[:5]), self.kind,
                     self.domain, self.handle_nans, self.interpfn))


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
        return data.transform(domain)


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
        with data.unlocked_reference(data.X):
            data.X = X
        return data


######################################### XAS normalization ##########
class XASnormalizationFeature(SelectColumn):
    InheritEq = True


class _XASnormalizationCommon(CommonDomainOrderUnknowns):

    def __init__(self, edge, preedge_dict, postedge_dict, domain):
        super().__init__(domain)
        self.edge = edge
        self.preedge_params = preedge_dict
        self.postedge_params = postedge_dict

    def transformed(self, X, energies):
        if X.shape[0] == 0:
            return np.zeros((0, X.shape[1] + 1))

        try:
            spectra, jump_vals = normal_xas.normalize_all(
                energies, X, self.edge, self.preedge_params, self.postedge_params)
            jump_vals = jump_vals.reshape(-1, 1)
        except Exception:
            # TODO handle meaningful exceptions with PreprocessException
            spectra = X
            jump_vals = np.zeros((len(X), 1))

        return np.hstack((spectra, jump_vals))


class XASnormalization(Preprocess):
    """
    XAS Athena like normalization with flattenning

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    def __init__(self, edge=None, preedge_dict=None, postedge_dict=None):
        self.edge = edge
        self.preedge_params = preedge_dict
        self.postedge_params = postedge_dict

    def __call__(self, data):
        common = _XASnormalizationCommon(self.edge, self.preedge_params,
                                         self.postedge_params, data.domain)
        newattrs = [ContinuousVariable(name=var.name,
                                       compute_value=XASnormalizationFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        newmetas = data.domain.metas + (ContinuousVariable(
            name='edge_jump', compute_value=XASnormalizationFeature(len(newattrs), common)),)

        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, newmetas)

        return data.from_table(domain, data)


######################################### EXAFS extraction #####
class NoEdgejumpProvidedException(PreprocessException):
    pass


class ExtractEXAFSFeature(SelectColumn):
    InheritEq = True


class EdgeJumpException(PreprocessException):
    pass


class _ExtractEXAFSCommon(CommonDomain):
    # not CommonDomainOrderUnknowns because E -> K
    # and because transformed needs Edge jumps

    def __init__(self, edge, extra_from, extra_to, poly_deg, kweight, m, k_interp, domain):
        super().__init__(domain)
        self.edge = edge
        self.extra_from = extra_from
        self.extra_to = extra_to
        self.poly_deg = poly_deg
        self.kweight = kweight
        self.m = m
        self.k_interp = k_interp

    def __call__(self, data):
        data = self.transform_domain(data)

        if "edge_jump" in data.domain:
            edges = data.transform(Orange.data.Domain([data.domain["edge_jump"]]))
            I_jumps = edges.X[:, 0]
        else:
            raise NoEdgejumpProvidedException(
                'Invalid meta data: Intensity jump at edge is missing')

        # order X by wavenumbers:
        # xs non ordered energies
        # xsind - indecies corresponding to the ordered energies
        # mon = True
        # X spectra as corresponding to the ordered energies
        xs, xsind, mon, X = transform_to_sorted_features(data)

        # for the missing data
        X, nans = nan_extend_edges_and_interpolate(xs[xsind], X)
        # TODO notify the user if some unknown values were interpolated

        # Replace remaining NaNs (where whole rows were NaN) with
        # with some values so that the function does not crash.
        # Results are going to be discarded later.
        nan_rows = bottleneck.allnan(X, axis=1)
        if np.any(nan_rows):  # if there were no nans X is a view, so do not modify
            X[nan_rows] = 1.

        # do the transformation
        X = self.transformed(X, xs[xsind], I_jumps)

        # discard nan rows
        X[nan_rows] = np.nan

        # k scores are always ordered, so do not restore order
        return X

    def transformed(self, X, energies, I_jumps):
        try:
            Km_Chi, Chi, bkgr = extra_exafs.extract_all(energies, X,
                                                        self.edge, I_jumps,
                                                        self.extra_from, self.extra_to,
                                                        self.poly_deg, self.kweight, self.m)
        except Exception as e:
            # extra_exafs should be fixed to return a specific exception
            if "jump at edge" in e.args[0]:
                raise EdgeJumpException("Problem with edge jump.")
            else:
                raise

        # this function always needs to return the expected input size - even
        # if the test data was empty - force the output size
        correct_shape = (X.shape[0], len(self.k_interp))
        if Km_Chi.shape != correct_shape:
            return np.full(correct_shape, np.nan)

        return Km_Chi


class ExtractEXAFS(Preprocess):
    """
    EXAFS extraction with polynomial background

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    def __init__(self, edge=None, extra_from=None, extra_to=None,
                 poly_deg=None, kweight=None, m=None):
        self.edge = edge
        self.extra_from = extra_from
        self.extra_to = extra_to
        self.poly_deg = poly_deg
        self.kweight = kweight
        self.m = m

    def __call__(self, data):

        if data.X.shape[1] > 0:
            # --- compute K
            energies = np.sort(getx(data))  # input data can be in any order
            start_idx, end_idx = extra_exafs.get_idx_bounds(energies, self.edge,
                                                            self.extra_from, self.extra_to)
            k_interp, k_points = extra_exafs.get_K_points(energies, self.edge,
                                                          start_idx, end_idx)
            # ----------

            common = _ExtractEXAFSCommon(self.edge, self.extra_from, self.extra_to,
                                         self.poly_deg, self.kweight, self.m, k_interp,
                                         data.domain)

            newattrs = [ContinuousVariable(name=str(var),
                                           compute_value=ExtractEXAFSFeature(i, common))
                        for i, var in enumerate(k_interp)]
        else:
            newattrs = []

        domain = Orange.data.Domain(newattrs, data.domain.class_vars, data.domain.metas)
        return data.transform(domain)


class ShiftAndScaleFeature(SelectColumn):
    InheritEq = True


class _ShiftAndScaleCommon(CommonDomain):

    def __init__(self, offset, scale, domain):
        super().__init__(domain)
        self.offset = offset
        self.scale = scale


    def transformed(self, data):
        return self.scale * data.X + self.offset


class ShiftAndScale(Preprocess):
    """
    Shift and scale of the data into the form:
    y = scale * x + offset

    Parameters
    ----------
    offset : float
    scale  : float
    """

    def __init__(self, offset=0., scale=1.):
        self.offset = offset
        self.scale = scale

    def __call__(self, data):
        common = _ShiftAndScaleCommon(self.offset, self.scale, data.domain)
        atts = [a.copy(compute_value=ShiftAndScaleFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class DespikeFeature(SelectColumn):
    InheritEq = True


class _DespikeCommon(CommonDomainOrderUnknowns):
    def __init__(self, threshold, cutoff, dis, domain):
        super().__init__(domain)
        self.threshold = threshold
        # threshold sets up a limit for the modified_z_scores test cutoff for spikes
        self.cutoff = cutoff
        # cutoff allows for spiked spectra to be determined from a large
        # multispectral image
        self.dis = dis
        # dis sets the distance over which to interpolate spiked areas

    def transformed(self, data, X):
        def interpolatespikes(spikes, row_in):
            row_out = row_in.copy()
            non_nan_nor_spike = None
            for i in np.nonzero(spikes)[0]:
                w = np.arange(max(i - self.dis, 0), min(i + self.dis + 1, len(spikes)))
                w2 = w[spikes[w] == 0]
                row_out[i] = np.nanmean(row_in[w2])
                if np.isnan(row_out[i]):  # no valid points within distance, find closest value
                    if non_nan_nor_spike is None:  # compute it only once
                        non_nan_nor_spike = np.nonzero(~(np.isnan(row_in) | (spikes > 0)))[0]
                    # find nearest non-nan indices
                    valid_ind = []
                    insertp = np.searchsorted(non_nan_nor_spike, i, side="left")
                    if insertp > 0:
                        valid_ind.append(non_nan_nor_spike[insertp-1])  # left non-nan
                    if insertp < len(non_nan_nor_spike):
                        valid_ind.append(non_nan_nor_spike[insertp])  # right non-nan
                    valid_ind = np.array(valid_ind)
                    diff_i = np.abs(valid_ind - i)
                    # closest non-nan indices (or pair of indices)
                    closest_ind = valid_ind[np.nonzero(diff_i == np.min(diff_i))[0]]
                    row_out[i] = np.nanmean(row_in[closest_ind])
            return row_out

        out = []
        Series = data
        if Series.size > 0:
            for row in Series:
                Distance = np.abs(np.diff(row, n=1))
                # Spiked spectra are processed and non spiked are passed through
                if np.any(Distance > self.cutoff):
                    median1 = np.median(row)
                    mad_int = np.median(np.abs(row - median1))
                    modified_z_scores = 0.6745 * (row - median1) / mad_int
                    spikes = (abs(modified_z_scores) > self.threshold)
                    despiked = interpolatespikes(spikes, row)
                    out.append(despiked)
                else:
                    out.append(row)
            return np.array(out)
        else:
            return data


class Despike(Preprocess):

    def __init__(self, threshold=7, cutoff=100, dis=5):
        self.threshold = threshold
        self.cutoff = cutoff
        self.dis = dis

    def __call__(self, data):
        common = _DespikeCommon(self.threshold, self.cutoff,
                                self.dis, data.domain)
        atts = [a.copy(compute_value=DespikeFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)



class SpSubtractFeature(SelectColumn):
    InheritEq = True


class _SpSubtractCommon(CommonDomainRef):

    def __init__(self, amount, reference, domain):
        super().__init__(reference, domain)
        self.amount = amount

    def transformed(self, data):
        ref_X = self.interpolate_extend_to(self.reference, getx(data))
        result = data.X - self.amount * ref_X
        return result


class SpSubtract(Preprocess):
    """
    Subtract reference spectrum with a multiplication factor

    Set reference

    Parameters
    ----------
    amount    : multiplier for reference to subtract from each spectrum (float)
    reference : reference single-channel (Orange.data.Table)
    """

    def __init__(self, reference, amount=0.):
        if reference is None or len(reference) != 1:
            raise WrongReferenceException("Reference data should have length 1")
        self.reference = reference
        self.amount = amount

    def __call__(self, data):
        common = _SpSubtractCommon(self.amount, self.reference, data.domain)
        atts = [a.copy(compute_value=SpSubtractFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)

class BaselineShiftFeature(SelectColumn):
    InheritEq = True


class _BaselineShiftCommon(CommonDomainOrderUnknowns):
    """
    Vectorized baseline shift transformation logic.
    """

    def __init__(self, method, wn=None, wn_range=None, domain=None):
        super().__init__(domain)
        self.method = method
        self.wn = wn
        self.wn_range = wn_range

    def transformed(self, X, x):
        """
        Apply baseline shift per spectrum using fully vectorized operations.
        X: (n_samples, n_wavenumbers)
        x: (n_wavenumbers,)
        """
        # Defensive: return unchanged if parameters are missing
        if self.method in ("specific", "min_range", "max_range", "avg_range"):
            if self.method == "specific" and self.wn is None:
                return X
            if self.method in ("min_range", "max_range", "avg_range") and self.wn_range is None:
                return X

        # Case 1: Minimum over whole spectrum (fastest)
        if self.method == "minimum":
            baselines = np.nanmin(X, axis=1, keepdims=True)  # shape (n_samples, 1)

        # Case 2: Specific wavenumber per spectrum
        elif self.method == "specific":
            idx = np.argmin(np.abs(x - self.wn))  # index of nearest wn
            baselines = X[:, idx:idx+1]  # broadcast shape (n_samples, 1)

        # Cases 3-5: Range-based operations
        elif self.method in ("min_range", "max_range", "avg_range"):
            mask = (x >= self.wn_range[0]) & (x <= self.wn_range[1])
            if not np.any(mask):  # No valid range? Return unchanged
                return X

            X_range = X[:, mask]  # restrict to wn range, shape (n_samples, n_points)
            if self.method == "min_range":
                baselines = np.nanmin(X_range, axis=1, keepdims=True)
            elif self.method == "max_range":
                baselines = np.nanmax(X_range, axis=1, keepdims=True)
            else:  # avg_range
                baselines = np.nanmean(X_range, axis=1, keepdims=True)

        else:
            raise ValueError(f"Unknown baseline shift method: {self.method}")

        # Vectorized subtraction
        return X - baselines


class BaselineShift(Preprocess):
    """
    Baseline shift preprocessor: subtracts a baseline level determined by
    a given method (min, specific wavenumber, min/max/avg in range).
    """

    METHODS = ("minimum", "specific", "min_range", "max_range", "avg_range")

    def __init__(self, method="minimum", wn=None, wn_range=None):
        """
        :param method: baseline shift method
        :param wn: specific wavenumber (if method == 'specific')
        :param wn_range: tuple (min_wn, max_wn) if method involves range
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}, must be one of {self.METHODS}")
        self.method = method
        self.wn = wn
        self.wn_range = wn_range

    def __call__(self, data):
        common = _BaselineShiftCommon(self.method, self.wn, self.wn_range, data.domain)
        atts = [
            a.copy(compute_value=BaselineShiftFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)