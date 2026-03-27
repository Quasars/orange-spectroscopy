import numpy as np
from scipy.signal import savgol_filter
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

import Orange
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import (
    SelectColumn,
    CommonDomainOrderUnknowns,
    interp1d_with_unknowns_numpy,
    nan_extend_edges_and_interpolate,
    MissingReferenceException,
)


class _AtmCorr(CommonDomainOrderUnknowns):
    """
    Atmospheric gas correction. Removes reference spectrum (or spectra) from
    the input spectra.

    correct_range and spline_range are lists of [x1, x2] that define
    non-overlapping x-value ranges that are individually corrected. For each
    range in correct_ranges and for each spectrum, the amount of reference
    subtracted from (or added to) the spectrum is chosen such that the
    resulting spectrum is maximally smooth; the sum of squares of the first
    derivative (differences between consecutive points) is minimized.
    If multiple references are used (mean_reference=False), the subtracted
    reference is a weighted sum of all the references. (In practice, this may
    be a poor replacement for arbitrary linear combinations of the references.)
    Optionally, the ranges in correct_ranges are smoothed with a Savitzky-Golay
    filter (poly-order 3, window width smooth_win).
    For each range in spline_ranges, data are replaced with a non-overshooting
    spline that merges gradually with the data at its edges (using a Tukey
    window with alpha=0.3).
    """

    def __init__(
        self,
        reference,
        *,
        correct_ranges,
        spline_ranges,
        smooth_win,
        spline_base_win,
        mean_reference,
        domain,
    ):
        super().__init__(domain)
        self.reference = reference
        if correct_ranges is not None:
            self.correct_ranges = [[min(r), max(r)] for r in correct_ranges]
        else:
            self.correct_ranges = [[1300, 2100], [3410, 3850]]
        if spline_ranges is not None:
            self.spline_ranges = [[min(r), max(r)] for r in spline_ranges]
        else:
            self.spline_ranges = [[2190, 2480]]
        self.smooth_win = smooth_win
        self.spline_base_win = spline_base_win
        self.spline_tukey_param = 0.2
        self.mean_reference = mean_reference

    def transformed(self, X, wavenumbers):
        def interpolate_to_data(other_xs, other_data):
            # all input data needs to be interpolated (and NaNs removed)
            interpolated = interp1d_with_unknowns_numpy(
                other_xs, other_data, wavenumbers
            )
            # we know that X is not NaN. same handling of reference as of X
            interpolated, _ = nan_extend_edges_and_interpolate(
                wavenumbers, interpolated
            )
            return interpolated

        def find_wn_ranges(wn, ranges):
            # Find indexes of a list of ranges of wavenumbers.
            if not len(ranges):
                return np.zeros((0, 0))
            ranges = np.asarray(ranges)
            r = np.stack(
                (
                    np.searchsorted(wn, ranges[:, 0]),
                    np.searchsorted(wn, ranges[:, 1], 'right'),
                ),
                1,
            )
            # r[r == len(wn)] = len(wn) - 1
            return r

        y = X.copy()

        if y.size == 0:
            return y

        ranges = find_wn_ranges(wavenumbers, self.correct_ranges)
        ranges = [[p, q] for p, q in ranges if q - p > 1]
        if ranges:
            if self.mean_reference:
                atms = np.atleast_2d(spectra_mean(self.reference.X))
            else:
                atms = np.atleast_2d(self.reference.X)
            atms = interpolate_to_data(getx(self.reference), atms)

            # remove baseline in reference (skip this?)
            for p, q in ranges:
                atms[:, p:q] -= interp1d(wavenumbers[[p, q - 1]], atms[:, [p, q - 1]])(
                    wavenumbers[p:q]
                )

            # Basic removal of atmospheric spectrum
            dy = X[:, :-1] - X[:, 1:]
            az = np.zeros((len(atms), len(X), len(ranges)))
            for ia, atm in enumerate(atms):
                dh = atm[:-1] - atm[1:]
                dh2 = np.cumsum(dh * dh)
                dhdy = np.cumsum(dy * dh, 1)
                for i, (p, q) in enumerate(ranges):
                    r = q - 2
                    az[ia, :, i] = (
                        ((dhdy[:, r] - dhdy[:, p - 1]) / (dh2[r] - dh2[p - 1]))
                        if p > 0
                        else (dhdy[:, r] / dh2[r])
                    )
            az2sum = (az * az).sum(0)
            az = az**3 / az2sum
            for ia, atm in enumerate(atms):
                for i, (p, q) in enumerate(ranges):
                    y[:, p:q] -= az[ia, :, i, None] @ atm[None, p:q]

        # Smoothing of atmospheric regions
        for p, q in ranges:
            if q - p >= self.smooth_win > 3:
                y[:, p:q] = savgol_filter(y[:, p:q], self.smooth_win, 3, axis=1)

        # Replace (CO2) region(s) with spline(s)
        for srange in self.spline_ranges:
            P, Q = find_wn_ranges(wavenumbers, [srange]).flatten()
            Q = min(Q, len(wavenumbers) - 1)
            # TODO: Handle small Q - P better
            if P >= Q:
                continue
            Pw, Qw = wavenumbers[[P, Q]]
            bw = int(self.spline_base_win) // 2
            cr = [
                max(P - bw, 0),
                min(P + bw + 1, len(wavenumbers)),
                max(Q - bw, 0),
                min(Q + bw + 1, len(wavenumbers)),
            ]

            a = np.empty((4, len(y)))
            a[0:2, :] = np.polyfit(
                wavenumbers[cr[0] : cr[1]] - Pw, y[:, cr[0] : cr[1]].T, deg=1
            )
            a[2:4, :] = np.polyfit(
                wavenumbers[cr[2] : cr[3]] - Qw, y[:, cr[2] : cr[3]].T, deg=1
            )
            a[::2, :] = a[::2, :] * (Qw - Pw)
            t = np.interp(wavenumbers[P : Q + 1], [Pw, Qw], [0, 1])
            tt = np.array(
                [
                    t**3 - 2 * t**2 + t,
                    2 * t**3 - 3 * t**2 + 1,
                    t**3 - t**2,
                    -2 * t**3 + 3 * t**2,
                ]
            )
            pt = a.T @ tt
            y[:, P : Q + 1] += (pt - y[:, P : Q + 1]) * tukey(
                len(t), self.spline_tukey_param
            )

        return y


class AtmCorr(Preprocess):
    def __init__(
        self,
        reference=None,
        correct_ranges=None,
        spline_ranges=None,
        smooth_win=0,
        spline_base_win=9,
        mean_reference=True,
    ):
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.correct_ranges = correct_ranges
        self.spline_ranges = spline_ranges
        self.smooth_win = smooth_win
        self.spline_base_win = spline_base_win
        self.mean_reference = mean_reference

    def __call__(self, data):
        # creates function for transforming data
        common = _AtmCorr(
            reference=self.reference,
            correct_ranges=self.correct_ranges,
            spline_ranges=self.spline_ranges,
            smooth_win=self.smooth_win,
            spline_base_win=self.spline_base_win,
            mean_reference=self.mean_reference,
            domain=data.domain,
        )
        # takes care of domain column-wise, by above transformation function
        atts = [
            a.copy(compute_value=SelectColumn(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]

        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
