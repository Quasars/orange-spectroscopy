"""
Iterative concave rubberband baseline correction as a Preprocess subclass.

Algorithm ported from algorithm.py (Samuel Pinilla). No Orange/Qt imports
at the algorithm layer; GUI integration is handled in baseline.py.
"""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import ConvexHull, QhullError

import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.preprocess.utils import (
    SelectColumn,
    CommonDomainOrderUnknowns,
)

_PARALLEL_THRESHOLD = 32  # below this, skip thread-pool dispatch overhead


def _correct_spectrum(
    spectrum: np.ndarray, n_iter: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply concave rubberband correction to one spectrum.

    :param spectrum: 1-D spectral intensities, shape (N,).
    :param n_iter: number of hull-correction iterations (>= 1).
    :returns: (corrected, baseline), both shape (N,) float64.
    """
    spectrum = np.ravel(spectrum)
    N = len(spectrum)

    baseline = np.zeros(N)
    s = spectrum.copy()

    for _ in range(n_iter):
        points = np.column_stack((np.arange(1, N + 1), s))
        try:
            hull = ConvexHull(points)
        except (QhullError, ValueError):
            break  # degenerate (e.g. flat) spectrum; stop iterating
        k_hull = hull.vertices

        p1 = np.where(k_hull == 0)[0][0]
        k_hull = np.concatenate((k_hull[p1:], k_hull[:p1]))

        diffs = np.diff(k_hull)
        stop_candidates = np.where(diffs <= 0)[0]
        stop = stop_candidates[0] + 1 if stop_candidates.size > 0 else len(k_hull)
        hull_idx = k_hull[:stop]

        # Recover flat-edge interior points dropped by ConvexHull: without
        # them a single Case-B segment spans the flat region and its parabolic
        # arch incorrectly absorbs spectral peaks rising above the minimum.
        _extra = []
        for _i in range(len(hull_idx) - 1):
            _ia, _ib = int(hull_idx[_i]), int(hull_idx[_i + 1])
            if s[_ia] == s[_ib]:
                _interior = np.where(s[_ia + 1 : _ib] == s[_ia])[0] + (_ia + 1)
                _extra.extend(_interior.tolist())
        if _extra:
            hull_idx = np.sort(
                np.concatenate([hull_idx, np.array(_extra, dtype=hull_idx.dtype)])
            )

        nv = len(hull_idx)
        B = np.zeros(N)

        for k in range(nv - 1):
            i1 = hull_idx[k]
            i2 = hull_idx[k + 1]
            y1 = s[i1]
            y2 = s[i2]
            span = i2 - i1

            idx = np.arange(i1, i2 + 1)
            L = y1 + (y2 - y1) * (idx - i1) / span  # linear baseline

            if span <= np.ceil(N / 3):
                B[idx] = L
            else:
                P = (idx - i1) * (idx - i2)  # <= 0 inside
                R = s[idx] - L  # >= 0

                int_mask = slice(1, span)
                a_bound = np.max(R[int_mask] / P[int_mask])  # <= 0

                B[idx] = L + a_bound * P
                B[idx] = np.minimum(B[idx], s[idx])

        baseline += B
        s = spectrum - baseline

    return s, baseline


def _correct_chunk(X_chunk: np.ndarray, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Correct every row in X_chunk; called inside a ThreadPoolExecutor worker."""
    out = np.empty_like(X_chunk, dtype=np.float64)
    baselines = np.empty_like(X_chunk, dtype=np.float64)
    for i, row in enumerate(X_chunk):
        out[i], baselines[i] = _correct_spectrum(row, n_iter)
    return out, baselines


def _correct_batch(X: np.ndarray, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply _correct_spectrum row-wise; uses a thread pool when M >= _PARALLEL_THRESHOLD.

    :returns: (corrected, baselines), both shape (M, N) float64.
    """
    M = len(X)
    out = np.empty((M, X.shape[1]), dtype=np.float64)
    baselines = np.empty((M, X.shape[1]), dtype=np.float64)

    if M < _PARALLEL_THRESHOLD:
        for i, row in enumerate(X):
            out[i], baselines[i] = _correct_spectrum(row, n_iter)
        return out, baselines

    chunk_size = max(1, M // (os.cpu_count() * 4))
    offsets = list(range(0, M, chunk_size))
    chunks = [X[off : off + chunk_size] for off in offsets]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        future_to_off = {
            ex.submit(_correct_chunk, chunk, n_iter): off
            for chunk, off in zip(chunks, offsets, strict=True)
        }
        for fut in as_completed(future_to_off):
            off = future_to_off[fut]
            c, b = fut.result()
            out[off : off + len(c)] = c
            baselines[off : off + len(b)] = b

    return out, baselines


class ConcaveRubberbandBaselineFeature(SelectColumn):
    InheritEq = True


class _ConcaveRubberbandBaselineCommon(CommonDomainOrderUnknowns):
    def __init__(self, n_iter: int, sub: int, domain):
        super().__init__(domain)
        self.n_iter = n_iter
        self.sub = sub

    def transformed(self, X: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
        """Return corrected spectra (sub=0) or baselines (sub=1), shape (M, N)."""
        corrected, baselines = _correct_batch(X, self.n_iter)
        return (
            corrected if self.sub == ConcaveRubberbandBaseline.Subtract else baselines
        )

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.n_iter == other.n_iter
            and self.sub == other.sub
        )

    def __hash__(self):
        return hash((super().__hash__(), self.n_iter, self.sub))


class ConcaveRubberbandBaseline(Preprocess):
    """Iterative concave rubberband baseline correction.

    :param n_iter: number of hull-correction iterations (>= 1, default 3).
    :param sub: Subtract (0) subtracts baseline and returns corrected spectrum;
        View (1) returns the baseline itself.
    """

    Subtract, View = 0, 1

    def __init__(self, n_iter: int = 3, sub: int = Subtract):
        self.n_iter = n_iter
        self.sub = sub

    def __call__(self, data: Orange.data.Table) -> Orange.data.Table:
        common = _ConcaveRubberbandBaselineCommon(self.n_iter, self.sub, data.domain)
        atts = [
            a.copy(compute_value=ConcaveRubberbandBaselineFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
