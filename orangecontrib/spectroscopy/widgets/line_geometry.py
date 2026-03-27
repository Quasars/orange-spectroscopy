import bottleneck
import numpy as np

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None


def rolling_window(a, window):
    """
    Make an ndarray with a rolling window of the last dimension

    This used to use the trick from
    http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html,
    but numpy 1.20+ has newer primitives.

    Parameters
    ----------
    a : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    """
    return np.lib.stride_tricks.sliding_window_view(a, window, axis=-1)


def intersect_line_segments(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Line segment intersection implemented after
    http://paulbourke.net/geometry/pointlineplane/

    This implementation does not handle colinear points.

    This implementation builds intermediate arrays that are
    bigger than the input array.
    """
    D = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    with np.errstate(divide='ignore'):
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / D
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / D
    return (D != 0) * (ua < 1) * (0 < ua) * (ub < 1) * (0 < ub)


def intersect_curves(x, ys, q1, q2):
    """
    Intersection between multiple curves described by points
    and a single line segment. Returns an array of booleans
    describing whether a line segment (q1 to q2) intersects
    a curve.

    :param x: x values of curves (they have to be sorted).
    :param ys: y values of multiple curves sharing x values.
    :param q1: point of the line segment (x, y)
    :param q2: point of the line segment (x, y)
    :return:
    """

    # convert curves into a series of startpoints and endpoints
    xp = rolling_window(x, 2)
    ysp = rolling_window(ys, 2)

    r = intersect_line_segments(
        xp[:, 0], ysp[:, :, 0], xp[:, 1], ysp[:, :, 1], q1[0], q1[1], q2[0], q2[1]
    )
    return np.any(r, axis=1)


def intersect_curves_chunked_numpy(x, ys, ys_sind, q1, q2, xmin, xmax):
    """
    Processes data in chunks, othewise same as intersect
    curves. Decreases maximum memory use.
    """
    rs = []
    x = x[xmin:xmax]
    for ysc in np.array_split(ys, 100):
        ysc = ysc[:, ys_sind]
        ysc = ysc[:, xmin:xmax]
        ic = intersect_curves(x, ysc, q1, q2)
        rs.append(ic)
    ica = np.concatenate(rs)
    return ica


def intersect_curves_chunked(x, ys, ys_sind, q1, q2, xmin, xmax):
    if isinstance(ys, np.ndarray):
        return intersect_curves_chunked_numpy(x, ys, ys_sind, q1, q2, xmin, xmax)
    elif dask and isinstance(ys, da.Array):
        x = x[xmin:xmax]
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            ys = ys[:, ys_sind]
        ys = ys[:, xmin:xmax]
        ic = intersect_curves(x, ys, q1, q2)
        return dask.compute(ic)
    else:
        raise RuntimeError("unsupported input type")


def distance_line_segment(x1, y1, x2, y2, x3, y3):
    """
    The distance to the line segment [ (x1, y1), (x2, y2) ]
    to a point (x3, y3).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / (
            (x1 - x2) ** 2 + (y1 - y2) ** 2
        )
        xc = x1 + u * (x2 - x1)
        yc = y1 + u * (y2 - y1)
        return np.where(
            (u <= 1) * (u >= 0),
            ((x3 - xc) ** 2 + (y3 - yc) ** 2) ** 0.5,  # distance to a point on line
            np.fmin(
                ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5,  # closest point
                ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5,
            ),
        )


def distance_curves(x, ys, q1):
    """
    Distances to the curves.

    :param x: x values of curves (they have to be sorted).
    :param ys: y values of multiple curves sharing x values.
    :param q1: a point to measure distance to.
    :return:
    """
    if dask and isinstance(ys, da.Array):
        raise RuntimeError("distance_curves does not support dask arrays")

    # convert curves into a series of startpoints and endpoints
    xp = rolling_window(x, 2)
    ysp = rolling_window(ys, 2)

    r = bottleneck.nanmin(
        distance_line_segment(
            xp[:, 0], ysp[:, :, 0], xp[:, 1], ysp[:, :, 1], q1[0], q1[1]
        ),
        axis=1,
    )

    return r


def is_left(l0x, l0y, l1x, l1y, px, py):
    return (l1x - l0x) * (py - l0y) - (px - l0x) * (l1y - l0y)


def in_polygon(point, polygon):
    """
    Test if a point is inside a polygon with a winding number algorithm.

    After "Inclusion of a Point in a Polygon" by Dan Sunday,
    http://geomalgorithms.com/a03-_inclusion.html

    :param point: a 2D point or list of points
    :param polygon: a list of polygon edges
    :return: if a point is inside a polygon
    """
    polygon = np.asarray(polygon)
    point = np.asarray(point)
    x = point[..., 0]
    y = point[..., 1]
    if x.shape:  # to support multiple points
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    pp = rolling_window(polygon.T, 2)
    left = is_left(pp[0][:, 0], pp[1][:, 0], pp[0][:, 1], pp[1][:, 1], x, y)
    upward_crossing = (pp[1][:, 0] <= y) * (y < pp[1][:, 1])
    downward_crossing = (pp[1][:, 0] > y) * (y >= pp[1][:, 1])
    wn = np.sum((left > 0) * (upward_crossing), axis=-1) - np.sum(
        (left < 0) * (downward_crossing), axis=-1
    )
    return wn != 0


if __name__ == "__main__":
    import Orange
    from orangecontrib.spectroscopy.data import getx
    import time
    import sys

    data = Orange.data.Table("collagen.csv")
    x = getx(data)
    sort = np.argsort(x)
    x = x[sort]
    print("sizeof", sys.getsizeof(data.X))
    ys = data.X[:, sort]
    print("sizeof", sys.getsizeof(ys))
    print(ys.shape)
    ys = np.tile(ys, (1, 1)).copy()
    print(ys.shape)
    print("sizeof ys", sys.getsizeof(ys))

    t = time.time()
    intc = np.where(intersect_curves(x, ys, np.array([0, 1.0]), np.array([3000, 1.0])))
    print(time.time() - t)
    print(intc)

    t = time.time()
    # dists = [ distance_curves(x, ys[i:i+1], np.array([910, 1.0])) for i in range(len(ys)-1) ]
    dists = distance_curves(x, ys, np.array([910, 1.0]))
    print(time.time() - t)
    print(dists)
