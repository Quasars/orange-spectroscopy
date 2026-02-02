import numpy as np

from Orange.data import Domain

MAP_X_VAR = "map_x"
MAP_Y_VAR = "map_y"


def apply_columns_numpy(array, function, selector=None, chunk_size=10 ** 7, callback=None):
    """Split the array by columns, applies selection and then the function.
    Returns output equivalent to function(array[selector])
    """
    chunks_needed = array.size // chunk_size
    # min chunks is 1, max chunks is the number of columns
    chunks = max(min(chunks_needed, array.shape[1]), 1)
    parts = np.array_split(array, chunks, axis=1)
    res = []
    for p in parts:
        if callback:
            callback(0)
        res.append(function(p[selector]))
    return np.hstack(res)


def values_to_linspace(vals):
    """Find a near matching linspace for the values given.
    The problem is that some values can be missing and
    that they are inexact. The minumum and maximum values
    are kept as limits."""
    vals = np.asarray(vals[~np.isnan(vals)])
    if len(vals):
        vals = np.unique(vals)  # returns sorted array
        if len(vals) == 1:
            return vals[0], vals[0], 1
        minabsdiff = (vals[-1] - vals[0])/(len(vals)*100)
        diffs = np.diff(vals)
        diffs = diffs[diffs > minabsdiff]
        first_valid = diffs[0]
        # allow for a percent mismatch
        diffs = diffs[diffs < first_valid*1.01]
        step = np.mean(diffs)
        size = int(round((vals[-1]-vals[0])/step) + 1)
        return vals[0], vals[-1], size
    return None


def location_values(vals, linspace):
    vals = np.asarray(vals)
    if linspace is None or linspace[2] == 1:  # everything is the same value
        width = 1
    else:
        width = (linspace[1] - linspace[0]) / (linspace[2] - 1)
    start = 0
    if linspace is not None:
        start = linspace[0]
    return (vals - start) / width


def index_values(vals, linspace):
    """ Remap values into index of array defined by linspace. """
    return index_values_nan(vals, linspace)[0]


def index_values_nan(vals, linspace):
    """ Remap values into index of array defined by linspace.
    Returns two arrays: first contains the indices, the second invalid values."""
    positions = location_values(vals, linspace)
    return np.round(positions).astype(int), np.isnan(positions)


class NanInsideHypercube(Exception):
    pass


class InvalidAxisException(Exception):
    pass


def axes_to_ndim_linspace(datam, attrs):
    ls = []
    indices = []

    for i, axis in enumerate(attrs):
        coor = datam.X[:, i]
        lsa = values_to_linspace(coor)
        if lsa is None:
            raise InvalidAxisException(axis.name)
        ls.append(lsa)
        indices.append(index_values(coor, lsa))

    return ls, tuple(indices)


def get_ndim_hyperspec(data, attrs):
    """
    Reshape table array into a n-dimensional hyperspectral array with respect to
    provided (n-1) ContinuousVariable attributes.

    The hypercube is organized [ attr0, attr1, ..., wavelengths ].
    Linspace tuple indexes correspond to original attr index.

    Args:
        data (Table): Hyperspectral data Table
        attrs (List): Attributes to build array dimensions along

    Returns:
        (hyperspec, [ls]): Hypercube numpy array and list linspace tuples
    """
    try:
        ndom = Domain(attrs)
    except TypeError:
        raise InvalidAxisException("Axis cannot be None")
    datam = data.transform(ndom)

    ls, indices = axes_to_ndim_linspace(datam, attrs)

    # set data
    new_shape = tuple([lsa[2] for lsa in ls]) + (data.X.shape[1],)
    hyperspec = np.ones(new_shape) * np.nan

    hyperspec[indices] = data.X

    return hyperspec, ls


def get_hypercube(data, xat, yat):
    """
    Reshape table array into a hypercube array according to x and y attributes.
    The hypercube is organized [ rows, columns, wavelengths ].

    Args:
        data (Table): Hyperspectral data Table
        xat (ContinuousVariable): x coordinate attribute
        yat (ContinuousVariable): y coordinate attribute

    Returns:
        (hypercube, lsx, lsy): Hypercube numpy array and linspace tuples
    """
    attrs = [yat, xat]
    hypercube, (lsy, lsx) = get_ndim_hyperspec(data, attrs)
    return hypercube, lsx, lsy


def split_to_size(size, interval):
    pos = 0
    intervals = []
    while pos < size:
        intervals.append(slice(pos, pos + min(size - pos, interval)))
        pos += min(size, interval)
    return intervals
