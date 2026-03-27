import bottleneck
import numpy as np


def getx(data):
    """
    Return x of the data. If all attribute names are numbers,
    return their values. If not, return indices.
    """
    x = np.arange(len(data.domain.attributes), dtype=np.float64)
    try:
        x = np.array([float(a.name) for a in data.domain.attributes])
    except:  # noqa: E722
        pass
    return x


def spectra_mean(X):
    return bottleneck.nanmean(X, axis=0)
