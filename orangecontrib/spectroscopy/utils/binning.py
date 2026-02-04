import bottleneck
import numpy as np
from Orange.data import Domain, Table

from orangecontrib.spectroscopy.utils import get_ndim_hyperspec, axes_to_ndim_linspace
from orangecontrib.spectroscopy.utils.skimage.shape import view_as_blocks


class InvalidBlockShape(Exception):
    pass


def get_coords(data, bin_attrs):
    ndom = Domain(bin_attrs)
    datam = data.transform(ndom)

    ls, indices = axes_to_ndim_linspace(datam, bin_attrs)

    # set data
    new_shape = tuple([lsa[2] for lsa in ls]) + (len(bin_attrs),)
    coords = np.ones(new_shape) * np.nan
    coords[indices] = datam.X

    return coords


def bin_mean(data, bin_shape, n_attrs):
    try:
        view = view_as_blocks(data, block_shape=bin_shape + (n_attrs,))
    except ValueError as e:
        raise InvalidBlockShape(str(e))
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1, n_attrs)
    mean_view = bottleneck.nanmean(flatten_view, axis=2)
    return mean_view


def bin_hyperspectra(data, bin_attrs, bin_shape):
    """
    Bin a Table with respect to specified attributes and block shape.

    Args:
        in_data (Table): Hyperspectral data Table
        bin_attrs (List): Attributes to bin along
        bin_shape (Tuple): Shape where block index corresponds to
                           attribute index in bin_attrs

    Returns:
        (Orange.data.Table): Binned data Table
    """
    hyperspec, _ = get_ndim_hyperspec(data, bin_attrs)
    n_attrs = len(data.domain.attributes)
    mean_view = bin_mean(hyperspec, bin_shape, n_attrs)

    coords = get_coords(data, bin_attrs)
    mean_coords = bin_mean(coords, bin_shape, len(bin_attrs))

    table_view = mean_view.reshape(-1, n_attrs)
    table_view_coords = mean_coords.reshape(-1, len(bin_attrs))

    domain = Domain(data.domain.attributes, metas=bin_attrs)
    return Table.from_numpy(domain, X=table_view, metas=table_view_coords)
