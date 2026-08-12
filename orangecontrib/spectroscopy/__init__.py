import Orange.data
import os.path

from . import io  # register file formats


def get_sample_datasets_dir():
    thispath = os.path.dirname(__file__)
    dataset_dir = os.path.join(thispath, 'datasets')
    return os.path.realpath(dataset_dir)


Orange.data.table.dataset_dirs.append(get_sample_datasets_dir())


try:
    import dask
    import dask.distributed
except ImportError:
    dask = None

_dask_client = None


def get_dask_client():
    """Return the shared dask.distributed client, creating it on first use."""
    global _dask_client
    if dask is None:
        return None
    if _dask_client is None:
        _dask_client = dask.distributed.Client(
            processes=False, n_workers=2, set_as_default=False, dashboard_address=None
        )
    return _dask_client
