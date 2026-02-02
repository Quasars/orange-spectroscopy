import Orange.data
import os.path

from . import io  # register file formats # noqa: F401


def get_sample_datasets_dir():
    thispath = os.path.dirname(__file__)
    dataset_dir = os.path.join(thispath, 'datasets')
    return os.path.realpath(dataset_dir)


Orange.data.table.dataset_dirs.append(get_sample_datasets_dir())


try:
    import dask
    import dask.distributed
    dask_client = dask.distributed.Client(processes=False, n_workers=2,
                                          set_as_default=False,
                                          dashboard_address=None)
except ImportError:
    dask = None
    dask_client = None