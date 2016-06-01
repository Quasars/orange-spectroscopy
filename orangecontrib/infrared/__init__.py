import Orange.data
import os.path
from . import data #register file formats


def get_sample_datasets_dir():
    thispath = os.path.dirname(__file__)
    dataset_dir = os.path.join(thispath, 'datasets')
    return os.path.realpath(dataset_dir)


Orange.data.table.dataset_dirs.append(get_sample_datasets_dir())
