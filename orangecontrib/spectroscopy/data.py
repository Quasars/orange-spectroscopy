# pylint: disable=unused-import
# ruff: noqa: F401

# import for compatibility before the restructure
import Orange.data.io  # legacy import so that file readers are registered
from orangecontrib.spectroscopy.io.util import (
    _metatable_maplocs,
    _spectra_from_image,
    _spectra_from_image_2d,
    build_spec_table,
)

from orangecontrib.spectroscopy.util import getx, spectra_mean

# imports for workflow compatibility when a reader was explicitly selected
# In the owfile settings, explicitly selected selected readers are stored with
# full module and class name. Moving but not adding imports would show
# the missing reader error.
from orangecontrib.spectroscopy.io.agilent import (
    AgilentImageReader,
    AgilentImageIFGReader,
    agilentMosaicReader,
    agilentMosaicIFGReader,
    agilentMosaicTileReader,
)
from orangecontrib.spectroscopy.io.ascii import AsciiColReader, AsciiMapReader
from orangecontrib.spectroscopy.io.diamond import NXS_STXM_Diamond_I08
from orangecontrib.spectroscopy.io.envi import EnviMapReader
from orangecontrib.spectroscopy.io.gsf import GSFReader
from orangecontrib.spectroscopy.io.matlab import MatlabReader
from orangecontrib.spectroscopy.io.maxiv import HDRReader_STXM
from orangecontrib.spectroscopy.io.meta import DatMetaReader, HDRMetaReader
from orangecontrib.spectroscopy.io.neaspec import NeaReader, NeaReaderGSF
from orangecontrib.spectroscopy.io.omnic import OmnicMapReader, SPCReader, SPAReader
from orangecontrib.spectroscopy.io.opus import OPUSReader
from orangecontrib.spectroscopy.io.ptir import PTIRFileReader
from orangecontrib.spectroscopy.io.soleil import (
    SelectColumnReader,
    HDF5Reader_HERMES,
    HDF5Reader_ROCK,
)
from orangecontrib.spectroscopy.io.wire import WiREReaders
