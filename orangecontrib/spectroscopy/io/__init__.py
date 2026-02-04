# FileFormat readers must be imported here

# Meta-readers to handle extension conflicts
from .meta import DatMetaReader, HDRMetaReader, HDF5MetaReader

# General readers
from .ascii import AsciiColReader, AsciiMapReader
from .envi import EnviMapReader
from .gsf import GSFReader
from .matlab import MatlabReader

# Instrument-specific readers
from .agilent import (
    AgilentImageReader,
    AgilentImageIFGReader,
    agilentMosaicReader,
    agilentMosaicIFGReader,
    agilentMosaicTileReader,
)
from .neaspec import NeaReader, NeaReaderGSF, NeaReaderMultiChannel
from .omnic import OmnicMapReader, SPAReader, SPCReader
from .opus import OPUSReader
from .ptir import PTIRFileReader
from .wire import WiREReaders
from .perkinelmer import PerkinElmerReader

# Facility-specific readers
from .diamond import NXS_STXM_Diamond_I08
from .maxiv import HDRReader_STXM, HDF5Reader_SoftiMAX
from .soleil import SelectColumnReader, HDF5Reader_HERMES, HDF5Reader_ROCK
from .cls import HDF5Reader_SGM
