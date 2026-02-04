from typing import List

import spectral.io
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.agilent import AgilentImageReader
from orangecontrib.spectroscopy.io.ascii import AsciiColReader
from orangecontrib.spectroscopy.io.cls import HDF5Reader_SGM
from orangecontrib.spectroscopy.io.envi import EnviMapReader
from orangecontrib.spectroscopy.io.maxiv import HDRReader_STXM
from orangecontrib.spectroscopy.io.soleil import HDF5Reader_ROCK


class DatMetaReader(FileFormat):
    """Meta-reader to handle agilentImageReader and AsciiColReader name clash
    over .dat extension."""

    EXTENSIONS = ('.dat',)
    DESCRIPTION = 'Spectra ASCII or Agilent Single Tile Image'
    PRIORITY = min(AsciiColReader.PRIORITY, AgilentImageReader.PRIORITY) - 1

    def read(self):
        try:
            # agilentImage requires the .bsp file to be present as well
            return AgilentImageReader(filename=self.filename).read()
        except OSError:
            return AsciiColReader(filename=self.filename).read()


class HDF5MetaReader(FileFormat):
    """Meta-reader to handle HDF5 (currently just .h5) extension(s)"""

    EXTENSIONS = ('.h5',)
    DESCRIPTION = "HDF5 files (ROCK/SGM)"
    PRIORITY = min(HDF5Reader_ROCK.PRIORITY, HDF5Reader_SGM.PRIORITY) - 1

    @property
    def sheets(self) -> List:
        sheets = []
        for reader in [HDF5Reader_SGM, HDF5Reader_ROCK]:
            try:
                sheets.extend(reader(self.filename).sheets)
            except:  # noqa: E722
                pass
        return sheets

    def read(self):
        try:
            reader = HDF5Reader_SGM(self.filename)
            reader.sheet = self.sheet
            return reader.read()
        except (ValueError, IndexError):
            return HDF5Reader_ROCK(filename=self.filename).read()


class HDRMetaReader(FileFormat):
    """Meta-reader to handle EnviMapReader and HDRReader_STXM name clash
    over .hdr extension."""

    EXTENSIONS = ('.hdr',)
    DESCRIPTION = 'Envi hdr or STXM hdr+xim files'
    PRIORITY = min(EnviMapReader.PRIORITY, HDRReader_STXM.PRIORITY) - 1

    def read(self):
        try:
            return EnviMapReader(filename=self.filename).read()
        except spectral.io.envi.FileNotAnEnviHeader:
            return HDRReader_STXM(filename=self.filename).read()
