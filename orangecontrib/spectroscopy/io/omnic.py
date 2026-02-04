import struct

import Orange
import numpy as np
from Orange.data import FileFormat, Domain, ContinuousVariable

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image
from orangecontrib.spectroscopy.utils import spc
from orangecontrib.spectroscopy.utils.pymca5 import OmnicMap


class OmnicMapReader(FileFormat, SpectralFileFormat):
    """Reader for files with two columns of numbers (X and Y)"""

    EXTENSIONS = ('.map',)
    DESCRIPTION = 'Omnic map'

    def read_spectra(self):
        om = OmnicMap.OmnicMap(self.filename)
        info = om.info
        X = om.data

        try:
            lv = info['OmnicInfo']['Last X value']
            fv = info['OmnicInfo']['First X value']
            features = np.linspace(fv, lv, num=X.shape[-1])
        except KeyError:
            # just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            loc_first = info['OmnicInfo']["First map location"]
            loc_last = info['OmnicInfo']["Last map location"]
            x_locs = np.linspace(
                min(loc_first[0], loc_last[0]),
                max(loc_first[0], loc_last[0]),
                X.shape[1],
            )
            y_locs = np.linspace(
                min(loc_first[1], loc_last[1]),
                max(loc_first[1], loc_last[1]),
                X.shape[0],
            )
        except KeyError:
            x_locs = np.arange(X.shape[1])
            y_locs = np.arange(X.shape[0])

        return _spectra_from_image(X, features, x_locs, y_locs)


class SPAReader(FileFormat, SpectralFileFormat):
    # based on code by Zack Gainsforth

    EXTENSIONS = (".spa", ".SPA", ".srs")
    DESCRIPTION = 'SPA'

    saved_sections = None
    type = None

    def sections(self):
        if self.saved_sections is None:
            with open(self.filename, 'rb') as f:
                name = struct.unpack("30s", f.read(30))[0].decode("ascii")
                extended = "Exte" in name
                f.seek(288)
                ft, v, _, n = struct.unpack('<hhhh', f.read(8))
                self.saved_sections = []
                self.type = ft
                for i in range(n):
                    # Go to the section start.
                    f.seek(304 + (22 if extended else 16) * i)
                    t, offset, length = struct.unpack(
                        '<hqi' if extended else '<hii', f.read(14 if extended else 10)
                    )
                    self.saved_sections.append((i, t, offset, length))
        return self.saved_sections

    SPEC_HEADER = 2

    def find_indextype(self, t):
        for i, a, _, _ in self.sections():
            if a == t:
                return i

    @property
    def sheets(self):
        return ()

    def read_spec_header(self):
        info = self.find_indextype(self.SPEC_HEADER)
        _, _, offset, length = self.sections()[info]
        with open(self.filename, 'rb') as f:
            f.seek(offset)
            dataType, numPoints, xUnits, yUnits, firstX, lastX, noise = struct.unpack(
                '<iiiifff', f.read(28)
            )
            return (
                numPoints,
                firstX,
                lastX,
            )

    def read_spectra(self):
        self.sections()

        if self.type == 1:
            type = 3
        else:
            type = 3  # TODO handle others differently

        numPoints, firstX, lastX = self.read_spec_header()

        _, _, offset, length = self.sections()[self.find_indextype(type)]

        with open(self.filename, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype='float32', count=length // 4)

        if len(data) == numPoints:
            domvals = np.linspace(firstX, lastX, numPoints)
        else:
            domvals = np.arange(len(data))

        data = np.array([data])
        return domvals, data, None


class SPCReader(FileFormat):
    EXTENSIONS = (
        '.spc',
        '.SPC',
    )
    DESCRIPTION = 'Galactic SPC format'

    def read(self):
        spc_file = spc.File(self.filename)
        if spc_file.talabs:
            table = self.multi_x_reader(spc_file)
        else:
            table = self.single_x_reader(spc_file)
        return table

    def single_x_reader(self, spc_file):
        domvals = spc_file.x  # first column is attribute name
        domain = Domain([ContinuousVariable.make("%f" % f) for f in domvals], None)
        y_data = [sub.y for sub in spc_file.sub]
        y_data = np.array(y_data)
        table = Orange.data.Table.from_numpy(domain, y_data.astype(float, order='C'))
        return table

    def multi_x_reader(self, spc_file):
        # use x-values as domain
        all_x = []
        for sub in spc_file.sub:
            x = sub.x
            # assume values in x do not repeat
            all_x = np.union1d(all_x, x)
        domain = Domain([ContinuousVariable.make("%f" % f) for f in all_x], None)

        instances = []
        for sub in spc_file.sub:
            x, y = sub.x, sub.y
            newinstance = np.ones(len(all_x)) * np.nan
            ss = np.searchsorted(all_x, x)  # find positions to set
            newinstance[ss] = y
            instances.append(newinstance)

        y_data = np.array(instances).astype(float, order='C')
        return Orange.data.Table.from_numpy(domain, y_data)
