import Orange
import numpy as np
from Orange.data import FileFormat, ContinuousVariable, Table, Domain
from Orange.data.io import CSVReader

from orangecontrib.spectroscopy.util import getx
from orangecontrib.spectroscopy.io.util import SpectralFileFormat
from orangecontrib.spectroscopy.utils import MAP_X_VAR, MAP_Y_VAR


class AsciiColReader(FileFormat, SpectralFileFormat):
    """ Reader for files with multiple columns of numbers. The first column
    contains the wavelengths, the others contain the spectra. """
    EXTENSIONS = ('.dat', '.dpt', '.xy', '.csv')
    DESCRIPTION = 'Spectra ASCII'

    PRIORITY = CSVReader.PRIORITY + 1

    def read_spectra(self):
        tbl = None
        delimiters = [";", None, ":", ","]
        for d in delimiters:
            try:
                comments = [a for a in [";", "#"] if a != d]
                tbl = np.loadtxt(self.filename, ndmin=2, delimiter=d, comments=comments)
                break
            except ValueError:
                pass
        if tbl is None:
            raise ValueError('File should be delimited by <whitespace>, ";", ":", or ",".')
        wavenumbers = tbl.T[0]  # first column is attribute name
        datavals = tbl.T[1:]
        return wavenumbers, datavals, None

    @staticmethod
    def write_file(filename, data):
        xs = getx(data)
        xs = xs.reshape((-1, 1))
        table = np.hstack((xs, data.X.T))
        np.savetxt(filename, table, delimiter="\t", fmt="%g")


class AsciiMapReader(FileFormat):
    """ Reader ascii map files.

    First row contains wavelengths, then each row describes a spectrum, starting with (x, y)
    coordinates: http://www.cytospec.com/file.php#FileASCII3 """
    EXTENSIONS = ('.xyz',)
    DESCRIPTION = 'Hyperspectral map ASCII'

    def read(self):
        with open(self.filename, "rb") as f:
            # read first row separately because of two empty columns
            header = f.readline().decode("ascii").rstrip().split("\t")
            header = [a.strip() for a in header]
            assert header[0] == header[1] == ""
            dom_vals = [float(v) for v in header[2:]]
            domain = Orange.data.Domain([ContinuousVariable.make("%f" % f) for f in dom_vals], None)
            tbl = np.loadtxt(f, ndmin=2)
            data = Table.from_numpy(domain, X=tbl[:, 2:])
            metas = [ContinuousVariable.make(MAP_X_VAR), ContinuousVariable.make(MAP_Y_VAR)]
            domain = Orange.data.Domain(domain.attributes, None, metas=metas)
            data = data.transform(domain)
            with data.unlocked(data.metas):
                data[:, metas[0]] = tbl[:, 0].reshape(-1, 1)
                data[:, metas[1]] = tbl[:, 1].reshape(-1, 1)
            return data

    @staticmethod
    def write_file(filename, data):
        wavelengths = getx(data)
        map_x = data.domain[MAP_X_VAR] if MAP_X_VAR in data.domain else ContinuousVariable(MAP_X_VAR)
        map_y = data.domain[MAP_Y_VAR] if MAP_Y_VAR in data.domain else ContinuousVariable(MAP_Y_VAR)
        ndom = Domain([map_x, map_y] + list(data.domain.attributes))
        data = data.transform(ndom)
        with open(filename, "wb") as f:
            header = ["", ""] + [("%g" % w) for w in wavelengths]
            f.write(('\t'.join(header) + '\n').encode("ascii"))
            np.savetxt(f, data.X, delimiter="\t", fmt="%g")