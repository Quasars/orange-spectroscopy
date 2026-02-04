import numpy as np
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image


def reader_gsf(file_path):
    with open(file_path, "rb") as f:
        if not f.readline() == b'Gwyddion Simple Field 1.0\n':
            raise ValueError('Not a correct GSF file, wrong header.')

        meta = {}

        term = False  # there are mandatory fileds
        while term != b'\x00':
            l = f.readline().decode('utf-8')
            name, value = l.split("=")
            name = name.strip()
            value = value.strip()
            meta[name] = value
            term = f.read(1)
            f.seek(-1, 1)

        f.read(4 - f.tell() % 4)

        meta["XRes"] = XR = int(meta["XRes"])
        meta["YRes"] = YR = int(meta["YRes"])
        meta["XReal"] = XReal = float(meta.get("XReal", 1))
        meta["YReal"] = YReal = float(meta.get("YReal", 1))
        meta["XOffset"] = XOffset = float(meta.get("XOffset", 0))
        meta["YOffset"] = YOffset = float(meta.get("YOffset", 0))
        meta["Title"] = meta.get("Title", None)
        meta["XYUnits"] = meta.get("XYUnits", None)
        meta["ZUnits"] = meta.get("ZUnits", None)

        X = np.fromfile(f, dtype='float32', count=XR * YR).reshape(XR, YR)

        XRr = np.arange(XR)
        # TODO change this to the NeaSCAN orientation
        YRr = np.arange(
            YR - 1, -1, -1
        )  # needed to have the same orientation as in Gwyddion

        XRr = XOffset * 1e6 + (XReal * 1e6 / XR) * XRr
        YRr = YOffset * 1e6 + (YReal * 1e6 / YR) * YRr

        X = X.reshape((meta["YRes"], meta["XRes"]) + (1,))

    return X, XRr, YRr


class GSFReader(FileFormat, SpectralFileFormat):
    EXTENSIONS = (".gsf",)
    DESCRIPTION = 'Gwyddion Simple Field'

    def read_spectra(self):
        X, XRr, YRr = reader_gsf(self.filename)
        data = _spectra_from_image(X, np.array([1]), XRr, YRr)
        return data
