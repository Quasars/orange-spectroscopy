import numpy as np
import spectral.io
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image


class EnviMapReader(FileFormat, SpectralFileFormat):
    EXTENSIONS = ('.hdr',)
    DESCRIPTION = 'Envi'

    def read_spectra(self):
        a = spectral.io.envi.open(self.filename)
        X = np.array(a.load())
        try:
            lv = a.metadata["wavelength"]
            features = np.array(list(map(float, lv)))
        except KeyError:
            # just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        x_locs = np.arange(X.shape[1])
        y_locs = np.arange(X.shape[0])

        return _spectra_from_image(X, features, x_locs, y_locs)
