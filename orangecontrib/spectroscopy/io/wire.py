import Orange
import numpy as np
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.util import (
    SpectralFileFormat,
    _spectra_from_image_2d,
)


class WiREReaders(FileFormat, SpectralFileFormat):
    EXTENSIONS = ('.wdf', '.WDF')
    DESCRIPTION = 'Renishaw WiRE WDF reader'

    def read_spectra(self):
        # renishawWiRE is imported here so that its API changes would not block spectroscopy
        from renishawWiRE import WDFReader  # pylint: disable=import-outside-toplevel

        wdf_file = WDFReader(self.filename)

        try:
            if wdf_file.measurement_type == 1:  # single point spectra
                table = self.single_reader(wdf_file)
            elif wdf_file.measurement_type == 2:  # series scan
                table = self.series_reader(wdf_file)
            elif wdf_file.measurement_type == 3:  # line scan
                table = self.map_reader(wdf_file)
        finally:
            wdf_file.close()

        return table

    def single_reader(self, wdf_file):
        domvals = wdf_file.xdata  # energies
        y_data = wdf_file.spectra  # spectra

        return domvals, y_data, None

    def series_reader(self, wdf_file):
        domvals = wdf_file.xdata  # energies
        y_data = wdf_file.spectra  # spectra
        z_locs = wdf_file.zpos  # depth info
        z_locs = z_locs.reshape(-1, 1)

        domain = Orange.data.Domain(
            [], None, metas=[Orange.data.ContinuousVariable.make("map_z")]
        )

        data = Orange.data.Table.from_numpy(
            domain, X=np.zeros((len(y_data), 0)), metas=np.asarray(z_locs, dtype=object)
        )
        return domvals, y_data, data

    def map_reader(self, wdf_file):
        domvals = wdf_file.xdata
        X = wdf_file.spectra  # a (rows, cols, wn) array
        if X.ndim == 2:  # line scan
            spectra = X
        elif X.ndim == 3:  # a map, X is as (rows, cols, wn) array
            spectra = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        else:
            raise NotImplementedError
        x_locs = wdf_file.xpos
        y_locs = wdf_file.ypos
        return _spectra_from_image_2d(spectra, domvals, x_locs, y_locs)
