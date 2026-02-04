import numpy as np
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image


class NXS_STXM_Diamond_I08(FileFormat, SpectralFileFormat):
    """A case specific reader for hyperspectral imaging NXS HDF5
    files from the I08 beamline of the Diamond Light Source.
    These files have very little metadata and don't fully adhere to
    https://manual.nexusformat.org/classes/applications/NXstxm.html"""

    EXTENSIONS = ('.nxs',)
    DESCRIPTION = 'NXS HDF5 file @I08/Diamond Light Source'

    def read_spectra(self):
        import h5py

        hdf5_file = h5py.File(self.filename, mode='r')
        if (
            'entry1/definition' in hdf5_file
            and hdf5_file['entry1/definition'][()].astype('str') == 'NXstxm'
        ):
            grp = hdf5_file['entry1/Counter1']
            x_locs = np.array(grp['sample_x'])
            y_locs = np.array(grp['sample_y'])
            energy = np.array(grp['photon_energy'])
            order = [
                grp[n].attrs['axis'] - 1
                for n in ['sample_y', 'sample_x', 'photon_energy']
            ]
            intensities = np.array(grp['data']).transpose(order)
            return _spectra_from_image(intensities, energy, x_locs, y_locs)
        else:
            raise IOError("Not an NXS HDF5 @I08/Diamond file")
