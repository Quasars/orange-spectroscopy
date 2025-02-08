import numpy as np
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image

import yaml

class SelectColumnReader(FileFormat, SpectralFileFormat):
    """ Reader for files with multiple columns of numbers. The first column
    contains the wavelengths, the others contain the spectra. """
    EXTENSIONS = ('.txt',)
    DESCRIPTION = 'XAS ascii spectrum from ROCK'
    PRIORITY = 9999

    @property
    def sheets(self):

        with open(self.filename, 'rt', encoding="utf8") as dataf:
            l = ""
            for l in dataf:
                if not l.startswith('#'):
                    break
            col_nbrs = range(2, min(len(l.split()) + 1, 11))

        return list(map(str, col_nbrs))

    def read_spectra(self):

        if self.sheet:
            col_nb = int(self.sheet)
        else:
            col_nb = int(self.sheets[0])

        spectrum = np.loadtxt(self.filename, comments='#',
                              usecols=(0, col_nb - 1),
                              unpack=True)
        return spectrum[0], np.atleast_2d(spectrum[1]), None


class HDF5Reader_HERMES(FileFormat, SpectralFileFormat):
    """ A very case specific reader for HDF5 files from the HEREMES beamline in SOLEIL"""
    EXTENSIONS = ('.hdf5',)
    DESCRIPTION = 'HDF5 file @HERMRES/SOLEIL'

    def read_spectra(self):
        import h5py
        with h5py.File(self.filename, 'r') as hdf5_file:
            if 'entry1/collection/beamline' in hdf5_file and \
                    hdf5_file['entry1/collection/beamline'][()].astype('str') == 'Hermes':
                x_locs = np.array(hdf5_file['entry1/Counter0/sample_x'])
                y_locs = np.array(hdf5_file['entry1/Counter0/sample_y'])
                energy = np.array(hdf5_file['entry1/Counter0/energy'])
                intensities = np.array(hdf5_file['entry1/Counter0/data']).T
                return _spectra_from_image(intensities, energy, x_locs, y_locs)
            else:
                raise IOError("Not an HDF5 HERMES file")


class HDF5Reader_HERMES_auto(FileFormat, SpectralFileFormat):
    """ A very case specific reader for HDF5 files from the HEREMES beamline in SOLEIL configured through a yaml file"""
    EXTENSIONS = ('.hdf5',)
    DESCRIPTION = 'HDF5 auto file @HERMRES/SOLEIL'

    def get_config(self):
        # TODO should be set for the whole package somewhere in __init__
        h5_config_file = 'orangecontrib/spectroscopy/io/hdf5_reader_config.yaml'
        with open(h5_config_file) as h5f:
            all_config = yaml.safe_load(h5f)

        config_id = 'soleil_hermes_beamline_2024' # must be set for each reader separately

        return all_config['h5entries'][config_id]

    def read_spectra(self):
        import h5py # TODO why is this here and not in the top?

        h5_config = self.get_config()

        hdf5_file = h5py.File(self.filename)

        if h5_config['format_id'] in hdf5_file and \
                hdf5_file[h5_config['format_id']][()].astype('str') == h5_config['format_id_value']:
            x_locs = np.array(hdf5_file[h5_config['x_locs']])
            y_locs = np.array(hdf5_file[h5_config['y_locs']])
            energy = np.array(hdf5_file[h5_config['energies']])
            intensities = np.array(hdf5_file[h5_config['intensities']]).T
            return _spectra_from_image(intensities, energy, x_locs, y_locs)
        else:
            raise IOError("Not an HDF5 HERMES file")


class PreconfHDF5Reader(FileFormat, SpectralFileFormat):
    """ A very general reader for pre-configured HDF5 files"""
    EXTENSIONS = ('.hdf5, .nxs',)
    DESCRIPTION = 'HDF5 reader'

    def get_h5configs(self):
        # TODO should be set for the whole package somewhere in __init__
        h5_config_file = 'orangecontrib/spectroscopy/io/hdf5_reader_config.yaml'
        with open(h5_config_file) as h5f:
            all_configs = yaml.safe_load(h5f)

        return all_configs['h5entries']

    def read_spectra(self):
        import h5py # TODO why is this here and not in the top?

        all_configs = self.get_h5configs()

        hdf5_file = h5py.File(self.filename)

        for config in all_configs:
            if all_configs[config]['format_id'] in hdf5_file: # one entry might not be enough but a tree can be...
                # if hdf5_file[all_configs[config]['format_id']][()].astype('str') == all_configs[config]['format_id_value']:
                # we could do such test, but this is not general - testing the structure is probably better
                x_locs = np.array(hdf5_file[all_configs[config]['x_locs']])
                y_locs = np.array(hdf5_file[all_configs[config]['y_locs']])
                energy = np.array(hdf5_file[all_configs[config]['energies']])
                intensities = np.array(hdf5_file[all_configs[config]['intensities']]).T
                return _spectra_from_image(intensities, energy, x_locs, y_locs)
            else:
                raise IOError(f"IOError {self.filename}")


class HDF5Reader_ROCK(FileFormat, SpectralFileFormat):
    """ A very case specific reader for hyperspectral imaging HDF5
    files from the ROCK beamline in SOLEIL"""
    EXTENSIONS = ('.h5',)
    DESCRIPTION = 'HDF5 file @ROCK(hyperspectral imaging)/SOLEIL'

    @property
    def sheets(self):
        import h5py as h5

        with h5.File(self.filename, "r") as dataf:
            cube_nbrs = range(1, len(dataf["data"].keys())+1)

        return list(map(str, cube_nbrs))

    def read_spectra(self):

        import h5py as h5

        if self.sheet:
            cube_nb = int(self.sheet)
        else:
            cube_nb = 1

        with h5.File(self.filename, "r") as dataf:
            cube_h5 = dataf["data/cube_{:0>5d}".format(cube_nb)]

            # directly read into float64 so that Orange.data.Table does not
            # convert to float64 afterwards (if we would not read into float64,
            # the memory use would be 50% greater)
            cube_np = np.empty(cube_h5.shape, dtype=np.float64)
            cube_h5.read_direct(cube_np)

            energies = np.array(dataf['context/energies'])

        intensities = np.transpose(cube_np, (1, 2, 0))
        height, width, _ = np.shape(intensities)
        x_locs = np.arange(width)
        y_locs = np.arange(height)

        return _spectra_from_image(intensities, energies, x_locs, y_locs)