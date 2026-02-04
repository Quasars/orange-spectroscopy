import shlex

import numpy as np
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image
from orangecontrib.spectroscopy.io.soleil import HDF5Reader_HERMES


class HDRReader_STXM(FileFormat, SpectralFileFormat):
    """Reader for STXM/NEXAFS files from MAX IV and other synchrotrons.
    It is assumed that there are .xim files with plain text data in the same
    directory as the .hdr file. For foo.hdr these are called foo_a000.xim,
    foo_a001.xim etc.
    """

    EXTENSIONS = ('.hdr',)
    DESCRIPTION = 'STXM/NEXAFS .hdr+.xim files'

    def read_hdr_list(self):
        "Read a list of items (1,2,3,{...}) from self._lex"
        d = []
        while True:
            val = self._lex.get_token()
            assert val
            if val == ')':
                self._lex.push_token(')')
                assert int(d[0]) == len(d) - 1
                return d[1:]
            elif val == ',':
                pass
            elif val == '{':
                d.append(self.read_hdr_dict())
                assert self._lex.get_token() == '}'
            elif val[0] == '"':
                d.append(val[1:-1])
            else:
                v = []
                while val not in (')', ','):
                    v.append(val)
                    val = self._lex.get_token()
                    assert val
                self._lex.push_token(val)
                v = ''.join(v)
                try:
                    v = float(v)
                except ValueError:
                    pass
                d.append(v)

    def read_hdr_dict(self, inner=True):
        """Read a dict {name = 'value'; foo = (...);} from self._lex;
        inner=False for the outermost level.
        """
        d = {}
        while True:
            name = self._lex.get_token()
            if not name:
                assert not inner
                return d
            elif name == '}':
                assert inner
                self._lex.push_token(name)
                return d
            assert self._lex.get_token() == '='
            val = self._lex.get_token()
            if val == '{':
                d[name] = self.read_hdr_dict()
                assert self._lex.get_token() == '}'
            elif val == '(':
                d[name] = self.read_hdr_list()
                assert self._lex.get_token() == ')'
            elif val[0] == '"':
                d[name] = val[1:-1]
            else:
                v = []
                while val != ';':
                    v.append(val)
                    val = self._lex.get_token()
                self._lex.push_token(';')
                v = ''.join(v)
                try:
                    v = float(v)
                except ValueError:
                    pass
                d[name] = v
            assert self._lex.get_token() == ';'

    def read_spectra(self):
        with open(self.filename, 'rt', encoding="utf8") as f:
            # Parse file contents into dictionaries/lists
            self._lex = shlex.shlex(instream=f)
            try:
                hdrdata = self.read_hdr_dict(inner=False)
            except AssertionError as e:
                raise IOError('Error parsing hdr file ' + self.filename) from e
        regions = hdrdata['ScanDefinition']['Regions'][0]
        axes = [
            regions['QAxis'],
            regions['PAxis'],
            hdrdata['ScanDefinition']['StackAxis'],
        ]
        dims = [len(ax['Points']) for ax in axes]
        spectra = np.empty(dims)
        for nf in range(dims[2]):
            ximname = '%s_a%03d.xim' % (self.filename[:-4], nf)
            xim = np.loadtxt(ximname)
            spectra[..., nf] = xim

        x_loc = axes[1]['Points']
        y_loc = axes[0]['Points']
        features = np.asarray(axes[2]['Points'])
        return _spectra_from_image(spectra, features, x_loc, y_loc)


class HDF5Reader_SoftiMAX(FileFormat, SpectralFileFormat):
    """A very case specific reader for HDF5 files from the SoftiMAX beamline in MAX-IV"""

    EXTENSIONS = ('.hdf5',)
    DESCRIPTION = 'HDF5 file @SoftiMAX/MAX-IV'
    PRIORITY = HDF5Reader_HERMES.PRIORITY + 1

    def read_spectra(self):
        import h5py

        with h5py.File(self.filename, 'r') as hdf5_file:
            if 'entry1/collection/beamline' in hdf5_file and hdf5_file[
                'entry1/collection/beamline'
            ][()].astype('str') == ['SLS Sophie at Softimax MAXIV']:
                x_locs = np.array(hdf5_file['entry1/counter0/sample_x'])
                y_locs = np.array(hdf5_file['entry1/counter0/sample_y'])
                energy = np.array(hdf5_file['entry1/counter0/energy'])
                intensities = np.array(hdf5_file['entry1/counter0/data']).T
                return _spectra_from_image(intensities, energy, y_locs, x_locs)
            else:
                raise IOError("Not a SoftiMAX file")
