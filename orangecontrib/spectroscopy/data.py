from collections import defaultdict
from functools import reduce
import numbers
import struct

import numpy as np
import spectral.io.envi
from scipy.interpolate import interp1d
from scipy.io import matlab

import Orange
from Orange.data import \
    ContinuousVariable, StringVariable, TimeVariable, Domain, Table
from Orange.data.io import FileFormat
import Orange.data.io

from .pymca5 import OmnicMap
from .agilent import agilentImage, agilentMosaic, agilentImageIFG, agilentMosaicIFG

SET_flow = False    # Use for manual control flow in some function inter Class
                    # See example in class NeaSpec and GSFReader

class SpectralFileFormat:

    def read_spectra(self):
        """ Fast reading of spectra. Return spectral information
        in two arrays (wavelengths and values). Only additional
        attributes (usually metas) are returned as a Table.

        Return a triplet:
            - 1D numpy array,
            - 2D numpy array with the same last dimension as xs,
            - Orange.data.Table with only meta or class attributes
        """
        pass

    def read(self):
        return build_spec_table(*self.read_spectra())


class AsciiColReader(FileFormat, SpectralFileFormat):
    """ Reader for files with multiple columns of numbers. The first column
    contains the wavelengths, the others contain the spectra. """
    EXTENSIONS = ('.dat', '.dpt', '.xy',)
    DESCRIPTION = 'Spectra ASCII'

    def read_spectra(self):
        tbl = np.loadtxt(self.filename, ndmin=2)
        wavenumbers = tbl.T[0]  # first column is attribute name
        datavals = tbl.T[1:]
        return wavenumbers, datavals, None

    @staticmethod
    def write_file(filename, data):
        xs = getx(data)
        xs = xs.reshape((-1, 1))
        table = np.hstack((xs, data.X.T))
        np.savetxt(filename, table, delimiter="\t", fmt="%g")


class SelectColumnReader(FileFormat, SpectralFileFormat):
    """ Reader for files with multiple columns of numbers. The first column
    contains the wavelengths, the others contain the spectra. """
    EXTENSIONS = ('.txt',)
    DESCRIPTION = 'XAS ascii spectrum from ROCK'

    @property
    def sheets(self):

        with open(self.filename, 'r') as dataf:
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
            data = Orange.data.Table(domain, tbl[:, 2:])
            metas = [ContinuousVariable.make('map_x'), ContinuousVariable.make('map_y')]
            domain = Orange.data.Domain(domain.attributes, None, metas=metas)
            data = data.transform(domain)
            data[:, metas[0]] = tbl[:, 0].reshape(-1, 1)
            data[:, metas[1]] = tbl[:, 1].reshape(-1, 1)
            return data

    @staticmethod
    def write_file(filename, data):
        wavelengths = getx(data)
        try:
            ndom = Domain([data.domain["map_x"], data.domain["map_y"]] +
                          list(data.domain.attributes))
        except KeyError:
            raise RuntimeError('Data needs to include meta variables '
                               '"map_x" and "map_y"')
        data = data.transform(ndom)
        with open(filename, "wb") as f:
            header = ["", ""] + [("%g" % w) for w in wavelengths]
            f.write(('\t'.join(header) + '\n').encode("ascii"))
            np.savetxt(f, data.X, delimiter="\t", fmt="%g")


def _spectra_from_image(X, features, x_locs, y_locs):
    """
    Create a spectral format (returned by SpectralFileFormat.read_spectra)
    from 3D image organized [ rows, columns, wavelengths ]
    """
    X = np.asarray(X)
    x_locs = np.asarray(x_locs)
    y_locs = np.asarray(y_locs)

    # each spectrum has its own row
    spectra = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))

    # locations
    y_loc = np.repeat(np.arange(X.shape[0]), X.shape[1])
    x_loc = np.tile(np.arange(X.shape[1]), X.shape[0])
    metas = np.array([x_locs[x_loc], y_locs[y_loc]]).T

    domain = Orange.data.Domain([], None,
                                metas=[Orange.data.ContinuousVariable.make("map_x"),
                                       Orange.data.ContinuousVariable.make("map_y")]
                               )
    data = Orange.data.Table.from_numpy(domain, X=np.zeros((len(spectra), 0)),
                                        metas=np.asarray(metas, dtype=object))
    return features, spectra, data


class MatlabReader(FileFormat):
    EXTENSIONS = ('.mat',)
    DESCRIPTION = "Matlab"

    # Matlab 7.3+ files are not handled by scipy reader

    def read(self):
        who = matlab.whosmat(self.filename)
        if not who:
            raise IOError("Couldn't load matlab file " + self.filename)
        else:
            ml = matlab.loadmat(self.filename, chars_as_strings=True)
            ml = {a: b for a, b in ml.items() if isinstance(b, np.ndarray)}

            def num_elements(array):
                return reduce(lambda x, y: x * y, array.shape, 1)

            def find_biggest(arrays):
                sizes = []
                for n, c in arrays.items():
                    sizes.append((num_elements(c), n))
                return max(sizes)[1]

            def is_string_array(array):
                return issubclass(array.dtype.type, np.str_)

            def is_number_array(array):
                return issubclass(array.dtype.type, numbers.Number)

            numeric = {n: a for n, a in ml.items() if is_number_array(a)}

            # X is the biggest numeric array
            X = ml.pop(find_biggest(numeric)) if numeric else None

            # find an array with compatible shapes
            attributes = []
            if X is not None:
                name_array = None
                for name in sorted(ml):
                    con = ml[name]
                    if con.shape in [(X.shape[1],), (1, X.shape[1])]:
                        name_array = name
                        break
                names = ml.pop(name_array).ravel() if name_array else range(X.shape[1])
                names = [str(a).rstrip() for a in names]  # remove matlab char padding
                attributes = [ContinuousVariable.make(a) for a in names]

            meta_names = []
            metas = []

            meta_size = None
            if X is None:
                counts = defaultdict(list)
                for name, con in ml.items():
                    counts[len(con)].append(name)
                if counts:
                    meta_size = max(counts.keys(), key=lambda x: len(counts[x]))
            else:
                meta_size = len(X)
            if meta_size:
                for name, con in ml.items():
                    if len(con) == meta_size:
                        meta_names.append(name)

            meta_data = []
            for m in sorted(meta_names):
                f = ml[m]
                if is_string_array(f) and len(f.shape) == 1:  # 1D string arrays
                    metas.append(StringVariable.make(m))
                    f = np.array([a.rstrip() for a in f])  # remove matlab char padding
                    f.resize(meta_size, 1)
                    meta_data.append(f)
                elif is_number_array(f) and len(f.shape) == 2:
                    if f.shape[1] == 1:
                        names = [m]
                    else:
                        names = [m + "_" + str(i+1) for i in range(f.shape[1])]
                    for n in names:
                        metas.append(ContinuousVariable.make(n))
                    meta_data.append(f)

            meta_data = np.hstack(tuple(meta_data)) if meta_data else None

            domain = Domain(attributes, metas=metas)
            if X is None:
                X = np.zeros((meta_size, 0))
            return Orange.data.Table.from_numpy(domain, X, Y=None, metas=meta_data)


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
            #just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        x_locs = np.arange(X.shape[1])
        y_locs = np.arange(X.shape[0])

        return _spectra_from_image(X, features, x_locs, y_locs)


class HDF5Reader_HERMES(FileFormat, SpectralFileFormat):
    """ A very case specific reader for HDF5 files from the HEREMES beamline in SOLEIL"""
    EXTENSIONS = ('.hdf5',)
    DESCRIPTION = 'HDF5 file @HERMRES/SOLEIL'

    def read_spectra(self):
        import h5py
        hdf5_file = h5py.File(self.filename)
        if hdf5_file['entry1/collection/beamline'].value.astype('str') == 'Hermes':
            x_locs = np.array(hdf5_file['entry1/Counter0/sample_x'])
            y_locs = np.array(hdf5_file['entry1/Counter0/sample_y'])
            energy = np.array(hdf5_file['entry1/Counter0/energy'])
            intensities = np.array(hdf5_file['entry1/Counter0/data']).T
        return _spectra_from_image(intensities, energy, x_locs, y_locs)


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


class OmnicMapReader(FileFormat, SpectralFileFormat):
    """ Reader for files with two columns of numbers (X and Y)"""
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
            #just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            loc_first = info['OmnicInfo']["First map location"]
            loc_last = info['OmnicInfo']["Last map location"]
            x_locs = np.linspace(min(loc_first[0], loc_last[0]),
                                 max(loc_first[0], loc_last[0]), X.shape[1])
            y_locs = np.linspace(min(loc_first[1], loc_last[1]),
                                 max(loc_first[1], loc_last[1]), X.shape[0])
        except KeyError:
            x_locs = np.arange(X.shape[1])
            y_locs = np.arange(X.shape[0])

        return _spectra_from_image(X, features, x_locs, y_locs)


class AgilentImageReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA single tile image files"""
    EXTENSIONS = ('.dat',)
    DESCRIPTION = 'Agilent Single Tile Image'

    def read_spectra(self):
        ai = agilentImage(self.filename)
        info = ai.info
        X = ai.data

        try:
            features = info['wavenumbers']
        except KeyError:
            #just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        return _spectra_from_image(X, features, x_locs, y_locs)


class AgilentImageIFGReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA single tile image files (IFG)"""
    EXTENSIONS = ('.seq',)
    DESCRIPTION = 'Agilent Single Tile Image (IFG)'

    def read_spectra(self):
        ai = agilentImageIFG(self.filename)
        info = ai.info
        X = ai.data

        features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        features, data, additional_table = _spectra_from_image(X, features, x_locs, y_locs)

        import_params = ['Effective Laser Wavenumber',
                         'Under Sampling Ratio',
        ]
        new_attributes = []
        new_columns = []
        for param_key in import_params:
            try:
                param = info[param_key]
            except KeyError:
                pass
            else:
                new_attributes.append(ContinuousVariable.make(param_key))
                new_columns.append(np.full((len(data),), param))

        domain = Domain(additional_table.domain.attributes,
                        additional_table.domain.class_vars,
                        additional_table.domain.metas + tuple(new_attributes))
        table = additional_table.transform(domain)
        table[:, new_attributes] = np.asarray(new_columns).T

        return (features, data, table)


class agilentMosaicReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA mosaic image files"""
    EXTENSIONS = ('.dmt',)
    DESCRIPTION = 'Agilent Mosaic Image'

    def read_spectra(self):
        am = agilentMosaic(self.filename)
        info = am.info
        X = am.data

        try:
            features = info['wavenumbers']
        except KeyError:
            #just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        return _spectra_from_image(X, features, x_locs, y_locs)


class agilentMosaicIFGReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA mosaic image files"""
    EXTENSIONS = ('.dmt',)
    DESCRIPTION = 'Agilent Mosaic Image (IFG)'
    PRIORITY = agilentMosaicReader.PRIORITY + 1

    def read_spectra(self):
        am = agilentMosaicIFG(self.filename)
        info = am.info
        X = am.data

        features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        features, data, additional_table = _spectra_from_image(X, features, x_locs, y_locs)

        import_params = ['Effective Laser Wavenumber',
                         'Under Sampling Ratio',
        ]
        new_attributes = []
        new_columns = []
        for param_key in import_params:
            try:
                param = info[param_key]
            except KeyError:
                pass
            else:
                new_attributes.append(ContinuousVariable.make(param_key))
                new_columns.append(np.full((len(data),), param))

        domain = Domain(additional_table.domain.attributes,
                        additional_table.domain.class_vars,
                        additional_table.domain.metas + tuple(new_attributes))
        table = additional_table.transform(domain)
        table[:, new_attributes] = np.asarray(new_columns).T

        return (features, data, table)


class SPCReader(FileFormat):
    EXTENSIONS = ('.spc', '.SPC',)
    DESCRIPTION = 'Galactic SPC format'

    def read(self):
        try:
            import spc
        except ImportError:
            raise RuntimeError("To load spc files install spc python module "
                               "(https://github.com/rohanisaac/spc)")

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
            newinstance = np.ones(len(all_x))*np.nan
            ss = np.searchsorted(all_x, x)  # find positions to set
            newinstance[ss] = y
            instances.append(newinstance)

        y_data = np.array(instances).astype(float, order='C')
        return Orange.data.Table.from_numpy(domain, y_data)


class OPUSReader(FileFormat):
    """Reader for OPUS files"""

    EXTENSIONS = (".0*", ".1*", ".2*", ".3*", ".4*", ".5*", ".6*", ".7*", ".8*", ".9*")
    DESCRIPTION = 'OPUS Spectrum'

    _OPUS_WARNING = "Opus files require the opusFC module (https://pypi.org/project/opusFC/)"

    @property
    def sheets(self):
        try:
            import opusFC
        except ImportError:
            # raising an exception here would just show an generic error in File widget
            return ()
        dbs = []
        for db in opusFC.listContents(self.filename):
            dbs.append(db[0] + " " + db[1] + " " + db[2])
        return dbs

    def read(self):
        try:
            import opusFC
        except ImportError:
            raise RuntimeError(self._OPUS_WARNING)

        if self.sheet:
            db = self.sheet
        else:
            db = self.sheets[0]

        db = tuple(db.split(" "))
        dim = db[1]

        try:
            data = opusFC.getOpusData(self.filename, db)
        except Exception:
            raise IOError("Couldn't load spectrum from " + self.filename)

        attrs, clses, metas = [], [], []

        attrs = [ContinuousVariable.make(repr(data.x[i]))
                 for i in range(data.x.shape[0])]

        y_data = None
        meta_data = None

        if type(data) == opusFC.MultiRegionDataReturn:
            y_data = []
            meta_data = []
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y'),
                          StringVariable.make('map_region'),
                          TimeVariable.make('start_time')])
            for region in data.regions:
                y_data.append(region.spectra)
                mapX = region.mapX
                mapY = region.mapY
                map_region = np.full_like(mapX, region.title, dtype=object)
                start_time = region.start_time
                meta_region = np.column_stack((mapX, mapY,
                                               map_region, start_time))
                meta_data.append(meta_region.astype(object))
            y_data = np.vstack(y_data)
            meta_data = np.vstack(meta_data)

        elif type(data) == opusFC.MultiRegionTRCDataReturn:
            y_data = []
            meta_data = []
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y'),
                          StringVariable.make('map_region')])
            attrs = [ContinuousVariable.make(repr(data.labels[i]))
                     for i in range(len(data.labels))]
            for region in data.regions:
                y_data.append(region.spectra)
                mapX = region.mapX
                mapY = region.mapY
                map_region = np.full_like(mapX, region.title, dtype=object)
                meta_region = np.column_stack((mapX, mapY, map_region))
                meta_data.append(meta_region.astype(object))
            y_data = np.vstack(y_data)
            meta_data = np.vstack(meta_data)

        elif type(data) == opusFC.ImageDataReturn:
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y')])

            data_3D = data.spectra

            for i in np.ndindex(data_3D.shape[:1]):
                map_y = np.full_like(data.mapX, data.mapY[i])
                coord = np.column_stack((data.mapX, map_y))
                if y_data is None:
                    y_data = data_3D[i]
                    meta_data = coord.astype(object)
                else:
                    y_data = np.vstack((y_data, data_3D[i]))
                    meta_data = np.vstack((meta_data, coord))

        elif type(data) == opusFC.ImageTRCDataReturn:
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y')])

            attrs = [ContinuousVariable.make(repr(data.labels[i]))
                     for i in range(len(data.labels))]
            data_3D = data.traces

            for i in np.ndindex(data_3D.shape[:1]):
                map_y = np.full_like(data.mapX, data.mapY[i])
                coord = np.column_stack((data.mapX, map_y))
                if y_data is None:
                    y_data = data_3D[i]
                    meta_data = coord.astype(object)
                else:
                    y_data = np.vstack((y_data, data_3D[i]))
                    meta_data = np.vstack((meta_data, coord))

        elif type(data) == opusFC.TimeResolvedTRCDataReturn:
            y_data = data.traces

        elif type(data) == opusFC.TimeResolvedDataReturn:
            metas.extend([ContinuousVariable.make('z')])

            y_data = data.spectra
            meta_data = data.z

        elif type(data) == opusFC.SingleDataReturn:
            y_data = data.y[None, :]

        else:
            raise ValueError("Empty or unsupported opusFC DataReturn object: " + type(data))

        import_params = ['SRT', 'SNM']

        for param_key in import_params:
            try:
                param = data.parameters[param_key]
            except KeyError:
                pass  # TODO should notify user?
            else:
                try:
                    param_name = opusFC.paramDict[param_key]
                except KeyError:
                    param_name = param_key
                if param_key == 'SRT':
                    var = TimeVariable.make(param_name)
                elif type(param) is float:
                    var = ContinuousVariable.make(param_name)
                elif type(param) is str:
                    var = StringVariable.make(param_name)
                else:
                    raise ValueError #Found a type to handle
                metas.extend([var])
                params = np.full((y_data.shape[0],), param, np.array(param).dtype)
                if meta_data is not None:
                    # NB dtype default will be np.array(fill_value).dtype in future
                    meta_data = np.column_stack((meta_data, params.astype(object)))
                else:
                    meta_data = params

        domain = Orange.data.Domain(attrs, clses, metas)

        meta_data = np.atleast_2d(meta_data)

        table = Orange.data.Table.from_numpy(domain,
                                             y_data.astype(float, order='C'),
                                             metas=meta_data)

        return table


class SPAReader(FileFormat, SpectralFileFormat):
    #based on code by Zack Gainsforth

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
                    t, offset, length = struct.unpack('<hqi' if extended else '<hii',
                                                      f.read(14 if extended else 10))
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
            dataType, numPoints, xUnits, yUnits, firstX, lastX, noise = \
                struct.unpack('<iiiifff', f.read(28))
            return numPoints, firstX, lastX,

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
            data = np.fromfile(f, dtype='float32', count=length//4)

        if len(data) == numPoints:
            domvals = np.linspace(firstX, lastX, numPoints)
        else:
            domvals = np.arange(len(data))

        data = np.array([data])
        return domvals, data, None


class GSFReader(FileFormat, SpectralFileFormat):

    EXTENSIONS = (".gsf",)
    DESCRIPTION = 'Gwyddion Simple Field'

    def read_spectra(self):
        with open(self.filename, "rb") as f:
            if not (f.readline() == b'Gwyddion Simple Field 1.0\n'):
                raise ValueError('Not a correct GSF file, wrong header.')

            meta = {}

            term = False #there are mandatory fileds
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
            meta["XReal"] = float(meta.get("XReal", 1))
            meta["YReal"] = float(meta.get("YReal", 1))
            meta["XOffset"] = float(meta.get("XOffset", 0))
            meta["YOffset"] = float(meta.get("YOffset", 0))
            meta["Title"] = meta.get("Title", None)
            meta["XYUnits"] = meta.get("XYUnits", None)
            meta["ZUnits"] = meta.get("ZUnits", None)

            X = np.fromfile(f, dtype='float32', count=XR*YR).reshape(XR, YR)

            XRr = np.arange(XR)
            YRr = np.arange(YR-1, -1, -1) # needed to have the same orientation as in Gwyddion

            X = X.reshape((meta["YRes"], meta["XRes"]) + (1,))
            
            global SET_flow
            if SET_flow:    #True if from class NeaReader()
                return X

            data = _spectra_from_image(X, [1], XRr, YRr)

            return data


class NeaReader(FileFormat, SpectralFileFormat):

    EXTENSIONS = (".nea", ".txt", '.gsf')
    DESCRIPTION = 'NeaSPEC'

    global SET_flow 
    SET_flow = True     #Set up for change the flow in GSFReader 

    def read_v1(self):

        with open(self.filename, "rt") as f:
            next(f)  # skip header
            l = next(f)
            l = l.strip()
            l = l.split("\t")
            ncols = len(l)

            f.seek(0)
            next(f)
            datacols = np.arange(4, ncols)
            data = np.loadtxt(f, dtype="float", usecols=datacols)

            f.seek(0)
            next(f)
            metacols = np.arange(0, 4)
            meta = np.loadtxt(f,
                              dtype={'names': ('row', 'column', 'run', 'channel'),
                                     'formats': (np.int, np.int, np.int, "S10")},
                              usecols=metacols)

            # ASSUMTION: runs start with 0
            runs = np.unique(meta["run"])

            # ASSUMPTION: there is one M channel and multiple O?A and O?P channels,
            # both with the same number, both starting with 0
            channels = np.unique(meta["channel"])
            maxn = -1

            def channel_type(a):
                if a.startswith(b"O") and a.endswith(b"A"):
                    return "OA"
                elif a.startswith(b"O") and a.endswith(b"P"):
                    return "OP"
                else:
                    return "M"

            for a in channels:
                if channel_type(a) in ("OA", "OP"):
                    maxn = max(maxn, int(a[1:-1]))
            numharmonics = maxn+1

            rowcols = np.vstack((meta["row"], meta["column"])).T
            uniquerc = set(map(tuple, rowcols))

            di = {}  # dictionary of indices for each row and column

            min_intp, max_intp = None, None

            for i, (row, col, run, chan) in enumerate(meta):
                if (row, col) not in di:
                    di[(row, col)] = \
                        {"M": np.zeros((len(runs), len(datacols))) * np.nan,
                         "OA": np.zeros((numharmonics, len(runs), len(datacols))) * np.nan,
                         "OP": np.zeros((numharmonics, len(runs), len(datacols))) * np.nan}
                if channel_type(chan) == "M":
                    di[(row, col)][channel_type(chan)][run] = data[i]
                    if min_intp is None:  # we need the limits of common X for all
                        min_intp = np.min(data[i])
                        max_intp = np.max(data[i])
                    else:
                        min_intp = max(min_intp, np.min(data[i]))
                        max_intp = min(max_intp, np.max(data[i]))
                elif channel_type(chan) in ("OA", "OP"):
                    di[(row, col)][channel_type(chan)][int(chan[1:-1]), run] = data[i]

            X = np.linspace(min_intp, max_intp, num=len(datacols))

            final_metas = []
            final_data = []

            for row, col in uniquerc:
                cur = di[(row, col)]
                M, OA, OP = cur["M"], cur["OA"], cur["OP"]

                OAn = np.zeros(OA.shape) * np.nan
                OPn = np.zeros(OA.shape) * np.nan
                for run in range(len(M)):
                    f = interp1d(M[run], OA[:, run])
                    OAn[:, run] = f(X)
                    f = interp1d(M[run], OP[:, run])
                    OPn[:, run] = f(X)

                OAmean = np.mean(OAn, axis=1)
                OPmean = np.mean(OPn, axis=1)
                final_data.append(OAmean)
                final_data.append(OPmean)
                final_metas += [[row, col, "O%dA" % i] for i in range(numharmonics)]
                final_metas += [[row, col, "O%dP" % i] for i in range(numharmonics)]

            final_data = np.vstack(final_data)

            metas = [Orange.data.ContinuousVariable.make("row"),
                     Orange.data.ContinuousVariable.make("column"),
                     Orange.data.StringVariable.make("channel")]

            domain = Orange.data.Domain([], None, metas=metas)
            meta_data = Table.from_numpy(domain, X=np.zeros((len(final_data), 0)),
                                         metas=np.asarray(final_metas, dtype=object))
            return X, final_data, meta_data

    def read_v2(self):

        # Find line in which data begins
        count = 0
        with open(self.filename, "r") as f:
            while f:
                line = f.readline()
                count = count + 1
                if line[0] != '#':
                    break

            file = np.loadtxt(f)  # Slower part

        # Find the Wavenumber column
        line = line.strip().split('\t')

        for i, e in enumerate(line):
            if e == 'Wavenumber':
                index = i
                break

        # Channel need to have exactly 3 letters
        Channel = line[index + 1:]
        Channel = np.array(Channel)
        # Extract other data #
        Max_row = int(file[:, 0].max() + 1)
        Max_col = int(file[:, 1].max() + 1)
        Max_omega = int(file[:, 2].max() + 1)
        N_rows = Max_row * Max_col * Channel.size
        N_cols = Max_omega

        # Transform Actual Data
        M = np.full((int(N_rows), int(N_cols)), np.nan, dtype='float')

        for j in range(int(Max_row * Max_col)):
            row_value = file[j * (Max_omega):(j + 1) * (Max_omega), 0]
            assert np.all(row_value == row_value[0])
            col_value = file[j * (Max_omega):(j + 1) * (Max_omega), 1]
            assert np.all(col_value == col_value[0])
            for k in range(Channel.size):
                M[k + Channel.size * j, :] = file[j * (Max_omega):(j + 1) * (Max_omega), k + 4]

        Meta_data = np.zeros((int(N_rows), 3), dtype='object')

        alpha = 0
        beta = 0
        Ch_n = int(Channel.size)

        for i in range(0, N_rows, Ch_n):
            if beta == Max_row:
                beta = 0
                alpha = alpha + 1
            Meta_data[i:i + Ch_n, 2] = Channel
            Meta_data[i:i + Ch_n, 1] = alpha
            Meta_data[i:i + Ch_n, 0] = beta
            beta = beta + 1

        waveN = file[0:int(Max_omega), 3]
        metas = [Orange.data.ContinuousVariable.make("row"),
                 Orange.data.ContinuousVariable.make("column"),
                 Orange.data.StringVariable.make("channel")]

        domain = Orange.data.Domain([], None, metas=metas)
        meta_data = Table.from_numpy(domain, X=np.zeros((len(M), 0)),
                                     metas=Meta_data)
                            
        return waveN, M, meta_data

    def read_v3(self):

        file_channel = str(self.filename.split(' ')[-2]).strip()
        folder_file = str(self.filename.split(file_channel)[-2]).strip()

        if 'P' in file_channel:
            self.channel_p = file_channel
            self.channel_a = file_channel.replace('P', 'A')
            file_gsf_p = self.filename
            file_gsf_a = self.filename.replace('P raw.gsf', 'A raw.gsf')
            file_html = folder_file + '.html'
        elif 'A' in file_channel:
            self.channel_a = file_channel
            self.channel_p = file_channel.replace('A', 'P')
            file_gsf_a = self.filename
            file_gsf_p = self.filename.replace('A raw.gsf', 'P raw.gsf')
            file_html = folder_file + '.html'


        data_gsf_a = self._gsf_reader(file_gsf_a)
        data_gsf_p = self._gsf_reader(file_gsf_p)
        info = self._html_reader(file_html)

        final_data, parameters, final_metas = self._format_file(data_gsf_a, data_gsf_p, info)


        metas = [Orange.data.ContinuousVariable.make("run"),
                Orange.data.ContinuousVariable.make("row"),
                Orange.data.ContinuousVariable.make("column"),
                Orange.data.StringVariable.make("channel")]

        domain = Orange.data.Domain([], None, metas=metas)
        meta_data = Table.from_numpy(domain, X=np.zeros((len(final_data), 0)),
                                     metas=np.asarray(final_metas, dtype=object))
        
        meta_data.attributes = parameters

        depth = np.arange(0, int(parameters['Pixel Area (X, Y, Z)'][3]))

        return depth, final_data, meta_data

    def _format_file(self, gsf_a, gsf_p, parameters):
        
        info = {}
        for row in range(len(parameters)):
            info.update({parameters[row,0].replace(':', ''): parameters[row,1:]})

        averaging = int(info['Averaging'][1])
        px_x = int(info['Pixel Area (X, Y, Z)'][1]) # 10
        px_y = int(info['Pixel Area (X, Y, Z)'][2]) # 10
        px_z = number_of_points = int(info['Pixel Area (X, Y, Z)'][3]) # 1024
        description = info['Description'][1]

        data_complete = []
        final_metas = []
        for y in range(0, px_y):
            amplitude = gsf_a[y].reshape((1, px_x * px_z * averaging))[0]
            phase = gsf_p[y].reshape((1, px_x * px_z * averaging))[0]
            i = 0
            f = i + px_z
            for x in range(0, px_x):
                for run in range(0, averaging):
                    data_complete += [amplitude[i:f]]
                    data_complete += [phase[i:f]]
                    
                    final_metas += [[run,x, y, self.channel_a]]
                    final_metas += [[run,x, y, self.channel_p]]
                    i = f
                    f = i + px_z

        return np.asarray(data_complete), info, final_metas

    def _html_reader(self, path):
        from pandas import read_html
        try:
            parameters = np.asarray(read_html(path)[0])
        except ImportError:
            print('''Probably you need to install xlml\nTry\n "pip install lxml"\nor\n"conda install -c conda-forge lxml"''')
            pass

        return parameters

    
    def _gsf_reader(self, path):

        gsf = GSFReader(path)
        X = gsf.read_spectra()

        return np.asarray(X)


    def read_spectra(self):
        version = 1
        try:
            with open(self.filename, "rt") as f:
                if f.read(2) == '# ':
                    version = 2
        except:
            pass
        try:
            if self.filename.split('.')[-1] == 'gsf':
                version = 3
                return self.read_v3()
        except:
            pass

        if version == 1:
            return self.read_v1()
        elif version == 2:
            return self.read_v2()


def build_spec_table(domvals, data, additional_table=None):
    """Create a an Orange data table from a triplet:
        - 1D numpy array defining wavelengths (size m)
        - 2D numpy array (shape (n, m)) with values
        - Orange.data.Table with only meta or class attributes (size n)
    """
    data = np.atleast_2d(data)
    features = [Orange.data.ContinuousVariable.make("%f" % f) for f in domvals]
    if additional_table is None:
        domain = Orange.data.Domain(features, None)
        return Orange.data.Table(domain, data)
    else:
        domain = Domain(features,
                        class_vars=additional_table.domain.class_vars,
                        metas=additional_table.domain.metas)
        ret_data = Table.from_numpy(domain, X=data, Y=additional_table.Y,
                                    metas=additional_table.metas)
        return ret_data


def getx(data):
    """
    Return x of the data. If all attribute names are numbers,
    return their values. If not, return indices.
    """
    x = np.arange(len(data.domain.attributes), dtype="f")
    try:
        x = np.array([float(a.name) for a in data.domain.attributes])
    except:
        pass
    return x


class DatMetaReader(FileFormat):
    """ Meta-reader to handle agilentImageReader and AsciiColReader name clash
    over .dat extension. """
    EXTENSIONS = ('.dat',)
    DESCRIPTION = 'Spectra ASCII or Agilent Single Tile Image'
    PRIORITY = min(AsciiColReader.PRIORITY, AgilentImageReader.PRIORITY) - 1

    def read(self):
        try:
            # agilentImage requires the .bsp file to be present as well
            return AgilentImageReader(filename=self.filename).read()
        except OSError:
            return AsciiColReader(filename=self.filename).read()


def spectra_mean(X):
    return np.nanmean(X, axis=0, dtype=np.float64)
