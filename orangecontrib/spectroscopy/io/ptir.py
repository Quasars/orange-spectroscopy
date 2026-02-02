import Orange
import numpy as np
import io
from Orange.data import FileFormat, ContinuousVariable, Domain
from PIL import Image
import h5py

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image, \
    ConstantBytesVisibleImage
from orangecontrib.spectroscopy.utils import MAP_X_VAR, MAP_Y_VAR


class PTIRFileReader(FileFormat, SpectralFileFormat):
    """ Reader for .ptir HDF5 files from Photothermal systems"""
    EXTENSIONS = ('.ptir',)
    DESCRIPTION = 'PTIR Studio file'

    data_signal = ''

    def get_channels(self):
        hdf5_file = h5py.File(self.filename, 'r')
        keys = list(hdf5_file.keys())

        # map all unique data channels
        channel_map = {}
        for meas_name in filter(lambda s: s.startswith('Measurement'), keys):
            hdf5_meas = hdf5_file[meas_name]
            meas_keys = list(hdf5_meas.keys())
            meas_attrs = hdf5_meas.attrs

            # skip background measurements
            if meas_attrs.keys().__contains__('IsBackground') and meas_attrs['IsBackground'][0]:
                continue

            for chan_name in filter(lambda s: s.startswith('Channel'), meas_keys):
                hdf5_chan = hdf5_meas[chan_name]
                try:
                    signal = hdf5_chan.attrs['DataSignal']
                    if not channel_map.keys().__contains__(signal):
                        label = hdf5_chan.attrs['Label']
                        channel_map[signal] = label
                except: # noqa: E722
                    pass
        if len(channel_map) == 0:
            raise IOError("Error reading channels from " + self.filename)
        return channel_map

    @property
    def sheets(self):
        return [label.decode("utf-8") for label in self.get_channels().values()]

    def read_spectra(self):
        channels = self.get_channels()

        for c, label in channels.items():
            if label.decode("utf-8") == self.sheet:
                self.data_signal = c
                break
        else:
            self.data_signal = list(channels.keys())[0]

        import h5py
        hdf5_file = h5py.File(self.filename,'r')
        keys = list(hdf5_file.keys())

        hyperspectra = False
        intensities = []
        wavenumbers = []
        x_locs = []
        y_locs = []
        # for including focus information
        z_locs = []

        # map checked images in file
        image_channels = []
        if 'Heightmaps' in keys:
            for img_name in list(hdf5_file['Heightmaps'].keys()):
                img_attrs = hdf5_file['Heightmaps'][img_name].attrs
                if 'Checked' in img_attrs.keys()and img_attrs['Checked'][0] == 1:
                    image_channels.append(img_name)
        if 'Images' in keys:
            for img_name in list(hdf5_file['Images'].keys()):
                img_attrs = hdf5_file['Images'][img_name].attrs
                if 'Checked' in img_attrs.keys()and img_attrs['Checked'][0] == 1:
                    image_channels.append(img_name)

        # Load and add image with ConstantBytesVisibleImage
        visible_images = []
        for img_name in image_channels:
            if 'Image' in img_name:
                img = hdf5_file['Images'][img_name]
                im = Image.fromarray(img[:], 'RGBA')
            elif 'Heightmap' in img_name:
                img = hdf5_file['Heightmaps'][img_name]
                im = Image.fromarray((img[:]/np.max(img[:]) \
                                    + np.min(img[:])/np.max(img[:])) * 255, 'F')
                im = im.convert('L')
            else:
                continue
            img_bytes = io.BytesIO()
            im.save(img_bytes, format='PNG')
            vimage = ConstantBytesVisibleImage(name=str(img.attrs['Label'].decode('UTF-8')),
                                   pos_x=img.attrs['PositionX'][0]-img.attrs['SizeWidth'][0]/2,
                                   pos_y=img.attrs['PositionY'][0]-img.attrs['SizeHeight'][0]/2,
                                   size_x=img.attrs['SizeWidth'][0],
                                   size_y=img.attrs['SizeHeight'][0],
                                   image_bytes=img_bytes,
                                   )
            visible_images.append(vimage)

        # load measurements
        for meas_name in filter(lambda s: s.startswith('Measurement'), keys):
            hdf5_meas = hdf5_file[meas_name]

            meas_keys = list(hdf5_meas.keys())
            meas_attrs = hdf5_meas.attrs

            # check if this measurement contains the selected data channel
            selected_signal = False
            for chan_name in filter(lambda s: s.startswith('Channel'), meas_keys):
                hdf5_chan = hdf5_meas[chan_name]
                if (hdf5_chan.attrs.keys().__contains__('DataSignal') and
                        hdf5_chan.attrs['DataSignal'] == self.data_signal):
                    selected_signal = True
                    break
            if not selected_signal:
                continue

            # build range arrays
            spec_vals = []
            try:
                if meas_attrs.keys().__contains__('RangeWavenumberStart'):
                    wn_start = meas_attrs['RangeWavenumberStart'][0]
                    wn_end = meas_attrs['RangeWavenumberEnd'][0]
                    wn_points = meas_attrs['RangeWavenumberPoints'][0]
                    spec_vals = np.linspace(wn_start, wn_end, wn_points)
            except:  # noqa: E722
                raise IOError("Error reading wavenumber range from " + self.filename)

            pos_vals = []
            try:
                if meas_attrs.keys().__contains__('RangeXStart'):
                    x_start = meas_attrs['RangeXStart'][0]
                    x_points = meas_attrs['RangeXPoints'][0]
                    x_incr = meas_attrs['RangeXIncrement'][0]
                    x_end = x_start + x_incr * (x_points - 1)
                    x_min = min(x_start, x_end)
                    if meas_attrs.keys().__contains__('RangeYStart'):
                        y_start = meas_attrs['RangeYStart'][0]
                        y_points = meas_attrs['RangeYPoints'][0]
                        y_incr = meas_attrs['RangeYIncrement'][0]
                        y_end = y_start + y_incr * (y_points - 1)
                        y_min = min(y_start, y_end)

                        # construct the positions array
                        for iY in range(int(y_points)):
                            y = y_min + iY * abs(y_incr)
                            for iX in range(int(x_points)):
                                x = x_min + iX * abs(x_incr)
                                pos_vals.append([x, y])
                        pos_vals = np.array(pos_vals)
                else:
                    pos_vals = np.array([1])
            except:  # noqa: E722
                raise IOError("Error reading position data from " + self.filename)
            hyperspectra = pos_vals.shape[0] > 1

            # ignore backgrounds and unchecked data
            if not hyperspectra:
                if meas_attrs.keys().__contains__('IsBackground') and meas_attrs['IsBackground'][0]:
                    continue
                if meas_attrs.keys().__contains__('Checked') and not meas_attrs['Checked'][0]:
                    continue

            if len(wavenumbers) == 0:
                wavenumbers = spec_vals

            if hyperspectra:
                x_len = meas_attrs['RangeXPoints'][0]
                y_len = meas_attrs['RangeYPoints'][0]
                x_locs = pos_vals[:x_len,0]
                y_indices = np.round(np.linspace(0, pos_vals.shape[0] - 1, y_len)).astype(int)
                y_locs = pos_vals[y_indices,1]
                # adding focus positions
                z_locs = hdf5_meas['Dataset_Focus'][:]
            else:
                x_locs.append(meas_attrs['LocationX'][0])
                y_locs.append(meas_attrs['LocationY'][0])
                z_locs.append(meas_attrs['LocationZ'][0])

            # load channels
            for chan_name in filter(lambda s: s.startswith('Channel'), meas_keys):
                hdf5_chan = hdf5_meas[chan_name]
                chan_attrs = hdf5_chan.attrs

                signal = chan_attrs['DataSignal']
                if signal != self.data_signal:
                    continue

                data = hdf5_chan['Raw_Data']
                if hyperspectra:
                    rows = meas_attrs['RangeYPoints'][0]
                    cols = meas_attrs['RangeXPoints'][0]
                    # organized rows, columns, wavelengths
                    intensities = np.reshape(data, (rows,cols,data.shape[1])) 
                    break
                else:
                    intensities.append(data[0,:])

        spectra = np.array(intensities)
        features = np.array(wavenumbers)
        x_locs = np.array(x_locs).flatten()
        y_locs = np.array(y_locs).flatten()
        z_locs = np.array(z_locs).flatten()

        if hyperspectra:
            features, spectra, additional_table = _spectra_from_image(spectra, features, x_locs, y_locs)

            new_attributes = []
            new_columns = []
            new_attributes.append(ContinuousVariable.make('z-focus'))
            new_columns.append(np.full((len(z_locs),), z_locs))

            domain = Domain(additional_table.domain.attributes,
                            additional_table.domain.class_vars,
                            additional_table.domain.metas + tuple(new_attributes))
            data = additional_table.transform(domain)
            with data.unlocked():
                data[:, new_attributes] = np.asarray(new_columns).T

        else:
            # locations
            metas = np.vstack((x_locs, y_locs, z_locs)).T

            domain = Orange.data.Domain([], None,
                                        metas=[Orange.data.ContinuousVariable.make(MAP_X_VAR),
                                               Orange.data.ContinuousVariable.make(MAP_Y_VAR),
                                               Orange.data.ContinuousVariable.make("z-focus")]
                                        )
            data = Orange.data.Table.from_numpy(domain, X=np.zeros((len(spectra), 0)),
                                                metas=np.asarray(metas, dtype=object))

        # Add vis and other images to data
        if visible_images:
            data.attributes['visible_images'] = visible_images

        return features, spectra, data