import Orange
import numpy as np
import io
from Orange.data import FileFormat, ContinuousVariable, StringVariable, Domain
from PIL import Image
import h5py

from orangecontrib.spectroscopy.io.util import (
    SpectralFileFormat,
    _spectra_from_image,
    ConstantBytesVisibleImage,
)
from orangecontrib.spectroscopy.utils import MAP_X_VAR, MAP_Y_VAR


def _is_v5_file(filename):
    """Return True if this is a v5 .ptir file, detected via the root DocType attribute."""
    with h5py.File(filename, 'r') as f:
        doc_type = f.attrs.get('DocType', b'')
        if isinstance(doc_type, (bytes, np.bytes_)):
            doc_type = doc_type.decode('utf-8', errors='replace')
        return doc_type == 'PTIR5'


class PTIRFileReader(FileFormat, SpectralFileFormat):
    """Reader for .ptir HDF5 files from Photothermal systems (v4 and v5)"""

    EXTENSIONS = ('.ptir',)
    DESCRIPTION = 'PTIR Studio file'

    data_signal = ''

    def get_channels(self):
        if _is_v5_file(self.filename):
            return self._get_channels_v5()
        return self._get_channels_v4()

    def _get_channels_v4(self):
        hdf5_file = h5py.File(self.filename, 'r')
        keys = list(hdf5_file.keys())

        # map all unique data channels
        channel_map = {}
        for meas_name in filter(lambda s: s.startswith('Measurement'), keys):
            hdf5_meas = hdf5_file[meas_name]
            meas_keys = list(hdf5_meas.keys())
            meas_attrs = hdf5_meas.attrs

            # skip background measurements
            if (
                meas_attrs.keys().__contains__('IsBackground')
                and meas_attrs['IsBackground'][0]
            ):
                continue

            for chan_name in filter(lambda s: s.startswith('Channel'), meas_keys):
                hdf5_chan = hdf5_meas[chan_name]
                try:
                    signal = hdf5_chan.attrs['DataSignal']
                    if not channel_map.keys().__contains__(signal):
                        label = hdf5_chan.attrs['Label']
                        channel_map[signal] = label
                except:  # noqa: E722
                    pass
        if len(channel_map) == 0:
            raise IOError("Error reading channels from " + self.filename)
        return channel_map

    def _get_channels_v5(self):
        import ptir5 as ptir5lib

        channel_map = {}
        with ptir5lib.open(self.filename) as f:
            for m in f.measurements:
                signal = m.metadata.get('Channel.DataSignal')
                label = m.metadata.get('Channel.Label')
                if signal and label and signal not in channel_map:
                    channel_map[signal] = label
        if not channel_map:
            raise IOError("Error reading channels from " + self.filename)
        return channel_map

    @property
    def sheets(self):
        channels = self.get_channels()
        if _is_v5_file(self.filename):
            return list(channels.values())
        return [label.decode("utf-8") for label in channels.values()]

    def read_spectra(self):
        if _is_v5_file(self.filename):
            return self._read_spectra_v5()
        return self._read_spectra_v4()

    def _read_spectra_v4(self):
        channels = self._get_channels_v4()

        for c, label in channels.items():
            if label.decode("utf-8") == self.sheet:
                self.data_signal = c
                break
        else:
            self.data_signal = list(channels.keys())[0]

        hdf5_file = h5py.File(self.filename, 'r')
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
                if 'Checked' in img_attrs.keys() and img_attrs['Checked'][0] == 1:
                    image_channels.append(img_name)
        if 'Images' in keys:
            for img_name in list(hdf5_file['Images'].keys()):
                img_attrs = hdf5_file['Images'][img_name].attrs
                if 'Checked' in img_attrs.keys() and img_attrs['Checked'][0] == 1:
                    image_channels.append(img_name)

        # Load and add image with ConstantBytesVisibleImage
        visible_images = []
        for img_name in image_channels:
            if 'Image' in img_name:
                img = hdf5_file['Images'][img_name]
                im = Image.fromarray(img[:], 'RGBA')
            elif 'Heightmap' in img_name:
                img = hdf5_file['Heightmaps'][img_name]
                im = Image.fromarray(
                    (img[:] / np.max(img[:]) + np.min(img[:]) / np.max(img[:])) * 255,
                    'F',
                )
                im = im.convert('L')
            else:
                continue
            img_bytes = io.BytesIO()
            im.save(img_bytes, format='PNG')
            vimage = ConstantBytesVisibleImage(
                name=str(img.attrs['Label'].decode('UTF-8')),
                pos_x=img.attrs['PositionX'][0] - img.attrs['SizeWidth'][0] / 2,
                pos_y=img.attrs['PositionY'][0] - img.attrs['SizeHeight'][0] / 2,
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
                if (
                    hdf5_chan.attrs.keys().__contains__('DataSignal')
                    and hdf5_chan.attrs['DataSignal'] == self.data_signal
                ):
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
                if (
                    meas_attrs.keys().__contains__('IsBackground')
                    and meas_attrs['IsBackground'][0]
                ):
                    continue
                if (
                    meas_attrs.keys().__contains__('Checked')
                    and not meas_attrs['Checked'][0]
                ):
                    continue

            if len(wavenumbers) == 0:
                wavenumbers = spec_vals

            if hyperspectra:
                x_len = meas_attrs['RangeXPoints'][0]
                y_len = meas_attrs['RangeYPoints'][0]
                x_locs = pos_vals[:x_len, 0]
                y_indices = np.round(
                    np.linspace(0, pos_vals.shape[0] - 1, y_len)
                ).astype(int)
                y_locs = pos_vals[y_indices, 1]
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
                    intensities = np.reshape(data, (rows, cols, data.shape[1]))
                    break
                else:
                    intensities.append(data[0, :])

        spectra = np.array(intensities)
        features = np.array(wavenumbers)
        x_locs = np.array(x_locs).flatten()
        y_locs = np.array(y_locs).flatten()
        z_locs = np.array(z_locs).flatten()

        if hyperspectra:
            features, spectra, additional_table = _spectra_from_image(
                spectra, features, x_locs, y_locs
            )

            new_attributes = []
            new_columns = []
            new_attributes.append(ContinuousVariable.make('z-focus'))
            new_columns.append(np.full((len(z_locs),), z_locs))

            domain = Domain(
                additional_table.domain.attributes,
                additional_table.domain.class_vars,
                additional_table.domain.metas + tuple(new_attributes),
            )
            data = additional_table.transform(domain)
            with data.unlocked():
                data[:, new_attributes] = np.asarray(new_columns).T

        else:
            # locations
            metas = np.vstack((x_locs, y_locs, z_locs)).T

            domain = Orange.data.Domain(
                [],
                None,
                metas=[
                    Orange.data.ContinuousVariable.make(MAP_X_VAR),
                    Orange.data.ContinuousVariable.make(MAP_Y_VAR),
                    Orange.data.ContinuousVariable.make("z-focus"),
                ],
            )
            data = Orange.data.Table.from_numpy(
                domain,
                X=np.zeros((len(spectra), 0)),
                metas=np.asarray(metas, dtype=object),
            )

        # Add vis and other images to data
        if visible_images:
            data.attributes['visible_images'] = visible_images

        return features, spectra, data

    def _read_spectra_v5(self):
        import ptir5 as ptir5lib
        from ptir5.enums import DataShape

        channels = self._get_channels_v5()

        # Resolve selected channel signal
        for signal, label in channels.items():
            if label == self.sheet:
                self.data_signal = signal
                break
        else:
            self.data_signal = next(iter(channels))

        # Collect spectra as (x_values, data) pairs to handle inhomogeneous ranges
        spectra_list = []  # list of (x_values, spectrum_data)
        float_images = []  # FloatImage2D measurements for the selected channel
        x_locs = []
        y_locs = []
        z_locs = []
        labels = []
        m_types = []
        visible_images = []
        hyperspectra = False
        intensities = []
        wavenumbers = np.array([])

        with ptir5lib.open(self.filename) as f:
            for m in f.measurements:
                # Camera / byte images always become visible images
                if m.data_shape == DataShape.BYTE_IMAGE_2D:
                    vimage = self._camera_image_to_visible_v5(m)
                    if vimage is not None:
                        visible_images.append(vimage)
                    continue

                signal = m.metadata.get('Channel.DataSignal')
                if signal != self.data_signal:
                    continue

                if m.data_shape == DataShape.FLOAT_SPECTRUM_1D:
                    # Point spectrum — defer assembly to after collecting all
                    spectra_list.append((m.x_values, m.data))
                    x_locs.append(float(m.metadata.get('PositionX', 0.0)))
                    y_locs.append(float(m.metadata.get('PositionY', 0.0)))
                    z_locs.append(float(m.metadata.get('TopFocus', 0.0)))
                    labels.append(m.label)
                    m_types.append(
                        m.measurement_type.name
                        if hasattr(m.measurement_type, 'name')
                        else str(m.measurement_type)
                    )

                elif m.data_shape == DataShape.FLOAT_HYPERCUBE_3D:
                    # Hyperspectra: data shape is (num_points, height, width)
                    hyperspectra = True
                    wavenumbers = m.x_values
                    raw = m.data  # (num_points, height, width)
                    # Rearrange to (height, width, num_points) for _spectra_from_image
                    intensities = np.transpose(raw, (1, 2, 0))

                    img_w = float(m.metadata.get('ImageWidth', 0.0))
                    img_h = float(m.metadata.get('ImageHeight', 0.0))
                    pos_x = float(m.metadata.get('PositionX', 0.0))
                    pos_y = float(m.metadata.get('PositionY', 0.0))

                    x_locs = np.linspace(
                        pos_x - img_w / 2, pos_x + img_w / 2, m.pixel_width
                    )
                    y_locs = np.linspace(
                        pos_y - img_h / 2, pos_y + img_h / 2, m.pixel_height
                    )
                    # z_locs is broadcast per-row to all pixels after _spectra_from_image
                    z_locs = np.full(
                        m.pixel_height, float(m.metadata.get('TopFocus', 0.0))
                    )
                    labels.append(m.label)
                    m_types.append(
                        m.measurement_type.name
                        if hasattr(m.measurement_type, 'name')
                        else str(m.measurement_type)
                    )

                elif m.data_shape == DataShape.FLOAT_IMAGE_2D:
                    # Single-wavenumber spatial map — store data eagerly for later use
                    float_images.append(
                        {
                            'wn': float(m.metadata.get('Wavenumber', 0.0)),
                            'data': m.data,
                            'img_w': float(m.metadata.get('ImageWidth', 0.0)),
                            'img_h': float(m.metadata.get('ImageHeight', 0.0)),
                            'pos_x': float(m.metadata.get('PositionX', 0.0)),
                            'pos_y': float(m.metadata.get('PositionY', 0.0)),
                            'top_focus': float(m.metadata.get('TopFocus', 0.0)),
                            'label': m.label,
                            'm_type': m.measurement_type.name
                            if hasattr(m.measurement_type, 'name')
                            else str(m.measurement_type),
                        }
                    )

        # If no point spectra or hyperspectra were found, use FloatImage2D as spectral maps.
        # Otherwise, add FloatImage2D measurements as visible images.
        if not spectra_list and not hyperspectra and float_images:
            for fi in float_images:
                img_w, img_h = fi['img_w'], fi['img_h']
                pos_x, pos_y = fi['pos_x'], fi['pos_y']
                h, w = fi['data'].shape
                xs = np.linspace(pos_x - img_w / 2, pos_x + img_w / 2, w)
                ys = np.linspace(pos_y - img_h / 2, pos_y + img_h / 2, h)
                xg, yg = np.meshgrid(xs, ys)
                wn_arr = np.array([fi['wn']])
                for pix_val, px, py in zip(
                    fi['data'].flatten(), xg.flatten(), yg.flatten(), strict=True
                ):
                    spectra_list.append((wn_arr, np.array([pix_val])))
                    x_locs.append(float(px))
                    y_locs.append(float(py))
                    z_locs.append(fi['top_focus'])
                    labels.append(fi['label'])
                    m_types.append(fi['m_type'])
        else:
            for fi in float_images:
                vimage = self._float_image_dict_to_visible_v5(fi)
                if vimage is not None:
                    visible_images.append(vimage)

        # Assemble point spectra using the union (sparse) wavenumber range
        if spectra_list and not hyperspectra:
            # Find global x range: lowest start, highest end, shared increment
            x_min_all = min(xv[0] for xv, _ in spectra_list)
            x_max_all = max(xv[-1] for xv, _ in spectra_list)
            x_inc = (
                spectra_list[0][0][1] - spectra_list[0][0][0]
                if len(spectra_list[0][0]) > 1
                else 1.0
            )
            wavenumbers = np.arange(x_min_all, x_max_all + x_inc * 0.5, x_inc)
            intensities_padded = []
            for xv, sp in spectra_list:
                # Find indices in wavenumbers that correspond to xv
                i_start = int(round((xv[0] - x_min_all) / x_inc))
                i_end = i_start + len(sp)

                row = np.full(len(wavenumbers), np.nan)
                row[i_start:i_end] = sp
                intensities_padded.append(row)
            intensities = intensities_padded

        features = np.array(wavenumbers)
        spectra = np.array(intensities)
        x_locs = np.array(x_locs).flatten()
        y_locs = np.array(y_locs).flatten()
        z_locs = np.array(z_locs).flatten()

        if hyperspectra:
            features, spectra, additional_table = _spectra_from_image(
                spectra, features, x_locs, y_locs
            )

            new_attributes = [ContinuousVariable.make('z-focus')]
            new_metas = [
                StringVariable.make("Name"),
                StringVariable.make("Measurement Type"),
            ]
            domain = Domain(
                additional_table.domain.attributes,
                additional_table.domain.class_vars,
                additional_table.domain.metas
                + tuple(new_attributes)
                + tuple(new_metas),
            )
            data = additional_table.transform(domain)
            # z_locs has one value per y-row; tile across x-columns to match all pixels
            n_pixels = len(additional_table)
            n_rows = len(z_locs)
            n_cols = n_pixels // n_rows if n_rows > 0 else n_pixels
            z_tiled = (
                np.repeat(z_locs, n_cols)
                if n_rows > 1
                else np.full(n_pixels, float(z_locs[0]) if len(z_locs) else 0.0)
            )
            with data.unlocked():
                data[:, new_attributes] = z_tiled.reshape(-1, 1)
                data[:, new_metas[0]] = np.full(
                    n_pixels, labels[0] if labels else "", dtype=object
                ).reshape(-1, 1)
                data[:, new_metas[1]] = np.full(
                    n_pixels, m_types[0] if m_types else "", dtype=object
                ).reshape(-1, 1)
        else:
            if len(spectra) == 0:
                raise IOError("No spectral data found in " + self.filename)
            metas_float = np.vstack((x_locs, y_locs, z_locs)).T
            metas_str = np.vstack((labels, m_types)).T
            all_metas = np.hstack((metas_float, metas_str))
            domain = Orange.data.Domain(
                [],
                None,
                metas=[
                    Orange.data.ContinuousVariable.make(MAP_X_VAR),
                    Orange.data.ContinuousVariable.make(MAP_Y_VAR),
                    Orange.data.ContinuousVariable.make("z-focus"),
                    Orange.data.StringVariable.make("Name"),
                    Orange.data.StringVariable.make("Measurement Type"),
                ],
            )
            data = Orange.data.Table.from_numpy(
                domain,
                X=np.zeros((len(spectra), 0)),
                metas=np.asarray(all_metas, dtype=object),
            )

        if visible_images:
            data.attributes['visible_images'] = visible_images

        return features, spectra, data

    @staticmethod
    def _camera_image_to_visible_v5(m):
        """Convert a v5 ByteImage2D (CameraImage) to a ConstantBytesVisibleImage."""
        try:
            raw = m.data  # (height, width, bpp)
            pixel_format = m.metadata.get('PixelFormat', '')
            if pixel_format == 'Bgra32':
                # Reorder BGR(A) to RGB(A)
                raw = (
                    raw[:, :, [2, 1, 0, 3]]
                    if raw.shape[2] == 4
                    else raw[:, :, [2, 1, 0]]
                )
            if raw.shape[2] == 4:
                im = Image.fromarray(raw.astype(np.uint8), 'RGBA')
            else:
                im = Image.fromarray(raw.astype(np.uint8), 'RGB')
            img_bytes = io.BytesIO()
            im.save(img_bytes, format='PNG')
            img_w = float(m.metadata.get('ImageWidth', 0.0))
            img_h = float(m.metadata.get('ImageHeight', 0.0))
            pos_x = float(m.metadata.get('PositionX', 0.0))
            pos_y = float(m.metadata.get('PositionY', 0.0))
            return ConstantBytesVisibleImage(
                name=m.label,
                pos_x=pos_x - img_w / 2,
                pos_y=pos_y - img_h / 2,
                size_x=img_w,
                size_y=img_h,
                image_bytes=img_bytes,
            )
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _float_image_dict_to_visible_v5(fi):
        """Convert a dict of FloatImage2D data to a ConstantBytesVisibleImage."""
        try:
            img_data = fi['data'].astype(np.float32)
            min_val, max_val = img_data.min(), img_data.max()
            rng = max_val - min_val or 1.0
            img_norm = ((img_data - min_val) / rng * 255).astype(np.uint8)
            im = Image.fromarray(img_norm, 'L')
            img_bytes = io.BytesIO()
            im.save(img_bytes, format='PNG')
            return ConstantBytesVisibleImage(
                name=fi['label'],
                pos_x=fi['pos_x'] - fi['img_w'] / 2,
                pos_y=fi['pos_y'] - fi['img_h'] / 2,
                size_x=fi['img_w'],
                size_y=fi['img_h'],
                image_bytes=img_bytes,
            )
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _float_image_to_visible_v5(m):
        """Convert a v5 FloatImage2D (e.g. OPTIRImage) to a ConstantBytesVisibleImage."""
        try:
            img_data = m.data.astype(np.float32)
            min_val, max_val = img_data.min(), img_data.max()
            rng = max_val - min_val
            if rng == 0:
                rng = 1.0
            img_norm = ((img_data - min_val) / rng * 255).astype(np.uint8)
            im = Image.fromarray(img_norm, 'L')
            img_bytes = io.BytesIO()
            im.save(img_bytes, format='PNG')
            img_w = float(m.metadata.get('ImageWidth', 0.0))
            img_h = float(m.metadata.get('ImageHeight', 0.0))
            pos_x = float(m.metadata.get('PositionX', 0.0))
            pos_y = float(m.metadata.get('PositionY', 0.0))
            return ConstantBytesVisibleImage(
                name=m.label,
                pos_x=pos_x - img_w / 2,
                pos_y=pos_y - img_h / 2,
                size_x=img_w,
                size_y=img_h,
                image_bytes=img_bytes,
            )
        except Exception:  # noqa: BLE001
            return None
