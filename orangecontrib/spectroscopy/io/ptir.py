import Orange
import numpy as np
import io
from Orange.data import FileFormat, ContinuousVariable, Domain
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
        from ptir5.enums import DataShape

        channel_map = {}
        with ptir5lib.open(self.filename) as f:
            for m_type in ptir5lib.enums.MeasurementType:
                shape = ptir5lib.enums.TYPE_TO_SHAPE.get(m_type)
                if not shape:
                    continue
                if f.measurements_by_type(m_type):
                    if shape == DataShape.FLOAT_SPECTRUM_1D:
                        channel_map['Spectra'] = 'Spectra'
                    elif shape == DataShape.FLOAT_IMAGE_2D:
                        channel_map['Images'] = 'Images'
                    elif shape == DataShape.FLOAT_HYPERCUBE_3D:
                        channel_map['Hyperspectra'] = 'Hyperspectra'

        if not channel_map:
            raise IOError(f"Error reading channels from {self.filename}")
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

        target_sheet = (
            self.sheet
            if self.sheet in channels.values()
            else next(iter(channels.values()))
        )
        load_as_maps = target_sheet == "Images"
        self.data_signal = target_sheet  # V5 no longer uses signal labels for filtering

        spectra_blocks = []
        float_images = []
        x_locs, y_locs, z_locs, labels, m_types = [], [], [], [], []
        visible_images = []

        with ptir5lib.open(self.filename) as f:
            for m_type in ptir5lib.enums.MeasurementType:
                shape = ptir5lib.enums.TYPE_TO_SHAPE.get(m_type)
                if not shape:
                    continue

                for m in f.measurements_by_type(m_type):
                    if shape == DataShape.BYTE_IMAGE_2D:
                        vimage = self._camera_image_to_visible_v5(m)
                        if vimage is not None:
                            visible_images.append(vimage)
                        continue

                    if (
                        shape == DataShape.FLOAT_SPECTRUM_1D
                        and target_sheet == "Spectra"
                    ):
                        spectra_blocks.append((m.x_values, np.atleast_2d(m.data)))
                        x_locs.append(float(m.metadata.get('PositionX', 0.0)))
                        y_locs.append(float(m.metadata.get('PositionY', 0.0)))
                        z_locs.append(float(m.metadata.get('TopFocus', 0.0)))
                        labels.append(m.label)
                        m_types.append(
                            getattr(m.measurement_type, 'name', str(m.measurement_type))
                        )

                    elif (
                        shape == DataShape.FLOAT_HYPERCUBE_3D
                        and target_sheet == "Hyperspectra"
                    ):
                        # data is (num_points, height, width) -> (height, width, num_points)
                        intensities = np.transpose(m.data, (1, 2, 0))
                        h, w, n_pts = intensities.shape
                        flat_intensities = intensities.reshape(-1, n_pts)
                        spectra_blocks.append((m.x_values, flat_intensities))

                        img_w, img_h = (
                            float(m.metadata.get('ImageWidth', 0.0)),
                            float(m.metadata.get('ImageHeight', 0.0)),
                        )
                        pos_x, pos_y = (
                            float(m.metadata.get('PositionX', 0.0)),
                            float(m.metadata.get('PositionY', 0.0)),
                        )
                        xs = np.linspace(
                            pos_x - img_w / 2, pos_x + img_w / 2, m.pixel_width
                        )
                        ys = np.linspace(
                            pos_y - img_h / 2, pos_y + img_h / 2, m.pixel_height
                        )
                        xg, yg = np.meshgrid(xs, ys)

                        n_pixels = h * w
                        x_locs.extend(xg.flatten())
                        y_locs.extend(yg.flatten())
                        z_locs.extend(
                            [float(m.metadata.get('TopFocus', 0.0))] * n_pixels
                        )

                        labels.extend([m.label] * n_pixels)
                        m_types.extend(
                            [
                                getattr(
                                    m.measurement_type, 'name', str(m.measurement_type)
                                )
                            ]
                            * n_pixels
                        )

                    elif shape == DataShape.FLOAT_IMAGE_2D:
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
                                'm_type': getattr(
                                    m.measurement_type, 'name', str(m.measurement_type)
                                ),
                            }
                        )

        if load_as_maps and float_images:
            for fi in float_images:
                img_w, img_h = fi['img_w'], fi['img_h']
                pos_x, pos_y = fi['pos_x'], fi['pos_y']
                h, w = fi['data'].shape
                xg, yg = np.meshgrid(
                    np.linspace(pos_x - img_w / 2, pos_x + img_w / 2, w),
                    np.linspace(pos_y - img_h / 2, pos_y + img_h / 2, h),
                )
                wn_arr = np.array([fi['wn']])

                flat_data = fi['data'].flatten().reshape(-1, 1)
                spectra_blocks.append((wn_arr, flat_data))

                n_pixels = len(flat_data)
                x_locs.extend(xg.flatten())
                y_locs.extend(yg.flatten())
                z_locs.extend([fi['top_focus']] * n_pixels)
                labels.extend([fi['label']] * n_pixels)
                m_types.extend([fi['m_type']] * n_pixels)
        else:
            visible_images.extend(
                v
                for fi in float_images
                if (v := self._float_image_dict_to_visible_v5(fi)) is not None
            )

        if not spectra_blocks:
            raise IOError(f"No spectral data found in {self.filename}")

        # Estimate increment from the first block with >1 points
        x_inc = 1.0
        for xv, _ in spectra_blocks:
            if len(xv) > 1:
                x_inc = abs(xv[1] - xv[0])
                break

        # Assemble the master array with NaN padding
        # Since x_values is computed linearly as x_start + i * x_increment,
        # we only need to check the exact endpoints.
        x_min_all = min(min(xv[0], xv[-1]) for xv, _ in spectra_blocks)
        x_max_all = max(max(xv[0], xv[-1]) for xv, _ in spectra_blocks)

        wavenumbers = np.arange(x_min_all, x_max_all + x_inc * 0.5, x_inc)

        total_pixels = sum(sp.shape[0] for _, sp in spectra_blocks)
        intensities = np.full((total_pixels, len(wavenumbers)), np.nan)

        current_row = 0
        for xv, sp in spectra_blocks:
            n_rows = sp.shape[0]

            if len(xv) > 1 and xv[1] < xv[0]:
                xv = xv[::-1]
                sp = sp[:, ::-1]

            i_start = int(round((xv[0] - x_min_all) / x_inc))
            i_end = i_start + len(xv)
            intensities[current_row : current_row + n_rows, i_start:i_end] = sp
            current_row += n_rows

        features, spectra = wavenumbers, intensities

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

        all_metas = np.hstack(
            (np.vstack((x_locs, y_locs, z_locs)).T, np.vstack((labels, m_types)).T)
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
