import os

import numpy as np

from Orange.data import Table, Domain, FileFormat

from orangecontrib.spectroscopy.io.util import (
    SpectralFileFormat,
    _spectra_from_image_2d,
)
from orangecontrib.spectroscopy.utils.specio.specio import BlockReader, PerkinElmer

# This code is partially based on software developed in the Diamond synchrotron.


class PerkinElmerReader(FileFormat, SpectralFileFormat):
    EXTENSIONS = (
        ".sp",
        ".fsm",
    )
    DESCRIPTION = "Perkin Elmer"

    def read_sp(self):
        with open(self.filename, "rb") as f:
            data = f.read()

        reader = BlockReader(data)

        info = {
            "signature": reader.read(4, format="utf-8"),
            "description": reader.read(40, format="utf-8"),
        }

        decoders = {
            25739: PerkinElmer.decode25739,
            35698: PerkinElmer.decode35698,
            35699: PerkinElmer.decode35699,
            35700: PerkinElmer.decode35700,
            35701: PerkinElmer.decode35701,
            35708: PerkinElmer.decode35708,
        }

        stops = []
        spectrum = []

        block_id, block_size = reader.read(6, format="<Hi", expect_tuple=True)

        stops.append(reader.start + block_size)

        while block_id != 122 and not reader.atEnd(2):
            next_block_id = reader.peek(2)

            if next_block_id[1] == 117:
                reader.start = stops[-1]
                stops = stops[:-1]

                while reader.start >= stops[-1]:
                    stops = stops[:-1]

            else:
                block_id, block_size = reader.read(6, format="<Hi", expect_tuple=True)

                stops.append(reader.start + block_size)

        info.update(PerkinElmer.decode5104(reader.read(block_size)))

        reader.start = stops[1]

        while not reader.atEnd():
            block_id, block_size = reader.read(6, format="<Hi", expect_tuple=True)

            if block_id in decoders:
                decoded = decoders[block_id](reader.peek(block_size))

                if isinstance(decoded, dict):
                    info.update(decoded)

                else:
                    spectrum = decoded

            reader.start += block_size

        wavenumbers = np.linspace(
            info['min_wavelength'], info['max_wavelength'], info['n_points']
        )

        datavals = np.array(spectrum)[None, ...]

        domain = Domain([], None)
        meta_data = Table.from_numpy(domain, X=np.zeros((len(datavals), 0)))

        meta_data.attributes = info

        return wavenumbers, datavals, meta_data

    def read_fsm(self):
        with open(self.filename, "rb") as f:
            data = f.read()

        reader = BlockReader(data)

        meta = {
            "signature": reader.read(4, format="utf-8"),
            "description": reader.read(40, format="utf-8"),
        }

        decoders = {
            5100: PerkinElmer.decode5100,
            5104: PerkinElmer.decode5104,
            5105: PerkinElmer.decode5105,
        }

        spectrum = []

        while not reader.atEnd(6):
            block_id, block_size = reader.read(6, format="<Hi", expect_tuple=True)

            decoded = decoders[block_id](reader.read(block_size))

            if isinstance(decoded, dict):
                meta.update(decoded)
            else:
                spectrum.append(decoded)

        wavenumbers = np.arange(
            meta['z_start'], meta['z_end'] + meta['z_delta'], meta['z_delta']
        )

        datavals = np.squeeze(spectrum)

        domain = Domain([], None)
        meta_data = Table.from_numpy(domain, X=np.zeros((len(datavals), 0)))

        meta_data.attributes = meta

        return datavals, wavenumbers, meta_data

    def read_spectra(self):
        if os.path.splitext(self.filename)[1].lower() == ".sp":
            return self.read_sp()

        elif os.path.splitext(self.filename)[1].lower() == ".fsm":
            intensities, wn, m = self.read_fsm()

            atts = m.attributes
            # TODO position calculation should be done by a helper function that accepts units as well
            xpositions = atts['x_init'] + atts['x_delta'] * np.arange(atts['n_x'])
            ypositions = atts['y_init'] + atts['y_delta'] * np.arange(atts['n_y'])

            # TODO this is not nice - we need new helpers to build the coordinates
            y_loc = np.repeat(np.arange(atts['n_y']), atts['n_x'])
            x_loc = np.tile(np.arange(atts['n_x']), atts['n_y'])

            features, final_data, locs_table = _spectra_from_image_2d(
                intensities, wn, xpositions[x_loc], ypositions[y_loc]
            )

            return features, final_data, locs_table

        else:
            raise IOError("File can't be read: unsupported file type.")
