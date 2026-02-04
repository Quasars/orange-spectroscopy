import io
import warnings

import Orange
import numpy as np
from Orange.data import FileFormat, ContinuousVariable, Domain

from agilent_format import (
    agilentImage,
    agilentImageIFG,
    agilentMosaic,
    agilentMosaicIFG,
    agilentMosaicTiles,
)
from orangecontrib.spectroscopy.io.util import (
    SpectralFileFormat,
    _spectra_from_image,
    TileFileFormat,
    ConstantBytesVisibleImage,
)
from orangecontrib.spectroscopy.utils import MAP_X_VAR, MAP_Y_VAR


def load_visible_images(vis_img_list: list[dict]) -> list[ConstantBytesVisibleImage]:
    visible_images = []
    for img in vis_img_list:
        try:
            with open(img['image_ref'], 'rb') as fh:
                image_bytes = io.BytesIO(fh.read())
            vimage = ConstantBytesVisibleImage(
                name=img["name"],
                pos_x=img['pos_x'],
                pos_y=img['pos_y'],
                size_x=img['img_size_x'],
                size_y=img['img_size_y'],
                image_bytes=image_bytes,
            )
            visible_images.append(vimage)
        except (KeyError, OSError) as e:
            warnings.warn(f"Visible images load failed: {e}")  # noqa: B028
    return visible_images


class AgilentImageReader(FileFormat, SpectralFileFormat):
    """Reader for Agilent FPA single tile image files"""

    EXTENSIONS = ('.dat',)
    DESCRIPTION = 'Agilent Single Tile Image'

    def read_spectra(self):
        ai = agilentImage(self.filename)
        info = ai.info
        X = ai.data

        try:
            features = info['wavenumbers']
        except KeyError:
            # just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(0, X.shape[1] * px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0] * px_size, num=X.shape[0], endpoint=False)

        return _spectra_from_image(X, features, x_locs, y_locs)


class AgilentImageIFGReader(FileFormat, SpectralFileFormat):
    """Reader for Agilent FPA single tile image files (IFG)"""

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
        x_locs = np.linspace(0, X.shape[1] * px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0] * px_size, num=X.shape[0], endpoint=False)

        features, data, additional_table = _spectra_from_image(
            X, features, x_locs, y_locs
        )

        import_params = [
            'Effective Laser Wavenumber',
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

        domain = Domain(
            additional_table.domain.attributes,
            additional_table.domain.class_vars,
            additional_table.domain.metas + tuple(new_attributes),
        )
        table = additional_table.transform(domain)
        with table.unlocked():
            table[:, new_attributes] = np.asarray(new_columns).T

        return (features, data, table)


class agilentMosaicReader(FileFormat, SpectralFileFormat):
    """Reader for Agilent FPA mosaic image files"""

    EXTENSIONS = ('.dmt',)
    DESCRIPTION = 'Agilent Mosaic Image'

    def read_spectra(self):
        am = agilentMosaic(self.filename, dtype=np.float64)
        info = am.info
        X = am.data
        visible_images = load_visible_images(am.vis)

        try:
            features = info['wavenumbers']
        except KeyError:
            # just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(0, X.shape[1] * px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0] * px_size, num=X.shape[0], endpoint=False)

        features, data, additional_table = _spectra_from_image(
            X, features, x_locs, y_locs
        )

        if visible_images:
            additional_table.attributes['visible_images'] = visible_images
        return features, data, additional_table


class agilentMosaicIFGReader(FileFormat, SpectralFileFormat):
    """Reader for Agilent FPA mosaic image files"""

    EXTENSIONS = ('.dmt',)
    DESCRIPTION = 'Agilent Mosaic Image (IFG)'
    PRIORITY = agilentMosaicReader.PRIORITY + 1

    def read_spectra(self):
        am = agilentMosaicIFG(self.filename, dtype=np.float64)
        info = am.info
        X = am.data

        features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(0, X.shape[1] * px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(0, X.shape[0] * px_size, num=X.shape[0], endpoint=False)

        features, data, additional_table = _spectra_from_image(
            X, features, x_locs, y_locs
        )

        import_params = [
            'Effective Laser Wavenumber',
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

        domain = Domain(
            additional_table.domain.attributes,
            additional_table.domain.class_vars,
            additional_table.domain.metas + tuple(new_attributes),
        )
        table = additional_table.transform(domain)
        with table.unlocked():
            table[:, new_attributes] = np.asarray(new_columns).T

        return (features, data, table)


class agilentMosaicTileReader(FileFormat, TileFileFormat):
    """Tile-by-tile reader for Agilent FPA mosaic image files"""

    EXTENSIONS = ('.dmt',)
    DESCRIPTION = 'Agilent Mosaic Tile-by-tile'
    PRIORITY = agilentMosaicReader.PRIORITY + 100

    def __init__(self, filename):
        super().__init__(filename)
        self.preprocessor = None

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def preprocess(self, table):
        if self.preprocessor is not None:
            return self.preprocessor(table)
        else:
            return table

    def read_tile(self):
        am = agilentMosaicTiles(self.filename)
        info = am.info
        tiles = am.tiles
        xtiles = am.tiles.shape[0]
        ytiles = am.tiles.shape[1]

        features = info['wavenumbers']

        attrs = [Orange.data.ContinuousVariable.make("%f" % f) for f in features]
        domain = Orange.data.Domain(
            attrs,
            None,
            metas=[
                Orange.data.ContinuousVariable.make(MAP_X_VAR),
                Orange.data.ContinuousVariable.make(MAP_Y_VAR),
            ],
        )

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1

        for x in range(xtiles):
            # Iterate over y tiles in reversed order to give matching output to
            # the not-tiled reader
            for y in range(ytiles - 1, -1, -1):
                tile = tiles[x, y]()
                if np.isnan(tile).all():
                    # Return an empty Table if tile doesn't exist (instead of storing a tile of nans)
                    yield Orange.data.Table.from_domain(domain)
                x_size, y_size = tile.shape[1], tile.shape[0]
                x_locs = np.linspace(
                    x * x_size * px_size,
                    (x + 1) * x_size * px_size,
                    num=x_size,
                    endpoint=False,
                )
                y_locs = np.linspace(
                    (ytiles - y - 1) * y_size * px_size,
                    (ytiles - y) * y_size * px_size,
                    num=y_size,
                    endpoint=False,
                )

                _, data, additional_table = _spectra_from_image(
                    tile, None, x_locs, y_locs
                )
                data = np.asarray(
                    data, dtype=np.float64
                )  # Orange assumes X to be float64
                tile_table = Orange.data.Table.from_numpy(
                    domain, X=data, metas=additional_table.metas
                )
                yield tile_table
