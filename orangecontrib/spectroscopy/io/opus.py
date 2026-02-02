import io
import warnings

import Orange
import numpy as np
from Orange.data import FileFormat, ContinuousVariable, StringVariable, TimeVariable

from orangecontrib.spectroscopy.utils import MAP_X_VAR, MAP_Y_VAR

from .util import ConstantBytesVisibleImage



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

        try:
            data = opusFC.getOpusData(self.filename, db)
        except Exception:
            raise IOError("Couldn't load spectrum from " + self.filename)

        attrs, clses, metas = [], [], []

        attrs = [ContinuousVariable(str(data.x[i]))
                 for i in range(data.x.shape[0])]

        y_data = None
        meta_data = None

        if isinstance(data, opusFC.MultiRegionDataReturn):
            y_data = []
            meta_data = []
            metas.extend([ContinuousVariable.make(MAP_X_VAR),
                          ContinuousVariable.make(MAP_Y_VAR),
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

        elif isinstance(data, opusFC.MultiRegionTRCDataReturn) and len(data.regions):
            y_data = []
            meta_data = []
            metas.extend([ContinuousVariable.make(MAP_X_VAR),
                          ContinuousVariable.make(MAP_Y_VAR),
                          StringVariable.make('map_region')])
            attrs = [ContinuousVariable(str(data.labels[i]))
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

        elif isinstance(data, opusFC.ImageDataReturn):
            metas.extend([ContinuousVariable.make(MAP_X_VAR),
                          ContinuousVariable.make(MAP_Y_VAR)])

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

        elif isinstance(data, opusFC.ImageTRCDataReturn):
            metas.extend([ContinuousVariable.make(MAP_X_VAR),
                          ContinuousVariable.make(MAP_Y_VAR)])

            attrs = [ContinuousVariable(str(data.labels[i]))
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

        elif isinstance(data, opusFC.TimeResolvedTRCDataReturn):
            y_data = data.traces

        elif isinstance(data, opusFC.TimeResolvedDataReturn):
            metas.extend([ContinuousVariable.make('z')])

            y_data = data.spectra
            meta_data = data.z

        elif isinstance(data, opusFC.SingleDataReturn):
            y_data = data.y[None, :]

        else:
            warnings.warn(f"Empty or unsupported opusFC DataReturn object: {type(data)}") # noqa: B028
            return Orange.data.Table()

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

        visible_images = []
        try:
            opus_imgs = opusFC.getVisImages(self.filename)
        except Exception as e:
            warnings.warn(f"Visible images load failed: {e}") # noqa: B028
        else:
            for img in opus_imgs:
                try:
                    from PIL import Image
                    image_bytes = io.BytesIO(img['image'])
                    width, height = Image.open(image_bytes).size
                    vimage = ConstantBytesVisibleImage(name=img["Title"],
                                                       pos_x=img['Pos. X'] * img['PixelSizeX'],
                                                       pos_y=img['Pos. Y'] * img['PixelSizeY'],
                                                       size_x=width * img['PixelSizeX'],
                                                       size_y=height * img['PixelSizeY'],
                                                       image_bytes=image_bytes
                                                       )
                    visible_images.append(vimage)
                except (KeyError, OSError):
                    pass

        domain = Orange.data.Domain(attrs, clses, metas)

        meta_data = np.atleast_2d(meta_data)

        table = Orange.data.Table.from_numpy(domain,
                                             y_data.astype(float, order='C'),
                                             metas=meta_data)

        if visible_images:
            table.attributes['visible_images'] = visible_images

        return table