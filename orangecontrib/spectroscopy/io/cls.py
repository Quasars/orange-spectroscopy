import warnings
from typing import List

import h5py
import numpy as np
from Orange.data import FileFormat, ContinuousVariable

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image


class HDF5Reader_SGM(FileFormat, SpectralFileFormat):
    """ A very case specific reader for interpolated hyperspectral mapping HDF5
    files from the SGM beamline at the CLS"""
    EXTENSIONS = ('.h5',)
    DESCRIPTION = 'HDF5 file @SGM/CLS'

    @property
    def sheets(self) -> List:
        sheets = ["All"]
        with h5py.File(self.filename, 'r') as h5:
            NXentries = [str(x) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
            NXdata = [entry + "/" + str(x) for entry in NXentries for x in h5['/' + entry].keys()
                      if 'NXdata' in str(h5[entry + "/" + x].attrs.get('NX_class'))]
            for d in NXdata:
                sheets.extend([k for k, v in h5[d].items() if len(v.shape) == 3 and v.shape[-1] > 1])
        return sheets

    def read_spectra(self):
        if self.sheet is None:
            self.sheet = self.sheets[0]
        with h5py.File(self.filename, 'r') as h5:
            NXentries = [str(x) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
            NXdata = [entry + "/" + str(x) for entry in NXentries for x in h5['/' + entry].keys()
                      if 'NXdata' in str(h5[entry + "/" + x].attrs.get('NX_class'))]
            axes = [[str(nm) for nm in h5[nxdata].keys() for s in h5[nxdata].attrs.get('axes') if str(s) in str(nm) or
                     str(nm) in str(s)] for nxdata in NXdata]
            indep_shape = [v.shape for i, d in enumerate(NXdata) for k, v in h5[d].items() if k in axes[i][0]]
            data = [
                {k: np.squeeze(v[()]) for k, v in h5[d].items() if v.shape[0] == indep_shape[i][0] and k not in axes[i]}
                for i, d in
                enumerate(NXdata)]

            features_entries = []
            X_entries = []
            meta_table_entries = []
            for i, d in enumerate(NXdata):
                d = NXdata[i]

                if len(axes[i]) == 1:
                    warnings.warn(f"1D datasets not yet implemented: {d} not loaded.")
                x_locs = h5[d][axes[i][0]]
                y_locs = h5[d][axes[i][1]]
                en = h5[d]['en']
                emission = h5[d]['emission']

                X_data = {}
                meta_table = None
                meta_data = {}
                for k, v in data[i].items():
                    dims = len(v.shape)
                    _, spectra, meta_table = _spectra_from_image(np.transpose(np.atleast_3d(v), (1, 0, 2)),
                                                                 None, x_locs, y_locs)
                    if dims == len(axes[i]) + 1 and self.sheet in [k, "All"]:
                        # sdd-type 3D data
                        X_data[k] = spectra
                    elif dims == len(axes[i]):
                        # mapping scalars
                        meta_data[k] = spectra
                X = np.concatenate(list(X_data.values()), axis=-1)
                X_entries.append(X)

                features = np.arange(X.shape[-1])
                if len(features) == len(emission):
                    features = np.array(emission)
                features_entries.append(features)

                meta_table = meta_table.add_column(ContinuousVariable("en"), np.ones(X.shape[0]) * en, to_metas=True)
                for k, v in meta_data.items():
                    meta_table = meta_table.add_column(ContinuousVariable(k), v[:, 0], to_metas=True)
                meta_table_entries.append(meta_table)

            if not all(np.array_equal(f, features_entries[0]) for f in features_entries):
                warnings.warn("Multiple NXdata entries with incompatible shape.")
            try:
                X = np.vstack(X_entries)
            except:
                warnings.warn("Multiple NXdata entries with incompatible shape.\nLoading first entry only.")
                return features_entries[0], X_entries[0], meta_table_entries[0]
            meta_table = meta_table.concatenate(meta_table_entries)

            return features_entries[0], X, meta_table


class HDF5Reader_BioXASImaging(FileFormat, SpectralFileFormat):
    """ A very case specific reader for interpolated XRF mapping HDF5
    files from the BioXAS-Imaging beamline at the CLS"""
    EXTENSIONS = ('.hdf',)
    DESCRIPTION = 'HDF5 file @BioXAS-Imaging/CLS'

    @staticmethod
    def detector_group_map():
        """Return map of detector group name to entries"""
        return {
            "Macro_dtc (sum)": [f"vortex_{i}_dtc" for i in [0, 1, 2, 3]],
            "Micro_dtc (sum)": [f"vortex_{i}_dtc" for i in [4, 5, 6]],
            "Single_dtc": ["vortex_7_dtc"],
            "Macro (sum)": [f"vortex_{i}" for i in [0, 1, 2, 3]],
            "Micro (sum)": [f"vortex_{i}" for i in [4, 5, 6]],
            "Single": ["vortex_7"],
        }

    @property
    def sheets(self) -> list:
        vortex = []
        scalar = []
        with h5py.File(self.filename, 'r') as h5:
            entries = list(h5['/mono_0'].keys())
            for entry in entries:
                if entry.startswith('vortex_') and not entry.endswith('meta'):
                    vortex.append(entry)
                elif entry.startswith('scalar_'):
                    scalar.append(entry + "/after_float")

        groups = []
        for group, detectors in self.detector_group_map().items():
            if all(det in entries for det in detectors):
                groups.append(group)

        # Sort sheets
        default = []
        default_entry = "vortex_sum_dtc"
        if default_entry in vortex:
            vortex.pop(vortex.index(default_entry))
            default.append(default_entry)

        return default + groups + vortex + scalar

    def read_spectra(self):
        if self.sheet is None:
            self.sheet = self.sheets[0]

        entries = [self.sheet]
        detector_groups = self.detector_group_map()
        if entries[0] in detector_groups:
            entries = detector_groups[entries[0]]
        entries = ["/mono_0/" + entry for entry in entries]

        with h5py.File(self.filename, "r") as h5:
            features = np.array(h5['/dark_current/data/energies'])
            data = None
            for entry in entries:
                if data is None:
                    data = np.array(h5[entry])
                else:
                    data = np.sum([data, np.array(h5[entry])], axis=0)
            if data is None:
                return [], [], None
            y_locs = np.arange(data.shape[0])
            x_locs = np.arange(data.shape[1])

        return _spectra_from_image(data, features, x_locs, y_locs)
