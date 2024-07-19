import os.path
import unittest

import Orange
import numpy as np
from Orange.data import Table
from Orange.data.io import FileFormat
from Orange.preprocess.preprocess import PreprocessorList
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy import get_sample_datasets_dir
from orangecontrib.spectroscopy.tests.test_preprocess import PREPROCESSORS_INDEPENDENT_SAMPLES
from orangecontrib.spectroscopy.widgets.owintegrate import OWIntegrate
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess, \
    create_preprocessor

from orangecontrib.spectroscopy.widgets.owtilefile import OWTilefile

AGILENT_TILE = "agilent/5_mosaic_agg1024.dmt"

# EMSC test fails on this dataset with
# "ValueError: On entry to DLASCL parameter number 4 had an illegal value"
PREPROCESSORS_INDEPENDENT_SAMPLES_NO_EMSC = [
    p for p in PREPROCESSORS_INDEPENDENT_SAMPLES
    if type(p).__name__ not in ["EMSC", "ExtractEXAFSUsage", "ME_EMSC"]]

class TestTileReaders(unittest.TestCase):

    def test_tile_load(self):
        Orange.data.Table(AGILENT_TILE)

    def test_tile_reader(self):
        # Can be removed once the get_reader interface is no logner required.
        path = os.path.join(get_sample_datasets_dir(), AGILENT_TILE)
        reader = OWTilefile.get_tile_reader(path)
        reader.read()

    def test_match_not_tiled(self):
        path = os.path.join(get_sample_datasets_dir(), AGILENT_TILE)
        reader = OWTilefile.get_tile_reader(path)
        t = reader.read()
        t_orig = Table(path)
        np.testing.assert_array_equal(t.X, t_orig.X)
        np.testing.assert_array_equal(t[:, ["map_x", "map_y"]].metas,
                                      t_orig[:, ["map_x", "map_y"]].metas)


class TestTilePreprocessors(unittest.TestCase):

    def test_single_preproc(self):
        # TODO problematic interface design: should be able to use Orange.data.Table directly
        path = os.path.join(get_sample_datasets_dir(), AGILENT_TILE)
        reader = OWTilefile.get_tile_reader(path)
        for p in PREPROCESSORS_INDEPENDENT_SAMPLES_NO_EMSC:
            reader.set_preprocessor(p)
            reader.read()

    def test_preprocessor_list(self):
        # TODO problematic interface design: should be able to use Orange.data.Table directly
        path = os.path.join(get_sample_datasets_dir(), AGILENT_TILE)
        reader = OWTilefile.get_tile_reader(path)
        pp = PreprocessorList(PREPROCESSORS_INDEPENDENT_SAMPLES[0:7])
        reader.set_preprocessor(pp)
        t = reader.read()
        assert len(t.domain.attributes) == 3


class TestTileReaderWidget(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWTilefile)

    def test_load(self):
        path = os.path.join(get_sample_datasets_dir(), AGILENT_TILE)
        self.widget.add_path(path)
        self.widget.source = self.widget.LOCAL_FILE
        self.widget.load_data()
        self.wait_until_stop_blocking()
        self.assertNotEqual(self.get_output("Data"), None)

    def test_preproc_load(self):
        """ Test that loading a preprocessor signal in the widget works """
        # OWPreprocess test setup from test_owpreprocess.test_allpreproc_indv
        self.preproc_widget = self.create_widget(OWPreprocess)
        pa = list(self.preproc_widget._qname2ppdef.values())[0]
        self.preproc_widget.add_preprocessor(pa)
        self.preproc_widget.commit.now()
        pp_out = self.get_output("Preprocessor", widget=self.preproc_widget)
        self.send_signal("Preprocessor", pp_out, widget=self.widget)
        # Single Input
        self.assertEqual(self.widget.preprocessor.preprocessors[0], pp_out)
        # Preprocessor members match editor model
        pp_from_model = create_preprocessor(self.preproc_widget.preprocessormodel.item(0), None)
        pp_tile = self.widget.preprocessor.preprocessors[0].preprocessors[0]
        self.assertIsInstance(pp_tile, type(pp_from_model))
        # MultiInput with OWIntegrate
        self.int_widget = self.create_widget(OWIntegrate)
        self.int_widget.add_preprocessor(pa)
        self.int_widget.commit.now()
        pp_out_2 = self.get_output("Preprocessor", widget=self.int_widget)
        self.send_signal("Preprocessor", pp_out_2, 2, widget=self.widget)
        self.assertEqual(self.widget.preprocessor.preprocessors,
                         [pp_out, pp_out_2])
