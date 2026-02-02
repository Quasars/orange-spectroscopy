import os.path
import unittest

import Orange
import numpy as np
from Orange.data import Table
from Orange.preprocess.preprocess import PreprocessorList
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy import get_sample_datasets_dir
from orangecontrib.spectroscopy.preprocess import Interpolate, SavitzkyGolayFiltering, Cut, \
    GaussianSmoothing, Absorbance, Transmittance, Integrate
from orangecontrib.spectroscopy.widgets.owintegrate import OWIntegrate
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess, \
    create_preprocessor
from orangecontrib.spectroscopy.widgets.owtilefile import OWTilefile

AGILENT_TILE = "agilent/5_mosaic_agg1024.dmt"

# no need to test all preprocessors here because tile reading uses domain
# transformations that are already tested in TestConversionIndpSamplesMixin
PREPROCESSORS_SEQUENCE = [
    Interpolate(np.linspace(1000, 1700, 100)),
    SavitzkyGolayFiltering(window=9, polyorder=2, deriv=2),
    Cut(lowlim=1000, highlim=1800),
    GaussianSmoothing(sd=3.),
    Absorbance(),
    Transmittance(),
    Integrate(limits=[[900, 100], [1100, 1200], [1200, 1300]])
]


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
        for p in PREPROCESSORS_SEQUENCE:
            reader.set_preprocessor(p)
            reader.read()

    def test_preprocessor_list(self):
        # TODO problematic interface design: should be able to use Orange.data.Table directly
        path = os.path.join(get_sample_datasets_dir(), AGILENT_TILE)
        reader = OWTilefile.get_tile_reader(path)
        pp = PreprocessorList(PREPROCESSORS_SEQUENCE)
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
        self.preproc_widget.add_preprocessor(self.preproc_widget.PREPROCESSORS[0])
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
        self.int_widget.add_preprocessor(self.int_widget.PREPROCESSORS[0])
        self.int_widget.commit.now()
        pp_out_2 = self.get_output("Preprocessor", widget=self.int_widget)
        self.send_signal("Preprocessor", pp_out_2, 2, widget=self.widget)
        self.assertEqual(self.widget.preprocessor.preprocessors,
                         [pp_out, pp_out_2])
