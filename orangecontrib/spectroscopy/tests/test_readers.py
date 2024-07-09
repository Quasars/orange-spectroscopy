from importlib import resources
import unittest
from unittest.mock import patch

import numpy as np
import Orange
from Orange.data import dataset_dirs
from Orange.data.io import FileFormat
from Orange.tests import named_file
from Orange.widgets.data.owfile import OWFile
from orangecontrib.spectroscopy.data import getx, build_spec_table
from orangecontrib.spectroscopy.io.neaspec import NeaReader, NeaReaderGSF
from orangecontrib.spectroscopy.io.util import ConstantBytesVisibleImage
from orangecontrib.spectroscopy.io.soleil import SelectColumnReader, HDF5Reader_HERMES
from orangecontrib.spectroscopy.preprocess import features_with_interpolation
from orangecontrib.spectroscopy.io import SPAReader
from orangecontrib.spectroscopy.io.agilent import agilentMosaicIFGReader
from orangecontrib.spectroscopy.io.ptir import PTIRFileReader

try:
    import opusFC
except ImportError:
    opusFC = None


def initialize_reader(reader, fn):
    """
    Returns an initialized reader with the file that can be relative
    to Orange's default data set directories.
    """
    absolute_filename = FileFormat.locate(fn, Orange.data.table.dataset_dirs)
    return reader(absolute_filename)


# pylint: disable=protected-access
def check_attributes(table):
    """
    Checks output attributes conform to OWFile expectations
    Keys "Name" and "Description" must be strings, etc
    This should be added to tests for all readers that implement attributes
    """
    OWFile._describe(table)


class TestReaders(unittest.TestCase):

    def test_autointerpolate(self):
        d2 = Orange.data.Table("collagen.csv")
        d2x = getx(d2)
        ndom = Orange.data.Domain(features_with_interpolation(d2x), None)
        dround = d2.transform(ndom)
        np.testing.assert_allclose(dround.X, d2.X)


class TestDat(unittest.TestCase):

    @unittest.skipIf(opusFC is None, "opusFC module not installed")
    def test_peach_juice(self):
        d1 = Orange.data.Table("peach_juice.dpt")
        d2 = Orange.data.Table("peach_juice.0")
        # dpt file has rounded values
        np.testing.assert_allclose(d1.X, d2.X, atol=1e-5)

    def test_roundtrip(self):
        with named_file("", suffix=".dat") as fn:
            # a single spectrum
            d1 = Orange.data.Table("peach_juice.dpt")
            d1.save(fn)
            d2 = Orange.data.Table(fn)
            np.testing.assert_equal(d1.X, d2.X)

            # multiple spectra
            d1 = Orange.data.Table("collagen.csv")
            d1.save(fn)
            d2 = Orange.data.Table(fn)
            np.testing.assert_equal(d1.X, d2.X)

    def test_semicolon_comments(self):
        with named_file("15 500;comment1\n30 650; comment2\n", suffix=".dpt") as fn:
            d = Orange.data.Table(fn)
            np.testing.assert_equal(d.X, [[500., 650.]])

    def test_semicolon_delimiter(self):
        with named_file("15;500\n30;650\n", suffix=".dpt") as fn:
            d = Orange.data.Table(fn)
            np.testing.assert_equal(d.X, [[500., 650.]])

    def test_comma_delim(self):
        with named_file("15,500\n30,650\n", suffix=".dpt") as fn:
            d = Orange.data.Table(fn)
            np.testing.assert_equal(d.X, [[500., 650.]])


try:
    no_visible_image = FileFormat.locate("opus/no_visible_images.0",
                                         Orange.data.table.dataset_dirs)
except OSError:
    no_visible_image = False

try:
    one_visible_image = FileFormat.locate("opus/one_visible_image.0",
                                          Orange.data.table.dataset_dirs)
except OSError:
    one_visible_image = False


@unittest.skipIf(opusFC is None, "opusFC module not installed")
class TestOpusReader(unittest.TestCase):

    @unittest.skipIf(no_visible_image is False, "Missing opus/no_visible_images.0")
    def test_no_visible_image_read(self):
        d = Orange.data.Table("opus/no_visible_images.0")

        # visible_images is not a permanent key
        self.assertNotIn("visible_images", d.attributes)

    @unittest.skipIf(one_visible_image is False, "Missing opus/one_visible_image.0")
    def test_one_visible_image_read(self):
        d = Orange.data.Table("opus/one_visible_image.0")

        self.assertIn("visible_images", d.attributes)
        self.assertEqual(len(d.attributes["visible_images"]), 1)

        img_info = d.attributes["visible_images"][0]
        self.assertIsInstance(img_info, ConstantBytesVisibleImage)
        self.assertEqual(img_info.name, "Image 01")
        self.assertAlmostEqual(img_info.pos_x,
                               43552.0 * 0.9008849859237671)
        self.assertAlmostEqual(img_info.pos_y,
                               20727.0 * 0.8928490281105042)
        self.assertAlmostEqual(img_info.size_x, 600, places=0)
        self.assertAlmostEqual(img_info.size_y, 480, places=0)

        # test image
        img = np.array(img_info.image)
        self.assertEqual(img.shape, (538, 666, 3))


class TestHermesHDF5Reader(unittest.TestCase):

    def test_read(self):
        reader = initialize_reader(HDF5Reader_HERMES,
                                   "Hermes_HDF5/small_OK.hdf5")
        d = reader.read()
        self.assertEqual(d[0, 0], 1000.1)
        self.assertEqual(d[1, 0], 2000.1)
        self.assertEqual(min(getx(d)), 100.1)
        self.assertEqual(max(getx(d)), 101.1)
        self.assertEqual(d[1]["map_x"], 2.1)
        self.assertEqual(d[1]["map_y"], 11.1)


class TestNXS_STXM_Diamond_I08(unittest.TestCase):

    def test_read(self):
        d = Orange.data.Table("small_diamond_nxs.nxs")
        self.assertAlmostEqual(d[0]['map_x'], -1.77900021)
        self.assertAlmostEqual(d[0]['map_y'], -2.74319824)
        self.assertEqual(d[13, 2], 1373)


class TestOmnicMapReader(unittest.TestCase):

    def test_read(self):
        d = Orange.data.Table("small_Omnic.map")
        self.assertAlmostEqual(d[1, 0], 4.01309, places=5)
        self.assertAlmostEqual(d[0, 0], 3.98295, places=5)
        self.assertEqual(min(getx(d)), 1604.51001)
        self.assertEqual(max(getx(d)), 1805.074097)
        self.assertEqual(d[0]["map_x"], 0)
        self.assertEqual(d[1]["map_y"], 0)


class TestAsciiMapReader(unittest.TestCase):

    def test_read(self):
        d = Orange.data.Table("map_test.xyz")
        self.assertEqual(len(d), 16)
        self.assertEqual(d[1]["map_x"], 1)
        self.assertEqual(d[1]["map_y"], 7)
        self.assertEqual(d[1][1], 0.1243)
        self.assertEqual(d[2][2], 0.1242)
        self.assertEqual(min(getx(d)), 1634.84)
        self.assertEqual(max(getx(d)), 1641.69)

    def test_roundtrip(self):
        d1 = Orange.data.Table("map_test.xyz")
        with named_file("", suffix=".xyz") as fn:
            d1.save(fn)
            d2 = Orange.data.Table(fn)
            np.testing.assert_equal(d1.X, d2.X)
            np.testing.assert_equal(getx(d1), getx(d2))
            np.testing.assert_equal(d1.metas, d2.metas)

    def test_undefined_map_positions(self):
        d = Orange.data.Table("iris")
        with named_file("", suffix=".xyz") as fn:
            d.save(fn)
            d2 = Orange.data.Table(fn)
            np.testing.assert_equal(np.isnan(d2.metas), np.ones((150, 2)))


class TestRenishawReader(unittest.TestCase):

    def test_single_sp_reader(self):
        d = Orange.data.Table("renishaw_test_files/sp.wdf")
        self.assertEqual(d.X[0][4], 52.4945182800293)
        self.assertEqual(min(getx(d)), 1226.275269)
        self.assertEqual(max(getx(d)), 2787.514404)

    # tested on 20201103, now disabled because data was too large for the repo
    def disabled_test_depth_reader(self):
        d = Orange.data.Table("renishaw_test_files/depth.wdf")
        self.assertEqual(d.X[3][4], 1.8102257251739502)
        self.assertEqual(min(getx(d)), 1226.605347)
        self.assertEqual(max(getx(d)), 2787.782959)

    # tested on 20201103, now disabled because data was too large for the repo
    def disabled_test_map_reader(self):
        # this is a line map, but the 2D maps are the same structure
        d = Orange.data.Table("renishaw_test_files/line.wdf")
        self.assertEqual(d.X[3][4], 112.22956848144531)
        self.assertEqual(min(getx(d)), 1226.267578)
        self.assertEqual(max(getx(d)), 2787.509766)


class TestAgilentReader(unittest.TestCase):

    def test_image_read(self):
        d = Orange.data.Table("agilent/4_noimage_agg256.dat")
        self.assertEqual(len(d), 64)
        # Pixel sizes are 5.5 * 16 = 88.0 (binning to reduce test data)
        self.assertAlmostEqual(
            d[1]["map_x"] - d[0]["map_x"], 88.0)
        self.assertAlmostEqual(
            d[8]["map_y"] - d[7]["map_y"], 88.0)
        # Last pixel should start at (8 - 1) * 88.0 = 616.0
        self.assertAlmostEqual(d[-1]["map_x"], 616.0)
        self.assertAlmostEqual(d[-1]["map_y"], 616.0)
        self.assertAlmostEqual(d[1][1], 1.27181053)
        self.assertAlmostEqual(d[2][2], 1.27506005)
        self.assertEqual(min(getx(d)), 1990.178226)
        self.assertEqual(max(getx(d)), 2113.600132)

    def test_mosaic_read(self):
        d = Orange.data.Table("agilent/5_mosaic_agg1024.dmt")
        self.assertEqual(len(d), 32)
        # Pixel sizes are 5.5 * 32 = 176.0 (binning to reduce test data)
        self.assertAlmostEqual(
            d[1]["map_x"] - d[0]["map_x"], 176.0)
        self.assertAlmostEqual(
            d[4]["map_y"] - d[3]["map_y"], 176.0)
        # Last pixel should start at (4 - 1) * 176.0 = 528.0
        self.assertAlmostEqual(d[-1]["map_x"], 528.0)
        # 1 x 2 mosiac, (8 - 1) * 176.0 = 1232.0
        self.assertAlmostEqual(d[-1]["map_y"], 1232.0)
        self.assertAlmostEqual(d[1][1], 1.14792180)
        self.assertAlmostEqual(d[2][2], 1.14063489)
        self.assertEqual(min(getx(d)), 1990.178226)
        self.assertEqual(max(getx(d)), 2113.600132)

    def test_no_visible_image_read(self):
        # Test file in this repo has no visible image
        d = Orange.data.Table("agilent/5_mosaic_agg1024.dmt")

        # visible_images is not a permanent key
        self.assertNotIn("visible_images", d.attributes)

    @unittest.skipIf(not hasattr(resources, "files"),
                     "importlib.resources.files requires python>=3.9")
    def test_visible_image_read(self):
        # Test file in agilent_format has 2 visible images
        vis_mosaic = resources.files("agilent_format") / "datasets" / "5_mosaic_agg1024.dmt"
        d = Orange.data.Table.from_file(vis_mosaic)

        self.assertIn("visible_images", d.attributes)
        self.assertEqual(len(d.attributes["visible_images"]), 2)

        img_info = d.attributes["visible_images"][0]
        self.assertIsInstance(img_info, ConstantBytesVisibleImage)
        self.assertEqual(img_info.name, "IR Cutout")
        self.assertAlmostEqual(img_info.pos_x,
                               0)
        self.assertAlmostEqual(img_info.pos_y,
                               0)
        self.assertAlmostEqual(img_info.size_x, 701, places=0)
        self.assertAlmostEqual(img_info.size_y, 1444, places=0)

        # test image
        img = np.array(img_info.image)
        self.assertEqual(img.shape, (280, 140, 3))

    def test_envi_comparison(self):
        # Image
        d1_a = Orange.data.Table("agilent/4_noimage_agg256.dat")
        d1_e = Orange.data.Table("agilent/4_noimage_agg256.hdr")
        np.testing.assert_equal(d1_a.X, d1_e.X)
        # Wavenumbers are rounded in .hdr files
        np.testing.assert_allclose(getx(d1_a), getx(d1_e))
        # Mosaic
        d2_a = Orange.data.Table("agilent/5_mosaic_agg1024.dmt")
        d2_e = Orange.data.Table("agilent/5_Mosaic_agg1024.hdr")
        np.testing.assert_equal(d2_a.X, d2_e.X)
        np.testing.assert_allclose(getx(d2_a), getx(d2_e))

    def test_image_ifg_read(self):
        d = Orange.data.Table("agilent/4_noimage_agg256.seq")
        self.assertEqual(len(d), 64)
        self.assertEqual(len(d.domain.attributes), 311)
        # Pixel sizes are 5.5 * 16 = 88.0 (binning to reduce test data)
        self.assertAlmostEqual(
            d[1]["map_x"] - d[0]["map_x"], 88.0)
        self.assertAlmostEqual(
            d[8]["map_y"] - d[7]["map_y"], 88.0)
        self.assertAlmostEqual(d[-1]["map_x"], 616.0)
        self.assertAlmostEqual(d[-1]["map_y"], 616.0)
        self.assertAlmostEqual(d[9][0], 0.64558595)
        self.assertAlmostEqual(d[18][0], 0.5792696)
        # Metadata
        self.assertEqual(d.metas[0, 2], 1.57980039e+04)
        self.assertEqual(d.metas[0, 3], 4)

    def test_mosaic_ifg_read(self):
        # This reader will only be selected manually due to shared .dmt extension
        reader = initialize_reader(agilentMosaicIFGReader,
                                   "agilent/5_mosaic_agg1024.dmt")
        d = reader.read()
        self.assertEqual(len(d), 32)
        self.assertEqual(len(d.domain.attributes), 311)
        # Pixel sizes are 5.5 * 32 = 176.0 (binning to reduce test data)
        self.assertAlmostEqual(
            d[1]["map_x"] - d[0]["map_x"], 176.0)
        self.assertAlmostEqual(
            d[4]["map_y"] - d[3]["map_y"], 176.0)
        # Last pixel should start at (4 - 1) * 176.0 = 528.0
        self.assertAlmostEqual(d[-1]["map_x"], 528.0)
        # 1 x 2 mosiac, (8 - 1) * 176.0 = 1232.0
        self.assertAlmostEqual(d[-1]["map_y"], 1232.0)
        self.assertAlmostEqual(d[21][0], 0.7116039)
        self.assertAlmostEqual(d[26][0], 0.48532167)
        # Metadata
        self.assertEqual(d.metas[0, 2], 1.57980039e+04)
        self.assertEqual(d.metas[0, 3], 4)


class TestPTIRFileReader(unittest.TestCase):

    def test_get_channels(self):
        reader = initialize_reader(PTIRFileReader,
                                   "photothermal/Nodax_Spectral_Array.ptir")
        channel_map = reader.get_channels()
        signal = b'//ZI/*/DEMODS/0/R'
        label = b'OPTIR (mV)'
        self.assertTrue(channel_map.keys().__contains__(signal))
        self.assertEqual(channel_map[signal], label)

    def test_array_read(self):
        reader = initialize_reader(PTIRFileReader,
                                   "photothermal/Nodax_Spectral_Array.ptir")
        reader.data_signal = b'//ZI/*/DEMODS/0/R'
        d = reader.read()
        self.assertAlmostEqual(d[0][0], 0.21426094)
        self.assertAlmostEqual(d[1][0], 1.6351842)
        self.assertEqual(min(getx(d)), 801.0)
        self.assertEqual(max(getx(d)), 1797.0)
        self.assertAlmostEqual(d[0]["map_x"], 801.9500122070312)
        self.assertAlmostEqual(d[0]["map_y"], -500.1499938964844)

    def test_hyperspectral_read(self):
        reader = initialize_reader(PTIRFileReader,
                                   "photothermal/Hyper_Sample.ptir")
        reader.data_signal = b'//ZI/*/DEMODS/0/R'
        d = reader.read()
        self.assertEqual(len(d), 35)
        self.assertEqual(len(d.domain.attributes), 451)
        self.assertAlmostEqual(d[0][0], 0.0137912575)
        self.assertAlmostEqual(d[1][0], -0.08101661)
        self.assertEqual(min(getx(d)), 900.0)
        self.assertEqual(max(getx(d)), 1800.0)
        self.assertAlmostEqual(d[0]["map_x"], -4088.96337890625)
        self.assertAlmostEqual(d[0]["map_y"], -886.1981201171875)


class TestGSF(unittest.TestCase):

    def test_open_line(self):
        data = Orange.data.Table("Au168mA_nodisplacement.gsf")
        self.assertEqual(data.X.shape, (20480, 1))

    def test_open_2d(self):
        data = Orange.data.Table("whitelight.gsf")
        self.assertEqual(data.X.shape, (20000, 1))
        # check some pixel vaules
        self.assertAlmostEqual(data.X[235,0], 1.2788502, 7)
        np.testing.assert_array_equal(data.metas[235], [35, 98])

        self.assertAlmostEqual(data.X[1235,0], 1.2770579, 7)
        np.testing.assert_array_equal(data.metas[1235], [35, 93])

        self.assertAlmostEqual(data.X[11235,0], 1.2476133, 7)
        np.testing.assert_array_equal(data.metas[11235], [35, 43])


class TestNea(unittest.TestCase):

    def test_open_v1(self):
        data = Orange.data.Table("spectra20_small.nea")
        self.assertEqual(len(data), 12)
        self.assertEqual("channel", data.domain.metas[2].name)
        np.testing.assert_almost_equal(getx(data), [0.000295, 0.00092])
        self.assertEqual("O0A", data.metas[0][2])
        np.testing.assert_almost_equal(data.X[0, 0], 10.2608052)  # O0A
        self.assertEqual("O0P", data.metas[6][2])
        np.testing.assert_almost_equal(data.X[6, 0], 0)  # O0P

    def test_open_v2(self):
        fn = "nea_test_v2.txt"
        absolute_filename = FileFormat.locate(fn, dataset_dirs)
        data = NeaReader(absolute_filename).read()
        self.assertEqual(len(data), 12)
        self.assertEqual("channel", data.domain.metas[2].name)
        np.testing.assert_almost_equal(getx(data), [15., 89.])
        self.assertEqual("O0A", data.metas[0][2])
        np.testing.assert_almost_equal(data.X[0, 0], 92.0)
        self.assertEqual("O0A", data.metas[6][2])
        np.testing.assert_almost_equal(data.X[6, 0], 38.0)


class TestNeaGSF(unittest.TestCase):

    def test_read(self):
        fn = 'NeaReaderGSF_test/NeaReaderGSF_test O2P raw.gsf'
        absolute_filename = FileFormat.locate(fn, dataset_dirs)
        data = NeaReaderGSF(absolute_filename).read()
        self.assertEqual(len(data), 2)
        self.assertEqual("run", data.domain.metas[2].name)
        self.assertEqual("O2A", data.metas[0][3])
        np.testing.assert_almost_equal(data.X[0, 0], 0.734363853931427)
        self.assertEqual("O2P", data.metas[1][3])
        np.testing.assert_almost_equal(data.X[1, 43], 0.17290098965168)
        n_ifg = int(data.attributes['Pixel Area (X, Y, Z)'][3])
        self.assertEqual(n_ifg, 1024)
        self.assertEqual(n_ifg, len(data.domain.attributes))
        check_attributes(data)


class TestEnvi(unittest.TestCase):

    def test_read(self):
        data = Orange.data.Table("agilent/4_noimage_agg256.hdr")
        self.assertEqual(len(data), 64)
        xs = getx(data)
        np.testing.assert_almost_equal(xs[:3], [1990.178230, 2005.605960, 2021.033700])
        self.assertAlmostEqual(data[0][2], 1.30845487)
        self.assertAlmostEqual(data[-1][3], 1.35767233)
        np.testing.assert_equal(data.metas[:3], [[0, 0], [1, 0], [2, 0]])
        np.testing.assert_equal(data.metas[-3:], [[5, 7], [6, 7], [7, 7]])


class TestSpa(unittest.TestCase):

    def test_open(self):
        _ = Orange.data.Table("sample1.spa")

    def test_read_header(self):
        fn = Orange.data.Table("sample1.spa").__file__
        r = SPAReader(fn)
        points, _, _ = r.read_spec_header()
        self.assertEqual(points, 1738)


class TestSpc(unittest.TestCase):

    def test_multiple_x(self):
        data = Orange.data.Table("m_xyxy.spc")
        self.assertEqual(len(data), 512)
        self.assertAlmostEqual(float(data.domain.attributes[0].name), 8401.800003)
        self.assertAlmostEqual(float(data.domain.attributes[-1].name), 137768.800049)


class TestMatlab(unittest.TestCase):

    def test_simple(self):
        """
        octave --eval "A = [ 5:7; 4:6 ]; save -6 simple.mat A"
        """
        data = Orange.data.Table("matlab/simple.mat")
        np.testing.assert_equal(data.X, [[5, 6, 7], [4, 5, 6]])
        names = [a.name for a in data.domain.attributes]
        self.assertEqual(names, ["0", "1", "2"])

    def test_with_wavenumbers(self):
        """
        octave --eval "A = [ 5:7; 4:6 ];
                       W = [ 0, 0.5, 1.0 ];
                       save -6 wavenumbers.mat A W"
        """
        data = Orange.data.Table("matlab/wavenumbers.mat")
        names = [a.name for a in data.domain.attributes]
        self.assertEqual(["0.0", "0.5", "1.0"], names)

    def test_with_names(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       W = [ "aa"; "bb"; "cc" ];
                       save -6 names.mat A W'
        """
        data = Orange.data.Table("matlab/names.mat")
        names = [a.name for a in data.domain.attributes]
        self.assertEqual(["aa", "bb", "cc"], names)

    def test_string_metas(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       M = [ "first row"; "second row" ];
                       save -6 metas_string.mat A M'
        """
        data = Orange.data.Table("matlab/metas_string.mat")
        self.assertEqual(["first row", "second row"], list(data.metas[:, 0]))

    def test_numeric_metas(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       M = [ 11, 12; 21, 22 ];
                       N = [ 8; 9 ];
                       save -6 metas_numeric.mat A M N'
        """
        data = Orange.data.Table("matlab/metas_numeric.mat")
        names = [a.name for a in data.domain.metas]
        self.assertEqual(["M_1", "M_2", "N"], names)
        np.testing.assert_equal([[11, 12, 8], [21, 22, 9]], data.metas)

    def test_mixed_metas(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       M = [ "first row"; "second row" ];
                       N = [ 8; 9 ];
                       save -6 metas_mixed.mat A M N'
        """
        data = Orange.data.Table("matlab/metas_mixed.mat")
        names = [a.name for a in data.domain.metas]
        self.assertEqual(["M", "N"], names)
        self.assertEqual(["first row", "second row"], list(data.metas[:, 0]))
        # extract numbers as numeric variables
        extracted = data.transform(Orange.data.Domain([data.domain.metas[1]]))
        np.testing.assert_equal([[8], [9]], extracted.X)

    def test_only_annotations(self):
        """
        octave --eval 'M = [ "first row"; "second row" ];
                       save -6 only_annotations.mat M'
        """
        data = Orange.data.Table("matlab/only_annotations.mat")
        self.assertEqual("M", data.domain.metas[0].name)
        self.assertEqual(["first row", "second row"], list(data.metas[:, 0]))

    def test_IOError(self):
        with patch("scipy.io.matlab.whosmat", lambda x: []):
            with self.assertRaises(IOError):
                Orange.data.Table("matlab/simple.mat")


class TestDataUtil(unittest.TestCase):

    def test_build_spec_table_not_copy(self):
        """build_spec_table should not copy tables if not neccessary"""
        xs = np.arange(3)
        # float32 table will force a copy because Orange X is float64
        X = np.ones([4, 3], dtype=np.float32)
        data = build_spec_table(xs, X)
        self.assertFalse(np.may_share_memory(data.X, X))
        # float64 will not force a copy
        X = np.ones([4, 3], dtype=np.float64)
        data = build_spec_table(xs, X)
        self.assertTrue(np.may_share_memory(data.X, X))


class TestSelectColumn(unittest.TestCase):

    def test_select_column(self):
        # explicit reader selection because of shared extension
        reader = initialize_reader(SelectColumnReader, "rock.txt")

        # column 1 are the energies
        self.assertEqual(reader.sheets, list(map(str, range(2, 10))))

        reader.sheet = "3"
        d = reader.read()
        np.testing.assert_equal(d.X,
                                [[0.91213142, 0.89539732, 0.87925428, 0.86225812]])
        np.testing.assert_equal(getx(d), [6870, 6880, 6890, 6900])


class TestStxmHdrXim(unittest.TestCase):

    def test_read(self):
        data = Orange.data.Table("max_iv.hdr")
        self.assertEqual(len(data), 100)
        self.assertAlmostEqual(float(data.domain.attributes[0].name), 698)
        self.assertAlmostEqual(float(data.domain.attributes[-1].name), 700)
