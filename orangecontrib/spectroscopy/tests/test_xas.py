import unittest
import numpy
import Orange
from Orange.data import Table

from orangecontrib.spectroscopy.preprocess import XASnormalization, ExtractEXAFS, NoEdgejumpProvidedException


class TestXASnormalization(unittest.TestCase):

    def test_flat(self):
        domain = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                     for w in [6800., 6940., 7060., 7400., 7800., 8000.]])
        data = Table.from_numpy(domain, [[0.2, 0.2, 0.8, 0.8, 0.8, 0.8]])

        f = XASnormalization(edge=7000.,
                             preedge_dict={'from': 6800., 'to': 6950., 'deg': 1},
                             postedge_dict={'from': 7050., 'to': 8000., 'deg': 2})

        fdata = f(data)

        numpy.testing.assert_almost_equal(fdata.X, [[0., 0., 1., 1., 1., 1.]])
        numpy.testing.assert_almost_equal(fdata.metas, [[0.6]])


class TestExtractEXAFS(unittest.TestCase):

    def test_edgejump_exception(self):

        domain = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                     for w in [6800., 6940., 7060., 7400., 7800., 8000.]])
        spectra = [[0., 0., 1., 1., 1., 1.]]

        data = Table.from_numpy(domain, spectra)

        test_edge = 7000.
        test_extra_from = 7002.
        test_extra_to = 7500.
        test_poly_deg = 7
        test_kweight = 2
        test_m = 2

        with self.assertRaises(NoEdgejumpProvidedException):
            extra = ExtractEXAFS(edge=test_edge, extra_from=test_extra_from, extra_to=test_extra_to,
                                 poly_deg=test_poly_deg, kweight=test_kweight, m=test_m)
            _ = extra(data)

    def test_file(self):
        data = Table("exafs-test.tab")
        test_edge = 20020.
        test_extra_from = 20020.0
        test_extra_to = 20990.0
        test_poly_deg = 8
        test_kweight = 2
        test_m = 0

        extra = ExtractEXAFS(edge=test_edge, extra_from=test_extra_from, extra_to=test_extra_to,
                             poly_deg=test_poly_deg, kweight=test_kweight, m=test_m)
        exafs = extra(data)
        numpy.testing.assert_almost_equal(
            [-3.46450033e-01, -3.45888957e-01, -3.44362296e-01, -3.41912861e-01,
             -3.38582017e-01, -3.34409725e-01, -3.29434571e-01, -3.23693808e-01], exafs.X[0, :8], decimal=3)
