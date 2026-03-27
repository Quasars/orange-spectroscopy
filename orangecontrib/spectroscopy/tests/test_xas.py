import unittest
import numpy
import Orange
from Orange.data import Table
from Orange.preprocess import PreprocessorList

from orangecontrib.spectroscopy.preprocess import (
    XASnormalization,
    ExtractEXAFS,
    NoEdgejumpProvidedException,
)
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALLER_COLLAGEN,
)


xas_norm_collagen = XASnormalization(
    edge=1630,
    preedge_dict={'from': 1000, 'to': 1300, 'deg': 1},
    postedge_dict={'from': 1650, 'to': 1700, 'deg': 1},
)
extract_exafs = ExtractEXAFS(
    edge=1630, extra_from=1630, extra_to=1800, poly_deg=1, kweight=0, m=0
)


class ExtractEXAFSUsage(PreprocessorList):
    """ExtractEXAFS needs previous XAS normalization"""

    def __init__(self):
        super().__init__(preprocessors=[xas_norm_collagen, extract_exafs])


class TestXASnormalization(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [xas_norm_collagen, ExtractEXAFSUsage()]
    data = SMALLER_COLLAGEN

    def test_flat(self):
        domain = Orange.data.Domain(
            [
                Orange.data.ContinuousVariable(str(w))
                for w in [6800.0, 6940.0, 7060.0, 7400.0, 7800.0, 8000.0]
            ]
        )
        data = Table.from_numpy(domain, [[0.2, 0.2, 0.8, 0.8, 0.8, 0.8]])

        f = XASnormalization(
            edge=7000.0,
            preedge_dict={'from': 6800.0, 'to': 6950.0, 'deg': 1},
            postedge_dict={'from': 7050.0, 'to': 8000.0, 'deg': 2},
        )

        fdata = f(data)

        numpy.testing.assert_almost_equal(fdata.X, [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        numpy.testing.assert_almost_equal(fdata.metas, [[0.6]])


class TestExtractEXAFS(unittest.TestCase):
    def test_edgejump_exception(self):
        domain = Orange.data.Domain(
            [
                Orange.data.ContinuousVariable(str(w))
                for w in [6800.0, 6940.0, 7060.0, 7400.0, 7800.0, 8000.0]
            ]
        )
        spectra = [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]

        data = Table.from_numpy(domain, spectra)

        test_edge = 7000.0
        test_extra_from = 7002.0
        test_extra_to = 7500.0
        test_poly_deg = 7
        test_kweight = 2
        test_m = 2

        with self.assertRaises(NoEdgejumpProvidedException):
            extra = ExtractEXAFS(
                edge=test_edge,
                extra_from=test_extra_from,
                extra_to=test_extra_to,
                poly_deg=test_poly_deg,
                kweight=test_kweight,
                m=test_m,
            )
            _ = extra(data)

    def test_file(self):
        data = Table("exafs-test.tab")
        test_edge = 20020.0
        test_extra_from = 20020.0
        test_extra_to = 20990.0
        test_poly_deg = 8
        test_kweight = 2
        test_m = 0

        extra = ExtractEXAFS(
            edge=test_edge,
            extra_from=test_extra_from,
            extra_to=test_extra_to,
            poly_deg=test_poly_deg,
            kweight=test_kweight,
            m=test_m,
        )
        exafs = extra(data)
        numpy.testing.assert_almost_equal(
            [
                -3.46450033e-01,
                -3.45888957e-01,
                -3.44362296e-01,
                -3.41912861e-01,
                -3.38582017e-01,
                -3.34409725e-01,
                -3.29434571e-01,
                -3.23693808e-01,
            ],
            exafs.X[0, :8],
            decimal=3,
        )
