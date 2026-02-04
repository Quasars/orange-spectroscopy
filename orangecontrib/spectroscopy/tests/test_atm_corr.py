import unittest

import numpy as np

from orangecontrib.spectroscopy.preprocess.atm_corr import AtmCorr
from orangecontrib.spectroscopy.tests.util import spectra_table
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALL_COLLAGEN,
    add_edge_case_data_parameter,
)


class TestAtmCorr(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = list(
        add_edge_case_data_parameter(
            AtmCorr,
            "reference",
            SMALL_COLLAGEN[0:1],
            correct_ranges=[(1300, 2100)],
            smooth_win=5,
        )
    )
    data = SMALL_COLLAGEN

    def test_atm_corr(self):
        # Fake atmospheric spectrum
        def atm(wn):
            return np.sin(wn * 0.0373) ** 2 * (
                (np.abs(wn - 1700) < 300) + (np.abs(wn - 3630) < 150)
            )

        # Make some fake data with different resolution than the atm spectrum
        wn = np.arange(800, 4000, 3)
        awn = np.arange(800, 4000, 1)
        # The CO2 region happens to be nearly straight with this function
        sp = np.sin(wn * 0.005) ** 2 * (1 - ((wn - 2400) / 1600) ** 2)
        data = spectra_table(wn, [sp + 0.3 * atm(wn)])
        ref = spectra_table(awn, [atm(awn)])
        method = AtmCorr(reference=ref, smooth_win=9)
        process = method(data)
        delta = ((data.X[0] - process.X[0]) ** 2).sum()
        assert 10 < delta < 11
        method = AtmCorr(reference=ref, correct_ranges=[], spline_ranges=[])
        process = method(data)
        delta = ((data.X[0] - process.X[0]) ** 2).sum()
        assert delta == 0
        # Test with multiple references
        ref = spectra_table(awn, 3 * [atm(awn)])
        method = AtmCorr(reference=ref, smooth_win=9, mean_reference=False)
        process = method(data)
        delta = ((data.X[0] - process.X[0]) ** 2).sum()
        assert 10 < delta < 11
