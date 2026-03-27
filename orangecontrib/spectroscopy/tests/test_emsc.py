import unittest

import numpy as np
import Orange
from Orange.data import Table

from orangecontrib.spectroscopy.preprocess.emsc import (
    EMSC,
    MissingReferenceException,
    SelectionFunction,
    SmoothedSelectionFunction,
)
from orangecontrib.spectroscopy.preprocess.npfunc import Sum
from orangecontrib.spectroscopy.tests.util import spectra_table
from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALL_COLLAGEN,
    add_edge_case_data_parameter,
)


class TestEMSC(unittest.TestCase, TestCommonIndpSamplesMixin):
    different_reference = list(
        add_edge_case_data_parameter(EMSC, "reference", SMALL_COLLAGEN[0:1])
    )
    different_badspectra = list(
        add_edge_case_data_parameter(
            EMSC, "badspectra", SMALL_COLLAGEN[0:2], reference=SMALL_COLLAGEN[-1:]
        )
    )
    preprocessors = different_reference + different_badspectra
    data = SMALL_COLLAGEN

    def test_ab(self):
        data = Table.from_numpy(None, [[1.0, 2.0, 1.0, 1.0], [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], order=0, output_model=True)
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X, [[1.0, 2.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0]]
        )
        np.testing.assert_almost_equal(fdata.metas, [[0.0, 1.0], [1.0, 2.0]])
        self.assertEqual(fdata.domain.metas[0].name, "EMSC parameter 0")
        self.assertEqual(fdata.domain.metas[1].name, "EMSC scaling parameter")

    def test_abde(self):
        # TODO Find test values
        data = Table.from_numpy(None, [[1.0, 2.0, 1.0, 1.0], [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], order=2, output_model=True)
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X, [[1.0, 2.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0]]
        )

        np.testing.assert_almost_equal(
            fdata.metas, [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0]]
        )
        self.assertEqual(fdata.domain.metas[0].name, "EMSC parameter 0")
        self.assertEqual(fdata.domain.metas[1].name, "EMSC parameter 1")
        self.assertEqual(fdata.domain.metas[2].name, "EMSC parameter 2")
        self.assertEqual(fdata.domain.metas[3].name, "EMSC scaling parameter")

    def test_no_reference(self):
        # average from the data will be used
        data = Table.from_numpy(None, [[1.0, 2.0, 1.0, 1.0], [3.0, 5.0, 3.0, 3.0]])
        with self.assertRaises(MissingReferenceException):
            _ = EMSC()(data)

    def test_select_all(self):
        data = spectra_table([0, 1, 2, 3], [[1.0, 2.0, 1.0, 1.0], [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], order=2, output_model=True)
        noweights = f(data)

        # table obtained by now-removed ranges_to_weight_table([(0, 3, 1)])
        weight_table = spectra_table(
            [-5e-324, 0.0, 3.0, 3.0000000000000004], [[0, 1, 1, 0]]
        )
        f = EMSC(reference=data[0:1], order=2, output_model=True, weights=weight_table)
        weights = f(data)
        np.testing.assert_equal(weights.X, noweights.X)

        f = EMSC(
            reference=data[0:1],
            order=2,
            output_model=True,
            weights=SelectionFunction(0, 3, 1),
        )
        weights = f(data)
        np.testing.assert_equal(weights.X, noweights.X)

    def test_interpolate_wavenumbers(self):
        domain_ref = Orange.data.Domain(
            [Orange.data.ContinuousVariable(str(w)) for w in [1.0, 2.0, 3.0, 4.0]]
        )
        data_ref = Table.from_numpy(domain_ref, X=[[1.0, 3.0, 2.0, 3.0]])
        domain = Orange.data.Domain(
            [
                Orange.data.ContinuousVariable(str(w))
                for w in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            ]
        )
        data = Table.from_numpy(
            domain,
            [
                [2.0, 4.0, 6.0, 5.0, 4.0, 5.0, 6.0],
                [1.5, 2.0, 2.5, 2.25, 2.0, 2.25, 2.5],
            ],
        )

        f = EMSC(reference=data_ref[0:1], order=0, output_model=True)
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X,
            [[1.0, 2.0, 3.0, 2.5, 2.0, 2.5, 3.0], [1.0, 2.0, 3.0, 2.5, 2.0, 2.5, 3.0]],
        )
        np.testing.assert_almost_equal(fdata.metas, [[0.0, 2.0], [1.0, 0.5]])

    def test_order_wavenumbers(self):
        domain_ref = Orange.data.Domain(
            [Orange.data.ContinuousVariable(str(w)) for w in [4.0, 3.0, 2.0, 1.0]]
        )
        data_ref = data = Table.from_numpy(domain_ref, [[3.0, 2.0, 3.0, 1.0]])
        domain = Orange.data.Domain(
            [
                Orange.data.ContinuousVariable(str(w))
                for w in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            ]
        )
        data = data = Table.from_numpy(
            domain,
            [
                [2.0, 4.0, 6.0, 5.0, 4.0, 5.0, 6.0],
                [1.5, 2.0, 2.5, 2.25, 2.0, 2.25, 2.5],
            ],
        )

        f = EMSC(reference=data_ref[0:1], order=0, output_model=True)
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X,
            [[1.0, 2.0, 3.0, 2.5, 2.0, 2.5, 3.0], [1.0, 2.0, 3.0, 2.5, 2.0, 2.5, 3.0]],
        )
        np.testing.assert_almost_equal(fdata.metas, [[0.0, 2.0], [1.0, 0.5]])

    def test_badspectra(self):
        data = Table.from_numpy(
            None, [[0, 0.25, 4.5, 4.75, 1.0, 1.25, 7.5, 7.75, 2.0, 5.25, 5.5, 2.75]]
        )
        data_ref = Table.from_numpy(None, [[0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0]])
        badspec = Table.from_numpy(None, [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])

        f = EMSC(
            reference=data_ref[0:1], badspectra=badspec, order=1, output_model=True
        )
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X, [[0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]]
        )
        np.testing.assert_almost_equal(fdata.metas, [[1.375, 1.375, 3.0, 2.0]])

    def test_multiple_badspectra(self):
        data = Table.from_numpy(
            None, [[0, 0.25, 4.5, 4.75, 1.0, 1.25, 7.5, 7.75, 2.0, 5.25, 5.5, 2.75]]
        )
        data_ref = Table.from_numpy(None, [[0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]])
        badspec = Table.from_numpy(
            None,
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            ],
        )

        f = EMSC(
            reference=data_ref[0:1], badspectra=badspec, order=1, output_model=True
        )
        fdata = f(data)
        np.testing.assert_almost_equal(
            fdata.X, [[0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        np.testing.assert_almost_equal(fdata.metas, [[1.375, 1.375, 3.0, 6.0, 2.0]])

    def test_eq(self):
        data = Table.from_numpy(
            None, [[0, 0.25, 4.5, 4.75, 1.0, 1.25, 7.5, 7.75, 2.0, 5.25, 5.5, 2.75]]
        )
        data_ref = Table.from_numpy(None, [[0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0]])
        badspec = Table.from_numpy(None, [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])

        d1 = EMSC(
            reference=data_ref[0:1], badspectra=badspec, order=1, output_model=True
        )(data)
        d2 = EMSC(
            reference=data_ref[0:1], badspectra=badspec, order=1, output_model=True
        )(data)
        self.assertEqual(d1.domain, d2.domain)
        self.assertEqual(hash(d1.domain), hash(d2.domain))

        d2 = EMSC(
            reference=Table.from_numpy(None, [[1, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0]]),
            badspectra=badspec,
            order=1,
            output_model=True,
        )(data)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        d2 = EMSC(
            reference=data_ref[0:1],
            badspectra=Table.from_numpy(None, [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]]),
            order=1,
            output_model=True,
        )(data)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        d2 = EMSC(reference=data, badspectra=badspec, order=1, output_model=True)(data)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        weight_table = spectra_table(
            [-5e-324, 0.0, 3.0, 3.0000000000000004], [[0, 1, 1, 0]]
        )
        d2 = EMSC(
            reference=data_ref[0:1],
            badspectra=badspec,
            order=1,
            output_model=True,
            weights=weight_table,
        )(data)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        d3 = EMSC(
            reference=data_ref[0:1],
            badspectra=badspec,
            order=1,
            output_model=True,
            weights=weight_table,
        )(data)
        self.assertEqual(d3.domain, d2.domain)
        self.assertEqual(hash(d3.domain), hash(d2.domain))

        d2 = EMSC(
            reference=data_ref[0:1],
            badspectra=badspec,
            order=1,
            output_model=True,
            weights=SelectionFunction(0, 3, 1),
        )(data)
        self.assertNotEqual(d1.domain, d2.domain)
        self.assertNotEqual(hash(d1.domain), hash(d2.domain))

        d3 = EMSC(
            reference=data_ref[0:1],
            badspectra=badspec,
            order=1,
            output_model=True,
            weights=SelectionFunction(0, 3, 1),
        )(data)
        self.assertEqual(d3.domain, d2.domain)
        self.assertEqual(hash(d3.domain), hash(d2.domain))

        d3 = EMSC(
            reference=data_ref[0:1],
            badspectra=badspec,
            order=1,
            output_model=True,
            weights=SmoothedSelectionFunction(0, 3, 1, 0.5),
        )(data)
        self.assertNotEqual(d3.domain, d2.domain)
        self.assertNotEqual(hash(d3.domain), hash(d2.domain))


class TestSelectionFuctions(unittest.TestCase):
    def test_no_smoothing(self):
        fn = SelectionFunction(1, 2, 1)
        np.testing.assert_equal(fn(np.arange(0, 4, 1)), [0, 1, 1, 0])
        np.testing.assert_equal(fn(np.arange(0, 4, 0.5)), [0, 0, 1, 1, 1, 0, 0, 0])

    def test_overlap(self):
        fn = Sum(
            SelectionFunction(1, 2, 1), SelectionFunction(3, 4, 1)
        )  # non-overlapping regions
        a = fn([0.5, 1, 2, 2.5, 3.5, 5])
        np.testing.assert_equal(a, [0, 1, 1, 0, 1, 0])

        fn = Sum(
            SelectionFunction(1, 2, 1), SelectionFunction(1, 3, 1.3)
        )  # overlapping
        a = fn([0.5, 1, 1.5, 2, 2.0001, 2.5, 2.999, 3.01, 3.5])
        np.testing.assert_equal(a, [0, 2.3, 2.3, 2.3, 1.3, 1.3, 1.3, 0, 0])

    def test_smoothing(self):
        fn = SmoothedSelectionFunction(0, 10, 3, 4)
        tx = np.array([-5, -0.1, 0, 0.1, 1, 5])
        np.testing.assert_almost_equal(fn(tx), (np.tanh(tx / 3) + 1) / 2 * 4)
        np.testing.assert_almost_equal(fn(tx + 10), (-np.tanh(tx / 3) + 1) / 2 * 4)
