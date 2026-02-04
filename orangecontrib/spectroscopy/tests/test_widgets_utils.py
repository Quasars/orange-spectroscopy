import unittest
import array

import numpy as np

from orangecontrib.spectroscopy.widgets.utils import pack_selection, unpack_selection


class TestSelectionPacking(unittest.TestCase):
    def test_pack(self):
        # None
        self.assertEqual(pack_selection(None), None)
        # empty
        sel = np.zeros(10, dtype=np.uint8)
        self.assertEqual(pack_selection(sel), None)
        # with a few elements
        sel[[2, 4]] = [1, 3]
        r = pack_selection(sel)
        self.assertEqual(r, [(2, 1), (4, 3)])
        # bigger arrays
        sel = np.zeros(2000, dtype=np.uint8)
        sel[500:1000] = 1
        sel[1000:1500] = 2
        r = pack_selection(sel)
        self.assertTrue(isinstance(r, array.array))
        self.assertTrue((np.array(r[500:1000]) == 1).all())
        self.assertTrue((np.array(r[1000:1500]) == 2).all())

    def test_unpack(self):
        # None
        self.assertTrue(isinstance(unpack_selection(None), np.ndarray))
        np.testing.assert_equal(unpack_selection(None), [])
        # list of tuples
        r = unpack_selection([(2, 1), (4, 3)])
        np.testing.assert_equal(r, [0, 0, 1, 0, 3])
        # arrays
        ia = [0, 0, 1, 2]
        r = unpack_selection(array.array('B', ia))
        self.assertTrue(isinstance(r, np.ndarray))
        np.testing.assert_equal(r, ia)
