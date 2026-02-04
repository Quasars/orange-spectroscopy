from unittest import TestCase

from Orange.data import Table

from orangecontrib.spectroscopy.preprocess.utils import table_eq_x


class TestEq(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")
        cls.iris2 = Table("iris")
        cls.iris_changed = Table("iris")
        with cls.iris_changed.unlocked():
            cls.iris_changed[0][0] = 42

    def test_reference_eq_X_none(self):
        data = self.iris
        self.assertTrue(table_eq_x(None, None))
        self.assertFalse(table_eq_x(data, None))
        self.assertFalse(table_eq_x(None, data))

    def test_reference_eq_X_same(self):
        self.assertTrue(table_eq_x(self.iris, self.iris))
        self.assertTrue(table_eq_x(self.iris, self.iris2))
        self.assertFalse(table_eq_x(self.iris, self.iris_changed))
