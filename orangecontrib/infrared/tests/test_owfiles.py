import unittest

import AnyQt

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owmultifile import OWFiles
from Orange.data import FileFormat, dataset_dirs, Table


class TestOWFiles(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWFiles)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def load_files(self, *files):
        files = [FileFormat.locate(name, dataset_dirs) for name in files]
        self.widget.load_files(files)

    def test_load_files(self):
        self.load_files("iris", "titanic")
        self.widget.load_data()
        out = self.get_output("Data")
        iris = Table("iris")
        titanic = Table("titanic")
        for a in list(iris.domain) + list(titanic.domain):
            self.assertIn(a, out.domain)
        self.assertEqual(set(out.domain.class_vars),
                         set(iris.domain.class_vars) | set(titanic.domain.class_vars))
        self.assertEqual(len(out), len(iris) + len(titanic))

    def test_filename(self):
        self.load_files("iris", "titanic")
        self.widget.load_data()
        out = self.get_output("Data")
        iris = out[:len(Table("iris"))]
        fns = set([e["Filename"].value for e in iris])
        self.assertTrue(len(fns), 1)
        self.assertIn("iris", fns.pop().lower())
        titanic = out[len(Table("iris")):]
        fns = set([e["Filename"].value for e in titanic])
        self.assertTrue(len(fns), 1)
        self.assertIn("titanic", fns.pop().lower())

    @unittest.skipIf(AnyQt.USED_API == "pyqt5", "Test crashes with Ubuntu and pyqt5")
    def test_load_clear(self):
        self.load_files("iris")
        self.load_files("titanic")
        self.widget.load_data()
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("iris")) + len(Table("titanic")))
        self.widget.clear()
        out = self.get_output("Data")
        self.assertIsNone(out)
        self.load_files("iris", "titanic")
        self.widget.load_data()
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("titanic")))
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertIsNone(out)

    def test_sheet_file(self):
        self.load_files("peach_juice.0")
        self.widget.sheet_combo.setCurrentIndex(1)
        self.widget.select_sheet()