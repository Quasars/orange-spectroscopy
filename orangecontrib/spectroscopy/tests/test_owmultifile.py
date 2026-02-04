import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from AnyQt.QtCore import Qt

from Orange.widgets.tests.base import WidgetTest
from Orange.data import FileFormat, dataset_dirs, Table
from Orange.widgets.utils.filedialogs import format_filter
from Orange.data.io import TabReader
from Orange.tests import named_file

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.io import SPAReader
from orangecontrib.spectroscopy.io.ascii import AsciiColReader
from orangecontrib.spectroscopy.io.util import SpectralFileFormat
from orangecontrib.spectroscopy.widgets.owmultifile import (
    OWMultifile,
    numpy_union_keep_order,
    wns_to_unique_str,
    decimals_neeeded_for_unique_str,
)


class ReadImaginaryFile(SPAReader):
    read_count = -1

    def read_spectra(self):
        type(self).read_count += 1
        if type(self).read_count == 0:
            return np.array([1, 2]), np.array([[42, 41]]), None
        return np.array([1, 2 + 1e-10]), np.array([[43, 44]]), None


class TestOWFilesAuxiliary(unittest.TestCase):
    def test_numpy_union(self):
        A = np.array([2, 1, 3])
        B = np.array([1, 3])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [2, 1, 3])
        B = np.array([])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [2, 1, 3])
        B = np.array([5, 4, 6, 3])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [2, 1, 3, 5, 4, 6])
        A = np.array([])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [5, 4, 6, 3])

    def test_decimals_neeeded_for_unique_str(self):
        fn = decimals_neeeded_for_unique_str
        self.assertEqual(fn([1, 2]), 1)
        self.assertEqual(fn([1.5, 2.5]), 1)
        self.assertEqual(fn([1.49, 2.51]), 0)
        self.assertEqual(fn([10, 20]), 0)
        self.assertEqual(fn([10.0, 10.1]), 2)
        self.assertEqual(fn([10.0, 10.2]), 1)
        self.assertEqual(fn([10.00000001, 10.00000002]), 9)
        self.assertEqual(fn([10.00000001, 10.000000015, 10.00000002]), 9)
        self.assertEqual(fn([10.00000001, 10.000000014, 10.00000002]), 9)
        self.assertEqual(fn([10.00000001, 10.000000013, 10.00000002]), 9)
        self.assertEqual(fn([10.00000001, 10.000000012, 10.00000002]), 9)
        self.assertEqual(fn([10.00000001, 10.000000011, 10.00000002]), 10)

    def test_wns_to_unique_str(self):
        names = wns_to_unique_str([1, 2, 2 + 1e-10, 3])
        self.assertEqual(
            names, ['1.000000', '2.000000000000', '2.000000000100', '3.000000']
        )
        names = wns_to_unique_str([1, 2, 2 + 0.99e-10, 3])
        self.assertEqual(
            names, ['1.000000', '2.000000000000', '2.000000000099', '3.000000']
        )
        names = wns_to_unique_str([4, 1, 2, 2 + 0.99e-10, 3])
        self.assertEqual(
            names,
            ['4.000000', '1.000000', '2.000000000000', '2.000000000099', '3.000000'],
        )
        names = wns_to_unique_str([2 + 2e-10, 1, 2, 2 + 1e-10, 3])
        self.assertEqual(
            names,
            [
                '2.000000000200',
                '1.000000',
                '2.000000000000',
                '2.000000000100',
                '3.000000',
            ],
        )


class TestOWMultifile(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWMultifile)  # type: OWMultifile

    def tearDown(self):
        super().tearDown()
        # clear FileFormat cache so the optional new classes are thrown out
        FileFormat._ext_to_attr_if_attr2.cache_clear()

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def load_files(self, *files, reader=None):
        files = [FileFormat.locate(name, dataset_dirs) for name in files]

        def open_with_no_specific_format(a, b, c, filters, e):
            return files, filters.split(";;")[0]

        def open_with_specific_format(a, b, c, filters, e):
            return files, format_filter(reader)

        patchfn = (
            open_with_no_specific_format
            if reader is None
            else open_with_specific_format
        )

        # pretend that files were chosen in the open dialog
        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileNames", patchfn):
            self.widget.browse_files()

    def test_load_files(self):
        self.load_files("iris", "titanic")
        out = self.get_output("Data")
        iris = Table("iris")
        titanic = Table("titanic")
        for a in list(iris.domain.variables) + list(titanic.domain.variables):
            self.assertIn(a, out.domain)
        self.assertEqual(
            set(out.domain.class_vars),
            set(iris.domain.class_vars) | set(titanic.domain.class_vars),
        )
        self.assertEqual(len(out), len(iris) + len(titanic))

    def test_load_files_reader(self):
        self.load_files("iris")
        self.assertIs(self.widget.recent_paths[0].file_format, None)
        self.load_files("iris", reader=TabReader)
        self.assertEqual(
            self.widget.recent_paths[1].file_format, "Orange.data.io.TabReader"
        )

    def test_filename(self):
        self.load_files("iris", "titanic")
        out = self.get_output("Data")
        iris = out[: len(Table("iris"))]
        fns = set([e["Filename"].value for e in iris])
        self.assertTrue(len(fns), 1)
        self.assertIn("iris", fns.pop().lower())
        titanic = out[len(Table("iris")) :]
        fns = set([e["Filename"].value for e in titanic])
        self.assertTrue(len(fns), 1)
        self.assertIn("titanic", fns.pop().lower())

    def test_load_clear(self):
        self.load_files("iris")
        self.load_files("titanic")
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("iris")) + len(Table("titanic")))
        self.widget.clear()
        out = self.get_output("Data")
        self.assertIsNone(out)
        self.load_files("iris", "titanic")
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("titanic")))
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertIsNone(out)

    def test_saving_setting(self):
        self.load_files("iris")
        self.load_files("iris", reader=TabReader)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.widget = self.create_widget(OWMultifile, stored_settings=settings)
        self.assertEqual(self.widget.recent_paths[0].basename, "iris.tab")
        self.assertEqual(self.widget.recent_paths[0].file_format, None)
        self.assertEqual(self.widget.recent_paths[1].basename, "iris.tab")
        self.assertEqual(
            self.widget.recent_paths[1].file_format, "Orange.data.io.TabReader"
        )

    def test_files_relocated_on_saved_workflow(self):
        tempdir = tempfile.mkdtemp()
        try:
            oiris = FileFormat.locate("iris.tab", dataset_dirs)
            ciris = os.path.join(tempdir, "iris.tab")
            shutil.copy(oiris, ciris)
            with patch(
                "Orange.widgets.widget.OWWidget.workflowEnv",
                Mock(return_value={"basedir": tempdir}),
            ):
                self.load_files(ciris)
                self.assertEqual(self.widget.recent_paths[0].relpath, "iris.tab")
        finally:
            shutil.rmtree(tempdir)

    def test_files_relocated_after_workflow_save(self):
        tempdir = tempfile.mkdtemp()
        try:
            oiris = FileFormat.locate("iris.tab", dataset_dirs)
            ciris = os.path.join(tempdir, "iris.tab")
            shutil.copy(oiris, ciris)
            self.load_files(ciris)
            self.assertEqual(self.widget.recent_paths[0].relpath, None)
            with patch(
                "Orange.widgets.widget.OWWidget.workflowEnv",
                Mock(return_value={"basedir": tempdir}),
            ):
                self.widget.workflowEnvChanged(
                    "basedir", tempdir, None
                )  # run to propagate changes
                self.assertEqual(self.widget.recent_paths[0].relpath, "iris.tab")
        finally:
            shutil.rmtree(tempdir)

    def test_saving_domain_edit(self):
        self.load_files("iris")
        model = self.widget.domain_editor.model()
        model.setData(model.createIndex(0, 2), "skip", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(data.domain.attributes))
        # saving settings
        settings = self.widget.settingsHandler.pack_data(self.widget)
        # reloading
        self.widget = self.create_widget(OWMultifile, stored_settings=settings)
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(data.domain.attributes))

    def test_reset_domain_edit(self):
        self.load_files("iris")
        model = self.widget.domain_editor.model()
        model.setData(model.createIndex(0, 2), "skip", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(data.domain.attributes))
        self.widget.reset_domain_edit()
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(4, len(data.domain.attributes))

    def test_special_spectral_reading(self):
        class CountTabReader(TabReader):
            read_count = 0

            def read(self):
                type(self).read_count += 1
                return super().read()

        class CountSPAReader(SPAReader):
            read_count = 0
            read_spectra_count = 0

            def read(self):
                type(self).read_count += 1
                return super().read()

            def read_spectra(self):
                type(self).read_spectra_count += 1
                return super().read_spectra()

        # can not patch readers directly as they are already in registry
        with patch.object(
            FileFormat,
            "registry",
            {"TabReader": CountTabReader, "SPAReader": CountSPAReader},
        ):
            # clear LRU cache so that new classes get use
            FileFormat._ext_to_attr_if_attr2.cache_clear()
            self.load_files("titanic.tab", "sample1.spa")
            self.assertEqual(CountSPAReader.read_count, 0)
            self.assertEqual(CountSPAReader.read_spectra_count, 1)
            self.assertEqual(CountTabReader.read_count, 1)

    def test_spectra_almost_same_wavenumbers(self):
        with patch.object(FileFormat, "registry", {"SPAReader": ReadImaginaryFile}):
            # clear LRU cache so that new classes get use
            FileFormat._ext_to_attr_if_attr2.cache_clear()
            self.load_files("sample1.spa", "sample1.spa")
            out = self.get_output("Data")
            np.testing.assert_equal(out.X, [[42, 41, np.nan], [43, np.nan, 44]])
            self.assertEqual(
                [a.name for a in out.domain.attributes],
                ['1.000000', '2.000000000000', '2.000000000100'],
            )

    def test_report_on_empty(self):
        self.widget.send_report()

    def test_report_files(self):
        self.load_files("iris", "iris")
        self.widget.send_report()

    def test_missing_files_do_not_disappear(self):
        tempdir = tempfile.mkdtemp()
        try:
            oiris = FileFormat.locate("iris.tab", dataset_dirs)
            ciris = os.path.join(tempdir, "iris.tab")
            shutil.copy(oiris, ciris)
            self.load_files(ciris)
            settings = self.widget.settingsHandler.pack_data(self.widget)
        finally:
            shutil.rmtree(tempdir)
        self.widget = self.create_widget(OWMultifile, stored_settings=settings)
        assert not os.path.exists(ciris)
        self.assertEqual(1, len(self.widget.recent_paths))
        self.assertTrue(self.widget.Error.file_not_found.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertEqual("File not found.", self.widget.lb.item(0).toolTip())
        self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

    def test_reader_not_found_error(self):
        self.load_files("iris")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        with patch(
            "orangecontrib.spectroscopy.widgets.owmultifile._get_reader",
            side_effect=Exception(),
        ):
            self.widget.load_data()
            self.assertTrue(self.widget.Error.missing_reader.is_shown())
            self.assertIsNone(self.get_output(self.widget.Outputs.data))
            self.assertEqual("Reader not found.", self.widget.lb.item(0).toolTip())
            self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

    def test_unknown_reader_error(self):
        self.load_files("iris")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        with patch("Orange.data.io.TabReader.read", side_effect=Exception("test")):
            self.widget.load_data()
            self.assertTrue(self.widget.Error.read_error.is_shown())
            self.assertIsNone(self.get_output(self.widget.Outputs.data))
            self.assertEqual("Read error:\ntest", self.widget.lb.item(0).toolTip())
            self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

    def test_sheet_setting(self):
        self.load_files("rock.txt")
        self.widget.sheet_combo.setCurrentIndex(2)
        self.widget.select_sheet()
        self.assertEqual(self.widget.sheet, "3")
        out = self.get_output(self.widget.Outputs.data)
        self.assertAlmostEqual(0.91213142, out.X[0][0])
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.widget = self.create_widget(OWMultifile, stored_settings=settings)
        self.assertEqual(self.widget.sheet, "3")
        self.assertEqual(2, self.widget.sheet_combo.currentIndex())
        out = self.get_output(self.widget.Outputs.data)
        self.assertAlmostEqual(0.91213142, out.X[0][0])

    def test_repeated_discrete_disjuct_values(self):
        f1 = """\
        Disjunct values\tCommon values
        M      \tM
        M      \tF
        """
        f2 = """\
        Disjunct values\tCommon values
        F      \tF
        F      \tM
        """
        with named_file(f1, suffix=".tab") as fn1:
            with named_file(f2, suffix=".tab") as fn2:
                self.load_files(fn1, fn2)
                out = self.get_output(self.widget.Outputs.data)
                disjunct = [str(a[0]) for a in out]
                self.assertEqual(disjunct, ['M', 'M', 'F', "F"])
                common = [str(a[1]) for a in out]
                self.assertEqual(common, ['M', 'F', 'F', "M"])

    def test_special_and_normal(self):
        s1 = """\
        100\t3
        102\t3.10
        104\t3.20
        """
        s2 = """\
        100\t3
        101\t3.05
        104\t3.20
        """
        n = """\
        100.000000\t200.000000\t300.000000\tclass
        4\t5\t6\td
        """

        nan = float("nan")
        concat = np.array(
            [
                [3.0, 3.1, 3.2, nan, nan, nan, nan],
                [3.0, nan, 3.2, 3.05, nan, nan, nan],
                [4.0, nan, nan, nan, 5.0, 6.0, 0.0],
            ]
        )

        self.assertTrue(issubclass(AsciiColReader, SpectralFileFormat))
        with named_file(s1, suffix=".xy") as fn1:
            with named_file(s2, suffix=".xy") as fn2:
                with named_file(n, suffix=".tab") as fn:
                    self.load_files(fn1, reader=AsciiColReader)
                    self.load_files(fn2, reader=AsciiColReader)
                    out = self.get_output(self.widget.Outputs.data)
                    np.testing.assert_equal(getx(out), [100, 102, 104, 101])
                    np.testing.assert_equal(out.X, concat[:2, :4])
                    self.load_files(fn)
                    out = self.get_output(self.widget.Outputs.data)
                    atts = [a.name for a in out.domain.attributes]
                    self.assertEqual(
                        atts,
                        [
                            '100.000000',
                            '102.000000',
                            '104.000000',
                            '101.000000',
                            '200.000000',
                            '300.000000',
                            'class',
                        ],
                    )
                    out = self.get_output(self.widget.Outputs.data)
                    np.testing.assert_equal(out.X, concat)

                    # load the same files in a different order
                    self.widget.clear()
                    self.load_files(fn1, reader=AsciiColReader)
                    self.load_files(fn)
                    self.load_files(fn2, reader=AsciiColReader)
                    out = self.get_output(self.widget.Outputs.data)
                    np.testing.assert_equal(out.X, concat[[0, 2, 1]])

    def test_special_spectral_reader_metas(self):
        n = """\
        100.000000\t200.000000\t300.000000\tmeta
        \t\t\tstring\n
        \t\t\tmeta\n
        4\t5\t6\thello
        """
        with named_file(n, suffix=".tab") as fn:
            self.load_files("small_diamond_nxs.nxs")  # special spectral rader
            self.load_files(fn)
            out = self.get_output(self.widget.Outputs.data)
            self.assertEqual(
                [a.name for a in out.domain.metas[:3]], ["map_x", "map_y", "meta"]
            )
            self.assertAlmostEqual(out[0]['map_x'], -1.77900021)
            self.assertAlmostEqual(out[0]['map_y'], -2.74319824)
            self.assertEqual(out[0]['meta'].value, "")
            self.assertEqual(out[-1]['meta'].value, "hello")
