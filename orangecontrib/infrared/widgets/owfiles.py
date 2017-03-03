import os
from functools import reduce
from itertools import chain, repeat
from collections import Counter

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QSizePolicy as Policy, QGridLayout, QLabel, QMessageBox, QFileDialog, QApplication, QStyle

import Orange
import orangecontrib.infrared
from Orange.data.table import get_sample_datasets_dir
from Orange.data.io import FileFormat
from Orange.widgets import widget, gui
import Orange.widgets.data.owfile
from Orange.widgets.data.owconcatenate import domain_union, concat, append_columns
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.filedialogs import RecentPathsWidgetMixin, RecentPath

from warnings import catch_warnings


OLD_DOMAINEDITOR = not hasattr(DomainEditor, "set_domain")


class OWFiles(Orange.widgets.data.owfile.OWFile, RecentPathsWidgetMixin):
    name = "Files"
    id = "orangecontrib.infrared.widgets.files"
    icon = "icons/files.svg"
    description = "Read data from input files " \
                  "and send a data table to the output."

    file_idx = -1

    sheet = Orange.widgets.settings.Setting(None)
    label = Orange.widgets.settings.Setting("")
    recent_paths = Orange.widgets.settings.Setting([])

    def __init__(self):
        widget.OWWidget.__init__(self)
        RecentPathsWidgetMixin.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.sheets = []

        self.lb = gui.listBox(self.controlArea, self, "file_idx")

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)

        file_button = gui.button(
            None, self, '  ...', callback=self.browse_files, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(
            QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 0)

        remove_button = gui.button(
            None, self, 'Remove', callback=self.remove_item)

        clear_button = gui.button(
            None, self, 'Clear', callback=self.clear)

        layout.addWidget(remove_button, 0, 1)
        layout.addWidget(clear_button, 0, 2)

        reload_button = gui.button(
            None, self, "Reload", callback=self.load_data, autoDefault=False)
        reload_button.setIcon(self.style().standardIcon(
                QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 7)

        self.sheet_box = gui.hBox(None, addToLayout=False, margin=0)
        self.sheet_combo = gui.comboBox(None, self, "xls_sheet",
                                        callback=self.select_sheet,
                                        sendSelectedValue=True)
        self.sheet_combo.setSizePolicy(
            Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_label = QLabel()
        self.sheet_label.setText('Sheet')
        self.sheet_label.setSizePolicy(
            Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_box.layout().addWidget(
            self.sheet_label, Qt.AlignLeft)
        self.sheet_box.layout().addWidget(
            self.sheet_combo, Qt.AlignVCenter)
        layout.addWidget(self.sheet_box, 2, 1)
        self.sheet_box.hide()

        layout.addWidget(self.sheet_box, 0, 5)

        label_box = gui.hBox(None, addToLayout=False, margin=0)
        label = gui.lineEdit(label_box, self, "label", callback=self.set_label,
                             label="Label", orientation=Qt.Horizontal)
        layout.addWidget(label_box, 0, 6)

        layout.setColumnStretch(3, 2)

        box = gui.widgetBox(self.controlArea, "View loaded columns")
        if OLD_DOMAINEDITOR:
            self.domain_editor = DomainEditor(self.variables)
        else:
            self.domain_editor = DomainEditor(self)
        self.editor_model = self.domain_editor.model()
        box.layout().addWidget(self.domain_editor)

        for i, rp in enumerate(self.recent_paths):
            self.lb.addItem(rp.abspath)

        # TODO unresolved paths just disappear! Modify _relocate_recent_files

        self._update_sheet_combo()
        self.load_data()

    def set_label(self):
        self.load_data()

    def add_path(self, filename):
        recent = RecentPath.create(filename, self._search_paths())
        self.recent_paths.append(recent)

    def set_file_list(self):
        # need to define it for RecentPathsWidgetMixin
        pass

    def _select_active_sheet(self):
        if self.sheet:
            try:
                sheet_list = [ s[0] for s in self.sheets]
                idx = sheet_list.index(self.sheet)
                self.sheet_combo.setCurrentIndex(idx)
            except ValueError:
                # Requested sheet does not exist in this file
                self.sheet = None
        else:
            self.sheet_combo.setCurrentIndex(0)

    def _update_sheet_combo(self):
        sheets = Counter()

        for fn in self.current_filenames():
            reader = FileFormat.get_reader(fn)
            sheets.update(reader.sheets)

        sheets = sorted(sheets.items(), key=lambda x: x[0])

        self.sheets = [(s, s + " (" + str(n) + ")") for s, n in sheets]

        if len(sheets) < 2:
            self.sheet_box.hide()
            self.sheet = None
        else:
            self.sheets.insert(0, (None, "(None)"))
            self.sheet_combo.clear()
            self.sheet_combo.addItems([s[1] for s in self.sheets])
            self._select_active_sheet()
            self.sheet_box.show()

    def select_sheet(self):
        self.sheet = self.sheets[self.sheet_combo.currentIndex()][0]
        self.load_data()

    def remove_item(self):
        ri = [ i.row() for i in  self.lb.selectedIndexes() ]
        for i in sorted(ri, reverse=True):
            self.recent_paths.pop(i)
            self.lb.takeItem(i)
        self._update_sheet_combo()
        self.load_data()

    def clear(self):
        self.lb.clear()
        while self.recent_paths:
            self.recent_paths.pop()
        self._update_sheet_combo()
        self.load_data()

    def browse_files(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with documentation data sets")
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        filenames = QFileDialog.getOpenFileNames(
            self, 'Open Multiple Data Files', start_file, self.dlg_formats)

        if isinstance(filenames, tuple):  # has a file description
            filenames = filenames[0]

        if not filenames:
            return

        for f in filenames:
            self.add_path(f)
            self.lb.addItem(f)

        self._update_sheet_combo()
        self.load_data()

    def current_filenames(self):
        return [rp.abspath for rp in self.recent_paths]

    def load_data(self):
        self.closeContext()

        fns = self.current_filenames()

        data_list = []
        fnok_list = []

        for fn in fns:
            reader = FileFormat.get_reader(fn)
            if self.sheet in reader.sheets:
                reader.select_sheet(self.sheet)

            errors = []
            with catch_warnings(record=True) as warnings:
                try:
                    data_list.append(reader.read())
                    fnok_list.append(fn)
                except Exception as ex:
                    errors.append("An error occurred:")
                    errors.append(str(ex))
                    #FIXME show error in the list of data
                self.warning(warnings[-1].message.args[0] if warnings else '')

        #code below is from concatenate widget
        if data_list:
            tables = data_list
            domain = reduce(domain_union,
                        (table.domain for table in tables))
            tables = [Orange.data.Table.from_table(domain, table)
                      for table in tables]
            data = concat(tables)
            source_var = Orange.data.StringVariable.make("Filename")
            source_values = list(
                chain(*(repeat(fn, len(table))
                        for fn, table in zip(fnok_list, tables)))
                )
            label_var = Orange.data.StringVariable.make("Label")
            label_values = list(
                chain(*(repeat(self.label, len(table))
                        for fn, table in zip(fnok_list, tables)))
            )
            data = append_columns(
                data, **{"metas": [(source_var, source_values),
                                   (label_var, label_values)]})
            self.data = data
            if OLD_DOMAINEDITOR:
                self.editor_model.set_domain(data.domain)
            else:
                self.openContext(data.domain)
        else:
            self.data = None
            if OLD_DOMAINEDITOR:
                self.editor_model.reset()
            else:
                self.domain_editor.set_domain(None)

        self.send("Data", self.data)


if __name__ == "__main__":
    import sys
    a = QApplication(sys.argv)
    ow = OWFiles()
    ow.show()
    a.exec_()
    ow.saveSettings()