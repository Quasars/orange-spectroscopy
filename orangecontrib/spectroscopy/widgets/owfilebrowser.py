import os
import re
import logging
from warnings import catch_warnings
import pathlib
from fnmatch import fnmatch

from AnyQt.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QWidget,
    QLineEdit,
    QTreeView,
    QComboBox,
    QHeaderView,
    QSizePolicy as Policy,
)

# QFileSystemModel model moved to QtGui in PyQt6;
# use the following until AnyQt unifies it
try:
    from AnyQt.QtGui import QFileSystemModel
except ImportError:
    from AnyQt.QtWidgets import QFileSystemModel

from AnyQt.QtCore import QSize, QDir, QPoint, Qt, Signal, QSortFilterProxyModel

from orangewidget.utils.filedialogs import format_filter

from Orange.data.io import FileFormat, class_from_qualified_name
from Orange.data.io_base import MissingReaderException
from Orange.data.table import Table
from Orange.widgets import widget, gui
from Orange.widgets.data.owfile import add_origin
from Orange.widgets.settings import (
    PerfectDomainContextHandler,
)
# from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.widget import Msg, Output

import orangecontrib.spectroscopy

DEFAULT_READER_TEXT = "Automatically detect type"

log = logging.getLogger(__name__)


class AddressBarLabel(QLabel):
    clicked = Signal(QPoint)

    def __init__(self, txt, interact=True):
        self.interaction_allowed = interact
        super().__init__(f" {txt} ")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if self.interaction_allowed == True:
            self.setStyleSheet(
                """
                QLabel{ border-radius: 5px; }
                QLabel:hover{ background-color: #d4d2d2; }
            """
            )

        self.setSizePolicy(Policy.Preferred, Policy.Preferred)
        width = self.fontMetrics().boundingRect(self.text()).width()
        # width = self.fontMetrics().width(self.text())
        self.setFixedWidth(width)

    def setToBold(self):
        self.setStyleSheet("font-weight: bold")
        width = self.fontMetrics().width(self.text())
        width = int(1.2 * width)
        self.setFixedWidth(width)

    def mousePressEvent(self, ev):
        if self.interaction_allowed:
            if ev.button() == Qt.MouseButton.LeftButton:
                self.clicked.emit(self.mapToParent(ev.pos()))


class AddressBar(QWidget):
    directoryClicked = Signal(str)

    def __init__(self):
        super().__init__()

        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setObjectName("main_frame")
        self.setStyleSheet(
            """
            QWidget#main_frame { border: 2px grey;
                                border-radius: 5px;
                                background-color: white;}
        """
        )

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Create the sub-layout for the address bar
        self.sub_frame = QWidget()
        self.sub_layout = QHBoxLayout()
        self.sub_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sub_layout.setContentsMargins(5, 2, 5, 2)
        self.sub_frame.setLayout(self.sub_layout)
        self.sub_frame.setObjectName("sub_frame")
        self.sub_frame.setStyleSheet(
            """
            QWidget#sub_frame { border-radius: 5px;
                                background-color: white;}
        """
        )

        self.layout.addWidget(self.sub_frame)
        # self.layout.addStretch()
        self.setLayout(self.layout)

    def stripAddressBar(self):
        # Remove all widgets and spacer items from the sub-layout
        for i in reversed(range(self.sub_layout.count())):
            item = self.sub_layout.itemAt(i)
            if item.widget().isHidden():
                item.widget().setVisible(True)

            if item:
                item.widget().deleteLater()
                self.sub_layout.removeItem(item)
            else:
                pass

    def set_required_width(self):
        width = self.get_required_width()
        self.sub_frame.setFixedWidth(width + 10)
        return width

    def get_required_width(self):
        width = 0
        for i in range(self.sub_layout.count()):
            item = self.sub_layout.itemAt(i)
            if item.widget().isHidden():
                pass
            else:
                width = width + item.geometry().width()

        return width

    def updateAddressBar(self, path: pathlib.PurePath):
        # Remove any existing content from the address bar
        self.stripAddressBar()

        self.path_parts = path.parts
        self.displayed_parts = list(self.path_parts)
        for i, part in enumerate(self.displayed_parts):
            if "\\" in part:
                slashindex = part.find("\\")
                self.displayed_parts[i] = part[0:slashindex]

        # Add a QLabel widget for each subdirectory and a separator after each one
        ddd = AddressBarLabel("...", interact=False)
        self.sub_layout.addWidget(ddd)
        self.sub_layout.itemAt(0).widget().hide()

        for i, name in enumerate(self.displayed_parts):
            sub_dir_l = AddressBarLabel(name)
            sub_dir_l.clicked.connect(self.onSubDirectoryClicked)
            self.sub_layout.addWidget(sub_dir_l)

            if i < len(self.path_parts) - 1:
                sep = AddressBarLabel("/", interact=False)
                self.sub_layout.addWidget(sep)

        for i in range(self.sub_layout.count() - 1):
            self.sub_layout.itemAt(i + 1).widget().setVisible(True)

        self.sub_layout.setSpacing(0)
        self.set_required_width()
        while self.geometry().width() < self.get_required_width():
            self.resizeToFit(self.geometry().width())

    def onSubDirectoryClicked(self, ev):
        labelItem = self.childAt(ev)
        index = int((self.sub_layout.indexOf(labelItem) - 1) / 2)
        new_directory_path = pathlib.PurePath(*self.path_parts[0 : index + 1])
        if new_directory_path is not None:
            self.updateAddressBar(new_directory_path)
            self.directoryClicked.emit(str(new_directory_path))

    def resizeToFit(self, maxsize):
        if maxsize < self.sub_frame.geometry().width():
            self.sub_layout.itemAt(0).widget().show()
            idx = 1
            while self.sub_layout.itemAt(idx).widget().isHidden():
                idx += 1
            if idx == self.sub_layout.count() - 1 or idx == self.sub_layout.count() - 2:
                pass
            else:
                self.sub_layout.itemAt(idx).widget().hide()
                self.set_required_width()
        else:
            idx = 2
            while self.sub_layout.itemAt(idx).widget().isHidden():
                idx += 1
            idx = idx - 1

            if idx == 1:
                self.sub_layout.itemAt(0).widget().hide()
                width_needed = (
                    self.get_required_width()
                    + self.sub_layout.itemAt(idx).geometry().width()
                    + 10
                )
            else:
                width_needed = (
                    self.get_required_width()
                    + self.sub_layout.itemAt(idx).geometry().width()
                    + 10
                )

            if maxsize > width_needed:
                self.sub_layout.itemAt(idx).widget().show()
                self.sub_frame.setFixedWidth(width_needed)

    def resizeEvent(self, event):
        self.resizeToFit(event.size().width())


class FileFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_string = ""
        self.filtered_extensions = ()

    def setFilterString(self, filter_string):
        self.filter_string = filter_string
        self.invalidateFilter()  # Trigger re-filtering

    def setExtensionFilter(self, extensions_tuple):
        self.filtered_extensions = extensions_tuple
        self.invalidateFilter()  # Trigger re-filtering

    def filterAcceptsRow(self, source_row, source_parent):
        idx = self.sourceModel().index(source_row, 0, source_parent)
        name = idx.data()

        if self.filtered_extensions == ():
            pass
        else:
            try:
                if (
                    re.search(self.filter_string, name) == None
                    or os.path.splitext(name)[1] not in self.filtered_extensions
                ) and self.sourceModel().isDir(idx) == False:
                    return False
            except:
                pass

        return True


class OWFileBrowser(widget.OWWidget):
    name = "File Browser"
    id = "orangecontrib.protosec.widgets.filebrowser"
    icon = "icons/quickfile.svg"
    description = (
        "Read data from an input file selected from the file tree"
        "and send a data table to the output."
    )
    priority = 10

    class Outputs:
        data = Output(
            "Data", Table, doc="Attribute-valued dataset read from the input file."
        )

    want_main_area = False

    SIZE_LIMIT = 1e9

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    class Warning(widget.OWWidget.Warning):
        file_too_big = widget.Msg(
            "The file is too large to load automatically." " Press Reload to load."
        )
        load_warning = widget.Msg("Read warning:\n{}")
        performance_warning = widget.Msg(
            "Categorical variables with >100 values may decrease performance."
        )

    class Error(widget.OWWidget.Error):
        missing_reader = Msg("No reader for this file.")
        file_not_found = widget.Msg("File not found.")
        sheet_error = widget.Msg("Error listing available sheets.")
        unknown = widget.Msg("Read error:\n{}")

    class NoFileSelected:
        pass

    def __init__(self):
        super().__init__()

        self.domain = None
        self.data = None
        self.reader = None
        self.auto_reader = False
        self.sheet = None

        readers = [
            f
            for f in FileFormat.formats
            if getattr(f, "read", None) and getattr(f, "EXTENSIONS", None)
        ]

        def group_readers_per_addon_key(w):
            # readers from Orange.data.io should go first
            def package(w):
                package = w.qualified_name().split(".")[:-1]
                package = package[:2]
                if ".".join(package) == "Orange.data":
                    return ["0"]  # force "Orange" to come first
                return package

            return package(w), w.DESCRIPTION

        self.available_readers = sorted(set(readers), key=group_readers_per_addon_key)

        initial_directory = os.getcwd()
        self.selected_folder = initial_directory
        self.base_folder = ""
        self.file_list = []
        self.filter_string = ""

        layout = QVBoxLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)

        browse_layout = QHBoxLayout()
        layout.addLayout(browse_layout)

        size = 30
        self.adr_bar = AddressBar()
        self.adr_bar.setMinimumSize(QSize(100, size))
        self.adr_bar.setSizePolicy(Policy.Expanding, Policy.Fixed)
        self.adr_bar.updateAddressBar(pathlib.PurePath(initial_directory))
        self.adr_bar.directoryClicked.connect(self.jump_to_folder)
        browse_layout.addWidget(self.adr_bar)

        browse_button = gui.button(
            self, self, " ... ", callback=self.browse_folder, autoDefault=False
        )
        browse_layout.addWidget(browse_button)

        box = gui.hBox(None, addToLayout=False, margin=0)
        box.setSizePolicy(Policy.Expanding, Policy.Fixed)
        self.reader_combo = QComboBox(self)
        self.reader_combo.setSizePolicy(Policy.Expanding, Policy.Fixed)
        self.reader_combo.setMinimumSize(QSize(100, 1))
        self.reader_combo.activated[int].connect(self.select_reader)
        box.layout().addWidget(self.reader_combo)
        layout.addWidget(box)
        self._initialize_reader_combo()

        self.sheet_box = gui.hBox(None, addToLayout=False, margin=0)
        self.sheet_combo = QComboBox()
        self.sheet_combo.textActivated.connect(self.select_sheet)
        self.sheet_combo.setSizePolicy(Policy.Expanding, Policy.Fixed)
        self.sheet_combo.setMinimumSize(QSize(50, 1))
        self.sheet_label = QLabel()
        self.sheet_label.setText("Sheet")
        self.sheet_label.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_box.layout().addWidget(self.sheet_label, Qt.AlignLeft)
        self.sheet_box.layout().addWidget(self.sheet_combo, Qt.AlignVCenter)
        layout.addWidget(self.sheet_box)
        self.sheet_box.hide()

        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Enter a string to filter files...")
        self.filter_input.textChanged.connect(self.filter_files)
        layout.addWidget(self.filter_input)

        self.treeview = QTreeView(self)
        layout.addWidget(self.treeview)
        self.treeview.setSizePolicy(Policy.Expanding, Policy.Expanding)

        # Source model
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setReadOnly(True)
        self.fileSystemModel.setRootPath(initial_directory)
        self.fileSystemModel.setFilter(QDir.Files | QDir.AllDirs | QDir.NoDotAndDotDot)

        # Proxy model for filtering
        self.proxy_model = FileFilterProxyModel()
        self.proxy_model.setDynamicSortFilter(True)
        self.proxy_model.setSourceModel(self.fileSystemModel)
        if self.reader is not None:
            self.proxy_model.setExtensionFilter(self.reader.EXTENSIONS)

        self.treeview.setModel(self.proxy_model)
        self.treeview.setRootIndex(
            self.proxy_model.mapFromSource(
                self.fileSystemModel.index(initial_directory)
            )
        )
        self.treeview.clicked.connect(self.load_selected_file)
        self.treeview.doubleClicked.connect(self.on_double_click)

        self.treeview.header().setStretchLastSection(False)
        self.treeview.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.info_box = gui.vBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(self.info_box, "No data loaded.")
        self.warnings = gui.widgetLabel(self.info_box, "")

    def set_fixed_readers(self, reader_description: list):
        self.available_readers = [
            self.get_described_reader(r) for r in reader_description
        ]
        self._initialize_reader_combo()

    def toggle_infobox(self):
        if self.info_box.isHidden():
            self.info_box.setHidden(False)
        else:
            self.info_box.setHidden(True)

    def browse_folder(self):
        fname = QFileDialog(
            self, "Select folder", self.selected_folder
        ).getExistingDirectory()
        if fname:
            self.jump_to_folder(fname)
            self.adr_bar.updateAddressBar(pathlib.PurePath(self.selected_folder))

    def folder_jump_up(self):
        self.jump_to_folder(os.path.abspath(os.path.dirname(self.selected_folder)))

    def jump_to_folder(self, folder):
        self.selected_folder = folder
        self.treeview.setRootIndex(
            self.proxy_model.mapFromSource(
                self.fileSystemModel.index(self.selected_folder)
            )
        )

    def filter_files(self):
        filter_text = self.filter_input.text().strip()
        self.proxy_model.setFilterString(filter_text)
        if self.reader is not None and self.auto_reader is False:
            self.proxy_model.setExtensionFilter(self.reader.EXTENSIONS)
        else:
            self.proxy_model.setExtensionFilter(())

    def load_selected_file(self):
        idx = self.treeview.currentIndex()
        source_index = self.proxy_model.mapToSource(idx)
        indexItem = self.fileSystemModel.index(
            source_index.row(), 0, source_index.parent()
        )
        self.filepath = self.fileSystemModel.filePath(indexItem)

        if not self.fileSystemModel.isDir(indexItem):
            error = self._try_load()
            if error:
                error()
                self.data = None
                self.sheet_box.hide()
                self.Outputs.data.send(None)
                self.infolabel.setText("No data.")

    def on_double_click(self):
        idx = self.treeview.currentIndex()
        source_index = self.proxy_model.mapToSource(idx)
        indexItem = self.fileSystemModel.index(
            source_index.row(), 0, source_index.parent()
        )

        if self.proxy_model.sourceModel().isDir(source_index):
            self.jump_to_folder(self.fileSystemModel.filePath(indexItem))
            self.adr_bar.updateAddressBar(pathlib.PurePath(self.selected_folder))
            # self.treeview.setRootIndex(self.proxy_model.mapFromSource(self.fileSystemModel.index(self.selected_folder)))

    def auto_get_reader(self, filename):
        filename = pathlib.PurePath(filename)
        for reader in self.available_readers:
            if filename.suffix in reader.EXTENSIONS:
                # print("Reader found:" + reader.DESCRIPTION)
                return reader

        raise MissingReaderException(f'No readers for file "{filename}"')

    def _try_load(self):
        self.clear_messages()

        if self.filepath and not os.path.exists(self.filepath):
            return self.Error.file_not_found

        if (
            self.auto_reader
        ):  # if no reader is specified than try autofind it based of file extension
            try:
                self.reader_combo.setCurrentIndex(0)
                # reader = FileFormat.get_reader(self.filepath)
                reader = self.auto_get_reader(self.filepath)
                qname = reader.qualified_name()
                self.reader = class_from_qualified_name(qname)
                # self.reader = self.auto_get_reader(self.filepath)
            except:
                return self.Error.missing_reader

        if os.path.splitext(self.filepath)[1] in self.reader.EXTENSIONS:
            self.reader_obj = self.reader(self.filepath)

        if self.reader_obj is self.NoFileSelected:
            self.Outputs.data.send(None)
            return None

        if self.sheet:
            self.reader_obj.select_sheet(self.sheet)

        with catch_warnings(record=True) as warnings:
            try:
                data = self.reader_obj.read()
            except Exception as ex:
                log.exception(ex)
                return lambda x=ex: self.Error.unknown(str(x))
            if warnings:
                self.Warning.load_warning(warnings[-1].message.args[0])

        if os.path.getsize(self.filepath) > self.SIZE_LIMIT:
            return self.Warning.file_too_big

        try:
            self._update_sheet_combo()
        except Exception:
            return self.Error.sheet_error

        self.infolabel.setText(self._describe(data))

        add_origin(data, self.filepath)
        self.data = data
        self.openContext(data.domain)
        self.Outputs.data.send(self.data)

        return None

    ########## GET THE READERS ##########
    def get_described_reader(self, description):
        """Return reader instance that reads the file given by the read description.

        Parameters
        ----------
        filename : str

        Returns
        -------
        FileFormat
        """
        # print(description)
        try:
            reader = [
                f
                for f in FileFormat.formats
                if getattr(f, "DESCRIPTION", None) == description
                and getattr(f, "EXTENSIONS", None)
            ]
            qname = reader[0].qualified_name()
            reader_class = class_from_qualified_name(qname)
            return reader_class
        except:
            raise OSError(f"No readers for {description} files.")

    def select_reader(self, n):
        if n == 0:  # default
            self.auto_reader = True
            self.load_selected_file()
        elif n <= len(self.available_readers):
            self.auto_reader = False
            self.reader = self.available_readers[n - 1]
            self.load_selected_file()
        else:  # the rest include just qualified names
            self.auto_reader = False
            reader = self.reader_combo.itemText(n)
            qname = reader[0].qualified_name()
            self.reader = class_from_qualified_name(qname)
            self.load_selected_file()

        self.filter_files()

    def select_sheet(self):
        self.sheet = self.sheet_combo.currentText()
        self.load_selected_file()

    def _update_sheet_combo(self):
        if len(self.reader_obj.sheets) < 2:
            self.sheet_box.hide()
            self.reader_obj.select_sheet(None)
            return

        self.sheet_combo.clear()
        self.sheet_combo.addItems(self.reader_obj.sheets)
        self._select_active_sheet()
        self.sheet_box.show()

    def _select_active_sheet(self):
        try:
            idx = self.reader_obj.sheets.index(self.reader_obj.sheet)
            self.sheet_combo.setCurrentIndex(idx)
        except ValueError:
            # Requested sheet does not exist in this file
            self.reader_obj.select_sheet(None)
            self.sheet_combo.setCurrentIndex(0)

    def _initialize_reader_combo(self):
        self.reader_combo.clear()
        filters = [format_filter(f) for f in self.available_readers]
        self.reader_combo.addItems([DEFAULT_READER_TEXT] + filters)
        self.reader_combo.setCurrentIndex(0)
        self.auto_reader = True
        # self.reader_combo.setDisabled(True)
        # additional readers may be added in self._get_reader()

    @staticmethod
    def _describe(table):
        def missing_prop(prop):
            if prop:
                return f"({prop * 100:.1f}% missing values)"
            else:
                return "(no missing values)"

        domain = table.domain
        text = ""

        attrs = getattr(table, "attributes", {})
        descs = [attrs[desc] for desc in ("Name", "Description") if desc in attrs]
        if len(descs) == 2:
            descs[0] = f"<b>{descs[0]}</b>"
        if descs:
            text += f"<p>{'<br/>'.join(descs)}</p>"

        text += f"<p>{len(table)} instance(s)"

        missing_in_attr = missing_prop(
            table.has_missing_attribute() and table.get_nan_frequency_attribute()
        )
        missing_in_class = missing_prop(
            table.has_missing_class() and table.get_nan_frequency_class()
        )
        text += f"<br/>{len(domain.attributes)} feature(s) {missing_in_attr}"
        if domain.has_continuous_class:
            text += f"<br/>Regression; numerical class {missing_in_class}"
        elif domain.has_discrete_class:
            text += (
                "<br/>Classification; categorical class "
                f"with {len(domain.class_var.values)} values {missing_in_class}"
            )
        elif table.domain.class_vars:
            text += (
                "<br/>Multi-target; "
                f"{len(table.domain.class_vars)} target variables "
                f"{missing_in_class}"
            )
        else:
            text += "<br/>Data has no target variable."
        text += f"<br/>{len(domain.metas)} meta attribute(s)"
        text += "</p>"

        if "Timestamp" in table.domain:
            # Google Forms uses this header to timestamp responses
            text += (
                f"<p>First entry: {table[0, 'Timestamp']}<br/>"
                f"Last entry: {table[-1, 'Timestamp']}</p>"
            )
        return text


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWFileBrowser).run()
