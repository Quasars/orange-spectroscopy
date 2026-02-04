import os
import logging
from itertools import chain
from warnings import catch_warnings
from urllib.parse import urlparse
from typing import List

import numpy as np
from AnyQt.QtCore import Qt, QSize
from AnyQt.QtWidgets import (
    QStyle,
    QComboBox,
    QMessageBox,
    QGridLayout,
    QLabel,
    QSizePolicy as Policy,
    QCompleter,
)

from Orange.data.io import FileFormat, UrlReader, class_from_qualified_name
from Orange.data.table import Table
from Orange.preprocess.preprocess import Preprocess, PreprocessorList
from Orange.widgets import widget, gui
from Orange.widgets.data.owfile import NamedURLModel, LineEditSelectOnFocus, add_origin
from Orange.widgets.settings import (
    Setting,
    ContextSetting,
    PerfectDomainContextHandler,
    SettingProvider,
)
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.filedialogs import (
    RecentPathsWComboMixin,
    open_filename_dialog,
)
from Orange.widgets.widget import MultiInput, Msg, Output

# Backward compatibility (from owfile): class RecentPath used to be defined in this module,
# and it is used in saved (pickled) settings. It must be imported into the
# module's namespace so that old saved settings still work
from Orange.widgets.utils.filedialogs import RecentPath

from orangecontrib.spectroscopy import get_sample_datasets_dir


log = logging.getLogger(__name__)


class OWTilefile(widget.OWWidget, RecentPathsWComboMixin):
    name = "Tile File"
    id = "orangecontrib.spectroscopy.widgets.tilefile"
    icon = "icons/tilefile.svg"
    description = (
        "Read data tile-by-tile from input files, "
        "preprocess, and send a data table to the output."
    )
    priority = 10000
    replaces = ["orangecontrib.protospec.widgets.owtilefile.OWTilefile"]

    class Inputs:
        preprocessor = MultiInput("Preprocessor", Preprocess)

    class Outputs:
        data = Output(
            "Data", Table, doc="Preprocessed dataset read from the input files."
        )

    want_main_area = False

    SEARCH_PATHS = [("sample-datasets", get_sample_datasets_dir())]
    # Always trigger size limit warning (never auto-load)
    SIZE_LIMIT = 0
    LOCAL_FILE, URL = range(2)

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    # pylint seems to want declarations separated from definitions
    recent_paths: List[RecentPath]
    recent_urls: List[str]
    variables: list

    # Overload RecentPathsWidgetMixin.recent_paths to set defaults
    recent_paths = Setting(
        [
            RecentPath("", "sample-datasets", "agilent/5_mosaic_agg1024.dmt"),
        ]
    )
    recent_urls = Setting([])
    source = Setting(LOCAL_FILE)
    xls_sheet = ContextSetting("")
    sheet_names = Setting({})
    url = Setting("")

    variables = ContextSetting([])

    domain_editor = SettingProvider(DomainEditor)

    class Warning(widget.OWWidget.Warning):
        no_preprocessor = Msg("No preprocessor on input. Press Reload to load anyway.")
        new_preprocessor = Msg("Preprocessor has changed. Press Reload to apply.")
        file_too_big = widget.Msg(
            "The file is too large to load automatically. Press Reload to load."
        )
        load_warning = widget.Msg("Read warning:\n{}")
        performance_warning = widget.Msg(
            "Categorical variables with >100 values may decrease performance."
        )

    class Error(widget.OWWidget.Error):
        missing_reader = Msg("No tile-by-tile reader for this file.")
        file_not_found = widget.Msg("File not found.")
        sheet_error = widget.Msg("Error listing available sheets.")
        unknown = widget.Msg("Read error:\n{}")

    class NoFileSelected:
        pass

    UserAdviceMessages = [
        widget.Message(
            "Connect a Preprocessor which results in data-reduction ",
            "to best make use of this widget.",
        ),
    ]

    def __init__(self):
        self.preprocessor = PreprocessorList()
        super().__init__()
        ### owfile init code-copy ###
        RecentPathsWComboMixin.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.reader = None

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)
        vbox = gui.radioButtons(
            None,
            self,
            "source",
            box=True,
            addSpace=True,
            callback=self.load_data,
            addToLayout=False,
        )

        rb_button = gui.appendRadioButton(vbox, "File:", addToLayout=False)
        layout.addWidget(rb_button, 0, 0, Qt.AlignVCenter)

        box = gui.hBox(None, addToLayout=False, margin=0)
        box.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.activated[int].connect(self.select_file)
        box.layout().addWidget(self.file_combo)
        layout.addWidget(box, 0, 1)

        file_button = gui.button(
            None, self, '...', callback=self.browse_file, autoDefault=False
        )
        file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 2)

        reload_button = gui.button(
            None, self, "Reload", callback=self.load_data, autoDefault=False
        )
        reload_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 3)

        self.sheet_box = gui.hBox(None, addToLayout=False, margin=0)
        self.sheet_combo = gui.comboBox(
            None,
            self,
            "xls_sheet",
            callback=self.select_sheet,
            sendSelectedValue=True,
        )
        self.sheet_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_label = QLabel()
        self.sheet_label.setText('Sheet')
        self.sheet_label.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_box.layout().addWidget(self.sheet_label, Qt.AlignLeft)
        self.sheet_box.layout().addWidget(self.sheet_combo, Qt.AlignVCenter)
        layout.addWidget(self.sheet_box, 2, 1)
        self.sheet_box.hide()

        rb_button = gui.appendRadioButton(vbox, "URL:", addToLayout=False)
        layout.addWidget(rb_button, 3, 0, Qt.AlignVCenter)

        self.url_combo = url_combo = QComboBox()
        url_model = NamedURLModel(self.sheet_names)
        url_model.wrap(self.recent_urls)
        url_combo.setLineEdit(LineEditSelectOnFocus())
        url_combo.setModel(url_model)
        url_combo.setSizePolicy(Policy.Ignored, Policy.Fixed)
        url_combo.setEditable(True)
        url_combo.setInsertPolicy(url_combo.InsertAtTop)
        url_edit = url_combo.lineEdit()
        m = url_edit.textMargins()
        l, t, r, b = m.left(), m.top(), m.bottom(), m.right()
        url_edit.setTextMargins(l + 5, t, r, b)
        layout.addWidget(url_combo, 3, 1, 3, 3)
        url_combo.activated.connect(self._url_set)
        # whit completer we set that combo box is case sensitive when
        # matching the history
        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseSensitive)
        url_combo.setCompleter(completer)

        box = gui.vBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(box, 'No data loaded.')
        self.warnings = gui.widgetLabel(box, '')

        box = gui.widgetBox(self.controlArea, "Columns (Double click to edit)")
        self.domain_editor = DomainEditor(self)
        self.editor_model = self.domain_editor.model()
        box.layout().addWidget(self.domain_editor)

        box = gui.hBox(self.controlArea)
        gui.button(
            box,
            self,
            "Browse documentation datasets",
            callback=lambda: self.browse_file(True),
            autoDefault=False,
        )
        gui.rubber(box)

        gui.button(box, self, "Reset", callback=self.reset_domain_edit)
        self.apply_button = gui.button(
            box, self, "Apply", callback=self.apply_domain_edit
        )
        self.apply_button.setEnabled(False)
        self.apply_button.setFixedWidth(170)
        self.editor_model.dataChanged.connect(
            lambda: self.apply_button.setEnabled(True)
        )

        self.set_file_list()
        # Must not call open_file from within __init__. open_file
        # explicitly re-enters the event loop (by a progress bar)

        self.setAcceptDrops(True)
        ### End code copy ###

        box = gui.vBox(self.controlArea, "Preprocessor")
        self.info_preproc = gui.widgetLabel(box, 'No preprocessor on input.')

        self.Warning.file_too_big()

    ### owfile methods code-copy ###
    @staticmethod
    def sizeHint():
        return QSize(600, 550)

    def select_file(self, n):
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.source = self.LOCAL_FILE
            self.load_data()
            self.set_file_list()

    def select_sheet(self):
        self.recent_paths[0].sheet = self.sheet_combo.currentText()
        self.load_data()

    def _url_set(self):
        url = self.url_combo.currentText()
        pos = self.recent_urls.index(url)
        url = url.strip()

        if not urlparse(url).scheme:
            url = 'http://' + url
            self.url_combo.setItemText(pos, url)
            self.recent_urls[pos] = url

        self.source = self.URL
        self.load_data()

    # Open a file, create data from it and send it over the data channel
    def load_data(self):
        # We need to catch any exception type since anything can happen in
        # file readers
        self.closeContext()
        self.domain_editor.set_domain(None)
        self.apply_button.setEnabled(False)
        self.clear_messages()
        self.set_file_list()

        error = self._try_load()
        if error:
            error()
            self.data = None
            self.sheet_box.hide()
            self.Outputs.data.send(None)
            self.infolabel.setText("No data.")

    def _try_load(self):
        # pylint: disable=broad-except
        if self.last_path() and not os.path.exists(self.last_path()):
            return self.Error.file_not_found

        try:
            self.reader = self._get_reader()
            assert self.reader is not None
        except Exception:
            return self.Error.missing_reader

        if self.reader is self.NoFileSelected:
            self.Outputs.data.send(None)
            return None

        try:
            self._update_sheet_combo()
        except Exception:
            return self.Error.sheet_error

        with catch_warnings(record=True) as warnings:
            try:
                data = self.reader.read()
            except Exception as ex:
                log.exception(ex)
                return lambda x=ex: self.Error.unknown(str(x))
            if warnings:
                self.Warning.load_warning(warnings[-1].message.args[0])

        self.infolabel.setText(self._describe(data))

        self.loaded_file = self.last_path()
        add_origin(data, self.loaded_file)
        self.data = data
        self.openContext(data.domain)
        self.apply_domain_edit()  # sends data
        return None

    def _update_sheet_combo(self):
        if len(self.reader.sheets) < 2:
            self.sheet_box.hide()
            self.reader.select_sheet(None)
            return

        self.sheet_combo.clear()
        self.sheet_combo.addItems(self.reader.sheets)
        self._select_active_sheet()
        self.sheet_box.show()

    def _select_active_sheet(self):
        if self.reader.sheet:
            try:
                idx = self.reader.sheets.index(self.reader.sheet)
                self.sheet_combo.setCurrentIndex(idx)
            except ValueError:
                # Requested sheet does not exist in this file
                self.reader.select_sheet(None)
        else:
            self.sheet_combo.setCurrentIndex(0)

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

        if 'Timestamp' in table.domain:
            # Google Forms uses this header to timestamp responses
            text += (
                f"<p>First entry: {table[0, 'Timestamp']}<br/>"
                f"Last entry: {table[-1, 'Timestamp']}</p>"
            )
        return text

    def storeSpecificSettings(self):
        self.current_context.modified_variables = self.variables[:]

    def retrieveSpecificSettings(self):
        if hasattr(self.current_context, "modified_variables"):
            self.variables[:] = self.current_context.modified_variables

    def reset_domain_edit(self):
        self.domain_editor.reset_domain()
        self.apply_domain_edit()

    def _inspect_discrete_variables(self, domain):
        for var in chain(domain.variables, domain.metas):
            if var.is_discrete and len(var.values) > 100:
                self.Warning.performance_warning()

    def apply_domain_edit(self):
        self.Warning.performance_warning.clear()
        if self.data is None:
            table = None
        else:
            domain, cols = self.domain_editor.get_domain(self.data.domain, self.data)
            if not (domain.variables or domain.metas):
                table = None
            elif domain is self.data.domain:
                table = self.data
            else:
                X, y, m = cols
                table = Table.from_numpy(domain, X, y, m, self.data.W)
                table.name = self.data.name
                table.ids = np.array(self.data.ids)
                table.attributes = getattr(self.data, 'attributes', {})
                self._inspect_discrete_variables(domain)

        self.Outputs.data.send(table)
        self.apply_button.setEnabled(False)

    def get_widget_name_extension(self):
        _, name = os.path.split(self.loaded_file)
        return os.path.splitext(name)[0]

    def send_report(self):
        def get_ext_name(filename):
            try:
                return FileFormat.names[os.path.splitext(filename)[1]]
            except KeyError:
                return "unknown"

        if self.data is None:
            self.report_paragraph("File", "No file.")
            return

        if self.source == self.LOCAL_FILE:
            home = os.path.expanduser("~")
            if self.loaded_file.startswith(home):
                # os.path.join does not like ~
                name = (
                    "~"
                    + os.path.sep
                    + self.loaded_file[len(home) :].lstrip("/").lstrip("\\")
                )
            else:
                name = self.loaded_file
            if self.sheet_combo.isVisible():
                name += f" ({self.sheet_combo.currentText()})"
            self.report_items(
                "File", [("File name", name), ("Format", get_ext_name(name))]
            )
        else:
            self.report_items(
                "Data", [("Resource", self.url), ("Format", get_ext_name(self.url))]
            )

        self.report_data("Data", self.data)

    @staticmethod
    def dragEnterEvent(event):
        """Accept drops of valid file urls"""
        urls = event.mimeData().urls()
        if urls:
            try:
                FileFormat.get_reader(urls[0].toLocalFile())
                event.acceptProposedAction()
            except IOError:
                pass

    def dropEvent(self, event):
        """Handle file drops"""
        urls = event.mimeData().urls()
        if urls:
            self.add_path(urls[0].toLocalFile())  # add first file
            self.source = self.LOCAL_FILE
            self.load_data()

    def workflowEnvChanged(self, key, value, oldvalue):
        """
        Function called when environment changes (e.g. while saving the scheme)
        It make sure that all environment connected values are modified
        (e.g. relative file paths are changed)
        """
        self.update_file_list(key, value, oldvalue)

    #### End code copy ####

    @staticmethod
    def _is_preproc(p):
        """
        Tests that a preprocessor is not None or empty PreprocessorList
        """
        return not (
            p is None or (isinstance(p, PreprocessorList) and len(p.preprocessors) == 0)
        )

    @staticmethod
    def _format_preproc_str(preprocessor):
        pstrings = []
        for i in preprocessor.preprocessors:
            if isinstance(i, PreprocessorList):
                for pp in i.preprocessors:
                    pstrings.append(str(pp))
            else:
                pstrings.append(str(i))
        return "\n".join(ps for ps in pstrings if ps != "None")

    def warn_preprocessor(self):
        self.Warning.no_preprocessor.clear()
        self.Warning.new_preprocessor.clear()
        if not any(self._is_preproc(p) for p in self.preprocessor.preprocessors):
            self.info_preproc.setText("No preprocessor on input.")
            self.Warning.no_preprocessor()
            return True
        self.info_preproc.setText(
            "New preprocessor, reload file to use.\n"
            + self._format_preproc_str(self.preprocessor)
        )
        self.Warning.new_preprocessor()
        return False

    @Inputs.preprocessor.insert
    def insert_preprocessor(self, index: int, preprocessor: Preprocess):
        """Insert a Preprocessor or PreprocessorList at index"""
        self.preprocessor.preprocessors.insert(index, preprocessor)
        self.warn_preprocessor()

    @Inputs.preprocessor
    def set_preprocessor(self, index: int, preprocessor: Preprocess):
        """Set the input preprocessor at index"""
        self.preprocessor.preprocessors[index] = preprocessor
        self.warn_preprocessor()

    @Inputs.preprocessor.remove
    def remove_preprocessor(self, index: int):
        """Remove a preprocessor at index"""
        del self.preprocessor.preprocessors[index]
        self.warn_preprocessor()

    def browse_file(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QMessageBox.information(
                    None,
                    "File",
                    "Cannot find the directory with documentation datasets",
                )
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        readers = [
            f
            for f in FileFormat.formats
            if getattr(f, 'read_tile', None) and getattr(f, "EXTENSIONS", None)
        ]
        filename, reader, _ = open_filename_dialog(start_file, None, readers)
        if not filename:
            return
        self.add_path(filename)
        if reader is not None:
            self.recent_paths[0].file_format = reader.qualified_name()

        self.source = self.LOCAL_FILE

        if self.warn_preprocessor():
            return
        self.load_data()

    @classmethod
    def get_tile_reader(cls, filename):
        """Return reader instance that can be used to read a file tile-wise

        Parameters
        ----------
        filename : str

        Returns
        -------
        FileFormat
        """
        readers = [
            f
            for f in FileFormat.formats
            if getattr(f, 'read_tile', None) and getattr(f, "EXTENSIONS", None)
        ]
        for reader in readers:
            if os.path.splitext(filename)[1] in reader.EXTENSIONS:
                return reader(filename)

        raise IOError('No readers for file "{}"'.format(filename))

    def _get_reader(self):
        """
        Returns
        -------
        FileFormat
        """
        if self.source == self.LOCAL_FILE:
            path = self.last_path()
            if self.recent_paths and self.recent_paths[0].file_format:
                qname = self.recent_paths[0].file_format
                reader_class = class_from_qualified_name(qname)
                reader = reader_class(path)
            else:
                reader = self.get_tile_reader(path)
            if self.recent_paths and self.recent_paths[0].sheet:
                reader.select_sheet(self.recent_paths[0].sheet)
            # set preprocessor here
            if hasattr(reader, "read_tile"):
                reader.set_preprocessor(self.preprocessor)
                if self.preprocessor is not None:
                    self.info_preproc.setText(
                        self._format_preproc_str(self.preprocessor)
                    )
            else:
                # only allow readers with tile-by-tile support to run.
                reader = None
            return reader
        elif self.source == self.URL:
            url = self.url_combo.currentText().strip()
            if url:
                return UrlReader(url)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.spectroscopy.preprocess import Cut, LinearBaseline

    preproc = PreprocessorList([LinearBaseline(), Cut(lowlim=2000, highlim=2006)])
    WidgetPreview(OWTilefile).run(insert_preprocessor=(0, preproc))
