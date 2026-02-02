import math
import os
from functools import reduce
from itertools import chain, count, repeat
from collections import Counter, namedtuple, defaultdict
from typing import List

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QSizePolicy as Policy, QGridLayout, QLabel, \
    QFileDialog, QStyle, QListWidget

from Orange.data import Domain, Table, Variable, ContinuousVariable, StringVariable
from Orange.data.io import FileFormat, class_from_qualified_name
from Orange.data.util import get_unique_names_duplicates, get_unique_names
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, \
    PerfectDomainContextHandler, SettingProvider
from Orange.widgets.utils.annotated_data import add_columns
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.filedialogs import RecentPathsWidgetMixin, \
    RecentPath, open_filename_dialog
from Orange.widgets.utils.signals import Output

from orangecontrib.spectroscopy.io.util import SpectralFileFormat


def numpy_union_keep_order(A, B):
    """ Union of A and B. Elements not in A are
    added in the same order as in B."""
    # sorted list of elements missing in A
    to_add = np.setdiff1d(B, A)

    # find indices of new elements in the sorted array
    orig_sind = np.argsort(B)
    indices = np.searchsorted(B, to_add, sorter=orig_sind)

    # take the missing elements in the correct order
    to_add = B[sorted(orig_sind[indices])]

    return np.concatenate((A, to_add))


def decimals_neeeded_for_unique_str(l):
    min_diff = min(np.abs(np.diff(sorted(l))))
    log_diff = math.log(min_diff, 10)
    # We need to avoid possible degenerate cases like (1.5, 2.5):
    # the log of the difference (1.0) is 0, but rounded to zero
    # decimals we would get (2, 2) due to floating point rounding behavior.
    # Therefore, slightly increase the number of needed decimals.
    # The number was selected such that (1.5, 2.5) returned 1 decimal,
    # and (1.49, 2.51) returned 0.
    log_diff = log_diff - 0.008
    decimals = max(0, math.ceil(-log_diff))
    return decimals


def wns_to_unique_str(l):
    """ Convert a list on wns to a list of unique strings.

    Use 6 decimal places by default. If that is not sufficient,
    increase precision as needed for wns in conflict.
    """

    # 6 used to be the default: "%f" means "%0.6f"
    default_decimals = 6
    default_format = "%0." + str(default_decimals) + "f"

    same = defaultdict(list)
    for wn in l:
        same[default_format % wn].append(wn)

    decimals = {}
    for n in same:
        if len(same[n]) == 1:
            decimals[n] = default_decimals
        else:
            # add one decimal place because without it the error could be too big
            # for example, (1.49, 2.51) would round to 1 and 3, with an added decimal
            # it will show (1.5, 2.5). Max error before was 0.5*last_decimal_place,
            # which seems too big.
            decimals[n] = decimals_neeeded_for_unique_str(same[n]) + 1

    return [("%0." + str(decimals[default_format % wn]) + "f") % wn for wn in l]


def concatenate_data(tables, filenames, label):
    if not tables:
        return None

    orig_tables = tables

    # prepare xs from the spectral specific tables for join into a common domain
    spectral_specific_domains = []
    xss = [t.special_spectral_data[0] for t in tables
           if hasattr(t, "special_spectral_data")]
    xs = reduce(numpy_union_keep_order, xss, np.array([]))
    if len(xs):
        names = wns_to_unique_str(xs)
        attrs = [ContinuousVariable(n) for n in names]
        spectral_specific_domains = [Domain(attrs, None, None)]

    domain = _merge_domains(spectral_specific_domains + [table.domain for table in tables])
    name = get_unique_names(domain, "Filename")
    source_var = StringVariable(name)
    name = get_unique_names(domain, "Label")
    label_var = StringVariable(name)
    domain = add_columns(domain, metas=(source_var, label_var))

    # concatenate tables
    tables = [table.transform(domain) for table in tables]
    data = type(tables[0]).concatenate(tables)

    with data.unlocked():
        # fill in spectral data
        xs_sind = np.argsort(xs)
        xs_sorted = xs[xs_sind]
        pos = 0
        for table in orig_tables:
            if hasattr(table, "special_spectral_data"):
                special = table.special_spectral_data
                indices = xs_sind[np.searchsorted(xs_sorted, special[0])]
                data.X[pos:pos+len(table), indices] = special[1]
            pos += len(table)

        data[:, source_var] = np.array(list(
            chain(*(repeat(fn, len(table))
                    for fn, table in zip(filenames, tables)))  # noqa: B905
        )).reshape(-1, 1)
        data[:, label_var] = np.array(list(
            chain(*(repeat(label, len(table))
                    for _, table in zip(filenames, tables)))  # noqa: B905
        )).reshape(-1, 1)

    return data


def _merge_domains(domains):
    def fix_names(part):
        for i, attr, name in zip(count(), part, name_iter):
            if attr.name != name:
                part[i] = attr.renamed(name)

    parts = [_get_part(domains, set.union, part)
             for part in ("attributes", "class_vars", "metas")]
    all_names = [var.name for var in chain(*parts)]
    name_iter = iter(get_unique_names_duplicates(all_names))
    for part in parts:
        fix_names(part)
    return Domain(*parts)


def _get_part(domains, oper, part):
    # keep the order of variables: first compute union or intersections as
    # sets, then iterate through chained parts
    vars_by_domain = [getattr(domain, part) for domain in domains]
    valid = reduce(oper, map(set, vars_by_domain))
    valid_vars = [var for var in chain(*vars_by_domain) if var in valid]
    return _unique_vars(valid_vars)


def _unique_vars(seq: List[Variable]):
    AttrDesc = namedtuple(
        "AttrDesc",
        ("template", "original", "values", "number_of_decimals")
    )

    attrs = {}
    for el in seq:
        desc = attrs.get(el)
        if desc is None:
            attrs[el] = AttrDesc(el, True,
                                 el.is_discrete and el.values,
                                 el.is_continuous and el.number_of_decimals)
            continue
        if desc.template.is_discrete:
            sattr_values = set(desc.values)
            # don't use sets: keep the order
            missing_values = tuple(
                val for val in el.values if val not in sattr_values
            )
            if missing_values:
                attrs[el] = attrs[el]._replace(
                    original=False,
                    values=desc.values + missing_values)
        elif desc.template.is_continuous:
            if el.number_of_decimals > desc.number_of_decimals:
                attrs[el] = attrs[el]._replace(
                    original=False,
                    number_of_decimals=el.number_of_decimals)

    new_attrs = []
    for desc in attrs.values():
        attr = desc.template
        if desc.original:
            new_attr = attr
        elif desc.template.is_discrete:
            new_attr = attr.copy()
            for val in desc.values[len(attr.values):]:
                new_attr.add_value(val)
        else:
            assert desc.template.is_continuous
            new_attr = attr.copy(number_of_decimals=desc.number_of_decimals)
        new_attrs.append(new_attr)
    return new_attrs


class RelocatablePathsWidgetMixin(RecentPathsWidgetMixin):
    """
    Do not rearrange the file list as the RecentPathsWidgetMixin does.
    """

    def add_path(self, filename, reader):
        """Add (or move) a file name to the top of recent paths"""
        self._check_init()
        recent = RecentPath.create(filename, self._search_paths())
        if reader is not None:
            recent.file_format = reader.qualified_name()
        self.recent_paths.append(recent)

    def select_file(self, n):
        return NotImplementedError


class OWMultifile(widget.OWWidget, RelocatablePathsWidgetMixin):
    name = "Multifile"
    id = "orangecontrib.spectroscopy.widgets.files"
    icon = "icons/multifile.svg"
    description = "Read data from input files " \
                  "and send a data table to the output."
    priority = 10000
    replaces = ["orangecontrib.infrared.widgets.owfiles.OWFiles",
                "orangecontrib.infrared.widgets.owmultifile.OWMultifile",
                # next file: a file unintentionally added in one version
                "orangecontrib.spectroscopy.widgets.owmultifile_vesna.OWMultifile",
                ]
    keywords = ["file", "files", "multiple"]

    class Outputs:
        data = Output("Data", Table, doc="Concatenated input files.")

    want_main_area = False

    file_idx = []

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    recent_paths: List[RecentPath]
    variables: list

    sheet = Setting(None, schema_only=True)
    label = Setting("", schema_only=True)
    recent_paths = Setting([], schema_only=True)
    variables = ContextSetting([], schema_only=True)

    class Error(widget.OWWidget.Error):
        file_not_found = widget.Msg("File(s) not found.")
        missing_reader = widget.Msg("Missing reader(s).")
        read_error = widget.Msg("Read error(s).")

    domain_editor = SettingProvider(DomainEditor)

    def __init__(self):
        widget.OWWidget.__init__(self)
        RelocatablePathsWidgetMixin.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.sheets = []

        self.lb = gui.listBox(self.controlArea, self, "file_idx",
                              selectionMode=QListWidget.MultiSelection)
        self.default_foreground = None

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
        reload_button.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 7)

        self.sheet_box = gui.hBox(None, addToLayout=False, margin=0)
        self.sheet_index = 0
        self.sheet_combo = gui.comboBox(None, self, "sheet_index",
                                        callback=self.select_sheet)
        self.sheet_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_label = QLabel()
        self.sheet_label.setText('Sheet')
        self.sheet_label.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_box.layout().addWidget(self.sheet_label, Qt.AlignLeft)
        self.sheet_box.layout().addWidget(self.sheet_combo, Qt.AlignVCenter)
        layout.addWidget(self.sheet_box, 2, 1)
        self.sheet_box.hide()

        layout.addWidget(self.sheet_box, 0, 5)

        label_box = gui.hBox(None, addToLayout=False, margin=0)
        gui.lineEdit(label_box, self, "label", callback=self.set_label,
                     label="Label", orientation=Qt.Horizontal)
        layout.addWidget(label_box, 0, 6)

        layout.setColumnStretch(3, 2)

        box = gui.widgetBox(self.controlArea, "Columns (Double click to edit)")
        self.domain_editor = DomainEditor(self)
        self.editor_model = self.domain_editor.model()
        box.layout().addWidget(self.domain_editor)

        for rp in self.recent_paths:
            self.lb.addItem(rp.abspath)

        box = gui.hBox(self.controlArea)
        gui.rubber(box)

        gui.button(box, self, "Reset", callback=self.reset_domain_edit)
        self.apply_button = gui.button(
            box, self, "Apply", callback=self.apply_domain_edit)
        self.apply_button.setEnabled(False)
        self.apply_button.setFixedWidth(170)
        self.editor_model.dataChanged.connect(
            lambda: self.apply_button.setEnabled(True))

        self._update_sheet_combo()
        self.load_data()

    def set_label(self):
        self.load_data()

    def _select_active_sheet(self):
        if self.sheet:
            try:
                sheet_list = [s[0] for s in self.sheets]
                idx = sheet_list.index(self.sheet)
                self.sheet_combo.setCurrentIndex(idx)
            except ValueError:
                # Requested sheet does not exist in this file
                self.sheet = None
        else:
            self.sheet_combo.setCurrentIndex(0)

    def _update_sheet_combo(self):
        sheets = Counter()

        for rp in self.recent_paths:
            try:
                reader = _get_reader(rp)
                sheets.update(reader.sheets)
            except:  # noqa: E722
                pass

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
        ri = [i.row() for i in self.lb.selectedIndexes()]
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

    def browse_files(self):
        start_file = self.last_path() or os.path.expanduser("~/")

        readers = [f for f in FileFormat.formats if
                   getattr(f, 'read', None) and getattr(f, "EXTENSIONS", None)]
        filenames, reader, _ = \
            open_filename_dialog(start_file, None, readers,
                                 dialog=QFileDialog.getOpenFileNames)

        self.load_files(filenames, reader)

    def load_files(self, filenames, reader):
        if not filenames:
            return

        for f in filenames:
            self.add_path(f, reader)
            self.lb.addItem(f)

        self._update_sheet_combo()
        self.load_data()

    def load_data(self):
        self.closeContext()

        self.Error.file_not_found.clear()
        self.Error.missing_reader.clear()
        self.Error.read_error.clear()

        data_list = []
        fnok_list = []

        def show_error(li, msg):
            li.setForeground(Qt.red)
            li.setToolTip(msg)

        empty_domain = Domain(attributes=[])
        for i, rp in enumerate(self.recent_paths):
            fn = rp.abspath

            li = self.lb.item(i)
            li.setToolTip("")
            if self.default_foreground is None:
                self.default_foreground = li.foreground()
            li.setForeground(self.default_foreground)

            if not os.path.exists(fn):
                show_error(li, "File not found.")
                self.Error.file_not_found()
                continue

            try:
                reader = _get_reader(rp)
                assert reader is not None
            except Exception:  # pylint: disable=broad-except
                show_error(li, "Reader not found.")
                self.Error.missing_reader()
                continue

            try:
                if self.sheet in reader.sheets:
                    reader.select_sheet(self.sheet)
                if isinstance(reader, SpectralFileFormat):
                    xs, vals, additional = reader.read_spectra()
                    if additional is None:
                        additional = Table.from_domain(empty_domain, n_rows=len(vals))
                    additional.special_spectral_data = xs, vals
                    data_list.append(additional)
                else:
                    data_list.append(reader.read())
                fnok_list.append(fn)
            except Exception as ex:  # pylint: disable=broad-except
                show_error(li, "Read error:\n" + str(ex))
                self.Error.read_error()

        if not data_list or self.Error.file_not_found.is_shown() \
                or self.Error.missing_reader.is_shown() \
                or self.Error.read_error.is_shown():
            self.data = None
            self.domain_editor.set_domain(None)
        else:
            data = concatenate_data(data_list, fnok_list, self.label)
            self.data = data
            self.openContext(data.domain)

        self.apply_domain_edit()  # sends data

    def storeSpecificSettings(self):
        self.current_context.modified_variables = self.variables[:]

    def retrieveSpecificSettings(self):
        if hasattr(self.current_context, "modified_variables"):
            self.variables[:] = self.current_context.modified_variables

    def apply_domain_edit(self):
        if self.data is None:
            table = None
        else:
            domain, cols = self.domain_editor.get_domain(self.data.domain,
                                                         self.data)
            if not (domain.variables or domain.metas):
                table = None
            else:
                X, y, m = cols
                table = Table.from_numpy(domain, X, y, m, self.data.W)
                table.name = self.data.name
                table.ids = np.array(self.data.ids)
                table.attributes = getattr(self.data, 'attributes', {})

        self.Outputs.data.send(table)
        self.apply_button.setEnabled(False)

    def reset_domain_edit(self):
        self.domain_editor.reset_domain()
        self.apply_domain_edit()

    def send_report(self):
        def get_format_name(format):
            try:
                return format.DESCRIPTION
            except AttributeError:
                return format.__class__.__name__

        if self.data is None:
            self.report_paragraph("File", "No file.")
            return

        files = []

        for rp in self.recent_paths:
            format = _get_reader(rp)
            files.append([rp.abspath, get_format_name(format)])

        self.report_table("Files", table=files)

        self.report_data("Data", self.data)

    def workflowEnvChanged(self, key, value, oldvalue):
        """
        Function called when environment changes (e.g. while saving the scheme)
        It make sure that all environment connected values are modified
        (e.g. relative file paths are changed)
        """
        self.update_file_list(key, value, oldvalue)

    def update_file_list(self, key, value, oldvalue):
        if key == "basedir":
            self._relocate_recent_files()


def _get_reader(rp):
    if rp.file_format:
        reader_class = class_from_qualified_name(rp.file_format)
        return reader_class(rp.abspath)
    else:
        return FileFormat.get_reader(rp.abspath)


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=ungrouped-imports
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWMultifile).run()
