from collections.abc import Iterable
import random
import time
import traceback
import sys

import numpy as np

import Orange.data
from Orange import preprocess
from Orange.widgets import gui, settings
from Orange.widgets.settings import SettingsHandler
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.data.utils.preprocess import SequenceFlow, Controller, \
    StandardItemModel
from Orange.widgets.data.owpreprocess import (
    PreprocessAction, Description, icon_path, DescriptionRole, ParametersRole
)
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.overlay import OverlayWidget
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin, ConcurrentMixin

from AnyQt.QtCore import (
    Qt, QEvent, QSize, QMimeData, QTimer
)
from AnyQt.QtWidgets import (
    QWidget, QListView, QVBoxLayout, QSizePolicy, QStyle,
    QPushButton, QLabel, QMenu, QAction, QScrollArea, QGridLayout,
    QToolButton, QSplitter
)
from AnyQt.QtGui import (
    QIcon, QStandardItemModel, QStandardItem,
    QKeySequence, QFont, QColor
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot, QObject

from orangecontrib.spectroscopy.preprocess.utils import PreprocessException
from orangecontrib.spectroscopy.widgets.owspectra import CurvePlot, NoSuchCurve
from orangecontrib.spectroscopy.widgets.preprocessors.misc import SavitzkyGolayFilteringEditor
from orangecontrib.spectroscopy.widgets.preprocessors.utils import REFERENCE_DATA_PARAM
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors


BREWER_PALETTE8 = [(127, 201, 127), (190, 174, 212), (253, 192, 134), (255, 255, 153),
                   (56, 108, 176), (240, 2, 127), (191, 91, 23), (102, 102, 102)]
PREVIEW_COLORS = [QColor(*a).name() for a in BREWER_PALETTE8]


class ViewController(Controller):

    def createWidgetFor(self, index):
        w = super().createWidgetFor(index)
        w.parent_widget = self.parent()
        return w

    # ensure that view on the right
    # and the model are sychronized when on_modelchanged is called

    def _dataChanged(self, topleft, bottomright):
        super()._dataChanged(topleft, bottomright)
        self.parent().on_modelchanged()

    def _rowsInserted(self, parent, start, end):
        super()._rowsInserted(parent, start, end)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def _rowsRemoved(self, parent, start, end):
        super()._rowsRemoved(parent, start, end)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def _widgetMoved(self, from_, to):
        super()._widgetMoved(from_, to)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def setModel(self, model):
        super().setModel(model)
        self.view.reset_preview_colors()


class FocusFrame(SequenceFlow.Frame):
    preview_changed = Signal()

    def __init__(self, parent=None, **kwargs):
        self.title_label = None
        super().__init__(parent=parent, **kwargs)
        self.preview = False
        self.color = "lightblue"
        self._build_tw()
        self.setTitleBarWidget(self.tw)

    def _build_tw(self):
        self.tw = tw = QWidget(self)
        tl = QGridLayout(tw)
        self.title_label = QLabel(self._title, tw)
        self.title_label.setMinimumWidth(100)
        tl.addWidget(self.title_label, 0, 1)
        close_button = QToolButton(self)
        ca = QAction("close", self, triggered=self.closeRequested,
                     icon=QIcon(self.style().standardPixmap(QStyle.SP_DockWidgetCloseButton)))
        close_button.setDefaultAction(ca)
        self.preview_button = QToolButton(self)
        pa = QAction("preview", self, triggered=self.toggle_preview, checkable=True,
                     icon=QIcon(self.style().standardPixmap(QStyle.SP_MediaPlay)),
                     shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_P))
        pa.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(pa)
        self.preview_button.setDefaultAction(pa)
        self.set_preview(self.preview)
        tl.addWidget(close_button, 0, 0)
        tl.addWidget(self.preview_button, 0, 2)
        tl.setColumnStretch(1, 1)
        tl.setSpacing(2)
        tl.setContentsMargins(0, 0, 0, 0)
        tw.setLayout(tl)

    def set_preview(self, p):
        self.preview = p
        self.update_status()

    def set_color(self, c):
        self.color = c
        self.update_status()

    def update_status(self):
        self.preview_button.setChecked(self.preview)
        self.tw.setStyleSheet("background:" + self.color + ";" if self.preview else "")

    def toggle_preview(self):
        self.set_preview(not self.preview)
        self.preview_changed.emit()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        try: #active selection on preview
            self.widget().activateOptions()
        except AttributeError:
            pass

    def setTitle(self, title):
        self._title = title
        super().setTitle(title)
        if self.title_label:
            self.title_label.setText(title)


class SequenceFlow(SequenceFlow):
    """
    FIXME Ugly hack: using the same name for access to private variables!
    """
    def __init__(self, *args, preview_callback=None, multiple_previews=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_callback = preview_callback
        self.multiple_previews = multiple_previews
        self.preview_colors = multiple_previews

    def preview_n(self):
        """How many preprocessors to apply for the preview?"""
        ppos = [i for i, item in enumerate(self.layout_iter(self.__flowlayout)) if item.widget().preview]
        # if any, show the chosen preview
        if not self.multiple_previews:
            return ppos[-1] if ppos else None
        else:
            return ppos

    def set_preview_n(self, n):
        """Set the preview position"""
        n = set(n if isinstance(n, Iterable) else [n])
        for i, item in enumerate(self.layout_iter(self.__flowlayout)):
            f = item.widget()
            f.set_preview(i in n)

    def preview_changed(self):
        if not self.multiple_previews:  # disable other previews
            sender = self.sender()
            for item in self.layout_iter(self.__flowlayout):
                f = item.widget()
                if sender != f:
                    f.set_preview(False)

        self.preview_callback()

    def insertWidget(self, index, widget, title):
        """ Mostly copied to get different kind of frame """
        frame = FocusFrame(widget=widget, title=title) #changed
        frame.closeRequested.connect(self.__closeRequested)
        frame.preview_changed.connect(self.preview_changed)

        layout = self.__flowlayout

        frames = [item.widget() for item in self.layout_iter(layout)
                  if item.widget()]

        if 0 < index < len(frames):
            # find the layout index of a widget occupying the current
            # index'th slot.
            insert_index = layout.indexOf(frames[index])
        elif index == 0:
            insert_index = 0
        elif index < 0 or index >= len(frames):
            insert_index = layout.count()
        else:
            raise AssertionError

        layout.insertWidget(insert_index, frame)
        frame.installEventFilter(self)

    def __closeRequested(self):
        self.sender().widget().parent_widget.curveplot.clear_markings()
        super().__closeRequested()

    def minimumSizeHint(self):
        """ Add space below so that dragging to bottom works """
        psh = super().minimumSizeHint()
        return QSize(psh.width(), psh.height() + 100)

    def reset_preview_colors(self):
        if self.preview_colors:
            for i, item in enumerate(self.layout_iter(self.__flowlayout)):
                item = item.widget()
                item.set_color(PREVIEW_COLORS[i % len(PREVIEW_COLORS)])

    def preview_color(self, i):
        """ Return preview color of a specific widget. """
        w = self.__flowlayout.itemAt(i).widget()
        return w.color


class TimeoutLabel(QLabel):
    """ A label that disappears out after two seconds. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        self.hide()

    def setText(self, t):
        super().setText(t)
        self.show()
        self.timer.start(2000)

    def _timeout(self):
        self.hide()


def transfer_highlight(from_: CurvePlot, to: CurvePlot):
    """Highlight the curve that is highlighted on "from_" also on "to"."""
    if from_.data is None or to.data is None:
        return
    highlight = None
    from_index = from_.highlighted_index_in_data()
    if from_index is not None:
        index_with_same_id = np.flatnonzero(to.data.ids == from_.data.ids[from_index])
        if len(index_with_same_id):
            highlight = index_with_same_id[0]
    try:
        to.highlight_index_in_data(highlight, emit=False)  # do not emit to avoid recursion
    except NoSuchCurve:
        pass


class PrepareSavingSettingsHandler(SettingsHandler):
    """Calls storeSpecificSettings, which is currently not called from non-context handlers."""

    def pack_data(self, widget):
        widget.storeSpecificSettings()
        return super().pack_data(widget)


def prepare_params(params, reference):
    if not isinstance(params, dict):
        params = {}
    # add optional reference data
    params[REFERENCE_DATA_PARAM] = reference
    return params


def create_preprocessor(item, reference):
    desc = item.data(DescriptionRole)
    params = item.data(ParametersRole)
    params = prepare_params(params, reference)
    create = desc.viewclass.createinstance
    return create(params)


class InterruptException(Exception):
    pass


class PreviewRunner(QObject, ConcurrentMixin):

    preview_updated = Signal()

    def __init__(self, master):
        super().__init__(parent=master)
        ConcurrentMixin.__init__(self)
        self.master = master

        # save state in function
        self.last_text = ""
        self.show_info_anyway = None
        self.preview_pos = None
        self.preview_data = None
        self.after_data = None
        self.last_partial = None

    def on_partial_result(self, result):
        i, data, reference = result
        self.last_partial = i
        if self.preview_pos == i:
            self.preview_data = data
        if self.preview_pos == i-1:
            self.after_data = data
        widgets = self.master.flow_view.widgets()
        if i < len(widgets):
            widgets[i].set_reference_data(reference)
            widgets[i].set_preview_data(data)

    def on_exception(self, ex: Exception):
        if isinstance(ex, InterruptException):
            return

        self.master.curveplot.set_data(self.preview_data)
        self.master.curveplot_after.set_data(self.after_data)

        if isinstance(ex, PreprocessException):
            self.master.Error.preview(ex.message())
            widgets = self.master.flow_view.widgets()
            if self.last_partial is not None and self.last_partial < len(widgets):
                w = widgets[self.last_partial]
                if getattr(w, "Error", None):  # only BaseEditorOrange supports errors
                    w.Error.exception(ex.message())
            self.preview_updated.emit()
        else:
            raise ex

    def on_done(self, result):
        orig_data, after_data = result
        final_preview = self.preview_pos is None
        if final_preview:
            self.preview_data = orig_data
            self.after_data = after_data

        if self.preview_data is None:  # happens in OWIntegrate
            self.preview_data = orig_data

        self.master.curveplot.set_data(self.preview_data)
        self.master.curveplot_after.set_data(self.after_data)

        self.show_image_info(final_preview)

        self.preview_updated.emit()

    def show_image_info(self, final_preview):
        master = self.master

        if not master.preview_on_image:
            master.final_preview_toggle = final_preview
            if final_preview:
                new_text = None
            else:
                item = master.preprocessormodel.item(self.preview_pos)
                new_text = item.data(DescriptionRole).description.title

            if new_text != self.last_text or self.show_info_anyway:
                if new_text is None:
                    master.curveplot_info.setText('Original data')
                    master.curveplot_after_info.setText('Preprocessed data')
                else:
                    master.curveplot_info.setText('Input to "' + new_text + '"')
                    master.curveplot_after_info.setText('Output of "' + new_text + '"')

            self.last_text = new_text

    def show_preview(self, show_info_anyway=False):
        """ Shows preview and also passes preview data to the widgets """
        master = self.master
        self.preview_pos = master.flow_view.preview_n()
        self.last_partial = None
        self.show_info_anyway = show_info_anyway
        self.preview_data = None
        self.after_data = None
        pp_def = [master.preprocessormodel.item(i)
                  for i in range(master.preprocessormodel.rowCount())]
        if master.data is not None:
            data = master.sample_data(master.data)
            self.start(self.run_preview, data, master.reference_data,
                       pp_def, master.process_reference)
        else:
            master.curveplot.set_data(None)
            master.curveplot_after.set_data(None)

    @staticmethod
    def run_preview(data: Orange.data.Table, reference: Orange.data.Table,
                    pp_def, process_reference, state: TaskState):

        def progress_interrupt(i: float):
            if state.is_interruption_requested():
                raise InterruptException

        n = len(pp_def)
        orig_data = data
        for i in range(n):
            progress_interrupt(0)
            state.set_partial_result((i, data, reference))
            item = pp_def[i]
            pp = create_preprocessor(item, reference)
            data = pp(data)
            progress_interrupt(0)
            if process_reference and reference is not None and i != n - 1:
                reference = pp(reference)
        progress_interrupt(0)
        state.set_partial_result((n, data, None))
        return orig_data, data


class SpectraPreviews:

    curveplot = settings.SettingProvider(CurvePlot)
    curveplot_after = settings.SettingProvider(CurvePlot)

    def __init__(self):
        self.curveplot = CurvePlot(self)
        self.curveplot_after = CurvePlot(self)
        self.curveplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        self.curveplot_after.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden

        self.curveplot.highlight_changed.connect(
            lambda: transfer_highlight(self.curveplot, self.curveplot_after))
        self.curveplot_after.highlight_changed.connect(
            lambda: transfer_highlight(self.curveplot_after, self.curveplot))

    def shutdown(self):
        self.curveplot.shutdown()
        self.curveplot_after.shutdown()


class GeneralPreprocess(OWWidget, ConcurrentWidgetMixin,
                        openclass=True):

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        preprocessed_data = Output("Preprocessed Data", Orange.data.Table, default=True)
        preprocessor = Output("Preprocessor", preprocess.preprocess.Preprocess)

    settingsHandler = PrepareSavingSettingsHandler()

    storedsettings = settings.Setting({}, schema_only=True)
    autocommit = settings.Setting(False)
    preview_curves = settings.Setting(3)
    preview_n = settings.Setting(None, schema_only=True)

    # compatibility for old workflows when reference was not processed
    process_reference = settings.Setting(True, schema_only=True)

    # draw preview on top of current image
    preview_on_image = False

    editor_registry = None

    _max_preview_spectra = 10

    class Error(OWWidget.Error):
        loading = Msg("Error when loading preprocessors. {}")
        applying = Msg("Preprocessing error. {}")
        preview = Msg("Preview error. {}")
        preprocessor = Msg("Preprocessor error: see the widget for details.")

    class Warning(OWWidget.Warning):
        reference_compat = Msg("Reference is not processed for compatibility with the loaded "
                               "workflow. New instances of this widget will also process "
                               "the reference input.")
        preprocessor = Msg("Preprocessor warning: see the widget for details.")

    def _build_PREPROCESSORS(self):
        self.PREPROCESSORS = []
        qualnames = set()
        for editor in self.editor_registry.sorted():
            assert editor.qualname is not None
            assert editor.qualname not in qualnames
            pa = PreprocessAction(editor.name,
                                  editor.qualname,
                                  editor.name,
                                  Description(editor.name,
                                              editor.icon if editor.icon else
                                              icon_path("Discretize.svg")),
                                  editor)
            qualnames.add(editor.qualname)
            self.PREPROCESSORS.append(pa)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        if self.editor_registry is not None:
            self._build_PREPROCESSORS()

        self.preview_runner = PreviewRunner(self)

        self.data = None
        self.reference_data = None

        # List of available preprocessors (DescriptionRole : Description)
        self.preprocessors = QStandardItemModel()

        def mimeData(indexlist):
            assert len(indexlist) == 1
            index = indexlist[0]
            qname = index.data(DescriptionRole).qualname
            m = QMimeData()
            m.setData("application/x-qwidget-ref", qname)
            return m
        # TODO: Fix this (subclass even if just to pass a function
        # for mimeData delegate)
        self.preprocessors.mimeData = mimeData

        self.button = QPushButton(self.BUTTON_ADD_LABEL, self)
        self.controlArea.layout().addWidget(self.button)
        self.preprocessor_menu = QMenu(self)
        self.button.setMenu(self.preprocessor_menu)
        self.button.setAutoDefault(False)

        self.preprocessorsView = QListView(
            selectionMode=QListView.SingleSelection,
            dragEnabled=True,
            dragDropMode=QListView.DragOnly
        )

        self._qname2ppdef = {ppdef.qualname: ppdef for ppdef in self.PREPROCESSORS}

        # List of 'selected' preprocessors and their parameters.
        self.preprocessormodel = None

        self.flow_view = SequenceFlow(preview_callback=self._show_preview_info,
                                      multiple_previews=self.preview_on_image)
        self.controler = ViewController(self.flow_view, parent=self)

        self.scroll_area = QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)

        self.flow_view.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.scroll_area.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        splitter.addWidget(self.curveplot)
        splitter.addWidget(self.curveplot_after)
        self.mainArea.layout().addWidget(splitter)

        def overlay(widget):
            o = OverlayWidget(self)
            o.setAttribute(Qt.WA_TransparentForMouseEvents)
            o.setWidget(widget)
            o.setLayout(QVBoxLayout(o))
            l = TimeoutLabel("", parent=o, wordWrap=True)
            l.setAlignment(Qt.AlignCenter)
            font = QFont()
            font.setPointSize(20)
            l.setFont(font)
            l.setStyleSheet("color: lightblue")
            o.layout().addWidget(l)
            return l

        self.curveplot_info = overlay(self.curveplot)
        self.curveplot_after_info = overlay(self.curveplot_after)

        if not self.preview_on_image:
            self.curveplot_after.show()
        else:
            self.curveplot_after.hide()

        self.controlArea.layout().addWidget(self.scroll_area)
        self.mainArea.layout().addWidget(splitter)

        self.flow_view.installEventFilter(self)

        self.preview_settings_box = box = gui.widgetBox(self.controlArea, "Preview")
        self.final_preview_toggle = False
        if not self.preview_on_image:
            self.final_preview = gui.button(
                box, self, "Final preview", self.flow_view.preview_changed,
                toggleButton=True, value="final_preview_toggle", autoDefault=False)
        gui.spin(box, self, "preview_curves", 1, self._max_preview_spectra, label="Show spectra",
                 callback=self._update_preview_number)

        self.output_box = gui.widgetBox(self.controlArea, "Output")
        b = gui.auto_commit(self.output_box, self, "autocommit", "Commit", box=False)
        b.button.setAutoDefault(False)

        self._initialize()

    def _show_preview_info(self):
        self.show_preview(show_info_anyway=True)

    def _update_preview_number(self):
        self.show_preview(show_info_anyway=False)

    def sample_data(self, data):
        if data is not None and len(data) > self.preview_curves:
            sampled_indices = random.Random(0).sample(range(len(data)), self.preview_curves)
            return data[sampled_indices]
        else:
            return self.data

    def _reference_compat_warning(self):
        self.Warning.reference_compat.clear()
        if not self.process_reference and self.reference_data is not None:
            self.Warning.reference_compat()

    def show_preview(self, show_info_anyway=False):
        """ Shows preview and also passes preview data to the widgets """
        self._reference_compat_warning()
        self.Warning.preprocessor.clear()
        self.Error.preprocessor.clear()
        self.Error.preview.clear()
        for w in  self.flow_view.widgets():
            if getattr(w, "Error", None):  # only BaseEditorOrange supports errors
                w.Error.exception.clear()

        return self.preview_runner.show_preview(show_info_anyway=show_info_anyway)

    def _create_preprocessor_action(self, pp_def):
        description = pp_def.description
        if description.icon:
            icon = QIcon(description.icon)
        else:
            icon = QIcon()
        action = QAction(
            description.title, self, triggered=lambda x, p=pp_def: self.add_preprocessor(p)
        )
        action.setToolTip(description.summary or "")
        action.setIcon(icon)
        return action

    def _init_menu_flat(self):
        for pa in self.PREPROCESSORS:
            action = self._create_preprocessor_action(pa)
            self.preprocessor_menu.addAction(action)

    def _init_menu_registry(self):
        for category in self.editor_registry.categories():
            category_menu = self.preprocessor_menu \
                if category == "" \
                else self.preprocessor_menu.addMenu(category)
            for editor in self.editor_registry.sorted(category):
                pa = self._qname2ppdef[editor.qualname]
                action = self._create_preprocessor_action(pa)
                category_menu.addAction(action)

    def _initialize(self):
        for pp_def in self.PREPROCESSORS:
            description = pp_def.description
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item = QStandardItem(icon, description.title)
            item.setToolTip(description.summary or "")
            item.setData(pp_def, DescriptionRole)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                          Qt.ItemIsDragEnabled)
            self.preprocessors.appendRow([item])

        if self.editor_registry is None:
            self._init_menu_flat()
        else:
            self._init_menu_registry()


        try:
            model = self.load(self.storedsettings)
        except Exception as ex:
            # open Orange's report window so that users can report the problem
            sys.excepthook(type(ex), ex, ex.__traceback__)
            # show an error with the same content
            self.Error.loading(traceback.format_exc())
            # allow the widget to work
            model = self.load({})

        self.set_model(model)
        self.flow_view.set_preview_n(self.preview_n)

        if not model.rowCount():
            # enforce default width constraint if no preprocessors
            # are instantiated (if the model is not empty the constraints
            # will be triggered by LayoutRequest event on the `flow_view`)
            self.__update_size_constraint()

        # output the preprocessor even if there is no input data
        self.commit.now()

    def load(self, saved):
        """Load a preprocessor list from a dict."""
        name = saved.get("name", "")  # noqa: F841
        preprocessors = saved.get("preprocessors", [])
        model = StandardItemModel()

        def dropMimeData(data, action, row, column, parent):
            if data.hasFormat("application/x-qwidget-ref") and \
                    action == Qt.CopyAction:
                qname = bytes(data.data("application/x-qwidget-ref")).decode()

                ppdef = self._qname2ppdef[qname]
                item = QStandardItem(ppdef.description.title)
                item.setData({}, ParametersRole)
                item.setData(ppdef.description.title, Qt.DisplayRole)
                item.setData(ppdef, DescriptionRole)
                self.preprocessormodel.insertRow(row, [item])
                return True
            else:
                return False

        model.dropMimeData = dropMimeData

        for qualname, params in preprocessors:
            pp_def = self._qname2ppdef[qualname]
            description = pp_def.description
            item = QStandardItem(description.title)
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item.setIcon(icon)
            item.setToolTip(description.summary)
            item.setData(pp_def, DescriptionRole)
            item.setData(params, ParametersRole)

            model.appendRow(item)
        return model

    def save(self, model):
        """Save the preprocessor list to a dict."""
        d = {"name": ""}
        preprocessors = []
        for i in range(model.rowCount()):
            item = model.item(i)
            pp_def = item.data(DescriptionRole)
            params = item.data(ParametersRole)
            preprocessors.append((pp_def.qualname, params))

        d["preprocessors"] = preprocessors
        return d

    def set_model(self, ppmodel):
        if self.preprocessormodel:
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = ppmodel
        self.controler.setModel(ppmodel)

    def on_modelchanged(self):
        self.show_preview()
        self.commit.deferred()

    @Inputs.data
    @check_sql_input
    def set_data(self, data=None):
        """Set the input data set."""
        self.data = data

    def handleNewSignals(self):
        self.show_preview(True)
        self.commit.now()

    def add_preprocessor(self, action):
        self.Error.loading.clear()
        item = QStandardItem()
        item.setData({}, ParametersRole)
        item.setData(action.description.title, Qt.DisplayRole)
        item.setData(action, DescriptionRole)
        self.preprocessormodel.appendRow([item])

    def create_outputs(self):
        raise NotImplementedError()

    @gui.deferred
    def commit(self):
        self.show_preview()
        self.Error.applying.clear()
        self.create_outputs()

    def on_partial_result(self, _):
        pass

    def on_done(self, results):
        data, preprocessor = results
        self.Outputs.preprocessor.send(preprocessor)
        self.Outputs.preprocessed_data.send(data)

    def on_exception(self, ex):
        if isinstance(ex, InterruptException):
            return  # do not change outputs if interrupted

        if isinstance(ex, PreprocessException):
            self.Error.applying(ex.message())
        else:
            raise ex

        self.Outputs.preprocessor.send(None)
        self.Outputs.preprocessed_data.send(None)

    def eventFilter(self, receiver, event):
        if receiver is self.flow_view and event.type() == QEvent.LayoutRequest:
            QTimer.singleShot(0, self.__update_size_constraint)

        return super().eventFilter(receiver, event)

    def storeSpecificSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        self.preview_n = self.flow_view.preview_n()
        super().storeSpecificSettings()

    def onDeleteWidget(self):
        self.shutdown()
        self.preview_runner.shutdown()
        self.data = None
        self.set_model(None)
        super().onDeleteWidget()

    @Slot()
    def __update_size_constraint(self):
        # Update minimum width constraint on the scroll area containing
        # the 'instantiated' preprocessor list (to avoid the horizontal
        # scroll bar).
        sh = self.flow_view.minimumSizeHint()
        self.scroll_area.setMinimumWidth(max(sh.width() + 50, 200))

        # call updateGeometry on the controlArea's scroll area, to expand/squeeze
        # it with widget-base 4.12.0 if a big preprocessor widget is added
        parent = self.controlArea
        while parent is not None and not isinstance(parent, gui.VerticalScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            parent.updateGeometry()

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(sh.width(), 500))

    @classmethod
    def migrate_preprocessor(cls, preprocessor, version):
        """ Migrate a preprocessor. A preprocessor should migrate into a list of preprocessors. """
        name, settings = preprocessor
        return [((name, settings), version)]

    @classmethod
    def migrate_preprocessor_list(cls, preprocessors):
        pl = []
        for p, v in preprocessors:
            tl = cls.migrate_preprocessor(p, v)
            if tl != [(p, v)]:  # if changed, try another migration
                tl = cls.migrate_preprocessor_list(tl)
            pl.extend(tl)
        return pl

    @classmethod
    def migrate_preprocessors(cls, preprocessors, version):
        input = list(zip(preprocessors, [version]*len(preprocessors)))  # noqa: B905
        migrated = cls.migrate_preprocessor_list(input)
        return [p[0] for p in migrated], cls.settings_version

    @classmethod
    def migrate_settings(cls, settings_, version):
        # migrate individual preprocessors
        if "storedsettings" in settings_ and "preprocessors" in settings_["storedsettings"]:
            settings_["storedsettings"]["preprocessors"], _ = \
                cls.migrate_preprocessors(settings_["storedsettings"]["preprocessors"], version)


class SpectralPreprocess(GeneralPreprocess,
                         SpectraPreviews, openclass=True):

    def __init__(self):
        SpectraPreviews.__init__(self)
        super().__init__()

    def onDeleteWidget(self):
        super().onDeleteWidget()
        SpectraPreviews.shutdown(self)


class SpectralPreprocessReference(SpectralPreprocess, openclass=True):

    class Inputs(SpectralPreprocess.Inputs):
        reference = Input("Reference", Orange.data.Table)

    @Inputs.reference
    def set_reference(self, reference):
        self.reference_data = reference


class OWPreprocess(SpectralPreprocessReference):

    name = "Preprocess Spectra"
    description = "Construct a data preprocessing pipeline."
    icon = "icons/preprocess.svg"
    priority = 1000
    replaces = ["orangecontrib.infrared.widgets.owpreproc.OWPreprocess",
                "orangecontrib.infrared.widgets.owpreprocess.OWPreprocess"]

    settings_version = 9

    BUTTON_ADD_LABEL = "Add preprocessor..."
    editor_registry = preprocess_editors

    _max_preview_spectra = 100

    preview_curves = settings.Setting(25)

    # draw preview on top of current image
    preview_on_image = False

    def create_outputs(self):
        self._reference_compat_warning()
        pp_def = [self.preprocessormodel.item(i) for i in range(self.preprocessormodel.rowCount())]
        self.start(self.run_task, self.data, self.reference_data, pp_def, self.process_reference)

    @staticmethod
    def run_task(data: Orange.data.Table, reference: Orange.data.Table,
                 pp_def, process_reference, state: TaskState):

        def progress_interrupt(i: float):
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for _i in range(10):
            time.sleep(0.010)
            progress_interrupt(0)

        n = len(pp_def)
        plist = []
        for i in range(n):
            progress_interrupt(i/n*100)
            item = pp_def[i]
            pp = create_preprocessor(item, reference)
            plist.append(pp)
            if data is not None:
                data = pp(data)
            progress_interrupt((i/n + 0.5/n)*100)
            if process_reference and reference is not None and i != n - 1:
                reference = pp(reference)
        # if there are no preprocessors, return None instead of an empty list
        preprocessor = preprocess.preprocess.PreprocessorList(plist) if plist else None
        return data, preprocessor

    @classmethod
    def migrate_preprocessor(cls, preprocessor, version):
        name, settings = preprocessor
        settings = settings.copy()
        if name == "orangecontrib.infrared.rubberband" and version < 2:
            name = "orangecontrib.infrared.baseline"
            settings["baseline_type"] = 1
            version = 2
        if name == "orangecontrib.infrared.savitzkygolay" and version < 4:
            name = "orangecontrib.spectroscopy.savitzkygolay"
            # make window, polyorder, deriv valid, even if they were saved differently
            SGE = SavitzkyGolayFilteringEditor
            # some old versions saved these as floats
            window = int(settings.get("window", SGE.DEFAULT_WINDOW))
            polyorder = int(settings.get("polyorder", SGE.DEFAULT_POLYORDER))
            deriv = int(settings.get("deriv", SGE.DEFAULT_DERIV))
            if window % 2 == 0:
                window = window + 1
            window = max(min(window, SGE.MAX_WINDOW), SGE.MIN_WINDOW)
            polyorder = max(min(polyorder, window - 1), 0)
            deriv = max(min(SGE.MAX_DERIV, deriv, polyorder), 0)
            settings["window"] = window
            settings["polyorder"] = polyorder
            settings["deriv"] = deriv
            version = 4
        if name == "orangecontrib.infrared.absorbance" and version < 5:
            name = "orangecontrib.spectroscopy.transforms"
            settings["from_type"] = 1
            settings["to_type"] = 0
            version = 5
        if name == "orangecontrib.infrared.transmittance" and version < 5:
            name = "orangecontrib.spectroscopy.transforms"
            settings["from_type"] = 0
            settings["to_type"] = 1
            version = 5
        if name in ["orangecontrib.spectroscopy.preprocess.emsc",
                    "orangecontrib.spectroscopy.preprocess.me_emsc.me_emsc"] \
                and version < 7:
            ranges = settings.get("ranges", [])
            new_ranges = [[l, r, w, 0.0] for l, r, w in ranges]
            settings["ranges"] = new_ranges
            version = 7
        if name == "orangecontrib.infrared.cut":
            name = "orangecontrib.spectroscopy.cut"
            settings["inverse"] = False
        if name == "orangecontrib.infrared.cutinverse":
            name = "orangecontrib.spectroscopy.cut"
            settings["inverse"] = True
        if name == "orangecontrib.infrared.curveshift":
            name = "orangecontrib.spectroscopy.shiftandscale"
            settings["offset"] = settings.get("amount", 0.)
            settings["scale"] = 1.
        return [((name, settings), version)]

    @classmethod
    def migrate_settings(cls, settings_, version):
        # For backwards compatibility, set process_reference=False
        # but only if there were multiple preprocessors
        if "process_reference" not in settings_:
            settings_["process_reference"] = not(
                version <= 5
                and "storedsettings" in settings_
                and "preprocessors" in settings_["storedsettings"]
                and len(settings_["storedsettings"]["preprocessors"]) > 1
            )

        super().migrate_settings(settings_, version)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    data = Orange.data.Table("collagen.csv")
    WidgetPreview(OWPreprocess).run(set_data=data[:30], set_reference=data[10:11])
