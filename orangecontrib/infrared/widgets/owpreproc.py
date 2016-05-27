import sys
import bisect
import contextlib
import warnings

import pkg_resources

import numpy
from PyQt4.QtGui import (
    QWidget, QButtonGroup, QGroupBox, QRadioButton, QSlider,
    QDoubleSpinBox, QComboBox, QSpinBox, QListView,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpacerItem, QSizePolicy,
    QCursor, QIcon,  QStandardItemModel, QStandardItem, QStyle,
    QStylePainter, QStyleOptionFrame, QPixmap,
    QApplication, QDrag
)
from PyQt4 import QtGui
from PyQt4.QtCore import (
    Qt, QObject, QEvent, QSize, QModelIndex, QMimeData, QTimer
)
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


import Orange.data
from Orange import preprocess
from Orange.statistics import distribution
from Orange.preprocess import Continuize, ProjectPCA, \
    ProjectCUR, Randomize as Random
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.overlay import OverlayWidget
from Orange.widgets.utils.sql import check_sql_input

from Orange.widgets.data.owpreprocess import (
    SequenceFlow, Controller, StandardItemModel,
    PreprocessAction, Description, icon_path, DiscretizeEditor,
    DescriptionRole, ParametersRole, BaseEditor, blocked
)

import numpy as np

from scipy.ndimage.filters import gaussian_filter1d

import copy

class GaussianSmoothing():

    def __init__(self, sd=10.):
        #super().__init__(variable)
        self.sd = sd

    def __call__(self, data):
        #FIXME this filted does not do automatic domain conversions!
        #FIXME we need need data about frequencies:
        #what if frequencies are not sampled on equal intervals
        x = np.arange(len(data.domain.attributes))
        newd = gaussian_filter1d(data.X, sigma=self.sd, mode="nearest")
        data = copy.copy(data)
        data.X = newd
        return data


class Feri():
# this is the actual computation
    def __init__(self, sd=10.):
        #super().__init__(variable)
        self.sd = sd

    def __call__(self, data):
        #FIXME this filted does not do automatic domain conversions!
        #FIXME we need need data about frequencies:
        #what if frequencies are not sampled on equal intervals
        x = np.arange(len(data.domain.attributes))
        newd = gaussian_filter1d(data.X, sigma=self.sd, mode="nearest")
        data = copy.copy(data)
        data.X = newd
        return data

class FeriEditor(BaseEditor):
    """
    This is the little window on the right with options
    """

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        self.__sd = 10.

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.__sdspin = sdspin = QDoubleSpinBox(  #QT elements for UI
           minimum=0.0, maximum=100.0, singleStep=0.5, value=self.__sd)
        layout.addWidget(sdspin)

        sdspin.valueChanged[float].connect(self.setSd)
        sdspin.editingFinished.connect(self.edited)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def setSd(self, sd):
        if self.__sd != sd:
            self.__sd = sd
            with blocked(self.__sdspin):
                self.__sdspin.setValue(sd)
            self.changed.emit()

    def sd(self):
        return self.__sd

    def intervals(self):
        return self.__nintervals

    def setParameters(self, params):
        self.setSd(params.get("sd", 10.))

    def parameters(self):
        return {"sd": self.__sd}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        sd = params.get("sd", 10.)
        return Feri(sd=sd)


class GaussianSmoothingEditor(BaseEditor):
    """
    Editor for preprocess.Discretize.
    """

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        self.__sd = 10.

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.__sdspin = sdspin = QDoubleSpinBox(
           minimum=0.0, maximum=100.0, singleStep=0.5, value=self.__sd)
        layout.addWidget(sdspin)

        sdspin.valueChanged[float].connect(self.setSd)
        sdspin.editingFinished.connect(self.edited)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def setSd(self, sd):
        if self.__sd != sd:
            self.__sd = sd
            with blocked(self.__sdspin):
                self.__sdspin.setValue(sd)
            self.changed.emit()

    def sd(self):
        return self.__sd

    def intervals(self):
        return self.__nintervals

    def setParameters(self, params):
        self.setSd(params.get("sd", 10.))

    def parameters(self):
        return {"sd": self.__sd}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        sd = params.get("sd", 10.)
        return GaussianSmoothing(sd=sd)

class SavitzkyGolayFiltering():
    """
    Apply a Savitzky-Golay[1] Filter to the data using SciPy Library.
    """
    def __init__(self, window=5,polyorder=2,deriv=0):
        #super().__init__(variable)
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def __call__(self, data):
        x = np.arange(len(data.domain.attributes))
        from scipy.signal import savgol_filter

        #savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        newd = savgol_filter(data.X, window_length=self.window, polyorder=self.polyorder, deriv=self.deriv, mode="nearest")

        data = copy.copy(data)
        data.X = newd
        return data


class SavitzkyGolayFilteringEditor(BaseEditor):
    """
    Editor for preprocess.savitzkygolayfiltering.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.window = 5
        self.polyorder = 2
        self.deriv = 0

        form = QFormLayout()

        self.wspin = QDoubleSpinBox(
            minimum=3, maximum=100, singleStep=2,
            value=self.polyorder)
        self.wspin.valueChanged[float].connect(self.setW)
        self.wspin.editingFinished.connect(self.edited)

        self.pspin = QDoubleSpinBox(
            minimum=2, maximum=self.window, singleStep=1,
            value=self.polyorder)
        self.pspin.valueChanged[float].connect(self.setP)
        self.pspin.editingFinished.connect(self.edited)

        self.dspin = QDoubleSpinBox(
            minimum=0, maximum=3, singleStep=1,
            value=self.deriv)
        self.dspin.valueChanged[float].connect(self.setD)
        self.dspin.editingFinished.connect(self.edited)

        form.addRow("Window", self.wspin)
        form.addRow("Polynomial Order", self.pspin)
        form.addRow("Derivative Order", self.dspin)
        self.layout().addLayout(form)

    def setParameters(self, params):
        self.setW(params.get("window", 5))
        self.setP(params.get("polyorder", 2))
        self.setD(params.get("deriv", 0))

    def parameters(self):
        return {"window": self.window, "polyorder": self.polyorder, "deriv": self.deriv}

    def setW(self, window):
        if self.window != window:
            self.window = window
            self.wspin.setValue(window)
            self.changed.emit()

    def setP(self, polyorder):
        if self.polyorder != polyorder:
            self.polyorder = polyorder
            self.pspin.setValue(polyorder)
            self.changed.emit()

    def setD(self, deriv):
        if self.deriv != deriv:
            self.deriv = deriv
            self.dspin.setValue(deriv)
            self.changed.emit()

    @staticmethod
    def createinstance(params):
        window = params.get("window", 5)
        polyorder = params.get("polyorder",2)
        deriv = params.get("deriv", 0)
        return SavitzkyGolayFiltering(window=window,polyorder=polyorder,deriv=deriv)



# what is registered
PREPROCESSORS = [
    PreprocessAction(
        "Gaussian smoothing", "orangecontrib.infrared.gaussian", "Smoothing",
        Description("Smooth spectra (gaussian)",
        icon_path("Discretize.svg")),
        GaussianSmoothingEditor
    ),
    PreprocessAction(
        "Savitzky-Golay Filter", "orangecontrib.infrared.savitzkygolay", "Smoothing",
        Description("Savitzky-Golay Filter (smoothing and differentiation)",
        icon_path("Discretize.svg")),
        SavitzkyGolayFilteringEditor
    ),
    PreprocessAction(
        "Feri", "orangecontrib.infrared.Feri", "test",
        Description("blablabla",
        icon_path("Discretize.svg")),
        FeriEditor
    ),
]



class OWPreprocess(widget.OWWidget):
    name = "Preprocess"
    description = "Construct a data preprocessing pipeline."
    icon = "icons/preprocess.svg"
    priority = 2105

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Preprocessor", preprocess.preprocess.Preprocess),
               ("Preprocessed Data", Orange.data.Table)]

    storedsettings = settings.Setting({})
    autocommit = settings.Setting(False)

    def __init__(self):
        super().__init__()

        self.data = None
        self._invalidated = False

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

        box = gui.widgetBox(self.controlArea, "Preprocessors")

        self.preprocessorsView = view = QListView(
            selectionMode=QListView.SingleSelection,
            dragEnabled=True,
            dragDropMode=QListView.DragOnly
        )
        view.setModel(self.preprocessors)
        view.activated.connect(self.__activated)

        box.layout().addWidget(view)

        ####
        self._qname2ppdef = {ppdef.qualname: ppdef for ppdef in PREPROCESSORS}

        # List of 'selected' preprocessors and their parameters.
        self.preprocessormodel = None

        self.flow_view = SequenceFlow()
        self.controler = Controller(self.flow_view, parent=self)

        self.overlay = OverlayWidget(self)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.overlay.setWidget(self.flow_view)
        self.overlay.setLayout(QVBoxLayout())
        self.overlay.layout().addWidget(
            QtGui.QLabel("Drag items from the list on the left",
                         wordWrap=True))

        self.scroll_area = QtGui.QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)
        self.mainArea.layout().addWidget(self.scroll_area)
        self.flow_view.installEventFilter(self)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.auto_commit(box, self, "autocommit", "Commit", box=False)

        self._initialize()

    def _initialize(self):
        for pp_def in PREPROCESSORS:
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

        try:
            model = self.load(self.storedsettings)
        except Exception:
            model = self.load({})

        self.set_model(model)

        if not model.rowCount():
            # enforce default width constraint if no preprocessors
            # are instantiated (if the model is not empty the constraints
            # will be triggered by LayoutRequest event on the `flow_view`)
            self.__update_size_constraint()

        self.apply()

    def load(self, saved):
        """Load a preprocessor list from a dict."""
        name = saved.get("name", "")
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
            self.preprocessormodel.dataChanged.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsInserted.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsRemoved.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsMoved.disconnect(self.__on_modelchanged)
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = ppmodel
        self.controler.setModel(ppmodel)
        if ppmodel is not None:
            self.preprocessormodel.dataChanged.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsInserted.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsRemoved.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsMoved.connect(self.__on_modelchanged)

        self.__update_overlay()

    def __update_overlay(self):
        if self.preprocessormodel is None or \
                self.preprocessormodel.rowCount() == 0:
            self.overlay.setWidget(self.flow_view)
            self.overlay.show()
        else:
            self.overlay.setWidget(None)
            self.overlay.hide()

    def __on_modelchanged(self):
        self.__update_overlay()
        self.commit()

    @check_sql_input
    def set_data(self, data=None):
        """Set the input data set."""
        self.data = data

    def handleNewSignals(self):
        self.apply()

    def __activated(self, index):
        item = self.preprocessors.itemFromIndex(index)
        action = item.data(DescriptionRole)
        item = QStandardItem()
        item.setData({}, ParametersRole)
        item.setData(action.description.title, Qt.DisplayRole)
        item.setData(action, DescriptionRole)
        self.preprocessormodel.appendRow([item])

    def buildpreproc(self):
        plist = []
        for i in range(self.preprocessormodel.rowCount()):
            item = self.preprocessormodel.item(i)
            desc = item.data(DescriptionRole)
            params = item.data(ParametersRole)

            if not isinstance(params, dict):
                params = {}

            create = desc.viewclass.createinstance
            plist.append(create(params))

        if len(plist) == 1:
            return plist[0]
        else:
            return preprocess.preprocess.PreprocessorList(plist)

    def apply(self):
        # Sync the model into storedsettings on every apply.
        self.storeSpecificSettings()
        preprocessor = self.buildpreproc()

        if self.data is not None:
            self.error(0)
            try:
                data = preprocessor(self.data)
            except ValueError as e:
                self.error(0, str(e))
                return
        else:
            data = None

        self.send("Preprocessor", preprocessor)
        self.send("Preprocessed Data", data)

    def commit(self):
        if not self._invalidated:
            self._invalidated = True
            QApplication.postEvent(self, QEvent(QEvent.User))

    def customEvent(self, event):
        if event.type() == QEvent.User and self._invalidated:
            self._invalidated = False
            self.apply()

    def eventFilter(self, receiver, event):
        if receiver is self.flow_view and event.type() == QEvent.LayoutRequest:
            QTimer.singleShot(0, self.__update_size_constraint)

        return super().eventFilter(receiver, event)

    def storeSpecificSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        super().storeSpecificSettings()

    def saveSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        super().saveSettings()

    def onDeleteWidget(self):
        self.data = None
        self.set_model(None)
        super().onDeleteWidget()

    @Slot()
    def __update_size_constraint(self):
        # Update minimum width constraint on the scroll area containing
        # the 'instantiated' preprocessor list (to avoid the horizontal
        # scroll bar).
        sh = self.flow_view.minimumSizeHint()
        scroll_width = self.scroll_area.verticalScrollBar().width()
        self.scroll_area.setMinimumWidth(
            min(max(sh.width() + scroll_width + 2, self.controlArea.width()),
                520))

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(sh.width(), 500))


def test_main(argv=sys.argv):
    argv = list(argv)
    app = QtGui.QApplication(argv)

    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    w = OWPreprocess()
    w.set_data(Orange.data.Table(filename))
    w.show()
    w.raise_()
    r = app.exec_()
    w.set_data(None)
    w.saveSettings()
    w.onDeleteWidget()
    return r

if __name__ == "__main__":
    sys.exit(test_main())
