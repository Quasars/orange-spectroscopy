import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QVBoxLayout, QFormLayout, QLabel

from Orange.widgets import gui
from orangecontrib.spectroscopy.preprocess import BaselineShift
from orangecontrib.spectroscopy.widgets.gui import XPosLineEdit
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange


class BaselineShiftEditor(BaseEditorOrange):
    """
    Baseline shift correction editor with dynamically displayed labeled input fields.
    """
    name = "Baseline Shift Correction"
    qualname = "orangecontrib.infrared.baseline_shift"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())

        form = QFormLayout()

        # Method selection dropdown
        self.method = 0  # Default: minimum
        self.method_cb = gui.comboBox(
            None, self, "method",
            items=["Minimum", "Specific wavenumber", "Minimum in range", "Maximum in range", "Average in range"],
            callback=self._adapt_ui
        )
        form.addRow(QLabel("Shift Method:"), self.method_cb)

        # --- Specific wavenumber input (method 1) ---
        self.wn_label = QLabel("Wavenumber:")
        self.wn_edit = XPosLineEdit()
        self.wn_edit.edited.connect(self.edited)
        form.addRow(self.wn_label, self.wn_edit)

        # --- Range selection inputs (methods 2, 3, 4) ---
        self.range_min_label = QLabel("From:")
        self.range_min_edit = XPosLineEdit()
        self.range_max_label = QLabel("To:")
        self.range_max_edit = XPosLineEdit()
        self.range_min_edit.edited.connect(self.edited)
        self.range_max_edit.edited.connect(self.edited)
        form.addRow(self.range_min_label, self.range_min_edit)
        form.addRow(self.range_max_label, self.range_max_edit)

        self.controlArea.layout().addLayout(form)
        self._adapt_ui()  # Initialize UI

    def setParameters(self, params):
        """Restore parameters from saved state."""
        self.method = params.get("method", 0)
        if "wn" in params:
            self.wn_edit.position = params["wn"]
        if "range_min" in params:
            self.range_min_edit.position = params["range_min"]
        if "range_max" in params:
            self.range_max_edit.position = params["range_max"]
        self._adapt_ui()

    def _adapt_ui(self):
        """Show only relevant input fields based on selected method."""
        is_specific = self.method == 1
        is_range = self.method in (2, 3, 4)

        # Show/hide specific wavenumber field
        self.wn_label.setVisible(is_specific)
        self.wn_edit.setVisible(is_specific)

        # Show/hide range fields
        self.range_min_label.setVisible(is_range)
        self.range_min_edit.setVisible(is_range)
        self.range_max_label.setVisible(is_range)
        self.range_max_edit.setVisible(is_range)

    def parameters(self):
        """Collect parameters for BaselineShift."""
        params = super().parameters()
        params["method"] = self.method

        if self.method == 1:  # Specific wavenumber
            try:
                params["wn"] = float(self.wn_edit.position)
            except (TypeError, ValueError):
                params["wn"] = None

        elif self.method in (2, 3, 4):  # Range-based
            try:
                rmin = float(self.range_min_edit.position)
                rmax = float(self.range_max_edit.position)
            #  Auto-sort range so "From" is always the lower value
                params["range_min"], params["range_max"] = sorted([rmin, rmax])

            except (TypeError, ValueError):
                params["range_min"] = None
                params["range_max"] = None

        return params

    @staticmethod
    def createinstance(params):
        """Create BaselineShift preprocessor instance."""
        methods = ["minimum", "specific", "min_range", "max_range", "avg_range"]
        method = methods[params.get("method", 0)]

        wn = params.get("wn", None)
        range_min = params.get("range_min", None)
        range_max = params.get("range_max", None)
        wn_range = (range_min, range_max) if range_min is not None and range_max is not None else None

        return BaselineShift(method=method, wn=wn, wn_range=wn_range)

    def set_preview_data(self, data):
        self.preview_data = data


# Register the editor
preprocess_editors.register(BaselineShiftEditor, 126)
