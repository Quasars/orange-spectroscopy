import numpy as np
from scipy import ndimage

from AnyQt.QtCore import QItemSelectionModel
from AnyQt.QtWidgets import QAbstractItemView, QListView

from Orange.data import ContinuousVariable, DiscreteVariable, Table
from Orange.data.util import get_unique_names
from Orange.widgets import gui, settings, widget
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Output

from orangecontrib.spectroscopy.utils import index_values, values_to_linspace


class OWClusterConnectivity(widget.OWWidget):
    """Split each categorical cluster into spatially connected components."""

    name = "Cluster Connectivity"
    description = (
        "Split disconnected spatial regions of a categorical cluster "
        "assignment into separate categories."
    )
    icon = "icons/cluster_connectivity.svg"
    priority = 4010

    want_main_area = False
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    settingsHandler = DomainContextHandler()

    cluster_attr = ContextSetting(None)
    selected_axes = ContextSetting([])
    connectivity = settings.Setting(1)
    auto_commit = settings.Setting(True)

    class Error(widget.OWWidget.Error):
        invalid_axis = widget.Msg(
            "Coordinate variable '{}' cannot be mapped onto a regular grid."
        )
        duplicate_coordinates = widget.Msg(
            "Multiple rows have the same spatial coordinates."
        )
        invalid_cluster_codes = widget.Msg(
            "The selected cluster variable contains invalid category codes."
        )

    class Warning(widget.OWWidget.Warning):
        no_clusters = widget.Msg("No categorical variables are available.")
        no_axes = widget.Msg("No continuous coordinate variables are available.")
        missing_coordinates = widget.Msg(
            "{} row(s) with missing coordinates have an unknown output category."
        )

    class Information(widget.OWWidget.Information):
        select_cluster = widget.Msg("Select a cluster variable.")
        select_axes = widget.Msg("Select at least one coordinate variable.")

    def __init__(self):
        super().__init__()

        self.data: Table | None = None

        self.cluster_model = DomainModel(
            order=(
                DomainModel.ATTRIBUTES
                | DomainModel.CLASSES
                | DomainModel.METAS
            ),
            valid_types=DiscreteVariable,
        )
        # Spectral variables are normally continuous attributes. Restricting
        # coordinates to classes/metas avoids listing every wavelength.
        self.axis_model = DomainModel(
            order=DomainModel.CLASSES | DomainModel.METAS,
            valid_types=ContinuousVariable,
        )

        cluster_box = gui.vBox(self.controlArea, "Cluster")
        self.cluster_combo = gui.comboBox(
            cluster_box,
            self,
            "cluster_attr",
            model=self.cluster_model,
            searchable=True,
            callback=self.commit.deferred,
        )

        axes_box = gui.vBox(self.controlArea, "Spatial coordinates")
        self.axis_view = QListView()
        self.axis_view.setModel(self.axis_model)
        self.axis_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        axes_box.layout().addWidget(self.axis_view)
        self.axis_view.selectionModel().selectionChanged.connect(
            self._axis_selection_changed
        )

        options_box = gui.vBox(self.controlArea, "Connectivity")
        self.connectivity_spin = gui.spin(
            options_box,
            self,
            "connectivity",
            minv=1,
            maxv=1,
            step=1,
            label="Order: ",
            callback=self.commit.deferred,
        )
        gui.widgetLabel(
            options_box,
            "1 uses face-sharing neighbours; higher values also include "
            "increasingly diagonal neighbours.",
        )

        summary_box = gui.vBox(self.controlArea, "Summary")
        self.summary_label = gui.widgetLabel(summary_box, "Connected regions: 0")

        gui.auto_commit(
            self.controlArea,
            self,
            "auto_commit",
            "Apply",
        )

    def _axis_selection_changed(self, *_):
        self.selected_axes = [
            self.axis_model[index.row()]
            for index in self.axis_view.selectionModel().selectedRows()
        ]
        self._update_connectivity_control()
        self.commit.deferred()

    def _restore_axis_selection(self):
        selection_model = self.axis_view.selectionModel()
        selection_model.blockSignals(True)
        selection_model.clearSelection()
        selected = set(self.selected_axes)
        for row, variable in enumerate(self.axis_model):
            if variable in selected:
                selection_model.select(
                    self.axis_model.index(row, 0),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
        selection_model.blockSignals(False)

    def _update_connectivity_control(self):
        ndim = len(self.selected_axes)
        maximum = max(ndim, 1)
        self.connectivity_spin.setMaximum(maximum)
        if self.connectivity > maximum:
            self.connectivity = maximum
        self.connectivity_spin.setEnabled(ndim > 0)


    def _clear_input_messages(self):
        """Clear messages whose state depends on the input domain."""
        self.Warning.no_clusters.clear()
        self.Warning.no_axes.clear()


    def _clear_commit_messages(self):
        """Clear messages produced while computing the output."""
        self.Error.invalid_axis.clear()
        self.Error.duplicate_coordinates.clear()
        self.Error.invalid_cluster_codes.clear()
        self.Warning.missing_coordinates.clear()
        self.Information.select_cluster.clear()
        self.Information.select_axes.clear()

    @Inputs.data
    def set_data(self, data: Table | None):
        self.closeContext()
        
        self._clear_input_messages()
        self._clear_commit_messages()

        self.data = data

        domain = data.domain if data is not None else None
        self.cluster_model.set_domain(domain)
        self.axis_model.set_domain(domain)

        if data is not None:
            self.openContext(data)
            if len(self.cluster_model) == 0:
                self.cluster_attr = None
                self.Warning.no_clusters()
            elif self.cluster_attr is None:
                self.cluster_attr = self.cluster_model[0]

            if len(self.axis_model) == 0:
                self.selected_axes = []
                self.Warning.no_axes()

        self._restore_axis_selection()
        self._update_connectivity_control()
        self.commit.now()

    @staticmethod
    def _coordinate_indices(
        coordinates: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], tuple[int, ...], np.ndarray]:
        """Map finite coordinate rows to indices in a regular N-D grid."""
        valid = np.isfinite(coordinates).all(axis=1)
        valid_coordinates = coordinates[valid]

        indices = []
        shape = []
        for column in valid_coordinates.T:
            linspace = values_to_linspace(column)
            if linspace is None:
                raise ValueError
            index = index_values(column, linspace)
            if np.any(index < 0) or np.any(index >= linspace[2]):
                raise ValueError
            indices.append(index)
            shape.append(linspace[2])

        index_tuple = tuple(indices)
        if valid_coordinates.size:
            flat = np.ravel_multi_index(index_tuple, tuple(shape))
            if np.unique(flat).size != flat.size:
                raise RuntimeError("duplicate coordinates")

        return index_tuple, tuple(shape), valid

    @staticmethod
    def connected_components(
        cluster_codes: np.ndarray,
        coordinates: np.ndarray,
        cluster_variable: DiscreteVariable,
        connectivity: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Return output category codes and labels for connected regions.

        Rows with a missing coordinate or missing cluster value receive NaN.
        Components are ordered first by the input category code and then by
        SciPy's deterministic connected-component scan order.
        """
        indices, shape, valid_coordinates = (
            OWClusterConnectivity._coordinate_indices(coordinates)
        )

        finite_cluster = np.isfinite(cluster_codes)
        valid = valid_coordinates & finite_cluster

        integer_codes = np.full(len(cluster_codes), -1, dtype=int)
        integer_codes[finite_cluster] = cluster_codes[finite_cluster].astype(int)
        if np.any(
            (integer_codes[finite_cluster] < 0)
            | (integer_codes[finite_cluster] >= len(cluster_variable.values))
            | (cluster_codes[finite_cluster] != integer_codes[finite_cluster])
        ):
            raise IndexError("invalid cluster code")

        # _coordinate_indices contains all rows with finite coordinates. Build
        # the matching subset of grid indices for rows that also have a known
        # cluster value.
        finite_coordinate_rows = np.flatnonzero(valid_coordinates)
        keep = finite_cluster[finite_coordinate_rows]
        valid_indices = tuple(axis[keep] for axis in indices)
        valid_rows = finite_coordinate_rows[keep]

        grid = np.full(shape, -1, dtype=int)
        if valid_rows.size:
            grid[valid_indices] = integer_codes[valid_rows]

        output = np.full(len(cluster_codes), np.nan, dtype=float)
        labels: list[str] = []
        structure = ndimage.generate_binary_structure(len(shape), connectivity)
        next_code = 0

        for category_code, category_label in enumerate(cluster_variable.values):
            labelled, component_count = ndimage.label(
                grid == category_code,
                structure=structure,
            )
            for component in range(1, component_count + 1):
                labels.append(f"{category_label} ({component})")
                component_rows = labelled[valid_indices] == component
                output[valid_rows[component_rows]] = next_code
                next_code += 1

        return output, labels

    @gui.deferred
    def commit(self):
        self._clear_commit_messages()
        self.summary_label.setText("Connected regions: 0")

        if self.data is None:
            self.Outputs.data.send(None)
            return
        if self.cluster_attr is None:
            self.Information.select_cluster()
            self.Outputs.data.send(None)
            return
        if not self.selected_axes:
            self.Information.select_axes()
            self.Outputs.data.send(None)
            return

        cluster_codes = self.data.get_column(self.cluster_attr)
        coordinates = np.column_stack(
            [self.data.get_column(axis) for axis in self.selected_axes]
        )

        try:
            output_codes, labels = self.connected_components(
                cluster_codes,
                coordinates,
                self.cluster_attr,
                self.connectivity,
            )
        except RuntimeError:
            self.Error.duplicate_coordinates()
            self.Outputs.data.send(None)
            return
        except IndexError:
            self.Error.invalid_cluster_codes()
            self.Outputs.data.send(None)
            return
        except (ValueError, OverflowError):
            # Identify the first axis that does not map cleanly when possible.
            bad_axis = next(
                (
                    axis.name
                    for axis in self.selected_axes
                    if values_to_linspace(
                        self.data.get_column(axis)[
                            np.isfinite(self.data.get_column(axis))
                        ]
                    )
                    is None
                ),
                self.selected_axes[0].name,
            )
            self.Error.invalid_axis(bad_axis)
            self.Outputs.data.send(None)
            return

        missing_count = int(np.count_nonzero(np.isnan(output_codes)))
        if missing_count:
            self.Warning.missing_coordinates(missing_count)

        variable = DiscreteVariable(
            name=get_unique_names(self.data.domain, "Connected Cluster"),
            values=labels,
        )
        output_data = self.data.add_column(
            variable,
            output_codes,
            to_metas=True,
        )
        self.summary_label.setText(f"Connected regions: {len(labels)}")
        self.Outputs.data.send(output_data)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWClusterConnectivity).run()
