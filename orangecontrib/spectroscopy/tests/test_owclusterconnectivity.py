import unittest

import numpy as np

from AnyQt.QtCore import QItemSelectionModel
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owclusterconnectivity import (
    OWClusterConnectivity,
)


class TestOWClusterConnectivity(WidgetTest):
    """Tests for the Cluster Connectivity widget."""

    def setUp(self):
        self.widget = self.create_widget(OWClusterConnectivity)

    @staticmethod
    def make_table(
        coordinates,
        cluster_codes,
        *,
        cluster_values=("A", "B"),
        axis_names=None,
        extra_metas=(),
    ):
        """Create a table with continuous coordinate metas and a cluster meta.

        Orange discrete columns are represented by floating-point category
        codes. Missing values must therefore be supplied as ``np.nan`` rather
        than as strings such as ``"?"``.
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates[:, None]

        n_rows, n_dimensions = coordinates.shape
        if axis_names is None:
            axis_names = tuple("XYZ"[:n_dimensions])
        if len(axis_names) != n_dimensions:
            raise ValueError("axis_names must match the coordinate dimension")

        axes = [ContinuousVariable(name) for name in axis_names]
        cluster = DiscreteVariable("Cluster", values=cluster_values)
        extra_variables = [variable for variable, _ in extra_metas]

        domain = Domain([], metas=[*axes, cluster, *extra_variables])
        metas = np.empty(
            (n_rows, n_dimensions + 1 + len(extra_metas)),
            dtype=float,
        )
        metas[:, :n_dimensions] = coordinates
        metas[:, n_dimensions] = np.asarray(cluster_codes, dtype=float)
        for column, (_, values) in enumerate(
            extra_metas, start=n_dimensions + 1
        ):
            metas[:, column] = np.asarray(values, dtype=float)

        table = Table.from_numpy(
            domain,
            X=np.empty((n_rows, 0)),
            metas=metas,
        )
        return table, axes, cluster

    def configure_widget(self, data, cluster, axes, connectivity=1):
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.cluster_attr = cluster
        self.widget.selected_axes = list(axes)
        self.widget.connectivity = connectivity
        self.widget._restore_axis_selection()
        self.widget._update_connectivity_control()
        self.widget.commit.now()

    def output_codes_and_variable(self):
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(output)
        variable = output.domain.metas[-1]
        return output, output.get_column(variable), variable

    def test_no_input(self):
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_no_cluster_selected(self):
        data, axes, _ = self.make_table([0, 1], [0, 0])
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.cluster_attr = None
        self.widget.selected_axes = axes
        self.widget.commit.now()

        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertTrue(self.widget.Information.select_cluster.is_shown())

    def test_no_axes_selected(self):
        data, _, cluster = self.make_table([0, 1], [0, 0])
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.cluster_attr = cluster
        self.widget.selected_axes = []
        self.widget.commit.now()

        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertTrue(self.widget.Information.select_axes.is_shown())

    def test_warns_when_no_discrete_variables_exist(self):
        x = ContinuousVariable("X")
        data = Table.from_numpy(
            Domain([], metas=[x]),
            X=np.empty((2, 0)),
            metas=np.array([[0.0], [1.0]]),
        )

        self.send_signal(self.widget.Inputs.data, data)

        self.assertTrue(self.widget.Warning.no_clusters.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_warns_when_no_coordinate_variables_exist(self):
        cluster = DiscreteVariable("Cluster", values=("A", "B"))
        data = Table.from_numpy(
            Domain([], metas=[cluster]),
            X=np.empty((2, 0)),
            metas=np.array([[0.0], [1.0]]),
        )

        self.send_signal(self.widget.Inputs.data, data)

        self.assertTrue(self.widget.Warning.no_axes.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_one_dimensional_gap_splits_a_cluster(self):
        data, axes, cluster = self.make_table(
            [0, 1, 4, 5],
            [0, 0, 0, 0],
        )
        self.configure_widget(data, cluster, axes)

        output, codes, variable = self.output_codes_and_variable()
        np.testing.assert_array_equal(codes, [0, 0, 1, 1])
        self.assertEqual(variable.values, ("A (1)", "A (2)"))
        self.assertEqual(len(output), len(data))

    def test_adjacent_rows_in_same_cluster_stay_together(self):
        data, axes, cluster = self.make_table(
            [0, 1, 2, 3],
            [0, 0, 0, 0],
        )
        self.configure_widget(data, cluster, axes)

        _, codes, variable = self.output_codes_and_variable()
        np.testing.assert_array_equal(codes, [0, 0, 0, 0])
        self.assertEqual(variable.values, ("A (1)",))

    def test_different_categories_are_labelled_separately(self):
        data, axes, cluster = self.make_table(
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        )
        self.configure_widget(data, cluster, axes)

        _, codes, variable = self.output_codes_and_variable()
        np.testing.assert_array_equal(codes, [0, 0, 1, 1])
        self.assertEqual(variable.values, ("A (1)", "B (1)"))

    def test_unused_source_category_does_not_create_output_value(self):
        data, axes, cluster = self.make_table(
            [0, 1],
            [0, 0],
            cluster_values=("A", "B", "C"),
        )
        self.configure_widget(data, cluster, axes)

        _, codes, variable = self.output_codes_and_variable()
        np.testing.assert_array_equal(codes, [0, 0])
        self.assertEqual(variable.values, ("A (1)",))

    def test_diagonal_pixels_are_separate_with_connectivity_one(self):
        data, axes, cluster = self.make_table(
            [(0, 0), (1, 1)],
            [0, 0],
        )
        self.configure_widget(data, cluster, axes, connectivity=1)

        _, codes, variable = self.output_codes_and_variable()
        self.assertNotEqual(codes[0], codes[1])
        self.assertEqual(variable.values, ("A (1)", "A (2)"))

    def test_diagonal_pixels_join_with_connectivity_two(self):
        data, axes, cluster = self.make_table(
            [(0, 0), (1, 1)],
            [0, 0],
        )
        self.configure_widget(data, cluster, axes, connectivity=2)

        _, codes, variable = self.output_codes_and_variable()
        np.testing.assert_array_equal(codes, [0, 0])
        self.assertEqual(variable.values, ("A (1)",))

    def test_missing_coordinate_produces_unknown_output(self):
        data, axes, cluster = self.make_table(
            [0, np.nan, 1],
            [0, 0, 0],
        )
        self.configure_widget(data, cluster, axes)

        _, codes, _ = self.output_codes_and_variable()
        self.assertTrue(np.isnan(codes[1]))
        self.assertFalse(np.isnan(codes[[0, 2]]).any())
        self.assertTrue(self.widget.Warning.missing_coordinates.is_shown())

    def test_missing_cluster_produces_unknown_output(self):
        data, axes, cluster = self.make_table(
            [0, 1, 2],
            [0, np.nan, 0],
        )
        self.configure_widget(data, cluster, axes)

        _, codes, _ = self.output_codes_and_variable()
        self.assertTrue(np.isnan(codes[1]))
        self.assertTrue(self.widget.Warning.missing_coordinates.is_shown())

    def test_duplicate_coordinates_are_rejected(self):
        data, axes, cluster = self.make_table(
            [(0, 0), (0, 0)],
            [0, 1],
        )
        self.configure_widget(data, cluster, axes)

        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertTrue(self.widget.Error.duplicate_coordinates.is_shown())

    def test_invalid_cluster_code_is_rejected(self):
        data, axes, cluster = self.make_table(
            [0, 1],
            [0, 2],
            cluster_values=("A", "B"),
        )
        self.configure_widget(data, cluster, axes)

        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertTrue(self.widget.Error.invalid_cluster_codes.is_shown())

    def test_fractional_cluster_code_is_rejected(self):
        data, axes, cluster = self.make_table(
            [0, 1],
            [0, 0.5],
            cluster_values=("A", "B"),
        )
        self.configure_widget(data, cluster, axes)

        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertTrue(self.widget.Error.invalid_cluster_codes.is_shown())

    def test_output_adds_one_discrete_meta(self):
        data, axes, cluster = self.make_table(
            [0, 1],
            [0, 0],
        )
        original_meta_count = len(data.domain.metas)
        self.configure_widget(data, cluster, axes)

        output, _, variable = self.output_codes_and_variable()
        self.assertEqual(len(output.domain.metas), original_meta_count + 1)
        self.assertIsInstance(variable, DiscreteVariable)
        self.assertEqual(variable.name, "Connected Cluster")

    def test_output_preserves_input_values_and_row_order(self):
        data, axes, cluster = self.make_table(
            [(2, 0), (0, 0), (1, 0)],
            [1, 0, 0],
        )
        original_metas = data.metas.copy()
        self.configure_widget(data, cluster, axes)

        output, _, _ = self.output_codes_and_variable()
        np.testing.assert_array_equal(
            output.metas[:, : original_metas.shape[1]],
            original_metas,
        )

    def test_output_name_collision_is_resolved(self):
        existing = DiscreteVariable("Connected Cluster", values=("old",))
        data, axes, cluster = self.make_table(
            [0, 1],
            [0, 0],
            extra_metas=((existing, [0, 0]),),
        )
        self.configure_widget(data, cluster, axes)

        output, _, variable = self.output_codes_and_variable()
        self.assertNotEqual(variable.name, "Connected Cluster")
        self.assertTrue(variable.name.startswith("Connected Cluster"))
        self.assertEqual(len(output.domain.metas), len(data.domain.metas) + 1)

    def test_summary_reports_number_of_regions(self):
        data, axes, cluster = self.make_table(
            [0, 1, 4, 5],
            [0, 0, 0, 0],
        )
        self.configure_widget(data, cluster, axes)

        self.assertEqual(self.widget.summary_label.text(), "Connected regions: 2")

    def test_connectivity_limit_matches_selected_dimensions(self):
        data, axes, cluster = self.make_table(
            [(0, 0), (1, 1)],
            [0, 0],
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.cluster_attr = cluster
        self.widget.selected_axes = list(axes)
        self.widget._update_connectivity_control()

        self.assertEqual(self.widget.connectivity_spin.maximum(), 2)
        self.assertTrue(self.widget.connectivity_spin.isEnabled())

    def test_axis_view_selection_updates_selected_axes(self):
        data, axes, _ = self.make_table(
            [(0, 0), (1, 1)],
            [0, 0],
        )
        self.send_signal(self.widget.Inputs.data, data)

        selection_model = self.widget.axis_view.selectionModel()
        for row in range(len(axes)):
            selection_model.select(
                self.widget.axis_model.index(row, 0),
                QItemSelectionModel.Select | QItemSelectionModel.Rows,
            )

        self.assertEqual(self.widget.selected_axes, axes)

    def test_coordinate_indices_preserve_input_row_mapping(self):
        coordinates = np.array(
            [
                [1, 1],
                [0, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=float,
        )

        indices, shape, valid = OWClusterConnectivity._coordinate_indices(
            coordinates
        )

        self.assertEqual(shape, (2, 2))
        np.testing.assert_array_equal(valid, [True, True, True, True])
        np.testing.assert_array_equal(indices[0], [1, 0, 1, 0])
        np.testing.assert_array_equal(indices[1], [1, 0, 0, 1])

    def test_connected_components_can_be_tested_without_widget(self):
        cluster = DiscreteVariable("Cluster", values=("A", "B"))
        codes, labels = OWClusterConnectivity.connected_components(
            cluster_codes=np.array([0, 0, 0, 0], dtype=float),
            coordinates=np.array([[0], [1], [4], [5]], dtype=float),
            cluster_variable=cluster,
            connectivity=1,
        )

        np.testing.assert_array_equal(codes, [0, 0, 1, 1])
        self.assertEqual(labels, ["A (1)", "A (2)"])


if __name__ == "__main__":
    unittest.main()
