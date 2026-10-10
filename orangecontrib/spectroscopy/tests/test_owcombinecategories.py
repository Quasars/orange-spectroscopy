import numpy as np

from Orange.data import (
    Table,
    Domain,
    DiscreteVariable,
)

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owcombinecategories import (
    OWCombineCategories,
)


class TestOWCombineCategories(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWCombineCategories
        )

    @staticmethod
    def create_test_data():
        cluster = DiscreteVariable(
            "Cluster",
            values=["A", "B"],
        )
        tissue = DiscreteVariable(
            "Tissue",
            values=["Tumour", "Necrotic"],
        )

        domain = Domain(
            [],
            metas=[cluster, tissue],
        )

        # Discrete variables are represented by numeric category codes:
        #
        # Cluster: 0 -> A, 1 -> B
        # Tissue:  0 -> Tumour, 1 -> Necrotic
        metas = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=float,
        )

        return Table.from_numpy(
            domain,
            X=np.empty((4, 0)),
            metas=metas,
        )

    def select_all_categories(self):
        self.widget.selected_attrs = [
            self.widget.category_model[0],
            self.widget.category_model[1],
        ]

    def test_no_input(self):
        self.send_signal(
            self.widget.Inputs.data,
            None,
        )

        output = self.get_output(
            self.widget.Outputs.data
        )

        self.assertIsNone(output)

    def test_no_categories_selected(self):
        data = self.create_test_data()

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.widget.selected_attrs = []
        self.widget.commit.now()

        output = self.get_output(
            self.widget.Outputs.data
        )

        self.assertIs(output, data)

    def test_combines_two_categories(self):
        data = self.create_test_data()

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.select_all_categories()

        self.widget.commit.now()

        output = self.get_output(
            self.widget.Outputs.data
        )

        combined = output.domain.metas[-1]
        values = output.get_column(combined)

        self.assertEqual(
            len(np.unique(values)),
            3,
        )

    def test_adds_new_meta_column(self):
        data = self.create_test_data()

        n_metas_before = len(
            data.domain.metas
        )

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.select_all_categories()

        self.widget.commit.now()

        output = self.get_output(
            self.widget.Outputs.data
        )

        self.assertEqual(
            len(output.domain.metas),
            n_metas_before + 1,
        )

    def test_combination_labels(self):
        data = self.create_test_data()

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.select_all_categories()

        self.widget.commit.now()

        output = self.get_output(
            self.widget.Outputs.data
        )

        combined = output.domain.metas[-1]

        self.assertIn(
            "Cluster=A | Tissue=Tumour",
            combined.values,
        )

        self.assertIn(
            "Cluster=A | Tissue=Necrotic",
            combined.values,
        )

        self.assertIn(
            "Cluster=B | Tissue=Tumour",
            combined.values,
        )

    def test_combination_counts_output(self):
        data = self.create_test_data()

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.select_all_categories()

        self.widget.commit.now()

        table = self.get_output(
            self.widget.Outputs.combinations
        )

        counts = table.get_column("Count")

        self.assertEqual(
            sorted(counts.tolist()),
            [1, 1, 2],
        )

    def test_same_rows_receive_same_code(self):
        data = self.create_test_data()

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.select_all_categories()

        self.widget.commit.now()

        output = self.get_output(
            self.widget.Outputs.data
        )

        combined = output.domain.metas[-1]
        values = output.get_column(combined)

        self.assertEqual(
            values[0],
            values[1],
        )

    def test_missing_values(self):
        cluster = DiscreteVariable(
            "Cluster",
            values=["A", "B"],
        )
        tissue = DiscreteVariable(
            "Tissue",
            values=["Tumour", "Necrotic"],
        )

        domain = Domain(
            [],
            metas=[cluster, tissue],
        )

        metas = np.array(
            [
                [0, 0],
                [np.nan, 0],
                [0, 1],
            ],
            dtype=float,
        )

        data = Table.from_numpy(
            domain,
            X=np.empty((3, 0)),
            metas=metas,
        )

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )
        self.select_all_categories()
        self.widget.commit.now()

        output = self.get_output(
            self.widget.Outputs.data
        )

        combined = output.domain.metas[-1]
        labels = set(combined.values)

        self.assertIn(
            "Cluster=? | Tissue=Tumour",
            labels,
        )

    def test_algorithm_unique_inverse_mapping(self):
        arr = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 0],
            ]
        )

        codes, unique_rows = (
            OWCombineCategories.combine_categories(arr)
        )

        self.assertEqual(len(unique_rows), 3)

        self.assertEqual(codes[0], codes[1])
        self.assertNotEqual(codes[0], codes[2])
        self.assertNotEqual(codes[0], codes[3])
        self.assertNotEqual(codes[2], codes[3])

    def test_algorithm_all_rows_identical(self):
        arr = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )

        codes, unique_rows = (
            OWCombineCategories.combine_categories(arr)
        )

        self.assertEqual(len(unique_rows), 1)

        self.assertTrue(
            np.all(codes == 0)
        )

    def test_algorithm_all_rows_unique(self):
        arr = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        )

        codes, unique_rows = (
            OWCombineCategories.combine_categories(arr)
        )

        self.assertEqual(len(unique_rows), 4)
        self.assertEqual(len(np.unique(codes)), 4)

    def test_combination_counts_sum_to_input_size(self):
        data = self.create_test_data()

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.select_all_categories()

        self.widget.commit.now()

        table = self.get_output(
            self.widget.Outputs.combinations
        )

        counts = table.get_column("Count")

        self.assertEqual(
            int(np.sum(counts)),
            len(data),
        )
    def test_unique_name_collision(self):
        data = self.create_test_data()

        existing = DiscreteVariable(
            "Combined Category",
            values=["A"],
        )

        extended = data.add_column(
            existing,
            np.zeros(len(data), dtype=float),
            to_metas=True,
        )

        self.send_signal(
            self.widget.Inputs.data,
            extended,
        )
        self.select_all_categories()
        self.widget.commit.now()

        output = self.get_output(
            self.widget.Outputs.data
        )
        new_var = output.domain.metas[-1]

        self.assertNotEqual(
            new_var.name,
            "Combined Category",
        )
        self.assertTrue(
            new_var.name.startswith(
                "Combined Category"
            )
        )

        self.assertEqual(
            len(output.domain.metas),
            len(extended.domain.metas) + 1,
        )

    def test_context_restoration(self):
        data = self.create_test_data()

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.widget.selected_attrs = [
            self.widget.category_model[0],
            self.widget.category_model[1],
        ]

        stored_settings = (
            self.widget.settingsHandler.pack_data(
                self.widget
            )
        )

        widget2 = self.create_widget(
            OWCombineCategories,
            stored_settings=stored_settings,
        )

        self.send_signal(
            widget2.Inputs.data,
            data,
            widget=widget2,
        )

        self.assertEqual(
            list(widget2.selected_attrs),
            [
                widget2.category_model[0],
                widget2.category_model[1],
            ],
        )

    def test_large_combination_warning(self):
        attrs = [
            DiscreteVariable(
                f"C{i}",
                values=["0", "1"],
            )
            for i in range(10)
        ]

        domain = Domain(
            [],
            metas=attrs,
        )

        # Every ten-bit pattern occurs exactly once, giving
        # 2**10 == 1024 unique combinations.
        metas = np.array(
            [
                [
                    1 if i & (1 << j) else 0
                    for j in range(10)
                ]
                for i in range(1024)
            ],
            dtype=float,
        )

        data = Table.from_numpy(
            domain,
            X=np.empty((1024, 0)),
            metas=metas,
        )

        self.send_signal(
            self.widget.Inputs.data,
            data,
        )

        self.widget.selected_attrs = list(
            self.widget.category_model
        )
        self.widget.commit.now()

        self.assertTrue(
            self.widget.Warning
            .large_number_of_combinations
            .is_shown()
        )

        combinations = self.get_output(
            self.widget.Outputs.combinations
        )
        self.assertEqual(
            len(combinations),
            1024,
        )

    def test_make_labels_with_missing_values(self):
        cluster = DiscreteVariable(
            "Cluster",
            values=["A", "B"],
        )

        tissue = DiscreteVariable(
            "Tissue",
            values=["Tumour", "Normal"],
        )

        rows = np.array(
            [
                [-1, 0],
                [0, 1],
            ]
        )

        labels = (
            OWCombineCategories
            ._make_combination_labels(
                rows,
                [cluster, tissue],
            )
        )

        self.assertEqual(
            labels[0],
            "Cluster=? | Tissue=Tumour",
        )

        self.assertEqual(
            labels[1],
            "Cluster=A | Tissue=Normal",
        )