from collections import Counter

import numpy as np

import Orange.data
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable, StringVariable

from Orange.data.util import get_unique_names

from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import DomainContextHandler, ContextSetting

from Orange.widgets.utils.itemmodels import DomainModel
from AnyQt.QtWidgets import QListView, QAbstractItemView

class OWCombineCategories(widget.OWWidget):
    """
    Create a new discrete variable describing the unique
    combinations of several discrete variables.

    Example

        Cluster=A, Tissue=Tumour
        Cluster=A, Tissue=Necrotic
        Cluster=B, Tissue=Tumour

    becomes

        Combined Category
            Cluster=A | Tissue=Tumour
            Cluster=A | Tissue=Necrotic
            Cluster=B | Tissue=Tumour
    """

    name = "Combine Categories"
    description = (
        "Create a new categorical variable from combinations "
        "of selected categorical variables."
    )
    icon = "icons/combine_categories.svg"
    priority = 4000

    want_main_area = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        combinations = Output("Combinations", Table)

    settingsHandler = DomainContextHandler()

    selected_attrs = ContextSetting([])
    auto_commit = settings.Setting(True)

    class Warning(widget.OWWidget.Warning):
        no_categories = widget.Msg("No categorical variables available.")

        large_number_of_combinations = widget.Msg("Large number of unique combinations created.")

    class Information(widget.OWWidget.Information):
        select_categories = widget.Msg("Select one or more categorical variables.")

    def __init__(self):
        super().__init__()

        self.data = None

        self.category_model = DomainModel(
            order=(
                DomainModel.ATTRIBUTES
                | DomainModel.CLASSES
                | DomainModel.METAS
            ),
            valid_types=DiscreteVariable,
        )

        box = gui.vBox(self.controlArea, "Categories")

        self.category_view = QListView()

        self.category_view.setModel(self.category_model)

        self.category_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        box.layout().addWidget(self.category_view)

        self.category_view.selectionModel().selectionChanged.connect(self._selection_changed)

        info_box = gui.vBox(self.controlArea, "Summary")

        self.info_selected = gui.widgetLabel(info_box, "Selected variables: 0")

        self.info_combinations = gui.widgetLabel(info_box, "Unique combinations: 0")

        gui.auto_commit(
            self.controlArea,
            self,
            "auto_commit",
            "Apply",
        )
        
    def _selection_changed(self):
        self.selected_attrs = [
            self.category_model[index.row()]
            for index in self.category_view.selectedIndexes()
        ]
        self.commit.deferred()


    @Inputs.data
    def set_data(self, data):
        self.closeContext()

        self.Warning.no_categories.clear()
        self.Warning.large_number_of_combinations.clear()
        self.Information.select_categories.clear()

        self.data = data

        domain = data.domain if data is not None else None

        self.category_model.set_domain(domain)

        if data is not None:
            self.openContext(data)

            if len(self.category_model) == 0:
                self.Warning.no_categories()

        self.commit.now()
        

    @staticmethod
    def _extract_columns(table, attrs):
        """
        table: Table
        attrs: list[DiscreteVariable]

        Extract columns and replace NaNs by -1.

        This allows rows containing missing values
        to participate in np.unique.
        """

        cols = np.column_stack(
            [
                np.nan_to_num(
                    table.get_column(attr),
                    nan=-1,
                ).astype(np.int64)
                for attr in attrs
            ]
        )

        return cols

    @staticmethod
    def _make_combination_labels(unique_rows, attrs):
        """
        unique_rows: np.ndarray,
        attrs: list[DiscreteVariable]

        returns list[str]
        """
        labels = []

        for row in unique_rows:
            parts = []

            for code, attr in zip(row, attrs):

                if code == -1:
                    value = "?"
                else:
                    value = attr.values[int(code)]

                parts.append(
                    f"{attr.name}={value}"
                )

            labels.append(" | ".join(parts))

        return labels


    @staticmethod
    def _build_combination_table(labels, counts):
        """
        labels: list[str]
        counts: np.ndarray
        
        returns Table

        Output table:

            Combination      Count
            ----------------------
            A|B              12
            A|C              37
        """

        combination = StringVariable("Combination")
        count = ContinuousVariable("Count")

        domain = Domain(
            [count],
            metas=[combination],
        )

        metas = np.array(labels, dtype=object).reshape(-1, 1)

        X = counts.reshape(-1, 1)

        return Table.from_numpy(
            domain,
            X=X,
            metas=metas,
        )


    @staticmethod
    def combine_categories(columns):
        """
        Return

            inverse:
                row -> combination id

            unique_rows:
                unique category combinations
        """

        unique_rows, inverse = np.unique(
            columns,
            axis=0,
            return_inverse=True,
        )

        return inverse, unique_rows


    @gui.deferred
    def commit(self):

        self.Information.select_categories.clear()
        self.Warning.large_number_of_combinations.clear()

        self.info_selected.setText(
            f"Selected variables: "
            f"{len(self.selected_attrs)}"
        )

        self.info_combinations.setText(
            "Unique combinations: 0"
        )

        if self.data is None:
            self.Outputs.data.send(None)
            self.Outputs.combinations.send(None)
            return

        attrs = list(self.selected_attrs)

        if not attrs:
            self.Information.select_categories()

            self.Outputs.data.send(self.data)
            self.Outputs.combinations.send(None)
            return

        columns = self._extract_columns(
            self.data,
            attrs,
        )

        codes, unique_rows = (self.combine_categories(columns))

        labels = self._make_combination_labels(unique_rows, attrs)

        counts = np.bincount(codes,minlength=len(labels),)

        n_combinations = len(labels)

        self.info_combinations.setText(f"Unique combinations: {n_combinations}")

        if n_combinations > 1000:
            self.Warning.large_number_of_combinations()

        variable = DiscreteVariable(
            name=get_unique_names(
                self.data.domain,
                "Combined Category",
            ),
            values=labels,
        )

        output_data = self.data.add_column(variable, codes, to_metas=True)

        combination_table = (
            self._build_combination_table(
                labels,
                counts,
            )
        )

        self.Outputs.data.send(output_data)
        self.Outputs.combinations.send(combination_table)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import (
        WidgetPreview,
    )

    WidgetPreview(OWCombineCategories).run()
