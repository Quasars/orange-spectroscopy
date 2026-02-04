import bottleneck
import numpy as np

import Orange.data
from Orange.data.filter import SameValue, FilterDiscrete, Values
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import DomainModel


class OWAverage(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Average Spectra"

    # Short widget description
    description = "Calculates averages."

    icon = "icons/average.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        averages = Output("Averages", Orange.data.Table, default=True)

    settingsHandler = settings.DomainContextHandler()
    group_var = settings.ContextSetting(None)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.data = None

        self.group_vars = DomainModel(
            placeholder="None",
            separators=False,
            valid_types=Orange.data.DiscreteVariable,
        )
        self.group_view = gui.listView(
            self.controlArea,
            self,
            "group_var",
            box="Group by",
            model=self.group_vars,
            callback=self.grouping_changed,
        )

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")

    @Inputs.data
    def set_data(self, dataset):
        self.closeContext()
        self.data = dataset
        self.group_var = None
        if dataset is not None:
            self.group_vars.set_domain(dataset.domain)
            self.openContext(dataset.domain)

        self.commit.now()

    @staticmethod
    def average_table(table):
        """
        Return a features-averaged table.

        For metas and class_vars,
          - return average value of ContinuousVariable
          - return value of DiscreteVariable, StringVariable and TimeVariable
            if all are the same.
          - return unknown otherwise.
        """
        if len(table) == 0:
            return table
        mean = (
            bottleneck.nanmean(
                table.X,
                axis=0,
            )
            .reshape(1, -1)
            .copy()
        )
        avg_table = Orange.data.Table.from_numpy(
            table.domain,
            X=mean,
            Y=np.atleast_2d(table.Y[0]).copy(),
            metas=np.atleast_2d(table.metas[0]).copy(),
        )
        cont_vars = [
            var
            for var in table.domain.class_vars + table.domain.metas
            if isinstance(var, Orange.data.ContinuousVariable)
        ]

        with avg_table.unlocked():
            for var in cont_vars:
                index = table.domain.index(var)
                col = table.get_column(index)
                avg_table[0, index] = np.nanmean(col)

            other_vars = [
                var
                for var in table.domain.class_vars + table.domain.metas
                if not isinstance(var, Orange.data.ContinuousVariable)
            ]
            for var in other_vars:
                index = table.domain.index(var)
                col = table.get_column(index)
                val = var.to_val(avg_table[0, var])
                if not np.all(col == val):
                    avg_table[0, var] = Orange.data.Unknown

        return avg_table

    def grouping_changed(self):
        """Calls commit() indirectly to respect auto_commit setting."""
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        averages = None
        if self.data is not None:
            if self.group_var is None:
                averages = self.average_table(self.data)
            else:
                parts = []
                for value in self.group_var.values:
                    svfilter = SameValue(self.group_var, value)
                    v_table = self.average_table(svfilter(self.data))
                    parts.append(v_table)
                # Using "None" as in OWSelectRows
                # Values is required because FilterDiscrete doesn't have
                # negate keyword or IsDefined method
                deffilter = Values(
                    conditions=[FilterDiscrete(self.group_var, None)], negate=True
                )
                v_table = self.average_table(deffilter(self.data))
                parts.append(v_table)
                averages = Orange.data.Table.concatenate(parts, axis=0)
        self.Outputs.averages.send(averages)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWAverage).run(Orange.data.Table("iris"))
