import numpy as np

import Orange.data
from Orange.data.filter import SameValue, FilterDiscrete, Values
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import DomainModel
import Orange.data.filter as data_filter


class OWAverage(OWWidget):
    # Widget's name as displayed in the canvas
    name = "snr"

    # Short widget description
    description = (
        "Calculates averages.")

    icon = "icons/none.svg"

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

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")

    def __init__(self):
        super().__init__()

        self.data = None
        self.set_data(self.data)  # show warning

        self.group_vars = DomainModel(
            placeholder="None", separators=False,
            valid_types=Orange.data.DiscreteVariable)
        self.group_view = gui.listView(
            self.controlArea, self, "group_var", box="Select axis: x",
            model=self.group_vars, callback=self.grouping_changed)

        self.group_vars1 = DomainModel(
            placeholder="None", separators=False,
            valid_types=Orange.data.DiscreteVariable)
        self.group_view = gui.listView(
            self.controlArea, self, "group_var1", box="Select axis: y",
            model=self.group_vars, callback=self.grouping_changed)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")


    @Inputs.data
    def set_data(self, dataset):
        self.Warning.nodata.clear()
        self.closeContext()
        self.data = dataset
        self.group_var = None
        if dataset is None:
            self.Warning.nodata()
        else:
            self.group_vars.set_domain(dataset.domain)
            self.openContext(dataset.domain)

        self.commit()

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
        mean = np.nanmean(table.X, axis=0, keepdims=True)/np.std(table.X, axis=0, keepdims=True)
        avg_table = Orange.data.Table.from_numpy(table.domain,
                                                 X=mean,
                                                 Y=np.atleast_2d(table.Y[0].copy()),
                                                 metas=np.atleast_2d(table.metas[0].copy()))
        cont_vars = [var for var in table.domain.class_vars + table.domain.metas
                     if isinstance(var, Orange.data.ContinuousVariable)]
        for var in cont_vars:
            index = table.domain.index(var)
            col, _ = table.get_column_view(index)
            try:
                avg_table[0, index] = np.nanmean(col)
            except AttributeError:
                # numpy.lib.nanfunctions._replace_nan just guesses and returns
                # a boolean array mask for object arrays because object arrays
                # do not support `isnan` (numpy-gh-9009)
                # Since we know that ContinuousVariable values must be np.float64
                # do an explicit cast here
                avg_table[0, index] = np.nanmean(col, dtype=np.float64)

        other_vars = [var for var in table.domain.class_vars + table.domain.metas
                      if not isinstance(var, Orange.data.ContinuousVariable)]
        for var in other_vars:
            index = table.domain.index(var)
            col, _ = table.get_column_view(index)
            val = var.to_val(avg_table[0, var])
            if not np.all(col == val):
                avg_table[0, var] = Orange.data.Unknown

        return avg_table

    def grouping_changed(self):
        """Calls commit() indirectly to respect auto_commit setting."""
        self.commit()

    def commit(self):
        print("<-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/->")
        averages = None
        if self.data is not None:
            if self.group_var is None:
                averages = self.average_table(self.data)
            else:
                parts = []
                domain = self.data.domain
                for value in self.group_var.values:
                    print("<-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/->")
                    print('entrou em values', value)
                    attr_name = self.group_var
                    attr_index = domain.index(attr_name)
                    # attr_index = 10
                    filter = data_filter.FilterDiscrete(attr_index, list(str(value)))
                    filtro = []
                    filtro.append(filter)
                    filters = data_filter.Values(filtro)
                    temp = filters(self.data)
                    for value1 in self.group_var1.values:
                        print("<-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/->")
                        print('entrou em values1', value1)
                        attr_name1 = self.group_var1
                        attr_index1 = domain.index(attr_name1)
                        # attr_index1 = attr_index + 1
                        filter1 = data_filter.FilterDiscrete(attr_index1, value1)
                        filtro1 = []
                        filtro1.append(filter1)
                        filters1 = data_filter.Values(filtro1)
                        full = filters1(temp)
                        print(value, value1, full.X.shape, temp.X.shape)
                        v_table = self.average_table(full)
                        # print(v_table)
                        # print(value)
                        parts.append(v_table)
                        # self.Outputs.averages.send(full)


                # Using "None" as in OWSelectRows
                # Values is required because FilterDiscrete doesn't have
                # negate keyword or IsDefined method
                deffilter = Values(conditions=[FilterDiscrete(self.group_var, None)],
                                   negate=True)
                v_table = self.average_table(deffilter(self.data))
                parts.append(v_table)
                averages = Orange.data.Table.concatenate(parts, axis=0)
                # print(list(self.group_var))
        self.Outputs.averages.send(averages)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWAverage).run(Orange.data.Table("iris"))
