import numpy as np

import Orange.data
from Orange.data.filter import SameValue, FilterDiscrete, Values
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import DomainModel
import Orange.data.filter as data_filter


class OWSNR(OWWidget):
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
        averages = Output("SNR", Orange.data.Table, default=True)

    settingsHandler = settings.DomainContextHandler()
    group_x = settings.ContextSetting(None)
    group_y = settings.ContextSetting(None)

    autocommit = settings.Setting(False)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")
        nofloat_x = Msg("Group x, is not a float or int value")
        nofloat_y = Msg("Group y, is not a float or int value")

    def __init__(self):
        super().__init__()

        self.data = None
        self.set_data(self.data)  # show warning

        self.group_axis_x = DomainModel(
            placeholder="None", separators=False,
            valid_types=Orange.data.DiscreteVariable)
        self.group_view = gui.listView(
            self.controlArea, self, "group_x", box="Select axis: x",
            model=self.group_axis_x, callback=self.grouping_changed)

        self.group_axis_y = DomainModel(
            placeholder="None", separators=False,
            valid_types=Orange.data.DiscreteVariable)
        self.group_view = gui.listView(
            self.controlArea, self, "group_y", box="Select axis: y",
            model=self.group_axis_y, callback=self.grouping_changed)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")


    @Inputs.data
    def set_data(self, dataset):
        self.Warning.nodata.clear()
        self.closeContext()
        self.data = dataset
        self.group_x = None
        self.group_y = None
        if dataset is None:
            self.Warning.nodata()
        else:
            self.group_axis_x.set_domain(dataset.domain)
            self.group_axis_y.set_domain(dataset.domain)
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
        averages = None
        if self.data is not None:
            if self.group_x is None or self.group_y is None:
                averages = self.average_table(self.data)
            else:
                try:
                    self.Warning.nofloat_x()
                    if float(self.group_x.values[0]):
                        self.Warning.nofloat_x.clear()
                except ValueError:
                    self.group_x = None
                try:
                    self.Warning.nofloat_y()
                    if float(self.group_y.values[0]):
                        self.Warning.nofloat_y.clear()
                except ValueError:
                    self.group_y = None

                if self.group_x is None or self.group_y is None:
                    averages = self.average_table(self.data)
                else:
                    self.Warning.nofloat_x.clear()
                    self.Warning.nofloat_y.clear()
                    parts = []
                    domain = self.data.domain
                    for x in self.group_x.values:
                        attr_name = self.group_x
                        attr_index = domain.index(attr_name)
                        filter = data_filter.FilterDiscrete(attr_index, x)
                        filtro = []
                        filtro.append(filter)
                        filters = data_filter.Values(filtro)
                        temp = filters(self.data)
                        for y in self.group_y.values:
                            attr_name1 = self.group_y
                            attr_index1 = domain.index(attr_name1)
                            filter1 = data_filter.FilterDiscrete(attr_index1, y)
                            filtro1 = []
                            filtro1.append(filter1)
                            filters1 = data_filter.Values(filtro1)
                            full = filters1(temp)
                            v_table = self.average_table(full)
                            parts.append(v_table)


                    # Using "None" as in OWSelectRows
                    # Values is required because FilterDiscrete doesn't have
                    # negate keyword or IsDefined method
                    deffilter = Values(conditions=[FilterDiscrete(self.group_x, None)],
                                       negate=True)
                    v_table = self.average_table(deffilter(self.data))
                    parts.append(v_table)
                    averages = Orange.data.Table.concatenate(parts, axis=0)
        self.Outputs.averages.send(averages)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSNR).run(Orange.data.Table("iris"))
