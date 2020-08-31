import numpy as np

import Orange.data
from Orange.data.filter import SameValue, FilterDiscrete, Values
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import DomainModel


class OWSNR(OWWidget):
    # Widget's name as displayed in the canvas
    name = "snr"

    # Short widget description
    description = (
        "Calculates SNR, averages or Standard Deviation.")

    icon = "icons/snr.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        final_data = Output("SNR", Orange.data.Table, default=True)

    OUT_OPTIONS = {'Signal-to-noise ratio': 0, #snr
                   'Average': 1, # average
                   'Standard Deviation': 2} # std

    settingsHandler = settings.DomainContextHandler()
    group_x = settings.ContextSetting(None)
    group_y = settings.ContextSetting(None)
    group = settings.ContextSetting(None)
    out_choiced = settings.ContextSetting(0)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")

    def __init__(self):
        super().__init__()

        self.data = None
        self.set_data(self.data)  # show warning

        self.group_axis_x = DomainModel(
            placeholder="None", separators=False,
            valid_types=Orange.data.DiscreteVariable)
        self.group_view_x = gui.comboBox(
            self.controlArea, self, "group_x", box="Select axis: x",
            model=self.group_axis_x, callback=self.grouping_changed)

        self.group_axis_y = DomainModel(
            placeholder="None", separators=False,
            valid_types=Orange.data.DiscreteVariable)
        self.group_view_y = gui.comboBox(
            self.controlArea, self, "group_y", box="Select axis: y",
            model=self.group_axis_y, callback=self.grouping_changed)

        self.selected_out = gui.comboBox(
            self.controlArea, self, "out_choiced", box="Select Output:",
            items=self.OUT_OPTIONS, callback=self.out_choice_changed)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")


    @Inputs.data
    def set_data(self, dataset):
        self.Warning.nodata.clear()
        self.closeContext()
        self.data = dataset
        self.group_x = None
        self.group_y = None
        self.group = None
        if dataset is None:
            self.Warning.nodata()
        else:
            self.group_axis_x.set_domain(dataset.domain)
            self.group_axis_y.set_domain(dataset.domain)
            self.openContext(dataset.domain)

        self.commit()

    def calc_table(self, table):
        if len(table) == 0:
            return table
        if self.out_choiced == 0: #snr
            return self.make_table(np.nanmean(table.X, axis=0, keepdims=True) / np.std(table.X, axis=0, keepdims=True), table)
        elif self.out_choiced == 1: #avg
            return self.make_table(np.nanmean(table.X, axis=0, keepdims=True), table)
        else: # std
            return self.make_table(np.std(table.X, axis=0, keepdims=True), table)

    @staticmethod
    def make_table(data, table):

        new_table = Orange.data.Table.from_numpy(table.domain,
                                                 X=data,
                                                 Y=np.atleast_2d(table.Y[0].copy()),
                                                 metas=np.atleast_2d(table.metas[0].copy()))
        cont_vars = [var for var in table.domain.class_vars + table.domain.metas
                     if isinstance(var, Orange.data.ContinuousVariable)]
        for var in cont_vars:
            index = table.domain.index(var)
            col, _ = table.get_column_view(index)
            try:
                new_table[0, index] = np.nanmean(col)
            except AttributeError:
                # numpy.lib.nanfunctions._replace_nan just guesses and returns
                # a boolean array mask for object arrays because object arrays
                # do not support `isnan` (numpy-gh-9009)
                # Since we know that ContinuousVariable values must be np.float64
                # do an explicit cast here
                new_table[0, index] = np.nanmean(col, dtype=np.float64)

        other_vars = [var for var in table.domain.class_vars + table.domain.metas
                      if not isinstance(var, Orange.data.ContinuousVariable)]
        for var in other_vars:
            index = table.domain.index(var)
            col, _ = table.get_column_view(index)
            val = var.to_val(new_table[0, var])
            if not np.all(col == val):
                new_table[0, var] = Orange.data.Unknown

        return new_table

    def grouping_changed(self):
        """Calls commit() indirectly to respect auto_commit setting."""
        self.commit()
       
    def out_choice_changed(self):
        self.commit()

    def select_2coordinates(self):
        parts = []
        for x in self.group_x.values:
            svfilter_x = SameValue(self.group_x, x)
            data_x = svfilter_x(self.data)
            for y in self.group_y.values:
                svfilter_y = SameValue(self.group_y, y)
                data_y = svfilter_y(data_x)

                v_table = self.calc_table(data_y)
                parts.append(v_table)

        table_2_coord = Orange.data.Table.concatenate(parts, axis=0)
        return table_2_coord

    def select_1coordinate(self):
        parts = []
        for value in self.group.values:
            svfilter = SameValue(self.group, value)
            v_table = self.calc_table(svfilter(self.data))
            parts.append(v_table)
        table_1_coord = Orange.data.Table.concatenate(parts, axis=0)
        return table_1_coord
    
    def select_coordinate(self):
        if self.group_y is None and self.group_x is None:
            final_data = self.calc_table(self.data)
        elif None in [self.group_x, self.group_y]:
            if self.group_x is None:
                self.group = self.group_y
            else:
                self.group = self.group_x
            final_data = self.select_1coordinate()
        else:
            final_data = self.select_2coordinates()

        return final_data

    def commit(self):
        final_data = None
        if self.data is not None:
            final_data = self.select_coordinate()

        self.Outputs.final_data.send(final_data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    folder = """/home/levandoski/Documentos/ic-orange/\
interface_orange/orange-spectroscopy/\
orangecontrib/spectroscopy/datasets/"""
    file_name = "three_coordinates_data.csv"
    path = folder + file_name ### TODO open "three coordinates data.csv" without indicating the folder
    WidgetPreview(OWSNR).run(Orange.data.Table(path))
