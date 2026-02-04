import bottleneck
import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import DomainModel
from orangecontrib.spectroscopy.utils import values_to_linspace, index_values_nan


class OWSNR(OWWidget):
    # Widget's name as displayed in the canvas
    name = "SNR"

    # Short widget description
    description = "Calculates Signal-to-Noise Ratio (SNR), Averages or Standard Deviation by coordinates."

    icon = "icons/snr.svg"
    keywords = ["signal", "noise", "standard", "deviation", "average"]

    # Define inputs and outputs
    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        final_data = Output("SNR", Orange.data.Table, default=True)

    OUT_OPTIONS = {
        'Signal-to-noise ratio': 0,  # snr
        'Average': 1,  # average
        'Standard Deviation': 2,
    }  # std

    settingsHandler = settings.DomainContextHandler()
    group_x = settings.ContextSetting(
        None, exclude_attributes=True, exclude_class_vars=True
    )
    group_y = settings.ContextSetting(
        None, exclude_attributes=True, exclude_class_vars=True
    )
    out_choiced = settings.Setting(0)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")

    def __init__(self):
        super().__init__()

        self.data = None

        self.group_x = None
        self.group_y = None

        # methods in this widget assume group axes in metas
        self.xy_model = DomainModel(
            DomainModel.METAS,
            placeholder="None",
            separators=False,
            valid_types=Orange.data.ContinuousVariable,
        )
        self.group_view_x = gui.comboBox(
            self.controlArea,
            self,
            "group_x",
            box="Select axis: x",
            model=self.xy_model,
            callback=self.grouping_changed,
        )

        self.group_view_y = gui.comboBox(
            self.controlArea,
            self,
            "group_y",
            box="Select axis: y",
            model=self.xy_model,
            callback=self.grouping_changed,
        )

        self.selected_out = gui.comboBox(
            self.controlArea,
            self,
            "out_choiced",
            box="Select Output:",
            items=self.OUT_OPTIONS,
            callback=self.out_choice_changed,
        )

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")

        # prepare interface according to the new context
        self.contextAboutToBeOpened.connect(lambda x: self.init_attr_values(x[0]))

        self.set_data(self.data)  # show warning

    def init_attr_values(self, domain):
        self.xy_model.set_domain(domain)
        self.group_x = None
        self.group_y = None

    @Inputs.data
    def set_data(self, dataset):
        self.Warning.nodata.clear()
        self.closeContext()
        self.data = dataset
        self.group = None
        if dataset is None:
            self.Warning.nodata()
        else:
            self.openContext(dataset.domain)

        self.commit.now()

    def calc_table_np(self, array):
        if len(array) == 0:
            return array
        if self.out_choiced == 0:  # snr
            return self.make_table(
                (
                    bottleneck.nanmean(array, axis=0) / bottleneck.nanstd(array, axis=0)
                ).reshape(1, -1),
                self.data,
            )
        elif self.out_choiced == 1:  # avg
            return self.make_table(
                bottleneck.nanmean(array, axis=0).reshape(1, -1), self.data
            )
        else:  # std
            return self.make_table(
                bottleneck.nanstd(array, axis=0).reshape(1, -1), self.data
            )

    @staticmethod
    def make_table(array, data_table):
        new_table = Orange.data.Table.from_numpy(
            data_table.domain,
            X=array.copy(),
            Y=np.atleast_2d(data_table.Y[0]).copy(),
            metas=np.atleast_2d(data_table.metas[0]).copy(),
        )
        cont_vars = data_table.domain.class_vars + data_table.domain.metas
        with new_table.unlocked():
            for var in cont_vars:
                index = data_table.domain.index(var)
                col = data_table.get_column(index)
                val = var.to_val(new_table[0, var])
                if not np.all(col == val):
                    new_table[0, var] = Orange.data.Unknown

        return new_table

    def grouping_changed(self):
        self.commit.deferred()

    def out_choice_changed(self):
        self.commit.deferred()

    def select_2coordinates(self, attr_x, attr_y):
        xat = self.data.domain[attr_x]
        yat = self.data.domain[attr_y]

        def extract_col(data, var):
            nd = Orange.data.Domain([var])
            d = self.data.transform(nd)
            return d.X[:, 0]

        coorx = extract_col(self.data, xat)
        coory = extract_col(self.data, yat)

        lsx = values_to_linspace(coorx)
        lsy = values_to_linspace(coory)

        xindex, xnan = index_values_nan(coorx, lsx)
        yindex, ynan = index_values_nan(coory, lsy)

        # trick:
        # https://stackoverflow.com/questions/31878240/numpy-average-of-values-corresponding-to-unique-coordinate-positions

        coo = np.hstack([xindex.reshape(-1, 1), yindex.reshape(-1, 1)])
        sortidx = np.lexsort(coo.T)
        sorted_coo = coo[sortidx]
        unqID_mask = np.append(True, np.any(np.diff(sorted_coo, axis=0), axis=1))
        ID = unqID_mask.cumsum() - 1
        unq_coo = sorted_coo[unqID_mask]
        unique, counts = np.unique(ID, return_counts=True)

        pos = 0
        bins = []
        for size in counts:
            bins.append(sortidx[pos : pos + size])
            pos += size

        matrix = []
        for indices in bins:
            selection = self.data.X[indices]
            array = self.calc_table_np(selection)
            matrix.append(array)
        table_2_coord = Orange.data.Table.concatenate(matrix, axis=0)

        with table_2_coord.unlocked():
            table_2_coord[:, attr_x] = np.linspace(*lsx)[unq_coo[:, 0]].reshape(-1, 1)
            table_2_coord[:, attr_y] = np.linspace(*lsy)[unq_coo[:, 1]].reshape(-1, 1)
        return table_2_coord

    def select_1coordinate(self, attr):
        at = self.data.domain[attr]

        def extract_col(data, var):
            nd = Orange.data.Domain([var])
            d = self.data.transform(nd)
            return d.X[:, 0]

        coor = extract_col(self.data, at)
        ls = values_to_linspace(coor)
        index, _ = index_values_nan(coor, ls)
        coo = np.hstack([index.reshape(-1, 1)])
        sortidx = np.lexsort(coo.T)
        sorted_coo = coo[sortidx]
        unqID_mask = np.append(True, np.any(np.diff(sorted_coo, axis=0), axis=1))
        ID = unqID_mask.cumsum() - 1
        unq_coo = sorted_coo[unqID_mask]
        unique, counts = np.unique(ID, return_counts=True)

        pos = 0
        bins = []
        for size in counts:
            bins.append(sortidx[pos : pos + size])
            pos += size

        matrix = []
        for indices in bins:
            selection = self.data.X[indices]
            array = self.calc_table_np(selection)
            matrix.append(array)
        table_1_coord = Orange.data.Table.concatenate(matrix, axis=0)

        with table_1_coord.unlocked():
            table_1_coord[:, attr] = np.linspace(*ls)[unq_coo[:, 0]].reshape(-1, 1)

        return table_1_coord

    def select_coordinate(self):
        if self.group_y is None and self.group_x is None:
            final_data = self.calc_table_np(self.data.X)
        elif None in [self.group_x, self.group_y]:
            if self.group_x is None:
                group = self.group_y
            else:
                group = self.group_x
            final_data = self.select_1coordinate(group)
        else:
            final_data = self.select_2coordinates(self.group_x, self.group_y)

        return final_data

    @gui.deferred
    def commit(self):
        final_data = None
        if self.data is not None:
            final_data = self.select_coordinate()

        self.Outputs.final_data.send(final_data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSNR).run(Orange.data.Table("three_coordinates_data.csv"))
