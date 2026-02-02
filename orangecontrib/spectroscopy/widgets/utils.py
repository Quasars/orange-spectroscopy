import array

import numpy as np

from Orange.data import Table
from Orange.widgets.utils.annotated_data import \
    ANNOTATED_DATA_SIGNAL_NAME, create_annotated_table, create_groups_table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Msg, Output


def pack_selection(selection_group):
    if selection_group is None:
        return None
    nonzero_indices = np.flatnonzero(selection_group)
    if len(nonzero_indices) == 0:
        return None
    if len(nonzero_indices) < 1000:
        return [(a, b) for a, b in zip(nonzero_indices, selection_group[nonzero_indices], strict=True)]
    else:
        # much faster than array.array("B", selection_group)
        a = array.array("B")
        a.frombytes(selection_group.tobytes())
        return a


def unpack_selection(saved_selection):
    """ Return an numpy array of np.uint8 representing the selection.
    The array can be smaller than the size of data."""
    if saved_selection is None or len(saved_selection) == 0:
        return np.array([], dtype=np.uint8)
    if isinstance(saved_selection, array.array):
        return np.array(saved_selection, dtype=np.uint8)
    else:  # a list of tuples
        # the size that is needed for this number of elements
        a = np.array(saved_selection)
        maxi = np.max(a[:, 0])
        r = np.zeros(maxi + 1, dtype=np.uint8)
        r[a[:, 0]] = a[:, 1]
        return r


def selections_to_length(selection_group, length):
    """
    Make selection size equal to length (add zeros or remove elements)
    """
    add_zeros = length - len(selection_group)
    if add_zeros > 0:
        return np.append(selection_group, np.zeros(add_zeros, dtype=np.uint8))
    else:
        return selection_group[:length].copy()


def groups_or_annotated_table(data, selection):
    """
    Use either Orange's create_annotated_table (for at most 1 selected class
    or create_groups_table (for more selected classes)
    :param data: Orange data table
    :param selection: classes for selected indices (0 for unselected)
    :return: Orange data table with an added column
    """
    if len(selection) and np.max(selection) > 1:
        return create_groups_table(data, selection)
    else:
        return create_annotated_table(data, np.flatnonzero(selection))


class SelectionGroupMixin:
    selection_group_saved = Setting(None, schema_only=True)

    def __init__(self):
        self.selection_group = np.array([], dtype=np.uint8)
        # Remember the saved state to restore with the first open file
        self._pending_selection_restore = self.selection_group_saved

    def restore_selection_settings(self):
        self.selection_group = unpack_selection(self._pending_selection_restore)
        self.selection_group = selections_to_length(self.selection_group, len(self.data))
        self._pending_selection_restore = None

    def prepare_settings_for_saving(self):
        self.selection_group_saved = pack_selection(self.selection_group)


class SelectionOutputsMixin:

    # older versions did not include the "Group" feature
    # fot the selected output
    compat_no_group = Setting(False, schema_only=True)

    class Outputs:
        selected_data = Output("Selection", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Information(OWWidget.Information):
        compat_no_group = Msg("Compatibility mode: Selection does not output Groups.")

    def __init__(self):
        if self.compat_no_group:
            # We decided not to show the warning explicitly
            # self.Information.compat_no_group()
            pass

    def _send_selection(self, data, selection_group, no_group=False):
        annotated_data = groups_or_annotated_table(data, selection_group)
        self.Outputs.annotated_data.send(annotated_data)

        selected = None
        if data:
            if no_group and data:  # compatibility mode, the output used to lack the group column
                selection_indices = np.flatnonzero(selection_group)
                selected = data[selection_indices]
            else:
                selected = create_groups_table(data,
                                               selection_group, False, "Group")
        selected = selected if selected else None
        self.Outputs.selected_data.send(selected if selected else None)

        return annotated_data, selected

    def send_selection(self, data, selection_group):
        return self._send_selection(data, selection_group,
                                    no_group=self.compat_no_group)
