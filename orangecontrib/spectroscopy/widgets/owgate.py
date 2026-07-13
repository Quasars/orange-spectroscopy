from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings

from orangecontrib.flow.utils import objects_equal




class OWGate(OWWidget):
    name = "Gate"
    description = "A gate to control the flow of data."
    icon = "icons/gate.svg"
    id = "orangecontrib.flow.widgets.owgate"
    priority = 10


    class Inputs:
        data = Input("Data", object, default=True, auto_summary=False)

    class Outputs:
        data = Output("Data", object, default=True, auto_summary=False)

    class Warning(OWWidget.Warning):
        not_connected = Msg("New data pending.")


    resizing_enabled = False
    want_main_area = False

    autocommit = settings.Setting(False)


    def __init__(self):
        super().__init__()

        self.in_data = None
        self.out_data = None

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")
    

    def is_connected(self):
        return OWGate.is_equal(self.in_data, self.out_data)


    @Inputs.data
    def setData(self, data):
        self.in_data = data

        self.Warning.not_connected()

        if self.is_connected():
            self.Warning.not_connected.clear()
            
        self.commit.deferred()


    @gui.deferred
    def commit(self):
        self.Warning.not_connected.clear()

        if self.is_connected():
            return
        
        self.out_data = self.in_data
            
        self.Outputs.data.send(self.out_data)


    @staticmethod
    def is_equal(table_0, table_1):
        if table_0 is None and table_1 is None:
            return True

        if table_0 is None or table_1 is None:
            return False
        
        try:
            return objects_equal(table_0, table_1)
        except:
            return False




if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGate).run()
