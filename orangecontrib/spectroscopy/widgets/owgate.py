from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings
import numpy as np


class OWGate(OWWidget):
    name = "Gate"
    description = "A gate to control the flow of data."
    icon = "icons/gate.svg"
    id = "orangecontrib.spectroscopy.widgets.owgate"
    priority = 10
    replaces = "orangecontrib.flow.owgate"


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
        self.Warning.not_connected()


    @Inputs.data
    def setData(self, data):
        self.in_data = data

        self.Warning.not_connected()
            
        self.commit.deferred()


    @gui.deferred
    def commit(self):
        self.Warning.not_connected.clear()
        
        self.out_data = self.in_data
            
        self.Outputs.data.send(self.out_data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGate).run()
