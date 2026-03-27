"""
Example for orangecontrib.spectroscopy.widgets.gui.connect_settings

In this example widget, two settings are connected according to a function
that depends on another setting.
"""

import sys
from decimal import Decimal

import Orange
from Orange.widgets.widget import OWWidget

from orangecontrib.spectroscopy.widgets.gui import (
    connect_settings,
    lineEditFloatRange,
    ValueTransform,
    connect_line,
    MovableVline,
)
from orangecontrib.spectroscopy.widgets.owspectra import CurvePlot


class PlusAdd(ValueTransform):
    def __init__(self, contains_add):
        self.contains_add = contains_add

    def transform(self, v):
        return v + self.contains_add.add

    def inverse(self, v):
        return v - self.contains_add.add


class TestConnected(OWWidget):
    name = "Test"

    want_main_area = True

    def __init__(self):
        super().__init__()

        self.v = Decimal(1111)
        le = lineEditFloatRange(self, self, "v")
        self.controlArea.layout().addWidget(le)

        self.vplus = Decimal(0)
        le100 = lineEditFloatRange(self, self, "vplus")
        self.controlArea.layout().addWidget(le100)

        self.add = Decimal(100)

        def refresh():
            self.v = self.v  # just set one main so that dependant values are refreshed
            # self.vplus = self.vplus  # this would have the opposite effect

        leadd = lineEditFloatRange(self, self, "add", callback=refresh)

        self.controlArea.layout().addWidget(leadd)

        connect_settings(self, "v", "vplus", transform=PlusAdd(self))

        #
        # add MovableVLines ajd test this with spectra
        #

        self.curveplot = CurvePlot(self)
        self.mainArea.layout().addWidget(self.curveplot)
        self.curveplot.set_data(Orange.data.Table("collagen"))

        self.line1 = MovableVline(label="Value")
        connect_line(self.line1, self, "v")
        self.line2 = MovableVline(label="Value + something")
        connect_line(self.line2, self, "vplus")

        self.curveplot.add_marking(self.line1)
        self.curveplot.add_marking(self.line2)


def main():
    from AnyQt.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ow = TestConnected()
    ow.show()
    ow.raise_()
    app.exec_()
    return 0


if __name__ == "__main__":
    sys.exit(main())
