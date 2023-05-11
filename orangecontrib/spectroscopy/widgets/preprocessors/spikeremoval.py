from AnyQt.QtWidgets import QVBoxLayout
from Orange.widgets import gui

from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import \
    BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditDecimalOrNone
from orangecontrib.spectroscopy.preprocess import Despike


class SpikeRemovalEditor(BaseEditorOrange):
    """
    Spike Removal.
    """
    name = "Spike Removal"
    qualname = "preprocessors.spikeremoval"
    categories = ["Correction"]

    THRESHOLD = 7
    CUTOFF = 100
    DIS = 5

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.dis = self.DIS
        self.threshold = self.THRESHOLD
        self.cutoff = self.CUTOFF
        self.controlArea.setLayout(QVBoxLayout())
        box = gui.widgetBox(self.controlArea)

        self.cutoffline = lineEditDecimalOrNone(None, master=self,
                                                bottom=0, value='cutoff', default=self.CUTOFF,
                                                callback=self.edited.emit)
        self.cutoffline.setPlaceholderText(str(self.CUTOFF))
        gui.widgetLabel(box, label="Cutoff:", labelWidth=50)
        box.layout().addWidget(self.cutoffline)
        self.thresholdline = lineEditDecimalOrNone(None, master=self, bottom=0,
                                                   value='threshold', default=self.THRESHOLD,
                                                   callback=self.edited.emit)
        self.thresholdline.setPlaceholderText(str(self.THRESHOLD))
        gui.widgetLabel(box, label='Threshold:', labelWidth=60)
        box.layout().addWidget(self.thresholdline)
        self.distancespin = gui.spin(None, self, "dis", label="distance to average",
                                     minv=0, maxv=1000, callback=self.edited.emit)
        gui.widgetLabel(box, label='Distance to Average:')
        box.layout().addWidget(self.distancespin)

        self.user_changed = False

    def setParameters(self, params):
        if params:
            self.user_changed = True
        self.cutoff = params.get("cutoff", self.CUTOFF)
        self.threshold = params.get("threshold", self.THRESHOLD)
        self.dis = params.get("dis", self.DIS)

    @classmethod
    def createinstance(cls, params):
        threshold = params.get("threshold", None)
        cutoff = params.get('cutoff', None)
        if threshold is None:
            threshold = cls.THRESHOLD
        if cutoff is None:
            cutoff = cls.CUTOFF
        dis = params.get('dis', cls.DIS)
        return Despike(threshold=float(threshold), cutoff=float(cutoff), dis=dis)


preprocess_editors.register(SpikeRemovalEditor, 600)
