import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owpreprocess_spectra import OWPreprocess, PREPROCESSORS


class TestOWPreprocess(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)

    def test_load_unload(self):
        self.send_signal("Data", Orange.data.Table("iris.tab"))
        self.send_signal("Data", None)

    def test_allpreproc_indv(self):
        data = Orange.data.Table("peach_juice.dpt")
        for i,p in enumerate(PREPROCESSORS):
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(i)
            # direct calls the preview so that exceptions do not get lost in Qt
            self.widget.show_preview()

    def test_allpreproc_indv_empty(self):
        data = Orange.data.Table("peach_juice.dpt")[:0]
        for i, p in enumerate(PREPROCESSORS):
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(i)
            self.widget.show_preview()  # direct call
        # no attributes
        data = Orange.data.Table("peach_juice.dpt")
        data = Orange.data.Table(Orange.data.Domain([],
                class_vars=data.domain.class_vars,
                metas=data.domain.metas), data)
        for i, p in enumerate(PREPROCESSORS):
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(i)
            self.widget.show_preview()  # direct call
