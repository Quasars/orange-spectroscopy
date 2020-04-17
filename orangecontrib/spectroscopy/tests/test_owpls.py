from orangecontrib.spectroscopy.widgets.owpls import OWPLS
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, ParameterMapping


class TestOWPLS(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWPLS,
                                         stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            ParameterMapping('iters', self.widget.n_iters),
            ParameterMapping('n_components', self.widget.ncomps_spin)]