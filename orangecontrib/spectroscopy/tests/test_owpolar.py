import numpy as np
import Orange
from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.tests.base import WidgetTest, DummySignalManager
from orangecontrib.spectroscopy.widgets.owpolar import OWPolar
from orangewidget.widget import OWBaseWidget, Output, Input
from orangewidget.workflow.widgetsscheme import WidgetsScheme, WidgetsSignalManager
from orangewidget.workflow.tests.test_widgetsscheme import widget_description
from orangewidget.utils.signals import notify_input_helper
from AnyQt.QtCore import QObject, pyqtSignal, QElapsedTimer
from AnyQt.QtTest import QSignalSpy
import unittest
import time

class Multifile(OWBaseWidget):
    name = "Multifile"
    
    class Outputs:
        out = Output("output", Orange.data.Table)
        
class Data1(OWBaseWidget):
    name = "Data 1"
    
    class Outputs:
        out = Output("output", Orange.data.Table) 
        
class Data2(OWBaseWidget):
    name = "Data 2"
    
    class Outputs:
        out = Output("output", Orange.data.Table)     
    
class Data3(OWBaseWidget):
    name = "Data 3"
    
    class Outputs:
        out = Output("output", Orange.data.Table)
        
class Data4(OWBaseWidget):
    name = "Data 4"
    
    class Outputs:
        out = Output("output", Orange.data.Table)  

class MockIn(OWBaseWidget):
    name = "Results"
    
    class Inputs:
        polar = Input("polar data", Orange.data.Table)
        model = Input("model data", Orange.data.Table)
      
    @Inputs.polar
    def set_polar(self, dataset):
        self.polar_results = dataset
        
    @Inputs.model
    def set_model(self, dataset):
        self.model_results = dataset

class SigMan(WidgetsSignalManager, DummySignalManager):
    def __init__(self, scheme: WidgetsScheme):
        WidgetsSignalManager.__init__(self, scheme)
        DummySignalManager.__init__(self)        
  
class TestOWPolar(WidgetTest): 
    
    def reset_input_links(self):
        for j, i in enumerate(self.scheme.links):
            widget = self.scheme.widget_for_node(i.sink_node)
            inputs = vars(widget.Inputs)
            input_keys = list(inputs)

            for k in input_keys:
                if inputs[k].name == i.sink_channel.name:
                    key = k
            
            self.scheme.remove_link(i)         
            self._send_signal(widget, i.sink_channel.name, 
                              inputs[key].closing_sentinel, j)
            self.pol_widget.handleNewSignals()
            
    @classmethod
    def setUpClass(cls):
        super().setUpClass()        
        cls.scheme = WidgetsScheme()
        cls.signal_manager = SigMan(cls.scheme)
        cls.scheme.signal_manager = cls.signal_manager
        
        cls.multifile_node = cls.scheme.new_node(widget_description(Multifile))
        cls.multifile_widget = cls.scheme.widget_for_node(cls.multifile_node)
        cls.in1_node = cls.scheme.new_node(widget_description(Data1))
        cls.in1_widget = cls.scheme.widget_for_node(cls.in1_node)
        cls.in2_node = cls.scheme.new_node(widget_description(Data2))
        cls.in2_widget = cls.scheme.widget_for_node(cls.in2_node)
        cls.in3_node = cls.scheme.new_node(widget_description(Data3))
        cls.in3_widget = cls.scheme.widget_for_node(cls.in3_node)
        cls.in4_node = cls.scheme.new_node(widget_description(Data4)) 
        cls.in4_widget = cls.scheme.widget_for_node(cls.in4_node)
        cls.pol_node = cls.scheme.new_node(widget_description(OWPolar)) 
        cls.pol_widget = cls.scheme.widget_for_node(cls.pol_node)
        cls.pol_widget.signalManager = cls.scheme.signal_manager
        cls.pol_widget.__init__()
        cls.mock_in_node = cls.scheme.new_node(widget_description(MockIn))
        cls.mock_in_widget = cls.scheme.widget_for_node(cls.mock_in_node)
        cls.multifile = Orange.data.Table("polar/4-angle-ftir_multifile.tab")
        cls.in1 = Orange.data.Table("polar/4-angle-ftir_multiin1.tab")
        cls.in2 = Orange.data.Table("polar/4-angle-ftir_multiin2.tab")
        cls.in3 = Orange.data.Table("polar/4-angle-ftir_multiin3.tab")
        cls.in4 = Orange.data.Table("polar/4-angle-ftir_multiin4.tab")
        cls.multifile_polar = Orange.data.Table("polar/4-angle-ftir_multifile_polar-results.tab")
        cls.multifile_model = Orange.data.Table("polar/4-angle-ftir_multifile_model-results.tab")
        cls.multiin_polar = Orange.data.Table("polar/4-angle-ftir_multiin_polar-results.tab")
        cls.multiin_model = Orange.data.Table("polar/4-angle-ftir_multiin_model-results.tab")
        
    def test_multifile_init(self):
        print('test_multifile_init')
        self.reset_input_links()
        self.scheme.new_link(self.multifile_node, "output", self.pol_node, "Data")
        self.send_signal("Data", self.multifile, 0, widget=self.pol_widget)

        testfeats = [ft for ft in self.multifile.domain.metas 
                     if isinstance(ft, ContinuousVariable)]
        testfeats = testfeats + [ft for ft in self.multifile.domain.attributes 
                     if isinstance(ft, ContinuousVariable)]
        polfeats = [ft for ft in self.pol_widget.featureselect[:]
                    if isinstance(ft, ContinuousVariable)]
        self.assertEqual(polfeats, testfeats)
        testinputs = [inp for inp in self.multifile.domain
                      if isinstance(inp, DiscreteVariable)]
        self.assertEqual(self.pol_widget.anglemetas[:], testinputs)
        testxy = [xy for xy in self.multifile.domain.metas
                  if isinstance(xy, (ContinuousVariable, DiscreteVariable))]
        self.assertEqual(self.pol_widget.x_axis[:], testxy)
        self.assertEqual(self.pol_widget.y_axis[:], testxy)
        
    def test_multifile_in(self):
        print('test_multifile_in')
        self.reset_input_links()
        self.scheme.new_link(self.multifile_node, "output", self.pol_node, "Data")
        self.send_signal("Data", self.multifile, 0, widget=self.pol_widget)
        
        self.assertTrue(self.pol_widget.isEnabled())
        for i in self.pol_widget.multiin_labels:
            self.assertFalse(i.isEnabled())
        for i in self.pol_widget.multiin_lines:
            self.assertFalse(i.isEnabled())
        self.pol_widget.angles = self.pol_widget.anglemetas[0]
        self.assertEqual(self.pol_widget.angles, self.multifile.domain.metas[2])
        self.pol_widget._change_angles()
        self.assertEqual(len(self.pol_widget.labels), 4)
        self.assertEqual(len(self.pol_widget.lines), 4)
        self.assertEqual(self.pol_widget.polangles, list(np.linspace(0, 180, 5)[:4]))
        for i in self.pol_widget.labels:
            self.assertTrue(i.isEnabled())
        for i in self.pol_widget.lines:
            self.assertTrue(i.isEnabled())
        self.pol_widget.map_x = self.pol_widget.x_axis[0]
        self.assertEqual(self.pol_widget.map_x, self.multifile.domain.metas[0])
        self.pol_widget.map_y = self.pol_widget.y_axis[1]
        self.assertEqual(self.pol_widget.map_y, self.multifile.domain.metas[1])
        self.pol_widget.feats = [self.pol_widget.feat_view.model()[:][2], self.pol_widget.feat_view.model()[:][3]]
        self.assertEqual(self.pol_widget.feats[0], self.multifile.domain.metas[3])
        self.assertEqual(self.pol_widget.feats[1], self.multifile.domain.metas[4])
        self.pol_widget.alpha = 0
        self.pol_widget.invert_angles = True
        self.pol_widget.autocommit = True

        self.scheme.new_link(self.pol_node, "Polar Data", self.mock_in_node, "polar data")
        self.scheme.new_link(self.pol_node, "Curve Fit model data", self.mock_in_node, "model data")
        self.commit_and_wait(self.pol_widget, 20000)
        
        self.scheme.signal_manager.process_node(self.mock_in_node)
        polar = self.mock_in_widget.polar_results
        model = self.mock_in_widget.model_results
        
        np.testing.assert_equal(self.multifile_polar.metas, polar.metas)
        np.testing.assert_equal(self.multifile_polar.X, polar.X)
        np.testing.assert_equal(self.multifile_model.metas, model.metas)
        np.testing.assert_equal(self.multifile_model.X, model.X)   
                       
    def test_multi_inputs(self):
        print('test_multi_inputs')
        self.reset_input_links()
        self.scheme.new_link(self.in1_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in2_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in3_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in4_node, "output", self.pol_node, "Data")
        self.send_signal("Data", self.in1, 0, widget=self.pol_widget)
        self.send_signal("Data", self.in2, 1, widget=self.pol_widget)
        self.pol_widget.handleNewSignals()

        self.assertFalse(self.pol_widget.anglesel.isEnabled())
        for i in self.pol_widget.multiin_labels:
            self.assertFalse(i.isEnabled())
        for i in self.pol_widget.multiin_lines:
            self.assertFalse(i.isEnabled())
        self.send_signal("Data", self.in3, 2, widget=self.pol_widget)
        self.send_signal("Data", self.in4, 3, widget=self.pol_widget)
        self.pol_widget.handleNewSignals()
        self.assertFalse(self.pol_widget.anglesel.isEnabled())
        for i in self.pol_widget.multiin_labels:
            self.assertTrue(i.isEnabled())
        for i in self.pol_widget.multiin_lines:
            self.assertTrue(i.isEnabled())

        self.pol_widget.map_x = self.pol_widget.x_axis[0]
        self.assertEqual(self.pol_widget.map_x, self.in1.domain.metas[0])
        self.pol_widget.map_y = self.pol_widget.y_axis[1]
        self.assertEqual(self.pol_widget.map_y, self.in1.domain.metas[1])

        self.pol_widget.feats = [self.pol_widget.feat_view.model()[:][2], self.pol_widget.feat_view.model()[:][3]]
        self.assertEqual(self.pol_widget.feats[0], self.in1.domain.metas[2].copy(compute_value=None))
        self.assertEqual(self.pol_widget.feats[1], self.in1.domain.metas[3].copy(compute_value=None))
        self.pol_widget.alpha = 0
        self.pol_widget.invert_angles = True
        self.pol_widget.autocommit = True

        self.scheme.new_link(self.pol_node, "Polar Data", self.mock_in_node, "polar data")
        self.scheme.new_link(self.pol_node, "Curve Fit model data", self.mock_in_node, "model data")
        self.commit_and_wait(self.pol_widget, 20000)
        
        self.scheme.signal_manager.process_node(self.mock_in_node)
        polar = self.mock_in_widget.polar_results
        model = self.mock_in_widget.model_results
        multiin_polar_fixed_values = self.multiin_polar.metas[:,np.r_[0:2,3:7]]
        multiin_model_fixed_values = self.multiin_model.metas[:,np.r_[0:2,3:7]]
        multiin_polar_calc_values = self.multiin_polar.metas[:,7:]
        multiin_model_calc_values = self.multiin_model.metas[:,7:]
       
        np.testing.assert_equal(multiin_polar_fixed_values, polar.metas[:,np.r_[0:2,3:7]])
        np.testing.assert_equal(multiin_polar_calc_values, polar.metas[:,7:])
        np.testing.assert_equal(self.multiin_polar.X, np.flip(polar.X, axis=1))
        np.testing.assert_equal(multiin_model_fixed_values, model.metas[:,np.r_[0:2,3:7]])
        np.testing.assert_equal(multiin_model_calc_values, model.metas[:,7:])
        np.testing.assert_equal(self.multiin_model.X, np.flip(model.X, axis=1))
        
    def test_pixelsubset(self):
        #Test multi in with subset of pixels selected
        print('test_multi_inputs')
        self.reset_input_links()
        self.scheme.new_link(self.in1_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in2_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in3_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in4_node, "output", self.pol_node, "Data")
        rng = np.random.default_rng()
        sub_idx = rng.choice(4, size=(2), replace=False)
        subset = self.in1[sub_idx]

        self.send_signal("Data", subset, 0, widget=self.pol_widget)
        self.send_signal("Data", self.in2, 1, widget=self.pol_widget)
        self.send_signal("Data", self.in3, 2, widget=self.pol_widget)
        self.send_signal("Data", self.in4, 3, widget=self.pol_widget)
        
        self.pol_widget.map_x = self.pol_widget.x_axis[0]
        self.pol_widget.map_y = self.pol_widget.y_axis[1]
        self.pol_widget.feats = [self.pol_widget.feat_view.model()[:][2], self.pol_widget.feat_view.model()[:][3]]
        self.pol_widget.alpha = 0
        self.pol_widget.invert_angles = True
        self.pol_widget.autocommit = True       
        
        self.scheme.new_link(self.pol_node, "Polar Data", self.mock_in_node, "polar data")
        self.scheme.new_link(self.pol_node, "Curve Fit model data", self.mock_in_node, "model data")
        self.commit_and_wait(self.pol_widget, 20000)
        
        self.scheme.signal_manager.process_node(self.mock_in_node)
        polar = self.mock_in_widget.polar_results
        model = self.mock_in_widget.model_results
        
        self.assertEqual(len(polar), len(sub_idx)*4)
        self.assertEqual(len(model), len(sub_idx)*4)
    
    def test_multiin_mismatched_domain(self):
        # test multi in with different domains
        self.reset_input_links()
        self.scheme.new_link(self.in1_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in2_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in3_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in4_node, "output", self.pol_node, "Data")
        
        metadom = self.in1.domain.metas
        metadom = [i for i in metadom if type(i) is ContinuousVariable]
        attdom = self.in1.domain.attributes
        attdom = attdom[0::2]
        mismatched_domain = Domain(attdom, metas = metadom)
        mismatched_table = self.in1.transform(mismatched_domain)
        
        self.send_signal("Data", mismatched_table, 0, widget=self.pol_widget)
        self.send_signal("Data", self.in2, 1, widget=self.pol_widget)
        self.send_signal("Data", self.in3, 2, widget=self.pol_widget)
        self.send_signal("Data", self.in4, 3, widget=self.pol_widget)
        
        feat_len = len(metadom) + len(attdom) + 1
        XY_len = len(metadom)
        self.assertEqual(feat_len, len(self.pol_widget.feat_view.model()[:]))
        self.assertEqual(XY_len, len(self.pol_widget.x_axis[:]))
        self.assertEqual(XY_len, len(self.pol_widget.y_axis[:]))
        
        self.reset_input_links()
        self.scheme.new_link(self.in1_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in2_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in3_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in4_node, "output", self.pol_node, "Data")
        
        self.send_signal("Data", self.in2, 0, widget=self.pol_widget)
        self.send_signal("Data", self.in3, 1, widget=self.pol_widget)
        self.send_signal("Data", mismatched_table, 2, widget=self.pol_widget)
        self.send_signal("Data", self.in4, 3, widget=self.pol_widget)

        
        feat_len = len(metadom) + len(attdom) + 1
        XY_len = len(metadom)
        self.assertEqual(feat_len, len(self.pol_widget.feat_view.model()[:]))
        self.assertEqual(XY_len, len(self.pol_widget.x_axis[:]))
        self.assertEqual(XY_len, len(self.pol_widget.y_axis[:]))
        
    def test_custom_angles(self):
        # test inputting custom angles (multin and multifile)
        self.reset_input_links()
        self.scheme.new_link(self.multifile_node, "output", self.pol_node, "Data")
        self.send_signal("Data", self.multifile, 0, widget=self.pol_widget)
        angles = np.array([0, 22.5, 45.0, 90])
        
        for i, j in enumerate(self.pol_widget.lines):
            j.setText(str(angles[i]))
        self.pol_widget._send_angles()
        for i, j in enumerate(self.pol_widget.polangles):
            self.assertEqual(j, angles[i])       
            
        self.reset_input_links()
        self.scheme.new_link(self.in1_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in2_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in3_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in4_node, "output", self.pol_node, "Data")     
        self.send_signal("Data", self.in1, 0, widget=self.pol_widget)
        self.send_signal("Data", self.in2, 1, widget=self.pol_widget)
        self.send_signal("Data", self.in3, 2, widget=self.pol_widget)
        self.send_signal("Data", self.in4, 3, widget=self.pol_widget)
        
        for i, j in enumerate(self.pol_widget.multiin_lines):
            j.setText(str(angles[i]))
        self.pol_widget._send_ind_angles()
        for i, j in enumerate(self.pol_widget.polangles):
            self.assertEqual(j, angles[i]) 

    def test_warnings(self):
        #test all warnings
        #self.pol_widget.Warning.<NAME>.is_shown()
        self.reset_input_links()
        self.scheme.new_link(self.multifile_node, "output", self.pol_node, "Data")
        self.send_signal("Data", self.multifile, 0, widget=self.pol_widget)
        self.pol_widget.autocommit = True
        
        self.commit_and_wait(self.pol_widget)
        self.assertTrue(self.pol_widget.Warning.nofeat.is_shown())
        
        self.pol_widget.feats = [self.pol_widget.feat_view.model()[:][4]]
        self.pol_widget.map_x = None
        self.pol_widget.map_y = None
        self.commit_and_wait(self.pol_widget)
        self.assertTrue(self.pol_widget.Warning.noxy.is_shown())

        self.pol_widget.map_x = self.pol_widget.x_axis[0]
        self.pol_widget.map_y = self.pol_widget.y_axis[1]
        self.pol_widget.polangles = []
        self.commit_and_wait(self.pol_widget)
        self.assertTrue(self.pol_widget.Warning.pol.is_shown())
        self.pol_widget.polangles = [0.0,45.0,'hi',135.0]
        self.commit_and_wait(self.pol_widget)
        self.assertTrue(self.pol_widget.Warning.pol.is_shown())
        
        self.pol_widget.polangles = [0.0,45.0,90.0,135.0]
        self.pol_widget.feats = [self.pol_widget.feat_view.model()[:][0]]     
        self.commit_and_wait(self.pol_widget)
        self.assertTrue(self.pol_widget.Warning.XYfeat.is_shown())
        
        self.reset_input_links()
        self.scheme.new_link(self.in1_node, "output", self.pol_node, "Data")
        self.scheme.new_link(self.in2_node, "output", self.pol_node, "Data")   
        self.send_signal("Data", self.in1, 0, widget=self.pol_widget)
        self.send_signal("Data", self.in2, 1, widget=self.pol_widget)
        self.assertTrue(self.pol_widget.Warning.notenough.is_shown())

        self.scheme.new_link(self.in3_node, "output", self.pol_node, "Data")
        self.send_signal("Data", self.in3, 2, widget=self.pol_widget)
        self.assertTrue(self.pol_widget.Warning.notenough.is_shown())
        
        self.scheme.new_link(self.in4_node, "output", self.pol_node, "Data")  
        self.send_signal("Data", self.in4, 3, widget=self.pol_widget)
        self.assertFalse(self.pol_widget.Warning.notenough.is_shown())
        
    # def test_clearangles(self):
    #     #test clearing angles
    #     pass
        
if __name__ == "__main__":
    unittest.main()        

