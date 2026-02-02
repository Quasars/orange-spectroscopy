import os
import unittest
from unittest.mock import Mock, patch

try:
    import dask
    from Orange.tests.test_dasktable import temp_dasktable
except ImportError:
    dask = None

from AnyQt.QtCore import QRectF, Qt
from AnyQt.QtGui import QFont
from AnyQt.QtTest import QSignalSpy
import numpy as np
import pyqtgraph as pg

from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table, Domain, DiscreteVariable

from orangecontrib.spectroscopy.widgets.owspectra import OWSpectra, MAX_INSTANCES_DRAWN, \
    PlotCurvesItem, NoSuchCurve, MAX_THICK_SELECTED, CurvePlot
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.line_geometry import intersect_curves, \
    distance_line_segment
from orangecontrib.spectroscopy.tests.util import hold_modifiers, set_png_graph_save
from orangecontrib.spectroscopy.preprocess import Interpolate


NAN = float("nan")


def wait_for_graph(widget, timeout=5000):
    if not isinstance(widget, CurvePlot):
        widget = widget.curveplot
    av = widget.show_average_thread
    ind = widget.show_individual_thread
    if av.task is not None or ind.task is not None:
        spy = QSignalSpy(widget.graph_shown)
        assert spy.wait(timeout), "Failed to update graph in the specified timeout"


class TestOWSpectra(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.titanic = Table("titanic")
        cls.collagen = Table("collagen")
        cls.normal_data = [cls.iris, cls.collagen]
        # dataset with a single attribute
        iris1 = cls.iris.transform(Domain(cls.iris.domain[:1]))
        iris1.name = "iris1"
        # dataset without any attributes
        iris0 = cls.iris.transform(Domain([]))
        iris0.name = "iris0"
        # data set with no lines
        empty = cls.iris[:0]
        empty.name = "empty"
        # dataset with large blank regions
        irisunknown = Interpolate(np.arange(20))(cls.iris)
        irisunknown.name = "irisunknown"
        cls.unknown_last_instance = cls.iris.copy()
        with cls.unknown_last_instance.unlocked():
            cls.unknown_last_instance.X[73] = NAN  # needs to be unknown after sampling and permutation
        cls.unknown_last_instance.name = "unknown_last_instance"
        # dataset with mixed unknowns
        cls.unknown_pts = cls.collagen.copy()
        with cls.unknown_pts.unlocked():
            cls.unknown_pts[5] = np.nan
            cls.unknown_pts[8:10] = np.nan
            cls.unknown_pts[15] = np.inf
        cls.unknown_pts.name = "unknown_pts"
        # a data set with only infs
        cls.only_inf = iris1.copy()
        with cls.only_inf.unlocked():
            cls.only_inf.X *= np.inf
        cls.only_inf.name = "only_inf"
        cls.strange_data = [iris1, iris0, empty, irisunknown, cls.unknown_last_instance,
                            cls.only_inf, cls.unknown_pts]

    def setUp(self):
        self.widget = self.create_widget(OWSpectra)  # OWSpectra

    def test_PlotCurvesItem_bounds(self):
        pc = PlotCurvesItem()
        # test defaults
        np.testing.assert_equal(pc.boundingRect(), QRectF(0, 0, 1, 1))
        pc.add_curve(pg.PlotCurveItem(x=[0, 1], y=[NAN, NAN]))
        np.testing.assert_equal(pc.boundingRect(), QRectF(0, 0, 1, 1))
        pc.add_curve(pg.PlotCurveItem(x=[-1, 2], y=[NAN, NAN]))
        np.testing.assert_equal(pc.boundingRect(), QRectF(-1, 0, 3, 1))
        # valid y values should overwrite the defaults
        pc.add_curve(pg.PlotCurveItem(x=[-1, 2], y=[0.1, 0.2]))
        np.testing.assert_equal(pc.boundingRect(), QRectF(-1, 0.1, 3, 0.1))

    def test_is_last_instance_force_sampling_and_permutation(self):
        mi = "orangecontrib.spectroscopy.widgets.owspectra.MAX_INSTANCES_DRAWN"
        with patch(mi, 100):
            self.send_signal("Data", self.unknown_last_instance)
            wait_for_graph(self.widget)
            self.assertTrue(np.all(np.isnan(self.unknown_last_instance[self.widget.curveplot.sampled_indices].X[-1])))

    def do_mousemove(self):
        mr = self.widget.curveplot.MOUSE_RADIUS
        sbr = self.widget.curveplot.plot.sceneBoundingRect()
        self.widget.curveplot.MOUSE_RADIUS = 1000
        self.widget.curveplot.mouse_moved_closest(
            (sbr.center(),))
        if self.widget.curveplot.data is not None \
                and np.any(np.isfinite(self.widget.curveplot.data.X)):  # a valid curve exists
            self.assertIsNotNone(self.widget.curveplot.highlighted)
        else:  # no curve can be detected
            self.assertIsNone(self.widget.curveplot.highlighted)

        # assume nothing is at the chosen point
        # therefore nothing should be highlighted
        nothing_point = (sbr.center() + sbr.topLeft())/2
        self.widget.curveplot.MOUSE_RADIUS = 5  # locally, test_mouse_move failed with 30
        self.widget.curveplot.mouse_moved_closest(
            (nothing_point,))
        self.assertIsNone(self.widget.curveplot.highlighted)

        self.widget.curveplot.MOUSE_RADIUS = mr

    def test_average(self):
        for data in self.normal_data + self.strange_data:
            self.send_signal("Data", data)
            self.widget.curveplot.show_average()
            wait_for_graph(self.widget)

    def test_empty(self):
        self.send_signal("Data", None)
        self.do_mousemove()

    def test_mouse_move(self):
        for data in self.normal_data + self.strange_data:
            with self.subTest(data.name):
                self.send_signal("Data", data)
                wait_for_graph(self.widget)
                self.do_mousemove()

    def test_rescale_y(self):
        for data in self.normal_data + self.strange_data:
            self.send_signal("Data", data)
            self.widget.curveplot.rescale_current_view_y()

    def simulate_drag(self, vb, from_, to_):
        # from_ and to_ are in plot coordinates
        event = Mock()
        event.button.return_value = Qt.LeftButton
        mapToParent = vb.childGroup.mapToParent
        event.pos.return_value = mapToParent(to_)
        event.lastPos.return_value = mapToParent(from_)
        event.buttonDownPos.return_value = mapToParent(from_)
        event.isFinish.return_value = True
        vb.mouseDragEvent(event)

    def select_diagonal(self):
        vb = self.widget.curveplot.plot.vb
        vb.set_mode_select()
        pos_down, pos_up = vb.viewRect().topLeft(), vb.viewRect().bottomRight()
        self.simulate_drag(vb, pos_down, pos_up)

    def test_select_line(self):
        for data in self.normal_data:
            self.send_signal("Data", data)
            wait_for_graph(self.widget)
            out = self.get_output("Selection")
            self.assertIsNone(out, None)
            out = self.get_output(self.widget.Outputs.annotated_data)
            sa = out.transform(Domain([out.domain["Selected"]]))
            np.testing.assert_equal(sa.X, 0)
            self.select_diagonal()
            out = self.get_output("Selection")
            self.assertEqual(len(data), len(out))
            out = self.get_output(self.widget.Outputs.annotated_data)
            self.assertEqual(len(data), len(out))
            sa = out.transform(Domain([out.domain["Selected"]]))
            np.testing.assert_equal(sa.X, 1)

    def do_zoom_rect(self, invertX):
        """ Test zooming with two clicks. """
        self.send_signal("Data", self.iris)
        vb = self.widget.curveplot.plot.vb
        self.widget.curveplot.invertX = invertX
        self.widget.curveplot.invertX_apply()
        vb.set_mode_zooming()
        vr = vb.viewRect()
        tlw = vr.bottomRight() if self.widget.curveplot.invertX else vr.bottomLeft()
        brw = vr.center()
        self.simulate_drag(vb, tlw, brw)
        vr = vb.viewRect()
        self.assertAlmostEqual(vr.bottom(), tlw.y())
        self.assertAlmostEqual(vr.top(), brw.y())
        if self.widget.curveplot.invertX:
            self.assertAlmostEqual(vr.right(), tlw.x())
            self.assertAlmostEqual(vr.left(), brw.x())
        else:
            self.assertAlmostEqual(vr.left(), tlw.x())
            self.assertAlmostEqual(vr.right(), brw.x())

    def test_zoom_rect(self):
        self.do_zoom_rect(invertX=False)
        self.do_zoom_rect(invertX=True)

    def test_warning_no_x(self):
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Warning.no_x.is_shown())
        self.send_signal("Data", self.strange_data[1])
        self.assertTrue(self.widget.Warning.no_x.is_shown())
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Warning.no_x.is_shown())

    def test_information(self):
        assert len(self.titanic) > MAX_INSTANCES_DRAWN
        self.send_signal("Data", self.titanic[:MAX_INSTANCES_DRAWN])
        wait_for_graph(self.widget)
        self.assertFalse(self.widget.Information.showing_sample.is_shown())
        self.send_signal("Data", self.titanic)
        wait_for_graph(self.widget)
        self.assertTrue(self.widget.Information.showing_sample.is_shown())
        self.send_signal("Data", self.titanic[:MAX_INSTANCES_DRAWN])
        wait_for_graph(self.widget)
        self.assertFalse(self.widget.Information.showing_sample.is_shown())

    def test_information_average_mode(self):
        self.send_signal("Data", self.iris[:100])
        self.assertFalse(self.widget.Information.showing_sample.is_shown())
        self.widget.curveplot.show_average()
        wait_for_graph(self.widget)
        self.send_signal("Data", self.iris[:99])
        self.assertFalse(self.widget.Information.showing_sample.is_shown())

    def test_handle_floatname(self):
        self.send_signal("Data", self.collagen)
        wait_for_graph(self.widget)
        x, _ = self.widget.curveplot.curves[0]
        fs = sorted([float(f.name) for f in self.collagen.domain.attributes])
        np.testing.assert_equal(x, fs)

    def test_handle_nofloatname(self):
        self.send_signal("Data", self.iris)
        wait_for_graph(self.widget)
        x, _ = self.widget.curveplot.curves[0]
        np.testing.assert_equal(x, range(len(self.iris.domain.attributes)))

    def test_show_average(self):
        def numcurves(curves):
            return sum(len(a[1]) for a in curves)

        self.send_signal("Data", self.iris)
        curves_plotted = self.widget.curveplot.curves_plotted
        wait_for_graph(self.widget)
        self.assertEqual(numcurves(curves_plotted), 150)
        self.widget.curveplot.show_average()
        wait_for_graph(self.widget)
        curves_plotted = self.widget.curveplot.curves_plotted
        self.assertEqual(numcurves(curves_plotted), 3)
        self.widget.curveplot.show_individual()
        wait_for_graph(self.widget)
        curves_plotted = self.widget.curveplot.curves_plotted
        self.assertEqual(numcurves(curves_plotted), 150)

    def test_line_intersection(self):
        data = self.collagen
        x = getx(data)
        sort = np.argsort(x)
        x = x[sort]
        ys = data.X[:, sort]
        boola = intersect_curves(x, ys, np.array([0, 1.15]), np.array([3000, 1.15]))
        intc = np.asarray(np.flatnonzero(boola))
        np.testing.assert_equal(intc, [191, 635, 638, 650, 712, 716, 717, 726])

    def test_line_point_distance(self):
        # nan in point
        a = distance_line_segment(np.array([0, 0]), np.array([0, float("nan")]), 10, 10, 5, 5)
        np.testing.assert_equal(a, [0, float("nan")])

        # distance to the middle of the line segment
        a = distance_line_segment(np.array(0), 0, 10, 10, 10, 0)
        self.assertEqual(a, 50**0.5)

        # equal endpoints
        a = distance_line_segment(np.array(0), 0, 0, 0, 10, 0)
        self.assertEqual(a, 10)

    def test_grid(self):
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.curveplot.show_grid)
        self.widget.curveplot.grid_changed()
        self.assertTrue(self.widget.curveplot.show_grid)

    def test_subset(self):
        data = self.titanic
        assert len(data) > MAX_INSTANCES_DRAWN

        self.send_signal("Data", data)
        wait_for_graph(self.widget)
        sinds = self.widget.curveplot.sampled_indices
        self.assertEqual(len(sinds), MAX_INSTANCES_DRAWN)

        # the whole subset is drawn
        add_subset = data[:MAX_INSTANCES_DRAWN]
        self.send_signal("Data subset", add_subset)
        wait_for_graph(self.widget)
        sinds = self.widget.curveplot.sampled_indices
        self.assertTrue(set(add_subset.ids) <= set(data[sinds].ids))

        # the whole subset can not be drawn anymore
        add_subset = data[:MAX_INSTANCES_DRAWN+1]
        self.send_signal("Data subset", add_subset)
        wait_for_graph(self.widget)
        sinds = self.widget.curveplot.sampled_indices
        self.assertFalse(set(add_subset.ids) <= set(data[sinds].ids))

    def test_subset_connect_disconnect(self):
        self.send_signal("Data", self.collagen)
        self.assertFalse(np.any(self.widget.curveplot.subset_indices))
        self.send_signal("Data subset", self.collagen[:1])
        self.assertTrue(self.widget.curveplot.subset_indices[0])
        self.assertFalse(np.any(self.widget.curveplot.subset_indices[1:]))
        # connect some other data set
        self.send_signal("Data", self.iris)
        self.assertFalse(np.any(self.widget.curveplot.subset_indices))
        # reconnecting correct data set should have the same subset
        self.send_signal("Data", self.collagen)
        self.assertTrue(self.widget.curveplot.subset_indices[0])
        self.assertFalse(np.any(self.widget.curveplot.subset_indices[1:]))

    def test_subset_first(self):
        self.send_signal("Data subset", self.collagen[:1])
        self.send_signal("Data", self.collagen)
        self.assertTrue(self.widget.curveplot.subset_indices[0])
        self.assertFalse(np.any(self.widget.curveplot.subset_indices[1:]))

    def test_settings_color(self):
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.feature_color, None)
        self.widget.curveplot.feature_color = self.iris.domain.class_var
        iris_context = self.widget.settingsHandler.pack_data(self.widget)["context_settings"]
        self.send_signal("Data", self.titanic)
        self.assertEqual(self.widget.curveplot.feature_color, None)
        # because previous settings match any domain, use only context for iris
        self.widget = self.create_widget(OWSpectra,
                                         stored_settings={"context_settings": iris_context})
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.feature_color.name, "iris")

    def test_cycle_color(self):
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.feature_color, None)
        self.widget.curveplot.cycle_color_attr()
        self.assertEqual(self.widget.curveplot.feature_color, self.iris.domain.class_var)
        self.widget.curveplot.cycle_color_attr()
        self.assertEqual(self.widget.curveplot.feature_color, None)

    def test_color_individual(self):
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.color_individual, False)
        self.widget.curveplot.color_individual_changed()
        self.assertEqual(self.widget.curveplot.color_individual, True)
        # colorbrewer set1 has 9 colors, so we should have 9 possible pens
        self.assertEqual(len(self.widget.curveplot.pen_selected), 9)

    def test_open_selection(self):
        # saved selection in the file should be reloaded
        self.widget = self.create_widget(
            OWSpectra, stored_settings={"curveplot": {"selection_group_saved": [(0, 1)]}}
        )
        self.send_signal("Data", self.iris)
        out = self.get_output("Selection")
        self.assertEqual(out[0], self.iris[0])

    def test_migrate_selection(self):
        settings = {"curveplot": {"selected_indices": set([0])}}
        OWSpectra.migrate_settings(settings, 0)
        self.assertEqual(settings["curveplot"]["selection_group_saved"], [(0, 1)])

    def test_selection_changedata(self):
        # select something in the widget and see if it is cleared
        self.send_signal("Data", self.iris)
        wait_for_graph(self.widget)
        self.widget.curveplot.MOUSE_RADIUS = 1000
        self.widget.curveplot.mouse_moved_closest(
            (self.widget.curveplot.plot.sceneBoundingRect().center(),))
        self.widget.curveplot.select_by_click(None)
        out = self.get_output("Selection")
        self.assertEqual(len(out), 1)
        # resending the exact same data should not change the selection
        self.send_signal("Data", self.iris)
        wait_for_graph(self.widget)
        out2 = self.get_output("Selection")  # noqa: F841
        self.assertEqual(len(out), 1)
        # while resending the same data as a different object should
        self.send_signal("Data", self.iris.copy())
        wait_for_graph(self.widget)
        out = self.get_output("Selection")
        self.assertIsNone(out, None)

    def test_select_click_multiple_groups(self):
        data = self.collagen[:100]
        self.send_signal("Data", data)
        self.widget.curveplot.make_selection([1])
        with hold_modifiers(self.widget, Qt.ShiftModifier):
            self.widget.curveplot.make_selection([2])
        with hold_modifiers(self.widget, Qt.ShiftModifier):
            self.widget.curveplot.make_selection([3])
        with hold_modifiers(self.widget, Qt.ControlModifier):
            self.widget.curveplot.make_selection([4])
        out = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(out), 100)  # have a data table at the output
        newvars = out.domain.variables + out.domain.metas
        oldvars = data.domain.variables + data.domain.metas
        group_at = [a for a in newvars if a not in oldvars][0]
        unselected = group_at.to_val("Unselected")
        out = out[np.flatnonzero(out.transform(Domain([group_at])).X != unselected)]
        self.assertEqual(len(out), 4)
        np.testing.assert_equal([o for o in out], [data[i] for i in [1, 2, 3, 4]])
        np.testing.assert_equal([o[group_at].value for o in out], ["G1", "G2", "G3", "G3"])
        out = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_equal([o[out.domain["Group"]].value for o in out],
                                ["G1", "G2", "G3", "G3"])

        # remove one element
        with hold_modifiers(self.widget, Qt.AltModifier):
            self.widget.curveplot.make_selection([1])
        out = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_equal(len(out), 3)
        np.testing.assert_equal([o for o in out], [data[i] for i in [2, 3, 4]])
        np.testing.assert_equal([o[out.domain["Group"]].value for o in out],
                                ["G2", "G3", "G3"])

    def test_select_thick_lines(self):
        data = self.collagen[:100]
        assert MAX_INSTANCES_DRAWN >= len(data) > MAX_THICK_SELECTED
        self.send_signal("Data", data)
        wait_for_graph(self.widget)
        self.widget.curveplot.make_selection(list(range(MAX_THICK_SELECTED)))
        self.assertEqual(2, self.widget.curveplot.pen_selected[None].width())
        self.widget.curveplot.make_selection(list(range(MAX_THICK_SELECTED + 1)))
        self.assertEqual(1, self.widget.curveplot.pen_selected[None].width())
        self.widget.curveplot.make_selection(list(range(MAX_THICK_SELECTED)))
        self.assertEqual(2, self.widget.curveplot.pen_selected[None].width())

    def test_select_thick_lines_threshold(self):
        data = self.collagen[:100]
        assert MAX_INSTANCES_DRAWN >= len(data) > MAX_THICK_SELECTED
        threshold = MAX_THICK_SELECTED
        self.send_signal("Data", data)
        wait_for_graph(self.widget)
        set_curve_pens = 'orangecontrib.spectroscopy.widgets.owspectra.CurvePlot.set_curve_pens'
        with patch(set_curve_pens, Mock()) as m:

            def clen():
                return len(m.call_args[0][0])

            self.widget.curveplot.make_selection(list(range(threshold - 1)))
            self.assertEqual(threshold - 1, clen())
            with hold_modifiers(self.widget, Qt.ControlModifier):
                self.widget.curveplot.make_selection([threshold])
            self.assertEqual(1, clen())
            with hold_modifiers(self.widget, Qt.ControlModifier):
                self.widget.curveplot.make_selection([threshold + 1])
            self.assertEqual(threshold + 1, clen())  # redraw curves as thin
            with hold_modifiers(self.widget, Qt.ControlModifier):
                self.widget.curveplot.make_selection([threshold + 2])
            self.assertEqual(1, clen())
            with hold_modifiers(self.widget, Qt.AltModifier):
                self.widget.curveplot.make_selection([threshold + 2])
            self.assertEqual(1, clen())
            with hold_modifiers(self.widget, Qt.AltModifier):
                self.widget.curveplot.make_selection([threshold + 1])
            self.assertEqual(threshold + 1, clen())  # redraw curves as thick

    def test_unknown_feature_color(self):
        data = self.iris
        with data.unlocked():
            data[0][data.domain.class_var] = np.nan
        self.send_signal("Data", data)
        self.widget.curveplot.cycle_color_attr()
        self.assertEqual(self.widget.curveplot.feature_color, data.domain.class_var)
        self.widget.curveplot.show_average()
        wait_for_graph(self.widget)

    def test_curveplot_highlight(self):
        data = self.iris[:10]
        curveplot = self.widget.curveplot
        self.send_signal("Data", data)
        wait_for_graph(self.widget)
        self.assertIsNone(curveplot.highlighted)

        m = Mock()
        curveplot.highlight_changed.connect(m)
        curveplot.highlight(4)
        self.assertEqual(curveplot.highlighted, 4)
        self.assertEqual(m.call_count, 1)

        curveplot.highlight(0)
        curveplot.highlight(9)
        with self.assertRaises(NoSuchCurve):
            curveplot.highlight(-1)
        with self.assertRaises(NoSuchCurve):
            curveplot.highlight(10)

    def test_select_at_least_1(self):
        self.widget.curveplot.select_at_least_1 = True
        self.send_signal(self.widget.Inputs.data, self.iris[:3])
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(1, len(selected))
        self.assertEqual(self.iris[0], selected[0])
        self.send_signal(self.widget.Inputs.data, self.iris[:2])
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(1, len(selected))
        self.assertEqual(self.iris[0], selected[0])

    def test_new_data_clear_graph(self):
        curveplot = self.widget.curveplot
        curveplot.set_data(self.iris[:3], auto_update=False)
        curveplot.update_view()
        wait_for_graph(self.widget)
        self.assertEqual(3, len(curveplot.curves[0][1]))
        curveplot.set_data(self.iris[:3], auto_update=False)
        self.assertEqual([], curveplot.curves)
        curveplot.update_view()
        wait_for_graph(self.widget)
        self.assertEqual(3, len(curveplot.curves[0][1]))

    def test_save_graph(self):
        self.send_signal("Data", self.iris)
        wait_for_graph(self.widget)
        with set_png_graph_save() as fname:
            self.widget.save_graph()
            self.assertGreater(os.path.getsize(fname), 10000)

    def test_peakline_keep_precision(self):
        self.widget.curveplot.add_peak_label()  # precision 1 (no data)
        self.send_signal("Data", self.collagen)
        wait_for_graph(self.widget)
        self.widget.curveplot.add_peak_label()  # precision 2 (data range is collagen)
        label_text = self.widget.curveplot.peak_labels[0].label.textItem.toPlainText()
        label_text2 = self.widget.curveplot.peak_labels[1].label.textItem.toPlainText()
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.widget = self.create_widget(OWSpectra, stored_settings=settings)
        label_text_loaded = self.widget.curveplot.peak_labels[0].label.textItem.toPlainText()
        label_text2_loaded = self.widget.curveplot.peak_labels[1].label.textItem.toPlainText()
        self.assertEqual(label_text, label_text_loaded)
        self.assertEqual(label_text2, label_text2_loaded)

    def test_peakline_remove(self):
        self.widget.curveplot.add_peak_label()
        self.send_signal("Data", self.collagen)
        wait_for_graph(self.widget)
        self.widget.curveplot.add_peak_label()
        self.widget.curveplot.peak_labels[0].request_deletion()
        self.assertEqual(len(self.widget.curveplot.peak_labels), 1)
        info = self.widget.curveplot.peak_labels[0].save_info()
        self.assertAlmostEqual(info[0], 1351.9123)
        self.widget.curveplot.peak_labels[0].request_deletion()
        self.assertEqual(len(self.widget.curveplot.peak_labels), 0)

    def test_filter_unknowns(self):
        self.send_signal("Data", self.unknown_pts)
        curves_cont = self.widget.curveplot.curves_cont
        for obj in curves_cont.objs:
            self.assertTrue(np.isfinite(obj.yData).all())
            self.assertTrue(np.isfinite(obj.xData).all())

    def test_waterfall(self):
        self.send_signal("Data", self.iris)
        vb = self.widget.curveplot.plot.vb
        self.widget.curveplot.waterfall = False
        self.widget.curveplot.update_view()
        wait_for_graph(self.widget)
        self.assertTrue(vb.targetRect().bottom() < 10)
        self.widget.curveplot.waterfall_changed()
        wait_for_graph(self.widget)
        self.assertTrue(vb.targetRect().bottom() < 1000)
        self.assertTrue(vb.targetRect().bottom() > 800)

    def test_migrate_context_feature_color(self):
        # avoid class_vars in tests, because the setting does not allow them
        iris = self.iris.transform(
            Domain(self.iris.domain.attributes, None, self.iris.domain.class_vars))
        c = self.widget.settingsHandler.new_context(iris.domain, None, iris.domain.metas)
        c.values["curveplot"] = {"feature_color": ("iris", 1)}
        self.widget = self.create_widget(OWSpectra,
                                         stored_settings={"context_settings": [c]})
        self.send_signal("Data", iris)
        self.assertIsInstance(self.widget.curveplot.feature_color, DiscreteVariable)

    def test_compat_no_group(self):
        settings = {}
        OWSpectra.migrate_settings(settings, 3)
        self.assertEqual(settings, {})
        self.widget = self.create_widget(OWSpectra, stored_settings=settings)
        self.assertFalse(self.widget.compat_no_group)
        # We decided to remove the info box
        # self.assertFalse(self.widget.Information.compat_no_group.is_shown())

        settings = {}
        OWSpectra.migrate_settings(settings, 2)
        self.assertEqual(settings, {"compat_no_group": True})
        self.widget = self.create_widget(OWSpectra, stored_settings=settings)
        self.assertTrue(self.widget.compat_no_group)
        # We decided to remove the info box
        # self.assertTrue(self.widget.Information.compat_no_group.is_shown())

    def test_visual_settings(self, timeout=5):
        graph = self.widget.curveplot
        font = QFont()
        font.setItalic(True)
        font.setFamily("Helvetica")

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished(timeout=timeout)
        graph.cycle_color_attr()
        self.assertEqual(graph.feature_color, self.iris.domain.class_var)

        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(20)
        self.assertFontEqual(graph.parameter_setter.title_item.item.font(), font)

        key, value = ("Fonts", "Axis title", "Font size"), 14
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(14)
        for ax in ["bottom", "left"]:
            axis = graph.parameter_setter.getAxis(ax)
            self.assertFontEqual(axis.label.font(), font)

        key, value = ('Fonts', 'Axis ticks', 'Font size'), 15
        self.widget.set_visual_settings(key, value)
        key, value = ('Fonts', 'Axis ticks', 'Italic'), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(15)
        for ax in ["bottom", "left"]:
            axis = graph.parameter_setter.getAxis(ax)
            self.assertFontEqual(axis.style["tickFont"], font)

        key, value = ("Fonts", "Legend", "Font size"), 16
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Legend", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(16)
        legend_item = list(graph.parameter_setter.legend_items)[0]
        self.assertFontEqual(legend_item[1].item.font(), font)

        key, value = ("Annotations", "Title", "Title"), "Foo"
        self.widget.set_visual_settings(key, value)
        self.assertEqual(graph.parameter_setter.title_item.item.toPlainText(), "Foo")
        self.assertEqual(graph.parameter_setter.title_item.text, "Foo")

        key, value = ("Annotations", "x-axis title", "Title"), "Foo2"
        self.widget.set_visual_settings(key, value)
        axis = graph.parameter_setter.getAxis("bottom")
        self.assertEqual(axis.label.toPlainText().strip(), "Foo2")
        self.assertEqual(axis.labelText, "Foo2")

        key, value = ("Annotations", "y-axis title", "Title"), "Foo3"
        self.widget.set_visual_settings(key, value)
        axis = graph.parameter_setter.getAxis("left")
        self.assertEqual(axis.label.toPlainText().strip(), "Foo3")
        self.assertEqual(axis.labelText, "Foo3")

        self.assertFalse(self.widget.Information.view_locked.is_shown())
        key, value = ("View Range", "X", "xMin"), 1.
        self.widget.set_visual_settings(key, value)
        key, value = ("View Range", "X", "xMax"), 3.
        self.widget.set_visual_settings(key, value)
        vr = graph.plot.vb.viewRect()
        self.assertEqual(vr.left(), 1)
        self.assertEqual(vr.right(), 3)
        self.assertTrue(self.widget.Information.view_locked.is_shown())

        key, value = ("View Range", "Y", "yMin"), 2.
        self.widget.set_visual_settings(key, value)
        key, value = ("View Range", "Y", "yMax"), 42.
        self.widget.set_visual_settings(key, value)
        vr = graph.plot.vb.viewRect()
        self.assertEqual(vr.top(), 2)
        self.assertEqual(vr.bottom(), 42)

    def assertFontEqual(self, font1, font2):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())

    def test_migrate_visual_setttings(self):
        settings = {"curveplot":
                        {"label_title": "title",
                         "label_xaxis": "x",
                         "label_yaxis": "y"}
                    }
        OWSpectra.migrate_settings(settings, 4)
        self.assertEqual(settings["visual_settings"],
                         {('Annotations', 'Title', 'Title'): 'title',
                          ('Annotations', 'x-axis title', 'Title'): 'x',
                          ('Annotations', 'y-axis title', 'Title'): 'y'})
        settings = {}
        OWSpectra.migrate_settings(settings, 4)
        self.assertNotIn("visual_settings", settings)


@unittest.skipUnless(dask, "installed Orange does not support dask")
class TestOWSpectraWithDask(TestOWSpectra):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = temp_dasktable("iris")
        cls.titanic = temp_dasktable("titanic")
        cls.collagen = temp_dasktable("collagen")
        cls.normal_data = [temp_dasktable(d) for d in cls.normal_data]
        cls.unknown_last_instance = temp_dasktable(cls.unknown_last_instance)
        cls.unknown_pts = temp_dasktable(cls.unknown_pts)
        cls.only_inf = temp_dasktable(cls.only_inf)
        cls.strange_data = [temp_dasktable(d) for d in cls.strange_data]


if __name__ == "__main__":
    unittest.main()
