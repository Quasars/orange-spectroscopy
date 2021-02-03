import os
from unittest.mock import Mock, patch

from AnyQt.QtCore import QRectF, QPoint, Qt
from AnyQt.QtTest import QTest, QSignalSpy
import numpy as np
import pyqtgraph as pg

from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME, ANNOTATED_DATA_FEATURE_NAME

from orangecontrib.spectroscopy.widgets.owspectra import OWSpectra, MAX_INSTANCES_DRAWN, \
    PlotCurvesItem, NoSuchCurve, MAX_THICK_SELECTED
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.line_geometry import intersect_curves, \
    distance_line_segment
from orangecontrib.spectroscopy.tests.util import hold_modifiers, set_png_graph_save
from orangecontrib.spectroscopy.preprocess import Interpolate


NAN = float("nan")


def wait_for_graph(widget, timeout=5000):
    concurrent = widget.curveplot.show_average_thread
    if concurrent.task is not None:
        spy = QSignalSpy(concurrent.average_shown)
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
        # dataset without any attributes
        iris0 = cls.iris.transform(Domain([]))
        # data set with no lines
        empty = cls.iris[:0]
        # dataset with large blank regions
        irisunknown = Interpolate(np.arange(20))(cls.iris)
        cls.unknown_last_instance = cls.iris.copy()
        cls.unknown_last_instance.X[73] = NAN  # needs to be unknown after sampling and permutation
        # dataset with mixed unknowns
        cls.unknown_pts = cls.collagen.copy()
        cls.unknown_pts[5] = np.nan
        cls.unknown_pts[8:10] = np.nan
        cls.unknown_pts[15] = np.inf
        # a data set with only infs
        cls.only_inf = iris1.copy()
        cls.only_inf.X *= np.Inf
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
            self.assertTrue(np.all(np.isnan(self.unknown_last_instance[self.widget.curveplot.sampled_indices].X[-1])))

    def do_mousemove(self):
        mr = self.widget.curveplot.MOUSE_RADIUS
        self.widget.curveplot.MOUSE_RADIUS = 1000
        self.widget.curveplot.mouse_moved_closest(
            (self.widget.curveplot.plot.sceneBoundingRect().center(),))
        if self.widget.curveplot.data is not None \
                and np.any(np.isfinite(self.widget.curveplot.data.X)):  # a valid curve exists
            self.assertIsNotNone(self.widget.curveplot.highlighted)
        else:  # no curve can be detected
            self.assertIsNone(self.widget.curveplot.highlighted)

        # assume nothing is directly in the middle
        # therefore nothing should be highlighted
        self.widget.curveplot.MOUSE_RADIUS = 0.1
        self.widget.curveplot.mouse_moved_closest(
            (self.widget.curveplot.plot.sceneBoundingRect().center(),))
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
            self.send_signal("Data", data)
            self.do_mousemove()

    def select_diagonal(self):
        vb = self.widget.curveplot.plot.vb
        vb.set_mode_select()
        vr = vb.viewRect()
        tls = vr.bottomRight() if self.widget.curveplot.invertX else vr.bottomLeft()
        brs = vr.topLeft() if self.widget.curveplot.invertX else vr.topRight()
        tl = vb.mapViewToScene(tls).toPoint() + QPoint(2, 100)  # avoid menu button
        br = vb.mapViewToScene(brs).toPoint() - QPoint(2, 2)
        ca = self.widget.curveplot.childAt(tl)
        QTest.mouseClick(ca, Qt.LeftButton, pos=tl)
        self.widget.curveplot.plot.scene().sigMouseMoved.emit(tl)
        self.widget.curveplot.plot.scene().sigMouseMoved.emit(tl + QPoint(10, 10))
        QTest.qWait(1)
        QTest.mouseClick(ca, Qt.LeftButton, pos=br)

    def test_select_line(self):
        for data in self.normal_data:
            self.send_signal("Data", data)
            out = self.get_output("Selection")
            self.assertIsNone(out, None)
            out = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
            sa = out.transform(Domain([out.domain[ANNOTATED_DATA_FEATURE_NAME]]))
            np.testing.assert_equal(sa.X, 0)
            self.select_diagonal()
            out = self.get_output("Selection")
            self.assertEqual(len(data), len(out))
            out = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
            self.assertEqual(len(data), len(out))
            sa = out.transform(Domain([out.domain[ANNOTATED_DATA_FEATURE_NAME]]))
            np.testing.assert_equal(sa.X, 1)

    def test_Label_peaks(self):
        self.send_signal("Data", self.collagen)
        self.widget.curveplot.find_peak_variables()
        np.testing.assert_almost_equal(self.widget.curveplot.maximum_point, 1.2, 1)
        np.testing.assert_almost_equal(self.widget.curveplot.start_point, 1351, 0)
        np.testing.assert_almost_equal(self.widget.curveplot.minimum_point, 0)
        self.widget.curveplot.peak_apply_auto(prominence=None, maxHeight=None,
                                              minHeight=0.5, line_overlap=20)
        np.testing.assert_almost_equal(self.widget.curveplot.peak_locations,
                                       [1025, 1079, 1160, 1461, 1542, 1646, 1739], 0)
        #Add in test for peaks with specific values


    def do_zoom_rect(self, invertX):
        """ Test zooming with two clicks. """
        self.send_signal("Data", self.iris)
        vb = self.widget.curveplot.plot.vb
        self.widget.curveplot.invertX = invertX
        self.widget.curveplot.invertX_apply()
        vb.set_mode_zooming()
        vr = vb.viewRect()
        tls = vr.bottomRight() if self.widget.curveplot.invertX else vr.bottomLeft()
        # move down to avoid clicking on the menu button
        tl = vb.mapViewToScene(tls).toPoint() + QPoint(0, 100)
        br = vb.mapViewToScene(vr.center()).toPoint()
        tlw = vb.mapSceneToView(tl)
        brw = vb.mapSceneToView(br)
        ca = self.widget.curveplot.childAt(tl)
        QTest.mouseClick(ca, Qt.LeftButton, pos=tl)
        QTest.qWait(1)
        self.widget.curveplot.plot.scene().sigMouseMoved.emit(tl)
        QTest.qWait(1)
        self.widget.curveplot.plot.scene().sigMouseMoved.emit(tl + QPoint(10, 10))
        QTest.qWait(1)
        QTest.mouseClick(ca, Qt.LeftButton, pos=br)
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
        self.assertFalse(self.widget.Information.showing_sample.is_shown())
        self.send_signal("Data", self.titanic)
        self.assertTrue(self.widget.Information.showing_sample.is_shown())
        self.send_signal("Data", self.titanic[:MAX_INSTANCES_DRAWN])
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
        x, _ = self.widget.curveplot.curves[0]
        fs = sorted([float(f.name) for f in self.collagen.domain.attributes])
        np.testing.assert_equal(x, fs)

    def test_handle_nofloatname(self):
        self.send_signal("Data", self.iris)
        x, _ = self.widget.curveplot.curves[0]
        np.testing.assert_equal(x, range(len(self.iris.domain.attributes)))

    def test_show_average(self):
        def numcurves(curves):
            return sum(len(a[1]) for a in curves)

        self.send_signal("Data", self.iris)
        curves_plotted = self.widget.curveplot.curves_plotted
        self.assertEqual(numcurves(curves_plotted), 150)
        self.widget.curveplot.show_average()
        wait_for_graph(self.widget)
        curves_plotted = self.widget.curveplot.curves_plotted
        self.assertEqual(numcurves(curves_plotted), 3)
        self.widget.curveplot.show_individual()
        curves_plotted = self.widget.curveplot.curves_plotted
        self.assertEqual(numcurves(curves_plotted), 150)

    def test_limits(self):
        self.send_signal("Data", self.iris)
        vr = self.widget.curveplot.plot.viewRect()
        # there should ne no change
        self.widget.curveplot.set_limits()
        vr2 = self.widget.curveplot.plot.viewRect()
        self.assertEqual(vr, vr2)

    def test_line_intersection(self):
        data = self.collagen
        x = getx(data)
        sort = np.argsort(x)
        x = x[sort]
        ys = data.X[:, sort]
        boola = intersect_curves(x, ys, np.array([0, 1.15]), np.array([3000, 1.15]))
        intc = np.flatnonzero(boola)
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
        sinds = self.widget.curveplot.sampled_indices
        self.assertEqual(len(sinds), MAX_INSTANCES_DRAWN)

        # the whole subset is drawn
        add_subset = data[:MAX_INSTANCES_DRAWN]
        self.send_signal("Data subset", add_subset)
        sinds = self.widget.curveplot.sampled_indices
        self.assertTrue(set(add_subset.ids) <= set(data[sinds].ids))

        # the whole subset can not be drawn anymore
        add_subset = data[:MAX_INSTANCES_DRAWN+1]
        self.send_signal("Data subset", add_subset)
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
        self.send_signal("Data", Table("housing"))
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
        self.widget.curveplot.MOUSE_RADIUS = 1000
        self.widget.curveplot.mouse_moved_closest(
            (self.widget.curveplot.plot.sceneBoundingRect().center(),))
        self.widget.curveplot.select_by_click(None)
        out = self.get_output("Selection")
        self.assertEqual(len(out), 1)
        # resending the exact same data should not change the selection
        self.send_signal("Data", self.iris)
        out2 = self.get_output("Selection")
        self.assertEqual(len(out), 1)
        # while resending the same data as a different object should
        self.send_signal("Data", Table("iris"))
        out = self.get_output("Selection")
        self.assertIsNone(out, None)

    def test_select_click_multiple_groups(self):
        data = self.collagen[:100]
        self.send_signal("Data", data)
        self.widget.curveplot.make_selection([1])
        with hold_modifiers(self.widget, Qt.ControlModifier):
            self.widget.curveplot.make_selection([2])
        with hold_modifiers(self.widget, Qt.ShiftModifier):
            self.widget.curveplot.make_selection([3])
        with hold_modifiers(self.widget, Qt.ShiftModifier | Qt.ControlModifier):
            self.widget.curveplot.make_selection([4])
        out = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(len(out), 100)  # have a data table at the output
        newvars = out.domain.variables + out.domain.metas
        oldvars = data.domain.variables + data.domain.metas
        group_at = [a for a in newvars if a not in oldvars][0]
        unselected = group_at.to_val("Unselected")
        out = out[np.flatnonzero(out.transform(Domain([group_at])).X != unselected)]
        self.assertEqual(len(out), 4)
        np.testing.assert_equal([o for o in out], [data[i] for i in [1, 2, 3, 4]])
        np.testing.assert_equal([o[group_at].value for o in out], ["G1", "G2", "G3", "G3"])

        # remove one element
        with hold_modifiers(self.widget, Qt.AltModifier):
            self.widget.curveplot.make_selection([1])
        out = self.get_output("Selection")
        np.testing.assert_equal(len(out), 3)
        np.testing.assert_equal([o for o in out], [data[i] for i in [2, 3, 4]])

    def test_select_thick_lines(self):
        data = self.collagen[:100]
        assert MAX_INSTANCES_DRAWN >= len(data) > MAX_THICK_SELECTED
        self.send_signal("Data", data)
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
        data = Table("iris")
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
        self.assertEqual(3, len(curveplot.curves[0][1]))
        curveplot.set_data(self.iris[:3], auto_update=False)
        self.assertEqual([], curveplot.curves)
        curveplot.update_view()
        self.assertEqual(3, len(curveplot.curves[0][1]))

    def test_save_graph(self):
        self.send_signal("Data", self.iris)
        with set_png_graph_save() as fname:
            self.widget.save_graph()
            self.assertGreater(os.path.getsize(fname), 10000)

    def test_filter_unknowns(self):
        self.send_signal("Data", self.unknown_pts)
        curves_cont = self.widget.curveplot.curves_cont
        for obj in curves_cont.objs:
            self.assertTrue(np.isfinite(obj.yData).all())
            self.assertTrue(np.isfinite(obj.xData).all())

    def test_migrate_context_feature_color(self):
        c = self.widget.settingsHandler.new_context(self.iris.domain,
                                                    None, self.iris.domain.class_vars)
        c.values["curveplot"] = {"feature_color": ("iris", 1)}
        self.widget = self.create_widget(OWSpectra,
                                         stored_settings={"context_settings": [c]})
        self.send_signal("Data", self.iris)
        self.assertIsInstance(self.widget.curveplot.feature_color, DiscreteVariable)
