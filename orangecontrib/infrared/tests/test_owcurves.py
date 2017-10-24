import numpy as np
import Orange
import pyqtgraph as pg
from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table, Domain, ContinuousVariable
from orangecontrib.infrared.widgets.owspectra import OWCurves, MAX_INSTANCES_DRAWN, \
    PlotCurvesItem
from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.line_geometry import intersect_curves, \
    distance_line_segment
from orangecontrib.infrared.preprocess import Interpolate
from AnyQt.QtCore import QRectF, QPoint, Qt
from AnyQt.QtTest import QTest

try:
    qWaitForWindow = QTest.qWaitForWindowShown
except AttributeError:
    qWaitForWindow = QTest.qWaitForWindowActive

NAN = float("nan")

class TestOWCurves(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.collagen = Table("collagen")
        cls.normal_data = [cls.iris, cls.collagen]
        # dataset with a single attribute
        iris1 = Table(Domain(cls.iris.domain[:1]), cls.iris)
        # dataset without any attributes
        iris0 = Table(Domain([]), cls.iris)
        # dataset with large blank regions
        irisunknown = Interpolate(np.arange(20))(cls.iris)
        cls.unknown_last_instance = cls.iris.copy()
        cls.unknown_last_instance.X[73] = NAN  # needs to be unknown after sampling and permutation
        # a data set with features with the same names
        sfdomain = Domain([ContinuousVariable("1"), ContinuousVariable("1")])
        cls.same_features = Table(sfdomain, [[0, 1]])
        cls.strange_data = [iris1, iris0, irisunknown, cls.unknown_last_instance, cls.same_features]

    def setUp(self):
        self.widget = self.create_widget(OWCurves)  # OWCurves

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

    def test_is_last_instance(self):
        self.send_signal("Data", self.unknown_last_instance)
        self.assertTrue(np.all(np.isnan(self.unknown_last_instance[self.widget.curveplot.sampled_indices].X[-1])))

    def do_mousemove(self):
        mr = self.widget.curveplot.MOUSE_RADIUS
        self.widget.curveplot.MOUSE_RADIUS = 1000
        self.widget.curveplot.mouseMoved((self.widget.curveplot.plot.sceneBoundingRect().center(),))
        if self.widget.curveplot.data \
                and len(self.widget.curveplot.data.X) \
                and len(self.widget.curveplot.data_x):  # detect a curve if a validgi curve exists
            self.assertIsNotNone(self.widget.curveplot.highlighted)
        else:  # no curve can be detected
            self.assertIsNone(self.widget.curveplot.highlighted)

        # assume nothing is directly in the middle
        # therefore nothing should be highlighted
        self.widget.curveplot.MOUSE_RADIUS = 0.1
        self.widget.curveplot.mouseMoved((self.widget.curveplot.plot.sceneBoundingRect().center(),))
        self.assertIsNone(self.widget.curveplot.highlighted)

        self.widget.curveplot.MOUSE_RADIUS = mr

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
        QTest.qWait(100)
        tls = vr.bottomRight() if self.widget.curveplot.invertX else vr.bottomLeft()
        brs = vr.topLeft() if self.widget.curveplot.invertX else vr.topRight()
        tl = vb.mapViewToScene(tls).toPoint() + QPoint(2, 2)
        br = vb.mapViewToScene(brs).toPoint() - QPoint(2, 2)
        ca = self.widget.curveplot.childAt(tl)
        QTest.mouseClick(ca, Qt.LeftButton, pos=tl)
        QTest.mouseMove(self.widget.curveplot, pos=tl)  # test mouseMoved code
        QTest.mouseMove(self.widget.curveplot)  # test mouseMoved code
        QTest.qWait(1)
        QTest.mouseClick(ca, Qt.LeftButton, pos=br)

    def test_select_line(self):
        self.widget.show()
        qWaitForWindow(self.widget)
        for data in self.normal_data:
            self.send_signal("Data", data)
            out = self.get_output("Selection")
            self.assertIsNone(out, None)
            out = self.get_output("Data")
            sa = out.transform(Orange.data.Domain([out.domain["Selected"]]))
            np.testing.assert_equal(sa.X, 0)
            self.select_diagonal()
            out = self.get_output("Selection")
            self.assertEqual(len(data), len(out))
            out = self.get_output("Data")
            self.assertEqual(len(data), len(out))
            sa = out.transform(Orange.data.Domain([out.domain["Selected"]]))
            np.testing.assert_equal(sa.X, 1)
        self.widget.hide()

    def test_zoom_rect(self):
        """ Test zooming with two clicks. """
        self.widget.show()
        qWaitForWindow(self.widget)
        self.send_signal("Data", self.iris)
        vb = self.widget.curveplot.plot.vb
        vb.set_mode_zooming()
        vr = vb.viewRect()
        QTest.qWait(100)
        tls = vr.bottomRight() if self.widget.curveplot.invertX else vr.bottomLeft()
        tl = vb.mapViewToScene(tls).toPoint() + QPoint(2, 2)
        br = vb.mapViewToScene(vr.center()).toPoint()
        tlw = vb.mapSceneToView(tl)
        brw = vb.mapSceneToView(br)
        ca = self.widget.curveplot.childAt(tl)
        QTest.mouseClick(ca, Qt.LeftButton, pos=tl)
        QTest.qWait(1)
        QTest.mouseMove(self.widget.curveplot, pos=tl)  # test mouseMoved code
        QTest.qWait(1)
        QTest.mouseMove(self.widget.curveplot)  # test mouseMoved code
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
        self.widget.hide()

    def test_warning_no_x(self):
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Warning.no_x.is_shown())
        self.send_signal("Data", self.strange_data[1])
        self.assertTrue(self.widget.Warning.no_x.is_shown())
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Warning.no_x.is_shown())

    def test_information(self):
        self.send_signal("Data", self.iris[:100])
        self.assertFalse(self.widget.Information.showing_sample.is_shown())
        self.send_signal("Data", self.iris)
        self.assertTrue(self.widget.Information.showing_sample.is_shown())
        self.send_signal("Data", self.iris[:100])
        self.assertFalse(self.widget.Information.showing_sample.is_shown())

    def test_handle_floatname(self):
        self.send_signal("Data", self.collagen)
        x, cys = self.widget.curveplot.curves[0]
        ys = self.widget.curveplot.data.X
        self.assertEqual(len(ys), len(self.collagen))
        self.assertEqual(len(cys), MAX_INSTANCES_DRAWN)
        fs = sorted([float(f.name) for f in self.collagen.domain.attributes])
        np.testing.assert_equal(x, fs)

    def test_handle_nofloatname(self):
        self.send_signal("Data", self.iris)
        x, cys = self.widget.curveplot.curves[0]
        ys = self.widget.curveplot.data.X
        self.assertEqual(len(ys), len(self.iris))
        self.assertEqual(len(cys), MAX_INSTANCES_DRAWN)
        np.testing.assert_equal(x,
                                range(len(self.iris.domain.attributes)))

    def test_show_average(self):
        self.send_signal("Data", self.iris)
        curves_plotted = self.widget.curveplot.curves_plotted
        self.widget.curveplot.show_average()
        curves_plotted2 = self.widget.curveplot.curves_plotted
        def numcurves(curves):
            return sum(len(a[1]) for a in curves)
        self.assertLess(numcurves(curves_plotted2), numcurves(curves_plotted))
        self.widget.curveplot.show_individual()
        curves_plotted3 = self.widget.curveplot.curves_plotted
        self.assertEqual(numcurves(curves_plotted), numcurves(curves_plotted3))

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
        self.send_signal("Data", self.collagen)
        sinds = self.widget.curveplot.sampled_indices
        self.assertEqual(len(sinds), MAX_INSTANCES_DRAWN)

        # the whole subset is drawn
        add_subset = self.collagen[:MAX_INSTANCES_DRAWN]
        self.send_signal("Data subset", add_subset)
        sinds = self.widget.curveplot.sampled_indices
        self.assertTrue(set(add_subset.ids) <= set(self.collagen[sinds].ids))

        # the whole subset can not be drawn anymore
        add_subset = self.collagen[:MAX_INSTANCES_DRAWN+1]
        self.send_signal("Data subset", add_subset)
        sinds = self.widget.curveplot.sampled_indices
        self.assertFalse(set(add_subset.ids) <= set(self.collagen[sinds].ids))

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
        self.widget.curveplot.feature_color = "iris"
        self.send_signal("Data", Orange.data.Table("housing"))
        self.assertEqual(self.widget.curveplot.feature_color, None)
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.feature_color, "iris")

    def test_cycle_color(self):
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.curveplot.feature_color, None)
        self.widget.curveplot.cycle_color_attr()
        self.assertEqual(self.widget.curveplot.feature_color, "iris")
        self.widget.curveplot.cycle_color_attr()
        self.assertEqual(self.widget.curveplot.feature_color, None)

    def test_open_selection(self):
        # saved selection in the file should be reloaded
        self.widget.curveplot.selected_indices = set([0])
        self.send_signal("Data", self.iris)
        out = self.get_output("Selection")
        self.assertEqual(out[0], self.iris[0])

    def test_selection_changedata(self):
        # select something in the widget and see if it is cleared
        self.send_signal("Data", self.iris)
        self.widget.curveplot.MOUSE_RADIUS = 1000
        self.widget.curveplot.mouseMoved((self.widget.curveplot.plot.sceneBoundingRect().center(),))
        self.widget.curveplot.select_by_click(None, add=False)
        out = self.get_output("Selection")
        self.assertEqual(len(out), 1)
        # resending the same data should clear the selection
        self.send_signal("Data", self.iris)
        out = self.get_output("Selection")
        self.assertIsNone(out, None)
