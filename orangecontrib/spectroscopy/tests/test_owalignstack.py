import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.ndimage import sobel

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.data import build_spec_table
from orangecontrib.spectroscopy.io.util import _spectra_from_image
from orangecontrib.spectroscopy.widgets.owstackalign import (
    alignstack,
    RegisterTranslation,
    shift_fill,
    OWStackAlign,
    process_stack,
)


def test_image():
    return np.hstack((np.zeros((5, 1)), np.diag([1, 5.0, 3.0, 1, 1]), np.ones((5, 1))))


def _up(im, fill=0):
    return np.vstack((im[1:], np.ones_like(im[0]) * fill))


def _down(im, fill=0):
    return np.vstack((np.ones_like(im[0]) * fill, im[:-1]))


def _right(im, fill=0):
    return np.hstack((np.ones_like(im[:, :1]) * fill, im[:, :-1]))


def _left(im, fill=0):
    return np.hstack((im[:, 1:], np.ones_like(im[:, :1]) * fill))


class TestUtils(unittest.TestCase):
    def test_image_shift(self):
        im = test_image()
        calculate_shift = RegisterTranslation()
        s = calculate_shift(im, _up(im))
        np.testing.assert_equal(s, (1, 0))
        s = calculate_shift(im, _down(im))
        np.testing.assert_equal(s, (-1, 0))
        s = calculate_shift(im, _left(im))
        np.testing.assert_equal(s, (0, 1))
        s = calculate_shift(im, _right(im))
        np.testing.assert_equal(s, (0, -1))
        s = calculate_shift(im, _left(_left(im)))
        np.testing.assert_equal(s, (0, 2))

    def test_alignstack(self):
        im = test_image()
        _, aligned = alignstack(
            [im, _up(im), _down(im), _right(im)], shiftfn=RegisterTranslation()
        )
        self.assertEqual(aligned.shape, (4, 5, 7))

    def test_alignstack_calls_filterfn(self):
        filterfn = Mock()
        filterfn.side_effect = lambda x: x
        im = test_image()
        up = _up(im)
        down = _down(im)
        alignstack([im, up, down], shiftfn=RegisterTranslation(), filterfn=filterfn)
        for i, t in enumerate([im, up, down]):
            self.assertIs(filterfn.call_args_list[i][0][0], t)

    def test_shift_fill(self):
        im = test_image()

        # shift down
        a = shift_fill(im, (1, 0))
        np.testing.assert_almost_equal(a, _down(im, np.nan))
        a = shift_fill(im, (0.55, 0))
        np.testing.assert_equal(np.isnan(a), np.isnan(_down(im, np.nan)))
        a = shift_fill(im, (0.45, 0))
        np.testing.assert_equal(np.isnan(a), False)

        # shift up
        a = shift_fill(im, (-1, 0))
        np.testing.assert_almost_equal(a, _up(im, np.nan))
        a = shift_fill(im, (-0.55, 0))
        np.testing.assert_equal(np.isnan(a), np.isnan(_up(im, np.nan)))
        a = shift_fill(im, (-0.45, 0))
        np.testing.assert_equal(np.isnan(a), False)

        # shift right
        a = shift_fill(im, (0, 1))
        np.testing.assert_almost_equal(a, _right(im, np.nan))
        a = shift_fill(im, (0, 0.55))
        np.testing.assert_equal(np.isnan(a), np.isnan(_right(im, np.nan)))
        a = shift_fill(im, (0, 0.45))
        np.testing.assert_equal(np.isnan(a), False)

        # shift left
        a = shift_fill(im, (0, -1))
        np.testing.assert_almost_equal(a, _left(im, np.nan))
        a = shift_fill(im, (0, -0.55))
        np.testing.assert_equal(np.isnan(a), np.isnan(_left(im, np.nan)))
        a = shift_fill(im, (0, -0.45))
        np.testing.assert_equal(np.isnan(a), False)


def diamond():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 6, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 5, 1, 7, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 8, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )


def rectangle():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )


def fake_stxm_from_image(image):
    spectral = np.zeros(image.shape + (5,))
    spectral[:, :, 0] = diamond()
    spectral[:, :, 1] = _up(diamond())
    spectral[:, :, 2] = _down(diamond())
    spectral[:, :, 3] = _right(diamond())
    spectral[:, :, 4] = _down(_right(_down(diamond())))
    return spectral


def fake_stxm_from_rectangle(image):
    spectral = np.zeros(image.shape + (5,))
    spectral[:, :, 0] = rectangle()
    spectral[:, :, 1] = _up(rectangle())
    spectral[:, :, 2] = _down(rectangle())
    spectral[:, :, 3] = _right(rectangle())
    spectral[:, :, 4] = _down(_right(_down(rectangle())))
    return spectral


class SideEffect:
    def __init__(self, fn):
        self.fn = fn
        self.return_value = None

    def __call__(self, *args, **kwargs):
        self.return_value = self.fn(*args, **kwargs)
        return self.return_value


def lineedit_type(le, text):
    le.clear()
    le.insert(text)
    le.editingFinished.emit()


def orange_table_from_3d(image3d):
    info = _spectra_from_image(
        image3d,
        range(image3d.shape[2]),
        range(image3d[:, :, 0].shape[1]),
        range(image3d[:, :, 0].shape[0]),
    )
    data = build_spec_table(*info)
    return data


stxm_diamond = orange_table_from_3d(fake_stxm_from_image(diamond()))
stxm_rectangle = orange_table_from_3d(fake_stxm_from_rectangle(rectangle()))
stxm_rectangle_short = orange_table_from_3d(
    fake_stxm_from_rectangle(rectangle())[:, :, :3]
)


def orange_table_to_3d(data):
    nz = len(data.domain.attributes)
    minx = int(min(data.metas[:, 0]))
    miny = int(min(data.metas[:, 1]))
    maxx = int(max(data.metas[:, 0]))
    maxy = int(max(data.metas[:, 1]))
    image3d = np.ones((maxy - miny + 1, maxx - minx + 1, nz)) * np.nan
    for d in data:
        x, y = int(d.metas[0]), int(d.metas[1])
        image3d[y - miny, x - minx, :] = d.x
    return image3d


class TestOWStackAlign(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWStackAlign)  # type: OWStackAlign

    def test_add_remove_data(self):
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        out = self.get_output(self.widget.Outputs.newstack)
        self.assertIsInstance(out, Table)
        self.send_signal(self.widget.Inputs.data, None)
        out = self.get_output(self.widget.Outputs.newstack)
        self.assertIs(out, None)

    def test_output_aligned(self):
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        out = self.get_output(self.widget.Outputs.newstack)
        image3d = orange_table_to_3d(out)
        for z in range(1, image3d.shape[2]):
            np.testing.assert_almost_equal(image3d[:, :, 0], image3d[:, :, z])

    def test_output_aligned_with_ref(self):
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        self.send_signal(self.widget.Inputs.refdata, stxm_rectangle)
        self.widget.controls.use_refinput.toggle()
        out = self.get_output(self.widget.Outputs.newstack)
        image3d = orange_table_to_3d(out)
        for z in range(1, image3d.shape[2]):
            np.testing.assert_almost_equal(image3d[:, :, 0], image3d[:, :, z])

    def test_output_cropped(self):
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        out = self.get_output(self.widget.Outputs.newstack)
        image3d = orange_table_to_3d(out)
        # for a cropped image all have to be defined
        self.assertFalse(np.any(np.isnan(image3d)))
        # for diamond test data, extreme movement
        # in X was just one right,
        # in Y was one up and 2 down
        # try to crop manually to see if the obtained image is the same
        np.testing.assert_almost_equal(image3d[:, :, 0], diamond()[1:-2, :-1])

    def test_sobel_called(self):
        with patch(
            "orangecontrib.spectroscopy.widgets.owstackalign.sobel",
            Mock(side_effect=sobel),
        ) as mock:
            self.send_signal(self.widget.Inputs.data, stxm_diamond)
            _ = self.get_output(self.widget.Outputs.newstack)
            self.assertFalse(mock.called)
            self.widget.controls.sobel_filter.toggle()
            _ = self.get_output(self.widget.Outputs.newstack)
            self.assertTrue(mock.called)

    def test_report(self):
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        self.widget.send_report()

    def test_nan_in_image(self):
        data = stxm_diamond.copy()
        with data.unlocked():
            data.X[1, 2] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.nan_in_image.is_shown())
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        self.assertFalse(self.widget.Error.nan_in_image.is_shown())

    def test_with_class_columns(self):
        data = stxm_diamond
        cv = DiscreteVariable(name="class", values=["a", "b"])
        z = ContinuousVariable(name="z")
        domain = Domain(
            data.domain.attributes, class_vars=[cv], metas=data.domain.metas + (z,)
        )
        data = data.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.newstack)
        # The alignment rearranges table columns. Because we do not know, which
        # wavenumber they belong to, forget them for now.
        self.assertEqual(out.domain.class_vars, tuple())
        self.assertEqual(len(out.domain.metas), 2)

    def test_invalid_axis(self):
        data = stxm_diamond.copy()
        with data.unlocked():
            data.metas[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.invalid_axis.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.invalid_axis.is_shown())

    def test_reference_warning(self):
        data = stxm_diamond
        # only Data is set, reference input not used = No warning
        self.send_signal(self.widget.Inputs.data, data)
        self.assertFalse(self.widget.Warning.missing_reference.is_shown())
        # Toggle use reference input option = warning appears
        self.widget.controls.use_refinput.toggle()
        self.assertTrue(self.widget.Warning.missing_reference.is_shown())
        # Reference data is connected = warning disappears
        self.send_signal(self.widget.Inputs.refdata, data)
        self.assertFalse(self.widget.Warning.missing_reference.is_shown())

    def test_output_aligned_with_ref_unmatched(self):
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        self.send_signal(self.widget.Inputs.refdata, stxm_rectangle_short)
        self.widget.controls.use_refinput.toggle()
        self.assertTrue(self.widget.Error.wrong_reference.is_shown())
        self.widget.controls.use_refinput.toggle()
        self.assertFalse(self.widget.Error.wrong_reference.is_shown())

    def test_missing_metas(self):
        domain = Domain(stxm_diamond.domain.attributes)
        data = stxm_diamond.transform(domain)
        # this should not crash
        self.send_signal(self.widget.Inputs.data, data)

    def test_no_wavenumbers(self):
        domain = Domain(
            stxm_diamond.domain.attributes[:0], metas=stxm_diamond.domain.metas
        )
        data = stxm_diamond.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)

    def test_single_wavenumber(self):
        domain = Domain(
            stxm_diamond.domain.attributes[:1], metas=stxm_diamond.domain.metas
        )
        data = stxm_diamond.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.newstack)
        image3d = orange_table_to_3d(out)
        np.testing.assert_almost_equal(image3d[:, :, 0], diamond())

    def test_frame_changes_output(self):
        self.widget.ref_frame_num = 1
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        out1 = self.get_output(self.widget.Outputs.newstack)
        lineedit_type(self.widget.controls.ref_frame_num, "2")
        self.assertEqual(self.widget.ref_frame_num, 2)
        out2 = self.get_output(self.widget.Outputs.newstack)
        self.assertIsNot(out1, out2)
        # due to cropping we get the same output on this very simple problem
        np.testing.assert_equal(out1.X, out2.X)

    def test_frame_limits(self):
        self.send_signal(self.widget.Inputs.data, stxm_diamond)
        lineedit_type(self.widget.controls.ref_frame_num, "1")
        self.assertEqual(1, self.widget.ref_frame_num)
        # could not test input 0 for the first because lineedit_type does not use validatiors
        lineedit_type(self.widget.controls.ref_frame_num, "5")
        self.assertEqual(5, self.widget.ref_frame_num)
        lineedit_type(self.widget.controls.ref_frame_num, "6")
        self.assertEqual(5, self.widget.ref_frame_num)

    def test_frame_shifts(self):
        se = SideEffect(process_stack)
        with patch(
            "orangecontrib.spectroscopy.widgets.owstackalign.process_stack",
            Mock(side_effect=se),
        ) as mock:
            self.send_signal(self.widget.Inputs.data, stxm_diamond)
            lineedit_type(self.widget.controls.ref_frame_num, "1")
            self.assertEqual(2, mock.call_count)
            np.testing.assert_equal(se.return_value[0][0], (0, 0))
            lineedit_type(self.widget.controls.ref_frame_num, "3")
            self.assertEqual(3, mock.call_count)
            np.testing.assert_equal(se.return_value[0][2], (0, 0))
