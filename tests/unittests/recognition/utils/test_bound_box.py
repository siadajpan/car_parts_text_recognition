from unittest import TestCase

from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class TestBoundBox(TestCase):
    def setUp(self) -> None:
        self.bound_box = BoundBox(1, 2, 3, 4)

    def test___init__(self):
        self.assertEqual(self.bound_box.start_x, 1)
        self.assertEqual(self.bound_box.start_y, 2)
        self.assertEqual(self.bound_box.end_x, 3)
        self.assertEqual(self.bound_box.end_y, 4)

    def test___repr__(self):
        # given

        # when
        result = self.bound_box.__repr__()

        # then
        self.assertEqual('\nx0: 1, y0: 2, xe: 3, ye: 4',
                         result)

    def test___eq__(self):
        # given
        r1 = BoundBox(1, 2, 3, 4)
        r2 = BoundBox(7, 2, 3, 4)
        r3 = BoundBox(1, 7, 3, 4)
        r4 = BoundBox(1, 2, 7, 4)
        r5 = BoundBox(1, 2, 3, 7)
        r6 = BoundBox(1, 2, 3, 4)

        # then
        self.assertEqual(r1, r6)
        self.assertNotEqual(r1, r2)
        self.assertNotEqual(r1, r3)
        self.assertNotEqual(r1, r4)
        self.assertNotEqual(r1, r5)

    def test_width(self):
        # given
        self.bound_box = BoundBox(1, 2, 3, 4)

        # when
        width = self.bound_box.width

        # then
        self.assertEqual(width, 2)

    def test_height(self):
        # given
        self.bound_box = BoundBox(1, 2, 3, 4)

        # when
        height = self.bound_box.height

        # then
        self.assertEqual(height, 2)

    def test_mid_x(self):
        # given
        self.bound_box = BoundBox(1, 2, 3, 4)

        # when
        mid_x = self.bound_box.mid_x

        # then
        self.assertEqual(mid_x, 2)

    def test_mid_y(self):
        # given
        self.bound_box = BoundBox(1, 2, 3, 4)

        # when
        mid_y = self.bound_box.mid_y

        # then
        self.assertEqual(mid_y, 3)

    def test_from_width_height(self):
        # given
        r = BoundBox(1, 2, 4, 6)

        # when
        result = BoundBox.from_width_height(1, 2, 3, 4)

        # then
        self.assertEqual(r, result)
