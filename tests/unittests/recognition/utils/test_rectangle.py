from unittest import TestCase

from car_parts_text_recognition.recognition.utils.rectangle import Rectangle


class TestRectangle(TestCase):
    def setUp(self) -> None:
        self.rectangle = Rectangle(1, 2, 3, 4)

    def test___init__(self):
        self.assertEqual(self.rectangle.start_x, 1)
        self.assertEqual(self.rectangle.start_y, 2)
        self.assertEqual(self.rectangle.end_x, 3)
        self.assertEqual(self.rectangle.end_y, 4)

    def test___repr__(self):
        # given

        # when
        result = self.rectangle.__repr__()

        # then
        self.assertEqual('start x: 1, start y: 2,end x: 3, end y: 4',
                         result)

    def test___eq__(self):
        # given
        r1 = Rectangle(1, 2, 3, 4)
        r2 = Rectangle(7, 2, 3, 4)
        r3 = Rectangle(1, 7, 3, 4)
        r4 = Rectangle(1, 2, 7, 4)
        r5 = Rectangle(1, 2, 3, 7)
        r6 = Rectangle(1, 2, 3, 4)

        # then
        self.assertEqual(r1, r6)
        self.assertNotEqual(r1, r2)
        self.assertNotEqual(r1, r3)
        self.assertNotEqual(r1, r4)
        self.assertNotEqual(r1, r5)

    def test_from_width_height(self):
        # given
        r = Rectangle(1, 2, 4, 6)

        # when
        result = Rectangle.from_width_height(1, 2, 3, 4)

        # then
        self.assertEqual(r, result)
