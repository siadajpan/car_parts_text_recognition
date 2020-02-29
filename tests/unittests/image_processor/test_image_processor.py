from unittest import TestCase
from unittest.mock import MagicMock, patch, call

import cv2
import numpy as np

from car_parts_text_recognition.image_processor.image_processor import \
    ImageProcessor
from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class TestImageProcessor(TestCase):
    def setUp(self) -> None:
        self.image_processor = ImageProcessor()

    def test___init__(self):
        self.assertIsNone(self.image_processor.image)

    @patch('cv2.cvtColor')
    def test_to_gray(self, convert_mock):
        # given
        image = MagicMock()
        convert_mock.return_value = 'gray'

        # when
        result = self.image_processor.to_gray(image)

        # then
        self.assertEqual('gray', result)
        convert_mock.assert_called_with(image, cv2.COLOR_BGR2GRAY)

    @patch('cv2.threshold')
    def test_threshold_image(self, threshold_mock):
        # given
        threshold_mock.return_value = (None, 'thresh')

        # when
        result = self.image_processor.threshold_image(MagicMock())

        # then
        threshold_mock.assert_called()
        self.assertEqual('thresh', result)

    @patch('cv2.bitwise_not')
    def test_invert(self, invert_mock):
        # given
        invert_mock.return_value = 'inv'

        # when
        result = self.image_processor.invert(MagicMock())

        # then
        invert_mock.assert_called()
        self.assertEqual('inv', result)

    @patch('cv2.resize')
    def test_resize_picture_32(self, resize_mock):
        # given
        image = MagicMock()
        image.shape = (151, 251, 3)

        # when
        self.image_processor.resize_picture_32(image)

        # then
        resize_mock.assert_called_with(image, (256, 160))

    def test_cut_box(self):
        # given
        image = np.random.random((10, 10))
        box = BoundBox(1, 2, 3, 4)

        # when
        result = self.image_processor.cut_box(image, box)

        # then
        np.testing.assert_array_equal(image[2:4, 1:3], result)

    @patch('cv2.findContours')
    def test_find_contours(self, contours_mock):
        # given
        contours_mock.return_value = ('contours', 'hierarchy')

        # when
        result = self.image_processor.find_contours(MagicMock())

        # then
        contours_mock.assert_called()
        self.assertEqual('contours', result)

    @patch('cv2.drawContours')
    def test_fill_holes(self, draw_contours_mock):
        # given
        self.image_processor.find_contours = MagicMock(return_value=['cnt'])

        # when
        self.image_processor.fill_holes(MagicMock())

        # then
        self.image_processor.find_contours.assert_called()
        draw_contours_mock.assert_called()

    @patch('cv2.rectangle')
    def test_draw_box(self, rectangle_mock):
        # given
        rectangle = BoundBox(1, 2, 3, 4)
        image = MagicMock()

        # when
        self.image_processor.draw_box(image, rectangle)

        # then
        rectangle_mock.assert_called_with(image, (0, 1), (4, 5), (0, 255, 0), 1)

    def test_draw_boxes(self):
        # given
        box1, box2 = MagicMock(), MagicMock()
        image = MagicMock()
        self.image_processor.draw_box = MagicMock()

        # when
        self.image_processor.draw_boxes(image, [box1, box2])

        # then
        self.image_processor.draw_box.assert_has_calls(
            [call(image, box1), call(image, box2)])

