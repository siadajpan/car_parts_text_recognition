from unittest import TestCase
from unittest.mock import MagicMock, patch, call

import numpy as np

from car_parts_text_recognition.recognition.contours_analysis.bound_box_analyzer import \
    BoundBoxAnalyzer
from car_parts_text_recognition.recognition.contours_analysis.text_position_recognition_contours import \
    TextPositionRecognitionContours
from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class TestTextPositionRecognitionNoNet(TestCase):
    def setUp(self) -> None:
        self.recognition = TextPositionRecognitionContours()

    def test___init__(self):
        self.assertIsNone(self.recognition._image)
        self.assertIsNone(self.recognition._image_binary)
        self.assertIsInstance(self.recognition._box_analyzer,
                              BoundBoxAnalyzer)

    def test_update_image(self):
        # given
        self.recognition._pre_process_image = MagicMock()

        # when
        self.recognition._update_image('img')

        # then
        self.assertEqual('img', self.recognition._image)
        self.recognition._pre_process_image.assert_called()

    def test__pre_process_image(self):
        # given
        self.recognition.to_gray = MagicMock()
        self.recognition.threshold_image = MagicMock()
        self.recognition.invert = MagicMock(return_value='inv')

        # when
        self.recognition._pre_process_image()

        # then
        self.recognition.to_gray.assert_called()
        self.recognition.threshold_image.assert_called()
        self.recognition.invert.assert_called()
        self.assertEqual('inv', self.recognition._image_binary)

    @patch('cv2.rectangle')
    def test_draw_box(self, rectangle_mock):
        # given
        rectangle = BoundBox(1, 2, 3, 4)
        image = MagicMock()

        # when
        self.recognition.draw_box(image, rectangle)

        # then
        rectangle_mock.assert_called_with(image, (0, 1), (4, 5), (0, 255, 0), 1)

    def test_draw_boxes(self):
        # given
        box1, box2 = MagicMock(), MagicMock()
        image = MagicMock()
        self.recognition.draw_box = MagicMock()

        # when
        self.recognition.draw_boxes(image, [box1, box2])

        # then
        self.recognition.draw_box.assert_has_calls(
            [call(image, box1), call(image, box2)])

    def test__contours_to_boxes(self):
        # given
        contour1 = np.array([[[1, 4]], [[2, 15]]])
        contour2 = np.array([[[10, 5]], [[1, 6]], [[3, 25]]])

        # when
        bounds = self.recognition._contours_to_boxes([contour1, contour2])

        # then
        self.assertEqual([BoundBox(1, 4, 2, 15), BoundBox(1, 5, 10, 25)],
                         bounds)

    def test_find_text_boxes(self):
        # given
        self.recognition._update_image = MagicMock()
        self.recognition.fill_holes = MagicMock()
        self.recognition.find_contours = MagicMock()
        self.recognition._contours_to_boxes = MagicMock()
        self.recognition._box_analyzer = MagicMock()
        self.recognition._box_analyzer.analyze_bound_boxes = MagicMock(
            return_value=('letters', 'words'))

        # when
        result = self.recognition.find_text_boxes(MagicMock())

        # then
        self.recognition._update_image.assert_called()
        self.recognition.fill_holes.assert_called()
        self.recognition.find_contours.assert_called()
        self.recognition._contours_to_boxes.assert_called()
        self.recognition._box_analyzer.analyze_bound_boxes.assert_called()
        self.assertEqual(('letters', 'words'), result)
