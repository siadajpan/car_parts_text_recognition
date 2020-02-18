from unittest import TestCase
from unittest.mock import MagicMock, patch

import cv2

from car_parts_text_recognition.image_processor.image_processor import \
    ImageProcessor


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

    @patch('cv2.resize')
    def test_resize_picture_32(self, resize_mock):
        # given
        image = MagicMock()
        image.shape = (151, 251, 3)
        
        # when
        self.image_processor.resize_picture_32(image)
        
        # then
        resize_mock.assert_called_with(image, (256, 160))


