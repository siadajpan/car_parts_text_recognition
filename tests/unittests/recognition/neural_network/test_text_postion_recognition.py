from unittest import TestCase
from unittest.mock import MagicMock, patch, call

import numpy as np

from car_parts_text_recognition.image_processor.image_processor import \
    ImageProcessor
from car_parts_text_recognition.recognition.neural_network.text_postion_recognition_nn import \
    TextPositionRecognitionNN
from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class TestTextPositionRecognition(TestCase):
    @patch.object(TextPositionRecognitionNN, 'init_neural_network')
    def setUp(self, init_nn_mock) -> None:
        self.init_nn_mock = init_nn_mock
        self.recognition = TextPositionRecognitionNN()

    def test___init__(self):
        self.assertIsNone(self.recognition._net)
        self.init_nn_mock.assert_called()
        self.assertIsInstance(self.recognition._image_pre_processor,
                              ImageProcessor)
        self.assertEqual([], self.recognition._rectangles)
        self.assertEqual([], self.recognition._confidences)

    @patch('cv2.dnn.readNet')
    def test_init_neural_network(self, read_net_mock):
        # when
        self.recognition._init_neural_network()

        # then
        self.assertEqual(["feature_fusion/Conv_7/Sigmoid",
                          "feature_fusion/concat_3"],
                         self.recognition._layerNames)
        read_net_mock.assert_called()

    def test_init_pass(self):
        # given
        self.recognition._rectangles = [123]
        self.recognition._confidences = [123]

        # when
        self.recognition._init_pass()

        # then
        self.assertEqual([], self.recognition._rectangles)
        self.assertEqual([], self.recognition._confidences)

    def test_row_data_to_rect(self):
        # given
        row_index = 5
        col_index = 1
        row_data = [[1, 2, 3, 4], ] * 4
        
        # when
        result = self.recognition._row_data_to_rect(col_index, row_index,
                                                    row_data)
        
        # then
        self.assertEqual((2, 18, 6, 22), result)

    def test_get_row_and_confidence(self):
        # given
        row_index = 2
        scores = np.array([[[1, 2, 3, 4]]])
        geometry = np.array([[[1, 2, 3, 4]], ] * 4)

        # when
        result = self.recognition._get_row_and_confidence(
            scores, geometry, row_index
        )
        
        # then
        self.assertEqual((3, [3]), result)

    def test_update_rectangles(self):
        # given
        self.recognition._row_data_to_rect = MagicMock(return_value='rect')
        col_index = 1
        row_index = 4
        row_data = [1, 2, 3, 4]

        # when
        self.recognition._update_rectangles(col_index, row_index, row_data)

        # then
        self.recognition._row_data_to_rect.assert_called()
        self.assertEqual(['rect'], self.recognition._rectangles)

    def test_update_confidence(self):
        # given
        confidence = 0.2

        # when
        self.recognition._update_confidences(confidence)

        # then
        self.assertEqual([confidence], self.recognition._confidences)

    def test_process_columns_confidence_too_small(self):
        # given
        self.recognition._get_row_and_confidence = MagicMock(
            return_value=('row_data', [0.2]))
        self.recognition._update_rectangles = MagicMock()
        self.recognition._update_confidences = MagicMock()
        scores = MagicMock()
        scores.shape = [1, 1, 1, 1]
        geometry = MagicMock()

        # when
        self.recognition._process_columns(scores, geometry, 1)
        
        # then
        self.recognition._get_row_and_confidence.assert_called()
        self.recognition._update_confidences.assert_not_called()
        self.recognition._update_rectangles.assert_not_called()

    def test_process_columns_confidence(self):
        # given
        self.recognition._get_row_and_confidence = MagicMock(
            return_value=('row_data', [0.7]))
        self.recognition._update_rectangles = MagicMock()
        self.recognition._update_confidences = MagicMock()
        scores = MagicMock()
        scores.shape = [1, 1, 1, 1]
        geometry = MagicMock()

        # when
        self.recognition._process_columns(scores, geometry, 1)

        # then
        self.recognition._get_row_and_confidence.assert_called()
        self.recognition._update_confidences.assert_called()
        self.recognition._update_rectangles.assert_called()

    def test_decode_predictions(self):
        # given
        scores = MagicMock()
        scores.shape = [1, 1, 2]
        geometry = MagicMock()
        self.recognition._process_columns = MagicMock()

        # when
        self.recognition._decode_predictions(scores, geometry)
        
        # then
        self.recognition._process_columns.assert_has_calls([
            call(scores, geometry, 0),
            call(scores, geometry, 1)])

    def test_forward_pass_nn(self):
        # given
        self.recognition._net = MagicMock()
        self.recognition._net.forward = MagicMock(return_value=(1, 2))

        # when
        result = self.recognition._forward_pass_nn(MagicMock())

        # then
        self.recognition._net.setInput.assert_called()
        self.recognition._net.forward.assert_called()
        self.assertEqual((1, 2), result)

    @patch('imutils.object_detection.non_max_suppression')
    def test_process_rectangles(self, supression_mock):
        # given
        supression_mock.return_value = 'boxes'
        
        # when
        result = self.recognition._process_rectangles()
        
        # then
        supression_mock.assert_called()
        self.assertEqual('boxes', result)

    @patch('numpy.mean')
    @patch('cv2.dnn.blobFromImage')
    def test_blob_from_image(self, blob_mock, mean_mock):
        # given
        image = MagicMock()
        image.shape = [15, 14, 3]
        blob_mock.return_value = 'blob'
        
        # when
        result = self.recognition._blob_from_image(image)
        
        # then
        mean_mock.assert_called()
        blob_mock.assert_called()
        self.assertEqual('blob', result)

    def test_rescale_boxes(self):
        # given
        boxes = np.array([[1, 2, 3, 4]])
        
        # when
        result = self.recognition._rescale_boxes(boxes, 0.5, 0.7)
        
        # then
        np.testing.assert_array_equal([[2, 2, 6, 5]], result)

    def test_boxes_to_rectangles(self):
        # given
        boxes = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])

        # when
        result = self.recognition._arrays_to_bound_boxes(boxes)

        # then
        self.assertEqual(result, [BoundBox(1, 2, 3, 4), BoundBox(4, 5, 6, 7)])

    def test_predict_text_positions(self):
        # given
        image_old = MagicMock()
        image_old.shape = [15, 14, 3]
        image_new = MagicMock()
        image_new.shape = [10, 10, 3]

        self.recognition._image_pre_processor = MagicMock()
        self.recognition._image_pre_processor.resize_picture_32 = MagicMock(
            return_value=image_new
        )
        self.recognition._blob_from_image = MagicMock()
        self.recognition._forward_pass_nn = MagicMock(return_value=(1, 2))
        self.recognition._decode_predictions = MagicMock()
        self.recognition._process_rectangles = MagicMock()
        self.recognition._rescale_boxes = MagicMock()
        self.recognition._arrays_to_bound_boxes = MagicMock()

        # when
        self.recognition.predict_text_positions(image_old)

        # then
        self.recognition._image_pre_processor.resize_picture_32.assert_called()
        self.recognition._blob_from_image.assert_called()
        self.recognition._forward_pass_nn.assert_called()
        self.recognition._decode_predictions.assert_called()
        self.recognition._process_rectangles.assert_called()
        self.recognition._rescale_boxes.assert_called()
        self.recognition._arrays_to_bound_boxes.assert_called()

    @patch('cv2.rectangle')
    def test_draw_boxes(self, rectangle_mock):
        # given
        image = MagicMock()
        boxes = [[1, 2, 3, 4]]
        
        # when
        self.recognition.draw_boxes(image, boxes)
        
        # then
        rectangle_mock.assert_called()

