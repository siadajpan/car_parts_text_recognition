from typing import Optional, List

import cv2
import numpy as np
from imutils import object_detection

from car_parts_text_recognition import settings
from car_parts_text_recognition.image_processor.image_processor import \
    ImageProcessor
from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class TextPositionRecognitionNN(ImageProcessor):
    def __init__(self):
        super().__init__()
        self._net = None
        self._layerNames: Optional[List[str]] = None
        self._init_neural_network()
        self._image_pre_processor = ImageProcessor()
        self._rectangles = []
        self._confidences = []

    def _init_neural_network(self):
        self._layerNames = ["feature_fusion/Conv_7/Sigmoid",
                           "feature_fusion/concat_3"]

        self._net = cv2.dnn.readNet(settings.Locations.NEURAL_NET)

    def _init_pass(self):
        self._rectangles = []
        self._confidences = []

    @staticmethod
    def _row_data_to_rect(col_index, row_index, row_data):
        # compute the offset factor as our resulting feature
        # maps will be 4x smaller than the input image
        (offsetX, offsetY) = (col_index * 4.0, row_index * 4.0)

        # use the geometry volume to derive the width and height
        # of the bounding box
        h = row_data[0][col_index] + row_data[2][col_index]
        w = row_data[1][col_index] + row_data[3][col_index]

        # compute both the starting and ending (x, y)-coordinates
        # for the text prediction bounding box
        end_x = int(offsetX + row_data[1][col_index])
        end_y = int(offsetY + row_data[2][col_index])
        start_x = int(end_x - w)
        start_y = int(end_y - h)

        return start_x, start_y, end_x, end_y

    @staticmethod
    def _get_row_and_confidence(scores, geometry, row_index):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        confidence = scores[0, 0, row_index]
        row_data = geometry[0, 0:4, row_index]

        return row_data, confidence

    def _update_rectangles(self, col_index, row_index, row_data):
        rectangle = self._row_data_to_rect(col_index, row_index, row_data)
        self._rectangles.append(rectangle)

    def _update_confidences(self, confidence):
        self._confidences.append(confidence)

    def _process_columns(self, scores, geometry, row_index):
        row_data, confidence = self._get_row_and_confidence(
            scores, geometry, row_index
        )
        number_columns = scores.shape[3]

        for col_index in range(0, number_columns):
            # if our score does not have sufficient probability,
            # ignore it
            position_confidence = confidence[col_index]
            if position_confidence < 0.5:
                continue

            self._update_rectangles(col_index, row_index, row_data)
            self._update_confidences(position_confidence)

    def _decode_predictions(self, scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        number_rows = scores.shape[2]

        for row_index in range(0, number_rows):
            self._process_columns(scores, geometry, row_index)

    def _forward_pass_nn(self, blob):
        self._net.setInput(blob)
        (scores, geometry) = self._net.forward(self._layerNames)

        return scores, geometry

    def _process_rectangles(self):
        boxes = object_detection.non_max_suppression(
            np.array(self._rectangles), probs=self._confidences
        )

        return boxes

    @staticmethod
    def _blob_from_image(image):
        height, width = image.shape[:2]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        mean_px = (np.mean(image),) * 3
        blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), mean_px,
                                     swapRB=True, crop=False)

        return blob

    @staticmethod
    def _rescale_boxes(boxes: np.array, fx: float, fy: float):
        boxes = boxes.astype(float)
        boxes[:, 0] /= fx
        boxes[:, 1] /= fy
        boxes[:, 2] /= fx
        boxes[:, 3] /= fy
        return boxes.astype(int)

    @staticmethod
    def _arrays_to_bound_boxes(boxes: np.array):
        rectangles = [BoundBox(*box) for box in boxes]

        return rectangles

    def predict_text_positions(self, image: np.array):
        old_height, old_width = image.shape[:2]
        image = self._image_pre_processor.resize_picture_32(image)
        blob = self._blob_from_image(image)
        scores, geometry = self._forward_pass_nn(blob)
        self._decode_predictions(scores, geometry)
        boxes = self._process_rectangles()

        height, width = image.shape[:2]
        boxes = self._rescale_boxes(boxes, width / old_width,
                                    height / old_height)
        rectangles = self._arrays_to_bound_boxes(boxes)

        return rectangles

    @staticmethod
    def draw_boxes(image: np.array, boxes: List[BoundBox]):
        for rectangle in boxes:
            start_x, start_y = rectangle.start_x, rectangle.start_y
            end_x, end_y = rectangle.end_x, rectangle.end_y
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                          (0, 255, 0), 1)
