from typing import Optional, List, Tuple

import cv2
import numpy as np

from car_parts_text_recognition.image_processor.image_processor import \
    ImageProcessor
from car_parts_text_recognition.recognition.contours_analysis.bound_box_analyzer import \
    BoundBoxAnalyzer
from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class TextPositionRecognitionContours(ImageProcessor):
    """
    Recognition of text on the image with the use of contour analyzing.
    """
    def __init__(self):
        super().__init__()
        self._image: Optional[np.array] = None
        self._image_binary: Optional[np.array] = None
        self._box_analyzer = BoundBoxAnalyzer()

    def _update_image(self, image: np.array) -> None:
        self._image = image
        self._pre_process_image()

    def _pre_process_image(self) -> None:
        """
        Pre process updated image. Convert it to grayscale, threshold it to
        get binary image and invert it. Image inversion helps with function
        fill holes, that is looking for white blobs
        :return: None
        """
        gray = self.to_gray(self._image)
        binary = self.threshold_image(gray)
        inverted = self.invert(binary)

        self._image_binary = inverted

    @staticmethod
    def _contours_to_boxes(contours: np.array) -> List[BoundBox]:
        """
        Create Rectangle boxes around contours
        :param contours: np.array with 3 el shape
        :return:
        """
        bounds = []
        for i, contour in enumerate(contours):
            min_x = min(contour[:, 0, 0])
            max_x = max(contour[:, 0, 0])
            min_y = min(contour[:, 0, 1])
            max_y = max(contour[:, 0, 1])
            bounds.append(BoundBox(min_x, min_y, max_x, max_y))

        return bounds

    def find_text_boxes(self, image) \
            -> Tuple[List[List[BoundBox]], List[BoundBox]]:
        """
        Use image processing function fill holes, to fill all close shaped
        contours on the image. This helps reducing amount of contours to analyze
        Then, find contours on the image. Translate them into list of BoundBox
        rectangles and analyze them to remove all bounds that are not texts.
        :return: list of bounds around letters that are within each word,
        and list of bounds around each word
        """
        self._update_image(image)
        self.fill_holes(self._image_binary)
        contours = self.find_contours(self._image_binary)
        bound_boxes = self._contours_to_boxes(contours)
        letters, words = self._box_analyzer.analyze_bound_boxes(bound_boxes)

        return letters, words
