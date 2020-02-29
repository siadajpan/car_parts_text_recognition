from copy import copy
from typing import Optional, Tuple, List

import cv2
import numpy as np

from car_parts_text_recognition.image_processor.image_processor import \
    ImageProcessor
from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class PixelTextRecognition:
    def __init__(self):
        self._image: Optional[np.array] = None
        self._image_binary: Optional[np.array] = None
        self._boxes: Optional[List[BoundBox]] = None
        self._current_image: Optional[np.array] = None
        self._current_box: Optional[BoundBox] = None
        self._image_processor = ImageProcessor()

    def update_image(self, image: np.array):
        self._image = image
        self._pre_process_image()

    def _pre_process_image(self):
        gray = self._image_processor.to_gray(self._image)
        binary = self._image_processor.threshold_image(gray)
        inverted = self._image_processor.invert(binary)

        self._image_binary = inverted

    def update_boxes(self, boxes: List[BoundBox]):
        self._boxes = boxes

    def _get_contours_bounds(self):
        self._image_processor.fill_holes(self._current_image)
        contours = self._image_processor.find_contours(self._current_image)

        bounds = []

        for i, contour in enumerate(contours):
            min_x = min(contour[:, 0, 0])
            max_x = max(contour[:, 0, 0])
            min_y = min(contour[:, 0, 1])
            max_y = max(contour[:, 0, 1])
            bounds.append(BoundBox(min_x, min_y, max_x, max_y))

        return bounds

    @staticmethod
    def _analyze_contour_bounds(bounds: List[BoundBox]) -> Optional[BoundBox]:
        """
        Find median of y_starts and y_ends. Filter rectangles that have similar
        y_start and y_end
        :param bounds: list of rectangle bonds
        :return: rectangle final bonds
        """
        y_bounds = np.array([(rect.start_y, rect.end_y) for rect in bounds])
        y_start = np.median(y_bounds[:, 0])
        y_end = np.median(y_bounds[:, 1])

        rects_in = [rect for rect in bounds if
                    abs(rect.start_y - y_start) < 4
                    and abs(rect.end_y - y_end) < 4]

        if len(rects_in) < 3:
            return None

        x_bounds = np.array([(rect.start_x, rect.end_x) for rect in rects_in])
        x_start = np.min(x_bounds[:, 0])
        x_end = np.max(x_bounds[:, 1])

        bounds = BoundBox(x_start, int(y_start), x_end, int(y_end))

        return bounds

    def calculate_new_box(self, relative_box: BoundBox):
        width, height = self._current_box.width, self._current_box.height
        if relative_box.start_x < 5:
            self._current_box.start_x -= int(width / 5)
        if relative_box.start_y < 5:
            self._current_box.start_y -= int(height / 5)
        if abs(relative_box.end_x - width) < 5:
            self._current_box.end_x += int(width / 5)
        if abs(relative_box.end_y - height) < 5:
            self._current_box.end_y += int(height / 5)

    def relative_to_absolute(self, relative_box: BoundBox):
        self._current_box.start_x += relative_box.start_x - 2
        self._current_box.start_y += relative_box.start_y - 2
        self._current_box.end_x = \
            self._current_box.start_x + relative_box.width + 4
        self._current_box.end_y = \
            self._current_box.start_y + relative_box.height + 4

    def filter_current_bound(self, box: BoundBox):
        self._current_box = box

        for i in range(5):
            self._current_image = self._image_processor.cut_box(
                self._image_binary, self._current_box)
            bounds = self._get_contours_bounds()
            relative_box = self._analyze_contour_bounds(bounds)
            if relative_box is None:
                return False

            old_box = copy(self._current_box)
            self.calculate_new_box(relative_box)
            if self._current_box == old_box:
                self.relative_to_absolute(relative_box)
                return True

        return False

    def filter_bounds(self, image: np.array, boxes: List[BoundBox]):
        self.update_image(image)
        self.update_boxes(boxes)
        filtered_boxes = []

        for box in boxes:
            success = self.filter_current_bound(box)
            if success:
                filtered_boxes.append(self._current_box)

        self.update_boxes(filtered_boxes)

        return self._boxes

if __name__ == '__main__':
    img = cv2.imread('../files/1.jpg')

    p = PixelTextRecognition()
    p.update_image(img)
    box = BoundBox(240, 70, 276, 100)

    # rect = p.calculate_bonds(img)
    # print(rect)
    # p.update_boxes(rect)
    # p._image = p._image_processor.cut_box(img, rect)
    new_boxes = p.filter_bounds(img, [box])
    box = new_boxes[0]
    cv2.rectangle(p._image, (box.start_x, box.start_y),
                  (box.end_x, box.end_y), (155, 155, 129), 2)
    cv2.imshow('f', p._image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
