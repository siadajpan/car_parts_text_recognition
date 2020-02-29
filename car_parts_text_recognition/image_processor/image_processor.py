from typing import Optional, Tuple, List

import cv2
import numpy as np

from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class ImageProcessor:
    def __init__(self):
        self.image: Optional[np.array] = None

    @staticmethod
    def to_gray(image: np.array):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray

    @staticmethod
    def threshold_image(image: np.array):
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

        return threshold

    @staticmethod
    def invert(image: np.array):
        inverted = cv2.bitwise_not(image)

        return inverted

    @staticmethod
    def resize_picture_32(image: np.array):
        height, width = image.shape[:2]
        new_height = round(height / 32) * 32
        new_width = round(width / 32) * 32

        new_image = cv2.resize(image, (new_width, new_height))

        return new_image

    @staticmethod
    def cut_box(image: np.array, box: BoundBox):
        image_cut = image[box.start_y: box.end_y, box.start_x: box.end_x]

        return image_cut

    @staticmethod
    def find_contours(image):
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def fill_holes(self, image_binary):
        contours = self.find_contours(image_binary)

        for i, contour in enumerate(contours):
            cv2.drawContours(image_binary, contours, i, 255,
                             thickness=-1)

    @staticmethod
    def draw_box(image: np.array, rectangle: BoundBox,
                 color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """
        Draw box on the image with position from BoundBox
        :param image: image to draw on
        :param rectangle: BoundBox type, with x start, x end, y start, y end
        :param color: rgb color to draw rectangle
        :return: None
        """
        start_x, start_y = rectangle.start_x - 1, rectangle.start_y - 1
        end_x, end_y = rectangle.end_x + 1, rectangle.end_y + 1
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 1)

    def draw_boxes(self, image: np.array, boxes: List[BoundBox]) -> None:
        """
        Draw boxes on the image
        :param image:
        :param boxes: list of BoundBox with pixel positions
        :return: None
        """
        for rectangle in boxes:
            self.draw_box(image, rectangle)
