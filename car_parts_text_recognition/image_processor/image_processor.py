from typing import Optional

import cv2
import numpy as np


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
