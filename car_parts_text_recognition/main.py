import os

import cv2
import numpy as np

from car_parts_text_recognition import settings
from car_parts_text_recognition.recognition.text_postion_recognition import \
    TextPositionRecognition


class Main:
    def __init__(self):
        self.text_position_recognition = TextPositionRecognition()

    def predict_text_position(self, image: np.array):
        boxes = self.text_position_recognition.predict_text_positions(image)

        return boxes

    def draw_boxes(self, image, boxes):
        self.text_position_recognition.draw_boxes(image, boxes)


if __name__ == '__main__':
    img = cv2.imread(os.path.join(settings.Locations.FILES, '1.jpg'))

    main = Main()
    bxs = main.predict_text_position(img)
    main.draw_boxes(img, bxs)

    cv2.imshow('g', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
