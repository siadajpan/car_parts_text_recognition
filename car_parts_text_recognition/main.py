import os

import cv2
import numpy as np

from car_parts_text_recognition import settings
from car_parts_text_recognition.recognition.contours_analysis.text_position_recognition_contours import \
    TextPositionRecognitionContours
from car_parts_text_recognition.recognition.neural_network.pixel_text_recognition import \
    PixelTextRecognition
from car_parts_text_recognition.recognition.neural_network.text_postion_recognition_nn import \
    TextPositionRecognitionNN


class Main:
    def __init__(self):
        self.text_position_recognition = TextPositionRecognitionContours()

    def predict_text_position(self, image: np.array):
        boxes = self.text_position_recognition.find_text_boxes(image)

        return boxes

    def draw_boxes(self, image, boxes):
        self.text_position_recognition.draw_boxes(image, boxes)

    def update_boxes(self, image, boxes):
        boxes = self.pixel_position.filter_bounds(image, boxes)

        return boxes


if __name__ == '__main__':
    img = cv2.imread(os.path.join(settings.Locations.FILES, '1.jpg'))

    main = Main()
    bxs = main.predict_text_position(img)
    # main.draw_boxes(img, bxs)
    new_boxes = main.update_boxes(img, bxs)
    main.draw_boxes(img, new_boxes)
    cv2.imshow('g', img)
    while True:
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            break
