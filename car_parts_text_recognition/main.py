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
        self.text_position_recognition_nn = TextPositionRecognitionNN()
        self.pixel_position = PixelTextRecognition()

    def predict_text_position_nn(self, image: np.array):
        boxes = self.text_position_recognition_nn.predict_text_positions(image)

        return boxes

    def find_text_positions_contours(self, image: np.array):
        letters, words = self.text_position_recognition.find_text_boxes(image)

        return letters, words

    def draw_boxes(self, image, boxes):
        self.text_position_recognition.draw_boxes(image, boxes)

    def update_boxes(self, image, boxes):
        boxes = self.pixel_position.filter_bounds(image, boxes)

        return boxes


if __name__ == '__main__':
    IMAGE_PATH = os.path.join(settings.Locations.FILES, '1.jpg')

    img = cv2.imread(IMAGE_PATH)

    main = Main()
    bxs_letters, bxs_words = main.find_text_positions_contours(img)
    main.draw_boxes(img, bxs_words)
    cv2.imshow('g', img)

    while True:
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            break
