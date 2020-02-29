import os

import cv2

from car_parts_text_recognition import settings
from car_parts_text_recognition.recognition.contours_analysis. \
    text_position_recognition_contours import TextPositionRecognitionContours
from car_parts_text_recognition.recognition.neural_network.pixel_text_recognition import \
    PixelTextRecognition
from car_parts_text_recognition.recognition.neural_network.text_postion_recognition_nn import \
    TextPositionRecognitionNN

if __name__ == '__main__':
    img = cv2.imread(os.path.join(settings.Locations.FILES, '1.jpg'))
    t = TextPositionRecognitionNN()
    t1 = PixelTextRecognition()
    words = t.predict_text_positions(img)
    words_upgraded = t1.filter_bounds(img, words)
    t.draw_boxes(img, words_upgraded)

    cv2.imshow('g', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
