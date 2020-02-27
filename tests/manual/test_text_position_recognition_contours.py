import os

import cv2
import numpy as np

from car_parts_text_recognition import settings
from car_parts_text_recognition.recognition.text_position_recognition_contours import \
    TextPositionRecognitionContours

if __name__ == '__main__':
    img = cv2.imread(os.path.join(settings.Locations.FILES, '1.jpg'))
    t = TextPositionRecognitionContours()
    t.update_image(img)
    letters, words = t.find_text_boxes()

    t.draw_boxes(img, words)
    # for word_letters in letters:
    #     t.draw_boxes(img, word_letters)

    cv2.imshow('g', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
