import os
from pathlib import Path

import numpy as np


class Locations:
    ROOT = Path(__file__).parent
    FILES = os.path.join(ROOT, 'files')
    DIAMOND = os.path.join(FILES, 'diamond.png')
    NEURAL_NET = os.path.join(FILES, 'frozen_east_text_detection.pb')


class BoundBoxAnalysis:
    MAX_HORIZONTAL_DISTANCE = 10
    MAX_HEIGHT_DIFFERENCE = 2
    WINDOW_CUT_VERTICAL = 2
    WINDOW_CUT_HORIZONTAL = 100
    MAX_LETTERS_DISTANCE = 15
    FONT_HEIGHTS = np.array([8, 14])
    MIN_WORD_LENGTH = 2
