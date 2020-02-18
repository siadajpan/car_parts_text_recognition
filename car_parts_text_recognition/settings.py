import os
from pathlib import Path


class Locations:
    ROOT = Path(__file__).parent
    FILES = os.path.join(ROOT, 'files')
    DIAMOND = os.path.join(FILES, 'diamond.png')
    NEURAL_NET = os.path.join(FILES, 'frozen_east_text_detection.pb')
