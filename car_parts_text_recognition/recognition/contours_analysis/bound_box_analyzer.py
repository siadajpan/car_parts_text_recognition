from typing import List, Optional

import numpy as np

from car_parts_text_recognition.recognition.utils.bound_box import BoundBox
from car_parts_text_recognition.settings import BoundBoxAnalysis


class BoundBoxAnalyzer:
    def __init__(self):
        self._bound_boxes: Optional[List[BoundBox]] = None
        self._words: Optional[List[List[BoundBox]]] = []
        self._max_hor_distance = BoundBoxAnalysis.MAX_HORIZONTAL_DISTANCE
        self._max_height_difference = BoundBoxAnalysis.MAX_HEIGHT_DIFFERENCE
        self._window_cut_vertical = BoundBoxAnalysis.WINDOW_CUT_VERTICAL
        self._window_cut_horizontal = BoundBoxAnalysis.WINDOW_CUT_HORIZONTAL
        self._max_letter_distance = BoundBoxAnalysis.MAX_LETTERS_DISTANCE
        self._font_heights = BoundBoxAnalysis.FONT_HEIGHTS
        self._min_word_length = BoundBoxAnalysis.MIN_WORD_LENGTH

    def _sort_boxes_by_y_start(self):
        self._bound_boxes = sorted(self._bound_boxes,
                                   key=lambda rect: rect.start_y)

    def _position_filter(self, contour: BoundBox, ref: BoundBox):
        """
        Position check to use in self._find_near_contours
        :param contour:
        :param ref:
        :return:
        """
        y_filter = abs(ref.start_y - contour.start_y) < \
                   self._window_cut_vertical
        x_filter = abs(ref.start_x - contour.start_x) < \
                   self._window_cut_horizontal

        return x_filter and y_filter

    def _find_near_boxes(self, contour: BoundBox) -> List[BoundBox]:
        """
        Filter boxes using position filter to get ones in the same line
        and close to contour
        :param contour: reference contour to find the near ones
        :return: list of near contours
        """
        filtered = list(filter(lambda r: self._position_filter(contour, r),
                               self._bound_boxes))

        return filtered

    def _remove_boxes_by_height(self):
        """
        Updates self._bound_boxes.
        Remove all boxes that are different in height than any value from
        self._font_heights. Threshold is in self._max_height_difference
        :return: None
        """
        filtered = filter(
            lambda rect: any(
                np.less_equal(abs(self._font_heights - rect.height),
                              self._max_height_difference)),
            self._bound_boxes
        )
        self._bound_boxes = list(filtered)

    @staticmethod
    def _sort_boxes(same_line_boxes: List[BoundBox]) -> List[BoundBox]:
        sorted_bound_boxes = sorted(same_line_boxes,
                                    key=lambda rect: rect.start_x)

        return sorted_bound_boxes

    def _group_boxes(self, same_line_boxes: List[BoundBox]) \
            -> List[List[BoundBox]]:
        """
        Create groups of boxes that are close to another.
        Note, those boxes should already be filtered, so they should be in
        the same line. If two or more words are in the same line, we need to
        separate them.
        :param same_line_boxes: list of boxes that
        :return: list of grouped boxes (list of lists of boxes)
        """
        if len(same_line_boxes) <= 1:
            return [same_line_boxes]

        sorted_boxes = self._sort_boxes(same_line_boxes)

        bound_boxes_mid_x = np.array([contour.mid_x for contour in sorted_boxes])
        distances = bound_boxes_mid_x[1:] - bound_boxes_mid_x[:-1]

        group = [sorted_boxes[0]]
        groups = [group]

        for distance, contour in zip(distances, sorted_boxes[1:]):
            if distance <= self._max_letter_distance:
                group.append(contour)
            else:
                group = [contour]
                groups.append(group)

        return groups

    def _decide_on_group_length(self, group: List[BoundBox]):
        """
        Remove words that have only e.g. 1 or 2 letters - this is probably not
        a word
        :param group: list of boxes
        :return: None
        """
        if len(group) < self._min_word_length:
            for contour in group:
                self._bound_boxes.remove(contour)
        else:
            self._words.append(group)

    def _check_contour_part_of_word(self, contour, near_bound_boxes):
        groups = self._group_boxes(near_bound_boxes)
        for group in groups:
            if contour in group:
                self._decide_on_group_length(group)

    def _letters_to_text_boxes(self):
        text_boxes = [BoundBox(group[0].start_x, group[0].start_y,
                               group[-1].end_x, group[-1].end_y)
                      for group in self._words]

        return text_boxes

    def analyze_bound_boxes(self, bound_boxes):
        self._bound_boxes = bound_boxes
        self._remove_boxes_by_height()
        self._sort_boxes_by_y_start()

        for contour in self._bound_boxes:
            near_bound_boxes = self._find_near_boxes(contour)
            self._check_contour_part_of_word(contour, near_bound_boxes)

        text_boxes = self._letters_to_text_boxes()

        return self._words, text_boxes
