from unittest import TestCase
from unittest.mock import MagicMock, patch, call

import numpy as np

from car_parts_text_recognition import settings
from car_parts_text_recognition.recognition.contours_analysis.bound_box_analyzer import \
    BoundBoxAnalyzer
from car_parts_text_recognition.recognition.utils.bound_box import BoundBox


class TestBoundBoxAnalyzer(TestCase):
    def setUp(self) -> None:
        self.bound_box_analyzer = BoundBoxAnalyzer()

    def test___init__(self):
        self.assertIsNone(self.bound_box_analyzer._bound_boxes)
        self.assertEqual([], self.bound_box_analyzer._words)
        self.assertEqual(self.bound_box_analyzer._max_hor_distance,
                         settings.BoundBoxAnalysis.MAX_HORIZONTAL_DISTANCE)
        self.assertEqual(self.bound_box_analyzer._max_height_difference,
                         settings.BoundBoxAnalysis.MAX_HEIGHT_DIFFERENCE)
        self.assertEqual(self.bound_box_analyzer._window_cut_vertical,
                         settings.BoundBoxAnalysis.WINDOW_CUT_VERTICAL)
        self.assertEqual(self.bound_box_analyzer._window_cut_horizontal,
                         settings.BoundBoxAnalysis.WINDOW_CUT_HORIZONTAL)
        self.assertEqual(self.bound_box_analyzer._max_letter_distance,
                         settings.BoundBoxAnalysis.MAX_LETTERS_DISTANCE)
        np.testing.assert_array_equal(self.bound_box_analyzer._font_heights,
                                      settings.BoundBoxAnalysis.FONT_HEIGHTS)

    def test_sort_boxes_by_y_start(self):
        # given
        boxes = [BoundBox(1, 5, 2, 3), BoundBox(5, 7, 2, 1),
                 BoundBox(1, 3, 5, 2)]
        self.bound_box_analyzer._bound_boxes = boxes

        # when
        self.bound_box_analyzer._sort_boxes_by_y_start()

        # then
        self.assertEqual(boxes[0], self.bound_box_analyzer._bound_boxes[1])
        self.assertEqual(boxes[1], self.bound_box_analyzer._bound_boxes[2])
        self.assertEqual(boxes[2], self.bound_box_analyzer._bound_boxes[0])

    def test_position_filter_y_too_far(self):
        # given
        self.bound_box_analyzer._window_cut_horizontal = 10
        self.bound_box_analyzer._window_cut_vertical = 10
        contour = BoundBox(1, 2, 3, 4)
        reference = BoundBox(1, 22, 3, 4)

        # when
        result = self.bound_box_analyzer._position_filter(contour, reference)

        # then
        self.assertFalse(result)

    def test_position_filter_x_too_far(self):
        # given
        self.bound_box_analyzer._window_cut_horizontal = 10
        self.bound_box_analyzer._window_cut_vertical = 10
        contour = BoundBox(1, 2, 3, 4)
        reference = BoundBox(21, 2, 3, 4)

        # when
        result = self.bound_box_analyzer._position_filter(contour, reference)

        # then
        self.assertFalse(result)

    def test_position_filter(self):
        # given
        self.bound_box_analyzer._window_cut_horizontal = 10
        self.bound_box_analyzer._window_cut_vertical = 10
        contour = BoundBox(1, 2, 3, 4)
        reference = BoundBox(5, 6, 3, 4)

        # when
        result = self.bound_box_analyzer._position_filter(contour, reference)

        # then
        self.assertTrue(result)

    @patch('builtins.filter')
    @patch('builtins.list')
    def test_find_near_boxes(self, list_mock, filter_mock):
        # given
        box_ref = BoundBox(12, 3, 4, 5)
        filter_mock.return_value = MagicMock()
        list_mock.return_value = 'filtered'

        # when
        result = self.bound_box_analyzer._find_near_boxes(box_ref)

        # then
        filter_mock.assert_called()
        list_mock.assert_called()
        self.assertEqual('filtered', result)

    def test_remove_boxes_by_height(self):
        # given
        self.bound_box_analyzer._font_heights = np.array([5, 10])
        self.bound_box_analyzer._max_height_difference = 1
        self.bound_box_analyzer._bound_boxes = []
        for i in range(10):
            box = BoundBox(0, 0, 0, i)
            self.bound_box_analyzer._bound_boxes.append(box)

        # when
        self.bound_box_analyzer._remove_boxes_by_height()

        # then
        self.assertEqual(
            [4, 5, 6, 9],
            [box.end_y for box in self.bound_box_analyzer._bound_boxes]
        )

    def test_sort_boxes(self):
        # given
        x_starts = [5, 2, 7, 3, 6, 1, 4]
        boxes = []
        for x_start in x_starts:
            box = BoundBox(x_start, 0, 0, 0)
            boxes.append(box)

        # when
        result = self.bound_box_analyzer._sort_boxes(boxes)

        # then
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], [x.start_x for x in result])

    def test_group_boxes_no_boxes(self):
        # given
        boxes = []

        # when
        result = self.bound_box_analyzer._group_boxes(boxes)

        # then
        self.assertEqual([boxes], result)

    def test_group_boxes_one_box(self):
        # given
        boxes = [MagicMock()]

        # when
        result = self.bound_box_analyzer._group_boxes(boxes)

        # then
        self.assertEqual([boxes], result)

    def test_group_boxes_one_box(self):
        # given
        x_starts = [1, 2, 5, 6, 7, 9]
        boxes = []
        for x_start in x_starts:
            box = BoundBox(x_start, 0, x_start + 1, 0)
            boxes.append(box)
        self.bound_box_analyzer._sort_boxes = MagicMock(return_value=boxes)
        self.bound_box_analyzer._max_letter_distance = 1

        # when
        result = self.bound_box_analyzer._group_boxes(boxes)

        # then
        self.assertEqual([[1, 2], [5, 6, 7], [9]],
                         [[r.start_x for r in word] for word in result])

    def test_decide_on_group_length_too_small(self):
        # given
        box1, box2 = MagicMock(), MagicMock()
        group = [box1, box2]
        self.bound_box_analyzer._bound_boxes = MagicMock()
        self.bound_box_analyzer._min_word_length = 3

        # when
        self.bound_box_analyzer._decide_on_group_length(group)

        # then
        self.bound_box_analyzer._bound_boxes.remove.assert_has_calls([
            call(box1), call(box2)
        ])

    def test_decide_on_group_length(self):
        # given
        box1, box2 = MagicMock(), MagicMock()
        group = [box1, box2]
        self.bound_box_analyzer._words = MagicMock()
        self.bound_box_analyzer._min_word_length = 2

        # when
        self.bound_box_analyzer._decide_on_group_length(group)

        # then
        self.bound_box_analyzer._words.append.assert_has_calls([
            call([box1, box2])
        ])

    def test_check_contour_part_of_word(self):
        # given
        box1, box2, box3, box4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        groups = [[box1], [box2, box4], [box3]]
        self.bound_box_analyzer._group_boxes = MagicMock(return_value=groups)
        self.bound_box_analyzer._decide_on_group_length = MagicMock()
        # when
        self.bound_box_analyzer._check_contour_part_of_word(box2, [box1, box2])

        # then
        self.bound_box_analyzer._group_boxes.assert_called()
        self.bound_box_analyzer._decide_on_group_length.assert_called_with(
            [box2, box4])

    def test_letters_to_text_boxes(self):
        # given
        box1 = BoundBox(1, 2, 3, 4)
        box2 = BoundBox(5, 6, 7, 8)
        box3 = BoundBox(1, 2, 5, 6)
        self.bound_box_analyzer._words = [[box1, box2], [box3]]

        # when
        result = self.bound_box_analyzer._letters_to_text_boxes()

        # then
        self.assertEqual([BoundBox(1, 2, 7, 8), BoundBox(1, 2, 5, 6)], result)

    def test_analyze_bound_boxes(self):
        # given
        boxes = [MagicMock()]
        self.bound_box_analyzer._remove_boxes_by_height = MagicMock()
        self.bound_box_analyzer._sort_boxes_by_y_start = MagicMock()
        self.bound_box_analyzer._find_near_boxes = MagicMock()
        self.bound_box_analyzer._check_contour_part_of_word = MagicMock()
        self.bound_box_analyzer._letters_to_text_boxes = MagicMock(
            return_value=['boxes'])

        # when
        result = self.bound_box_analyzer.analyze_bound_boxes(boxes)

        # then
        self.bound_box_analyzer._remove_boxes_by_height.assert_called()
        self.bound_box_analyzer._sort_boxes_by_y_start.assert_called()
        self.bound_box_analyzer._find_near_boxes.assert_called()
        self.bound_box_analyzer._check_contour_part_of_word.assert_called()
        self.bound_box_analyzer._letters_to_text_boxes.assert_called()
        self.assertEqual(([], ['boxes']), result)

