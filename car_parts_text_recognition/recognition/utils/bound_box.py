from typing import Optional


class BoundBox:
    def __init__(self, start_x: int, start_y: int, end_x: int, end_y: int):
        self.start_x: Optional[int] = start_x
        self.start_y: Optional[int] = start_y
        self.end_x: Optional[int] = end_x
        self.end_y: Optional[int] = end_y

    def __repr__(self):
        representation = f'\nx0: {self.start_x}, y0: {self.start_y}, ' \
                         f'xe: {self.end_x}, ye: {self.end_y}'

        return representation

    def __eq__(self, other):
        same = self.start_x == other.start_x \
               and self.start_y == other.start_y \
               and self.end_x == other.end_x \
               and self.end_y == other.end_y

        return same

    @property
    def width(self):
        return self.end_x - self.start_x

    @property
    def height(self):
        return self.end_y - self.start_y

    @property
    def mid_x(self):
        return (self.start_x + self.end_x) / 2

    @property
    def mid_y(self):
        return (self.start_y + self.end_y) / 2

    @classmethod
    def from_width_height(cls, start_x, start_y, width, height):
        rectangle = cls(start_x, start_y, start_x + width, start_y + height)

        return rectangle
