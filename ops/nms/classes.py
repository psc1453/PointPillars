import math
from typing import Union, List

import torch
from torch import Tensor


class Point:
    def __init__(self, x=None, y=None):
        if isinstance(x, Tensor):
            self.x = x.item()
        else:
            self.x = x
        if isinstance(y, Tensor):
            self.y = y.item()
        else:
            self.y = y

    def __repr__(self):
        return f'Point({self.x}, {self.y})'

    def __add__(self, another):
        return Point(self.x + another.x, self.y + another.y)

    def __sub__(self, another):
        return Point(self.x - another.x, self.y - another.y)

    def __mul__(self, factor: Union[int, float]):
        return Point(self.x * factor, self.y * factor)

    def __truediv__(self, divisor: Union[int, float]):
        return Point(self.x / divisor, self.y / divisor)

    def is_none(self):
        if self.x is None or self.y is None:
            return True
        else:
            return False

    @property
    def angle(self):
        if self.is_none():
            return None
        else:
            return math.atan2(self.y, self.x)


class Line:
    def __init__(self, start: Point, end: Point):
        assert isinstance(start, Point) and isinstance(end, Point), 'Start and end must be of type Point'
        self.start = start
        self.end = end

    def __repr__(self):
        return f'{self.start} => {self.end}'


class RotatedRectangle:
    def __init__(self, start_point: Point, end_point: Point, rotate_angle: float = 0):
        self.start_point = start_point
        self.end_point = end_point
        self.rotate_angle = rotate_angle

        center_x = (self.start_point.x + self.end_point.x) / 2
        center_y = (self.start_point.y + self.end_point.y) / 2
        self._center = Point(center_x, center_y)

        self._width = math.fabs(end_point.x - start_point.x)
        self._height = math.fabs(end_point.y - start_point.y)

        self._box_h_w_r = Tensor([center_x, center_y, self._width, self._height, self.rotate_angle])
        self._rotated_corners_tensor = RotatedRectangle.box_rotator(self._box_h_w_r)

        self._rotated_corners = [Point(x, y) for x, y in self._rotated_corners_tensor]
        self._area = self._width * self._height

    @property
    def center(self) -> Point:
        return self._center

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def box_h_w_r(self) -> Tensor:
        return self._box_h_w_r

    @property
    def rotated_corners_tensor(self) -> Tensor:
        return self._rotated_corners_tensor

    @property
    def rotated_corners(self) -> List[Point]:
        return self._rotated_corners

    @property
    def area(self) -> float:
        return self._area

    @classmethod
    def generate_rotation_matrix(cls, r: float) -> Tensor:
        """
        :param r: Rotation angle in rad
        :return: Rotation matrix Tensor
        """
        cos_r = math.cos(r)
        sin_r = math.sin(r)

        rotation_matrix = Tensor([[cos_r, -sin_r],
                                  [sin_r, cos_r]])
        return rotation_matrix

    @classmethod
    def box_rotator(cls, box_h_w_r: Tensor) -> Tensor:
        """
        Generate rotated box
        :param box_h_w_r: Tensor of [center_x, center_y, width, height, rotation]
        :return: 4 x 2 Tensor: 4 x (x, y) corner coordinates after rotation []
        """
        center = box_h_w_r[:2].unsqueeze(-1)
        offset_pair = (box_h_w_r[2: 4] / 2).unsqueeze(-1)

        corner_gen = Tensor([
            [-1, 1, 1, -1],
            [-1, -1, 1, 1]
        ])
        offset_pair_set = corner_gen * offset_pair

        rotation_matrix = cls.generate_rotation_matrix(box_h_w_r[4].item())
        rotated_offset_pair_set = torch.matmul(rotation_matrix, offset_pair_set)

        rotated_box = rotated_offset_pair_set + center
        rotated_box_points = rotated_box.permute(1, 0).contiguous()
        return rotated_box_points