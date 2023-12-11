import math

from ops.nms.classes import Point, Line, RotatedRectangle
from ops.nms.overlap import box_overlap


def iou_box(box_a: RotatedRectangle, box_b: RotatedRectangle) -> float:
    intersect_area = box_overlap(box_a, box_b)

    union_area = box_a.area + box_b.area - intersect_area
    iou = intersect_area / union_area

    return iou


def test():
    box_1 = RotatedRectangle(Point(-1, -1), Point(1, 1))
    box_2 = RotatedRectangle(Point(-1, -1), Point(1, 1), math.pi / 4)
    iou = iou_box(box_1, box_2)
    print(iou)


if __name__ == '__main__':
    test()
