import math

from ops.nms.classes import Point, Line, RotatedRectangle
from ops.nms.overlap import box_overlap


def iou_two_boxes(box_a: RotatedRectangle, box_b: RotatedRectangle) -> float:
    intersect_area = box_overlap(box_a, box_b)

    union_area = box_a.area + box_b.area - intersect_area
    iou = intersect_area / union_area

    return iou


def test():
    box_1 = RotatedRectangle(Point(12.8248, -13.4288), Point(14.4717, -9.6939), -0.5242)
    box_2 = RotatedRectangle(Point(12.8257, -13.4140), Point(14.4805, -9.6871), -0.5248)
    iou = iou_two_boxes(box_1, box_2)
    print(f'IoU: {iou}')


if __name__ == '__main__':
    test()
