from typing import List
import math

from ops.nms.classes import Point, Line, RotatedRectangle


def vector_cross_magnitude(vector_a: Point, vector_b: Point) -> float:
    return vector_a.x * vector_b.y - vector_a.y * vector_b.x


def vector_cross_magnitude_with_origin(origin: Point, point_a: Point, point_b: Point) -> float:
    return (point_a.x - origin.x) * (point_b.y - origin.y) - (point_b.x - origin.x) * (point_a.y - origin.y)


def rect_has_intersection(line_a: Line, line_b: Line) -> bool:
    is_intersect = (min(line_a.start.x, line_a.end.x) <= max(line_b.start.x, line_b.end.x) and
                    min(line_b.start.x, line_b.end.x) <= max(line_a.start.x, line_a.end.x) and
                    min(line_a.start.y, line_a.end.y) <= max(line_b.start.y, line_b.end.y) and
                    min(line_b.start.y, line_b.end.y) <= max(line_a.start.y, line_a.end.y))
    return is_intersect


def line_intersection(line_a: Line, line_b: Line) -> Point:
    _EPS = 1e-8

    intersection = Point()

    # Fast exclusion
    if not rect_has_intersection(line_a, line_b):
        return intersection

    # Check cross standing
    s1 = vector_cross_magnitude_with_origin(line_b.start, line_a.end, line_a.start)
    s2 = vector_cross_magnitude_with_origin(line_a.end, line_b.end, line_a.start)
    s3 = vector_cross_magnitude_with_origin(line_a.start, line_b.end, line_b.start)
    s4 = vector_cross_magnitude_with_origin(line_b.end, line_a.end, line_b.start)

    if not (s1 * s2 > 0 and s3 * s4 > 0):
        return intersection

    # Calculate intersection of two lines
    s5 = vector_cross_magnitude_with_origin(line_b.end, line_a.end, line_a.start)
    if math.fabs(s5 - s1) > _EPS:
        intersection.x = (s5 * line_b.start.x - s1 * line_b.end.x) / (s5 - s1)
        intersection.y = (s5 * line_b.start.y - s1 * line_b.end.y) / (s5 - s1)
    else:
        a0 = line_a.start.y - line_a.end.y
        b0 = line_a.end.x - line_a.start.x
        c0 = line_a.start.x * line_a.end.y - line_a.end.x * line_a.start.y

        a1 = line_b.start.y - line_b.end.y
        b1 = line_b.end.x - line_b.start.x
        c1 = line_b.start.x * line_b.end.y - line_b.end.x * line_b.start.y

        d = a0 * b1 - a1 * b0

        intersection.x = (b0 * c1 - b1 * c0) / d
        intersection.y = (a1 * c0 - a0 * c1) / d

    return intersection


def point_is_in_box(point: Point, box: RotatedRectangle) -> bool:
    _MARGIN = 1e-5

    angle_cos = math.cos(-box.rotate_angle)
    angle_sin = math.sin(-box.rotate_angle)

    rotated_point_x = (point.x - box.center.x) * angle_cos + (point.y - box.center.y) * angle_sin + box.center.x
    rotated_point_y = -(point.x - box.center.x) * angle_sin + (point.y - box.center.y) * angle_cos + box.center.y

    is_in_box = (box.start_point.x - _MARGIN < rotated_point_x < box.end_point.x + _MARGIN and
                 box.start_point.y - _MARGIN < rotated_point_y < box.end_point.y + _MARGIN)

    return is_in_box


def sort_poly_corners(poly_corners: List[Point], poly_center: Point) -> List[Point]:
    corners = poly_corners.copy()

    num_corners = len(corners)
    assert num_corners > 0, 'No corners found'

    for i in range(num_corners):
        for j in range(num_corners - 1):
            vector_a = corners[j] - poly_center
            vector_b = corners[j + 1] - poly_center
            if vector_a.angle > vector_b.angle:
                corners[j], corners[j + 1] = corners[j + 1], corners[j]

    return corners


def box_overlap(box_a: RotatedRectangle, box_b: RotatedRectangle):
    intersect_poly_center_accumulator = Point(0, 0)

    # Get intersection of sides
    cross_points = []

    box_a_corners_iter = box_a.rotated_corners + [box_a.rotated_corners[0]]
    box_b_corners_iter = box_b.rotated_corners + [box_b.rotated_corners[0]]

    intersection_poly_corner_cnt = 0
    for corner_index_a in range(4):
        side_a = Line(box_a_corners_iter[corner_index_a], box_a_corners_iter[corner_index_a + 1])
        for corner_index_b in range(4):
            side_b = Line(box_b_corners_iter[corner_index_b], box_b_corners_iter[corner_index_b + 1])

            sides_intersection_point = line_intersection(side_a, side_b)
            if not sides_intersection_point.is_none():
                cross_points.append(sides_intersection_point)
                intersect_poly_center_accumulator += sides_intersection_point
                intersection_poly_corner_cnt += 1

    # Check corners inside the other box
    inside_points = []

    for corner_index in range(4):
        box_a_corner = box_a.rotated_corners[corner_index]
        box_b_corner = box_b.rotated_corners[corner_index]

        if point_is_in_box(box_a_corner, box_b):
            inside_points.append(box_a_corner)
            intersect_poly_center_accumulator += box_a_corner
            intersection_poly_corner_cnt += 1

        if point_is_in_box(box_b_corner, box_a):
            inside_points.append(box_b_corner)
            intersect_poly_center_accumulator += box_b_corner
            intersection_poly_corner_cnt += 1

    # Combine all corners of intersection polynomial
    intersect_poly_center = intersect_poly_center_accumulator / intersection_poly_corner_cnt
    intersect_poly_corners = cross_points + inside_points

    # Sort corners for computing area
    sorted_poly_corners = sort_poly_corners(intersect_poly_corners, intersect_poly_center)

    # Original CUDA code has a bug here
    sorted_poly_corners += [sorted_poly_corners[0]]

    # Calculate area
    area = 0
    for triangle_index in range(intersection_poly_corner_cnt):
        area += vector_cross_magnitude_with_origin(intersect_poly_center,
                                                   sorted_poly_corners[triangle_index],
                                                   sorted_poly_corners[triangle_index + 1])

    return math.fabs(area) / 2


def test():
    box_1 = RotatedRectangle(Point(-1, -1), Point(1, 1))
    box_2 = RotatedRectangle(Point(-1, -1), Point(1, 1), math.pi / 4)
    overlap = box_overlap(box_1, box_2)
    print(overlap)


if __name__ == '__main__':
    test()
