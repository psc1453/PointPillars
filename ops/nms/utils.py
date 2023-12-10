import math

import torch
from torch import Tensor


def generate_rotation_matrix(r: float) -> Tensor:
    """
    :param r: Rotation angle in rad
    :return: Rotation matrix Tensor
    """
    cos_r = math.cos(r)
    sin_r = math.sin(r)

    rotation_matrix = Tensor([[cos_r, -sin_r],
                              [sin_r, cos_r]])
    print(rotation_matrix)
    return rotation_matrix


def box_encoder(box_two_points_r: Tensor) -> Tensor:
    """
    Encode a box [x_1, y_1, x_2, y_2, rotation] to [center_x, center_y, height, width, rotation]
    :param box_two_points_r: Tensor of [x_1, y_1, x_2, y_2, rotation]
    :return: Tensor of [center_x, center_y, width, height, rotation]
    """
    center_x = torch.mean(box_two_points_r[[0, 2]])
    center_y = torch.mean(box_two_points_r[[1, 3]])
    width = box_two_points_r[2] - box_two_points_r[0]
    height = box_two_points_r[3] - box_two_points_r[1]
    box_h_w_r = Tensor([center_x, center_y, width, height, box_two_points_r[4]])
    return box_h_w_r


def box_rotator(box_h_w_r: Tensor) -> Tensor:
    """
    Generate rotated box
    :param box_h_w_r: Tensor of [center_x, center_y, width, height, rotation]
    :return: 4 x 2 Tensor: 4 x (x, y) corner coordinates after rotation
    """
    center = box_h_w_r[:2].unsqueeze(-1)
    offset_pair = (box_h_w_r[2: 4] / 2).unsqueeze(-1)

    corner_gen = Tensor([
        [1, 1, -1, -1],
        [1, -1, 1, -1]
    ])
    offset_pair_set = corner_gen * offset_pair

    rotation_matrix = generate_rotation_matrix(box_h_w_r[4].item())
    rotated_offset_pair_set = torch.matmul(rotation_matrix, offset_pair_set)

    rotated_box = rotated_offset_pair_set + center
    rotated_box_points = rotated_box.permute(1, 0).contiguous()

    return rotated_box_points


if __name__ == '__main__':
    box = Tensor([-1, -1, 1, 1, math.pi / 2])
    box_rotator(box_encoder(box))
