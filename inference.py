###
# Abbreviation:
#   pc: point cloud

import argparse
import os

import numpy as np
import torch

from model import PointPillars
from utils import read_points, keep_bbox_from_lidar_range, vis_pc


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    """
    :param pts: (N x 4) numpy array
    :param point_range: [x1, y1, z1, x2, y2, z2]
    :return: Filtered point cloud
    """
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts


def main(args):
    CLASSES = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2
    }
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    model = PointPillars(nclasses=len(CLASSES)).cuda()
    model.load_state_dict(torch.load(args.ckpt))

    if not os.path.exists(args.pc_path):
        raise FileNotFoundError
    # Binary point cloud file to 2D Numpy Array (N_points x 4) where 4 is (x, y, z, r) where r is reflection rate
    pc = read_points(args.pc_path)
    # Filter points that are out of range
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc).cuda()

    model.eval()
    with torch.no_grad():
        result_filter = model(batched_pts=[pc_torch], mode='test')[0]

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='development/reformat_state_dict.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', default='kitti/training/velodyne/000772.bin', help='your point cloud path')
    args = parser.parse_args()

    main(args)
