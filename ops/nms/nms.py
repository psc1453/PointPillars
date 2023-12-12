from typing import List

import torch
from torch import Tensor

from ops.nms.classes import Point, Line, RotatedRectangle
from ops.nms.iou import iou_two_boxes


def nms_core(boxes_tensor: Tensor, thresh: float) -> List[int]:
    """
    NMS core function
    :param boxes_tensor: List of box Tensors with the descent order of their scores. (N x [x1, y1, x2, y2, r])
    :param thresh: Threshold of NMS
    :return: List of indexes of input boxes which should be kept
    """
    # Mask for boxes, True for keep and False for abandon
    keep_mask = torch.ones(boxes_tensor.shape[0], dtype=torch.bool)
    num_of_boxes = boxes_tensor.shape[0]

    # Generate list for boxes with type of RotatedRectangle
    boxes = [RotatedRectangle(start_point=Point(box[0], box[1]),
                              end_point=Point(box[2], box[3]),
                              rotate_angle=box[4])
             for box in boxes_tensor]

    # Iterate over every boxes
    for reference_index in range(num_of_boxes):
        # Cache as reference
        reference_box = boxes[reference_index]
        # If already set invalid by previous iteration, go to next box
        if not keep_mask[reference_index]:
            continue

        # Iterate over the rest of boxes
        for checking_index in range(reference_index + 1, num_of_boxes):
            # Cache
            current_box = boxes[checking_index]

            # If already set invalid by previous iteration, go to next box
            if not keep_mask[checking_index]:
                continue

            # Compute IoU
            iou = iou_two_boxes(reference_box, current_box)
            # If IoU is large enough, deactivate current box
            # Since reference box has smaller index which means larger confidence score
            if iou > thresh:
                keep_mask[checking_index] = False

    # Convert the mask to index list
    keep_indexes = torch.nonzero(keep_mask).squeeze(-1)

    return keep_indexes


def nms(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    NMS without GPU requirement
    :param boxes: (torch.Tensor): Input boxes with the shape of [N, 5] ([x1, y1, x2, y2, ry])
    :param scores: (torch.Tensor): Scores of boxes with the shape of [N]
    :param thresh: (int): Threshold
    :param pre_maxsize: (int): Max size of boxes before nms. Default: None
    :param post_max_size: (int): Max size of boxes after nms. Default: None
    :return: torch.Tensor: Indexes of boxes that should be kept after NMS
    """
    # Order of boxes and scores by scores: descending
    order = scores.sort(0, descending=True)[1]

    # Max number of boxes filter before NMS
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    # Sort boxes
    boxes = boxes[order].contiguous()

    # Do NMS and get indexes of sorted boxes that should be kept
    keep_ordered_indexes = nms_core(boxes, thresh)
    # Map sorted indexes to original indexes
    keep_real_indexes = order[keep_ordered_indexes].contiguous()

    # Max number of boxes filter after NMS
    if post_max_size is not None:
        keep_real_indexes = keep_real_indexes[:post_max_size]

    return keep_real_indexes
