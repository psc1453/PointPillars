import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        # Voxel size x and y
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        # Center location of first voxel with index (0, 0)
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        # Voxel grid resolution
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        :param pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        :param coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        :param npoints_per_pillar: (p1 + p2 + ... + pb, )
        :return:  (bs, out_channel, y_l, x_l)
        """
        device = pillars.device
        # 1. Calculate offset of every point to the points center in each pillar
        # xyz coordinate of each points minus average coordinate
        offset_point_average_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1,
                                                                    keepdim=True) / npoints_per_pillar[:, None,
                                                                                    None]  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. Calculate offset of every point to the pillar center
        # x coordinate minus pillar center x coordinate
        x_offset_pillar_center = pillars[:, :, :1] - (
                coors_batch[:, None, 1:2] * self.vx + self.x_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        # y coordinate minus pillar center y coordinate
        y_offset_pillar_center = pillars[:, :, 1:2] - (
                coors_batch[:, None, 2:3] * self.vy + self.y_offset)  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. Encoder
        # Concat pillar points values, offset to pillar point average, offset to pillar center along the last dimension
        #       [Pillars x Num_Points x 4] # Point Values (x, y, z, r)
        #       [Pillars x Num_Points x 3] # Offset to Point Average in Pillar (offset_x, offset_y, offset_z)
        #       [Pillars x Num_Points x 1] # Offset to Pillar x Center (offset_x_p)
        #       [Pillars x Num_Points x 1] # Offset to Pillar y Center (offset_y_p)
        #   =>  [Pillars x Num_Points x 9]
        features = torch.cat([pillars, offset_point_average_center, x_offset_pillar_center, y_offset_pillar_center],
                             dim=-1)  # (p1 + p2 + ... + pb, num_points, 9)
        # TODO: This meaningless action is due to a legacy bug, try to remove it and re-train the network.
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150
        features[:, :, 0:1] = x_offset_pillar_center  # tmp
        features[:, :, 1:2] = y_offset_pillar_center  # tmp

        # 4. Generate mask and set the 9D data in unmasked(unselected) position to all (0, 0, ..., 0)
        # a very beautiful implementation
        #   Mask rule:
        #   Base rule: In feature, each row is a pillar, only first n are real points, n is recorded in npoints_per_pillar, the other are 0 paddings
        #   column 0: point[0] of pillars, if npoints_per_pillar > 0, reserve
        #   column 1: point[1] of pillars, if npoints_per_pillar > 1, reserve
        #   column 2: point[2] of pillars, if npoints_per_pillar > 2, reserve
        #   ...
        #   column N: point[N] of pillars, if npoints_per_pillar > N, reserve
        # Generate vector Tensor [0, 1, ..., max_num_in_pillar]
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        # Broadcast two Tensor, compare (np_n denotes number of points in pillar n, m denotes (max_num_in_pillar - 1)):
        # 0, 1, 2, ..., m             np_1, np_1, ..., np_1             [np_1 x True + Trailing False]
        # 0, 1, 2, ..., m             np_2, np_2, ..., np_2             [np_2 x True + Trailing False]
        # ...                  <                                 =      ...
        # ...                                                           ...
        # 0, 1, 2, ..., m             np_m, np_m, ..., np_m             [np_m x True + Trailing False]
        # Only first n are real points and are masked by True
        mask = (voxel_ids[None, :] < npoints_per_pillar[:, None]).contiguous()  # (p1 + p2 + ... + pb, num_points)
        # Mask feature, set all attributes to padding points to 0
        features *= mask[:, :, None]

        # 5. Embedding
        # Swap dimensions
        features = features.permute(0, 2, 1).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        # 1 x 1 convolution to map 9D feature to number of output channel => BN => ReLU
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        # Max filter in point dimension, takes the maximum feature value in every pillar.
        # Max function returns 2 values:(value, index), only takes value
        pooling_features = torch.max(features, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels)

        # 6. Pillar scatter
        batched_canvas = []
        # coors_batch: [num_points x 4] where 4 is [batch_index, x, y, z, r]
        bs = coors_batch[0, 0] + 1
        for i in range(bs):
            # Mask for pillars index which belong to current batch
            cur_coors_idx = coors_batch[:, 0] == i
            # Takes pillar coords and features which belong to current batch
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            # Create empty canvas space for [pillar_grip_size x out_channel] = [x_l x y_l x out_channel]
            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            # Since not every grid has valid pillar, put all pillars to canvas, keep 0 in other places
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            # [x_l x y_l x out_channel] => [out_channel x y_l x x_l]
            canvas = canvas.permute(2, 1, 0).contiguous()
            # Append to canvas list
            batched_canvas.append(canvas)
        # Stack the canvas list along fist dimension [[out_channel x y_l x x_l]] => [bs x out_channel x y_l x x_l]
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas
