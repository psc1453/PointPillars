import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import Voxelization


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        """
        :param voxel_size: 3D voxel size
        :param point_cloud_range: [x1, y1, z1, x2, y2, z2] used to crop the point cloud
        :param max_num_points: maximum number of points in a voxel
        :param max_voxels: maximum number of voxels to process
        """
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        """
        :param batched_pts: list[tensor], len(batched_pts) = bs
        :return:
            pillars: (p1 + p2 + ... + pb, num_points, c),
            coors_batch: (p1 + p2 + ... + pb, 1 + 3),
            num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            """
            This layer generates 3 outputs, they are:
            - voxels_out: (N x M x 4) Tensor
                - N is the number of voxels (only voxel containing points will be output)
                - M is the number of points in the voxel (always equals to the maximum allowed value, 0-padding or cropping)
                - 4 is the data representation of a point [x, y, z, r]
            - coors_out: (N x 3) Tensor
                - N is the number of voxels
                - 3 is the voxel 3D indices [x, y, z], where z = 0 since pillar voxel
            - num_points_per_voxel: Actual number of points per voxel without padding but no greater than the limit
            """
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)

            # Store every output of a batch to a list
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        # Stack pillars in and actual number of points in the batch along the pillar axis
        pillars = torch.cat(pillars, dim=0)  # ((N1 + N2 + ... + Nb) x M x 4)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)  # (N1 + N2 + ... + Nb)

        # Add batch index to pillar coordinate, [x, y, z] => [b, x, y, z]
        # Then, stack the batch along pillar axis.
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)  # ((N1 + N2 + ... + Nb) x (1 + 3))

        return pillars, coors_batch, npoints_per_pillar
