# This file is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/voxelize.py

import torch
import torch.nn as nn
from .voxel_op import hard_voxelize


class _Voxelization(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000,
                deterministic=True):
        """convert kitti points(N, >=3) to voxels.
        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        
        voxels = points.new_zeros(
            size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(
            size=(max_voxels, ), dtype=torch.int)
        voxel_num = hard_voxelize(points, voxels, coors,
                                    num_points_per_voxel, voxel_size,
                                    coors_range, max_points, max_voxels, 3,
                                    deterministic)
        # select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num].flip(-1) # (z, y, x) -> (x, y, z)
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


class Voxelization(nn.Module):
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

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels,
                 deterministic=True):
        """
        :param voxel_size: 3D voxel size
        :param point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] used to crop the point cloud
        :param max_num_points: maximum number of points in a voxel
        :param max_voxels: maximum number of voxels to process
        :param deterministic: if True, sampling will be the same in every run
        """
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.deterministic = deterministic

        # # Unused code fragment
        # point_cloud_range = torch.tensor(
        #     point_cloud_range, dtype=torch.float32)
        #
        # voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        # grid_size = (point_cloud_range[3:] -
        #              point_cloud_range[:3]) / voxel_size
        # grid_size = torch.round(grid_size).long()
        # input_feat_shape = grid_size[:2]
        # self.grid_size = grid_size
        # # the origin shape is as [x-len, y-len, z-len]
        # # [w, h, d] -> [d, h, w]
        # self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return _Voxelization.apply(input, self.voxel_size, self.point_cloud_range,
                                   self.max_num_points, max_voxels,
                                   self.deterministic)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr
