import torch
import torch.nn as nn
from torch import Tensor


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
        self.max_voxels = max_voxels[0] if self.training else max_voxels[1]
        self.deterministic = deterministic
        self.NDim = 3

    def forward(self, input):
        # Point cloud attributes
        num_dim_every_point = input.shape[1]

        # Voxelization attributes
        voxel_size = Tensor(self.voxel_size).to(input.device)
        point_cloud_range = Tensor(self.point_cloud_range).to(input.device)

        # Voxelized grid size
        grid_size = torch.round(
            (point_cloud_range[self.NDim:] - point_cloud_range[:self.NDim]) / voxel_size).to(input.device,
                                                                                             dtype=torch.int)
        # Create a table with dimension of (x, y, z) to store the index of every voxel (x is the lsb axis, put last)
        grid_coord_to_voxel_index = -torch.ones_like(grid_size, device=input.device, dtype=torch.int32)

        # Generate voxel grid coordinates for every point
        # Since voxel grids are generated from points, empty voxel will not be generated
        # input[:, :3] only takes [x, y, z] of point records, point_cloud_range[:3] only takes the starting coordinates
        points_voxel_coords = torch.floor((input[:, :3] - point_cloud_range[:3]) / voxel_size).to(torch.int32)

        # Generate mask for filter points
        points_inside_lower_bound_mask = torch.all(points_voxel_coords >= Tensor([0, 0, 0]).to(input.device), dim=1)
        points_inside_upper_bound_mask = torch.all(points_voxel_coords < grid_size, dim=1)
        points_inside_grid_bound_mask = torch.logical_and(points_inside_lower_bound_mask, points_inside_upper_bound_mask)

        # Save GPU memory
        del points_inside_lower_bound_mask, points_inside_upper_bound_mask
        torch.cuda.empty_cache()

        # Take only valid points inside bounds
        points_valid = input[points_inside_grid_bound_mask]
        points_voxel_coords_valid = points_voxel_coords[points_inside_grid_bound_mask]

        # Save GPU memory
        del points_inside_grid_bound_mask, points_voxel_coords
        torch.cuda.empty_cache()

        # Generate non-duplicated voxel coordinates
        voxels_coords = torch.unique(points_voxel_coords_valid, dim=0)
        # Count voxels
        num_voxels_valid = voxels_coords.shape[0]

        # Initialize Tensors for return
        voxels_points = input.new_zeros(size=(num_voxels_valid, self.max_num_points, num_dim_every_point))
        num_points_per_voxel = input.new_zeros(size=(num_voxels_valid,), dtype=torch.int)

        # Iterate over every voxel
        for index, voxel_coord in enumerate(voxels_coords):
            # Mask of points belonging to current voxel
            children_point_mask = torch.all(points_voxel_coords_valid == voxel_coord, dim=1)
            # Get all points belonging to current voxel
            children_points = points_valid[children_point_mask]
            # Count points
            num_of_children_in_voxel = children_points.shape[0]
            # Take maximum self.max_num_points points per voxel
            if num_of_children_in_voxel > self.max_num_points:
                num_of_children_in_voxel = self.max_num_points
                children_points = children_points[:32]
            # Store them
            voxels_points[index][:num_of_children_in_voxel] = children_points
            num_points_per_voxel[index] = num_of_children_in_voxel

        # Original realization with CUDA extension
        # # Variables used to return (after post-process)
        # voxels = input.new_zeros(
        #     size=(self.max_voxels, self.max_num_points, input.size(1)))
        # coors = input.new_zeros(size=(self.max_voxels, 3), dtype=torch.int)
        # num_points_per_voxel = input.new_zeros(
        #     size=(self.max_voxels,), dtype=torch.int)
        #
        # voxel_num = hard_voxelize(input, voxels, coors,
        #                           num_points_per_voxel, self.voxel_size,
        #                           self.point_cloud_range, self.max_num_points, self.max_voxels, self.NDim,
        #                           self.deterministic)
        #
        # # Select the valid voxels inside the range
        # voxels_out = voxels[:voxel_num]
        # coors_out = coors[:voxel_num].flip(-1)  # (z, y, x) -> (x, y, z)
        # num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        #
        # return voxels_out, coors_out, num_points_per_voxel_out

        return voxels_points, voxels_coords, num_points_per_voxel

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr
