import torch
import torch.nn as nn

from model.components import Voxelization


class PointPillars(nn.Module):
    def __init__(self,
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    def forward(self, pts):
        voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)

        return voxels_out, coors_out, num_points_per_voxel_out


def main():
    model = PointPillars().cuda()

    pc = torch.load('development/pc_for_inference.pt', map_location='cuda')

    model.eval()
    with torch.no_grad():
        pillars = model(pc)

    # print(pillars)


if __name__ == '__main__':
    main()
