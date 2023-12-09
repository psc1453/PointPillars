import torch.nn as nn


class FPNEncoder(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        # Just a container
        # The difference between nn.ModuleList() and nn.Sequential() is that nn.ModuleList() does not have forward()
        self.multi_blocks = nn.ModuleList()

        # Length of layer_stride is the number of layers of the FPN
        for i in range(len(layer_strides)):
            blocks = []
            # Stride=n(typically 2) convolution, normalization and ReLU
            # For an (n x n) matrix, if n is even, output will be (n/2 x n/2)
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            # Do layer_nums[i] more times of Stride=2, keep channel convolution, normalization and ReLU
            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            # Input channel of the next iteration(next block) is set to current output channel for linking them
            in_channel = out_channels[i]
            # Add the unpacked currently generated block to block list
            self.multi_blocks.append(nn.Sequential(*blocks))

        # Parameter normalization
        # In consistent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        :param x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        :return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        # Output list
        outs = []
        # (Input -> Output -> Input) loop
        # Add output of every iteration to list
        # This is a typical FPN
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs
