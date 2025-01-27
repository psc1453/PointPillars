import torch
import torch.nn as nn


class FPNDecoder(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        # Just a container
        # The difference between nn.ModuleList() and nn.Sequential() is that nn.ModuleList() does not have forward()
        self.decoder_blocks = nn.ModuleList()

        # Length of layer_stride is the number of layers of the FPN
        for i in range(len(in_channels)):
            decoder_block = []
            # Deconvolution, normalization and ReLU for recover original shape
            decoder_block.append(nn.ConvTranspose2d(in_channels[i],
                                                    out_channels[i],
                                                    upsample_strides[i],
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            # Add the unpacked currently generated block to block list
            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # Parameter normalization
        # In consistent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        :param x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        :return: (bs, 384, 248, 216)
        """
        # Output list
        # (Input -> Output -> Input) loop
        # Concat output of every iteration along channel dimension
        # This is a typical FPN
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out
