import torch
import torch.nn as nn


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.Conv2d(in_channels=in_channels[i],
                                           out_channels=out_channels[i] * upsample_strides[i] * upsample_strides[i],
                                           kernel_size=3,
                                           padding=1,
                                           bias=False))
            decoder_block.append(nn.PixelShuffle(upsample_strides[i]))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        :param x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        :return: (bs, 384, 248, 216)
        """
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out
