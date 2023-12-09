import torch.nn as nn

from model.components import FPNEncoder, FPNDecoder


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn_encoder = FPNEncoder(in_channel=64,
                                      out_channels=[64, 128, 256],
                                      layer_nums=[3, 5, 5])
        self.fpn_decoder = FPNDecoder(in_channels=[64, 128, 256],
                                      upsample_strides=[1, 2, 4],
                                      out_channels=[128, 128, 128])

    def forward(self, x):
        # x:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        x = self.fpn_encoder(x)

        # x: (bs, 384, 248, 216)
        x = self.fpn_decoder(x)

        return x
