#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Space2Depth(nn.Module):
    def __init__(self, down_factor):
        super(Space2Depth, self).__init__()
        self.down_factor = down_factor

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.down_factor, stride=self.down_factor)
        return unfolded_x.view(n, c * self.down_factor ** 2, h // self.down_factor, w // self.down_factor)


def conv_bn_activation(in_channels, out_channels, kernel_size, stride, padding, use_bn, activation):
    module = nn.Sequential()
    module.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    module.add_module('bn', nn.BatchNorm2d(out_channels)) if use_bn else None
    module.add_module('activation', activation) if activation else None

    return module


class BlockStack(nn.Module):
    def __init__(self, channels, num_block, share_weight, connect_mode, use_bn, activation):
        # connect_mode: refer to "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks"
        super(BlockStack, self).__init__()

        self.num_block = num_block
        self.connect_mode = connect_mode

        self.blocks = nn.ModuleList()

        if share_weight is True:
            block = nn.Sequential(
                conv_bn_activation(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3, stride=1, padding=1,
                    use_bn=use_bn, activation=activation
                ),
                conv_bn_activation(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3, stride=1, padding=1,
                    use_bn=use_bn, activation=activation
                )
            )
            for i in range(num_block):
                self.blocks.append(block)

        else:
            for i in range(num_block):
                block = nn.Sequential(
                    conv_bn_activation(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3, stride=1, padding=1,
                        use_bn=use_bn, activation=activation
                    ),
                    conv_bn_activation(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3, stride=1, padding=1,
                        use_bn=use_bn, activation=activation
                    )
                )
                self.blocks.append(block)

    def forward(self, x):
        if self.connect_mode == 'no':
            for i in range(self.num_block):
                x = self.blocks[i](x)
        elif self.connect_mode == 'distinct_source':
            for i in range(self.num_block):
                x = self.blocks[i](x) + x
        elif self.connect_mode == 'shared_source':
            x0 = x
            for i in range(self.num_block):
                x = self.blocks[i](x) + x0
        else:
            print('"connect_mode" error!')
            exit(0)
        return x


class Model(nn.Module):  # Edge Adaptive Rendering
    def __init__(self, shuffle_rate=2, in_channels=6, out_channels=3, middle_channels=256, num_block=5, connect_mode='distinct_source', use_bn=False, activation='leaky_relu'):
        super(Model, self).__init__()

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            activation = nn.ELU(inplace=True)
        else:
            print('"activation" error!')
            exit(0)

        self.downsample = Space2Depth(shuffle_rate)
        self.conv0 = conv_bn_activation(
            in_channels=in_channels * shuffle_rate**2,
            out_channels=middle_channels,
            kernel_size=3, stride=1, padding=1,
            use_bn=use_bn, activation=activation
        )
        self.block_stack = BlockStack(
            channels=middle_channels,
            num_block=num_block, share_weight=False, connect_mode=connect_mode,
            use_bn=use_bn, activation=activation
        )
        self.conv1 = conv_bn_activation(
            in_channels=middle_channels,
            out_channels=out_channels * shuffle_rate ** 2,
            kernel_size=3, stride=1, padding=1,
            use_bn=False, activation=None
        )
        self.upsample = nn.PixelShuffle(shuffle_rate)

    def forward(self, x):
        y = self.downsample(x)
        y = self.conv0(y)
        y = self.block_stack(y)
        y = self.conv1(y)
        y = self.upsample(y)
        output = (x[:, 0:3] + x[:, 3:6]) / 2 + y

        return output