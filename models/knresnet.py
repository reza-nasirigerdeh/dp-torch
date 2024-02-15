"""
    MIT License
    Copyright (c) 2024 Reza NasiriGerdeh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch
from torch import nn
from layers.knconv import KNConv2d
from layers.kernel_norm import KernelNorm2d1x1


class ResidualBlockKN(nn.Module):
    def __init__(self, channels, inter_channels, block_dropout_p=0.05, activation=nn.Mish):
        super(ResidualBlockKN, self).__init__()
        self.activ1 = activation()
        self.conv1 = KNConv2d(channels, inter_channels, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=block_dropout_p)

        self.activ2 = activation()
        self.conv2 = KNConv2d(inter_channels, channels, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=True, dropout_p=block_dropout_p)

    def forward(self, input_t):
        identity = input_t
        out = self.activ1(input_t)
        out = self.conv1(out)

        out = self.activ2(out)
        out = self.conv2(out)

        out += identity
        return out


class TransBlockKN(nn.Module):
    def __init__(self, in_channels, out_channels, block_dropout_p=0.05, padding=(0, 0, 0, 0), activation=nn.Mish):
        super(TransBlockKN, self).__init__()
        self.activ = activation()
        self.conv = KNConv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(1, 1), padding=padding, bias=True, dropout_p=block_dropout_p)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    def forward(self, input_t):
        out = self.activ(input_t)
        out = self.conv(out)
        out = self.max_pool(out)

        return out


class ConvBlockKN(nn.Module):
    def __init__(self, in_channels, out_channels, padding, block_dropout_p=0.05, activation=nn.Mish):
        super(ConvBlockKN, self).__init__()
        self.activ = activation()
        self.conv = KNConv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(1, 1), padding=padding, bias=True, dropout_p=block_dropout_p)

    def forward(self, input_t):
        out = self.activ(input_t)
        out = self.conv(out)

        return out


class KNResNet18(nn.Module):
    def __init__(self, num_classes=1000, block_dropout_p=0.05, final_dropout_p=0.25, activation=nn.Mish, low_resolution=False):

        """ 18-layer kernel normalized residual network """
        super(KNResNet18, self).__init__()

        if low_resolution:
            self.block0 = nn.Sequential(
                KNConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=block_dropout_p)
            )
        else:
            self.block0 = nn.Sequential(
                KNConv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=True, dropout_p=block_dropout_p),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )

        self.res_block1 = ResidualBlockKN(channels=64, inter_channels=256, block_dropout_p=block_dropout_p, activation=activation)
        self.res_block2 = ResidualBlockKN(channels=64, inter_channels=256, block_dropout_p=block_dropout_p, activation=activation)

        self.trans_block1 = TransBlockKN(in_channels=64, out_channels=256, padding=(1, 0, 1, 0), block_dropout_p=block_dropout_p, activation=activation)
        self.res_block3 = ResidualBlockKN(channels=256, inter_channels=256, block_dropout_p=block_dropout_p, activation=activation)
        self.res_block4 = ResidualBlockKN(channels=256, inter_channels=256, block_dropout_p=block_dropout_p, activation=activation)

        self.trans_block2 = TransBlockKN(in_channels=256, out_channels=512, padding=(0, 1, 0, 1), block_dropout_p=block_dropout_p, activation=activation)
        self.res_block5 = ResidualBlockKN(channels=512, inter_channels=512, block_dropout_p=block_dropout_p, activation=activation)

        padding = (1, 0, 1, 0) if low_resolution else (2, 1, 2, 1)
        self.trans_block3 = TransBlockKN(in_channels=512, out_channels=724, padding=padding, block_dropout_p=block_dropout_p, activation=activation)
        self.res_block6 = ResidualBlockKN(channels=724, inter_channels=724, block_dropout_p=block_dropout_p, activation=activation)

        self.max_pool_f = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        padding = (1, 1) if low_resolution else (0, 0)
        self.conv_block_f = ConvBlockKN(in_channels=724, out_channels=512, padding=padding, block_dropout_p=block_dropout_p, activation=activation)

        self.kn_f = KernelNorm2d1x1(dropout_p=final_dropout_p)
        self.activ_f = activation()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_t):
        out = self.block0(input_t)

        out = self.res_block1(out)
        out = self.res_block2(out)

        out = self.trans_block1(out)
        out = self.res_block3(out)
        out = self.res_block4(out)

        out = self.trans_block2(out)
        out = self.res_block5(out)

        out = self.trans_block3(out)
        out = self.res_block6(out)

        out = self.max_pool_f(out)
        out = self.conv_block_f(out)

        out = self.kn_f(out)
        out = self.activ_f(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


def knresnet18(num_classes=1000, block_dropout_p=0.05, final_dropout_p=0.25, activation=nn.Mish, low_resolution=True):
    return KNResNet18(num_classes, block_dropout_p, final_dropout_p, activation, low_resolution)
