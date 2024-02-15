"""
    Copyright 2023 Reza NasiriGerdeh. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


import torch
from torch import nn
from layers.knconv import KNConv2d
from layers.kernel_norm import KernelNorm2d1x1


def knconv_block(in_channels, out_channels, activation=nn.ReLU, max_pool=False, knconv_dropout_p=0.1):
    layers = [
        activation(),
        KNConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=knconv_dropout_p),
    ]
    if max_pool:
        layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
    return nn.Sequential(*layers)


# def knconv_block1x1(in_channels, out_channels, activation=nn.ReLU, max_pool=False, knconv_dropout_p=0.1):
#     layers = [
#         activation(),
#         KNConv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True, dropout_p=knconv_dropout_p)
#     ]
#     if max_pool:
#         layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
#     return nn.Sequential(*layers)
#

class ResidualBlockKN(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, knconv_dropout_p=0.1):
        super(ResidualBlockKN, self).__init__()
        self.activ1 = activation()
        self.conv1 = KNConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=knconv_dropout_p)

        self.activ2 = activation()
        self.conv2 = KNConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=knconv_dropout_p)

    def forward(self, input_t):
        identity = input_t

        out = self.activ1(input_t)
        out = self.conv1(out)

        out = self.activ2(out)
        out = self.conv2(out)

        out_f = out + identity
        return out_f


class ResNet13KN(nn.Module):
    def __init__(self, num_classes=10, activation=nn.ReLU,  knconv_dropout_p=0.1, kn_dropout_p=0.5):

        """ 8-layer kernel normalized residual network """
        super(ResNet13KN, self).__init__()

        self.conv0 = KNConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=knconv_dropout_p)

        self.res_block1 = ResidualBlockKN(in_channels=64, out_channels=64, activation=activation, knconv_dropout_p=knconv_dropout_p)
        self.conv_block1 = knconv_block(in_channels=64, out_channels=64, activation=activation, max_pool=True, knconv_dropout_p=knconv_dropout_p)

        self.res_block2 = ResidualBlockKN(in_channels=64, out_channels=64, activation=activation, knconv_dropout_p=knconv_dropout_p)
        self.conv_block2 = knconv_block(in_channels=64, out_channels=128, activation=activation, max_pool=True, knconv_dropout_p=knconv_dropout_p)

        self.res_block3 = ResidualBlockKN(in_channels=128, out_channels=128, activation=activation, knconv_dropout_p=knconv_dropout_p)
        self.conv_block3 = knconv_block(in_channels=128, out_channels=256, activation=activation, max_pool=True, knconv_dropout_p=knconv_dropout_p)

        self.res_block4 = ResidualBlockKN(in_channels=256, out_channels=256, activation=activation, knconv_dropout_p=knconv_dropout_p)

        self.knorm = KernelNorm2d1x1(dropout_p=kn_dropout_p)
        self.activ = activation()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input_t):
        out = self.conv0(input_t)

        out = self.res_block1(out)
        out = self.conv_block1(out)

        out = self.res_block2(out)
        out = self.conv_block2(out)

        out = self.res_block3(out)
        out = self.conv_block3(out)

        out = self.res_block4(out)

        out = self.knorm(out)
        out = self.activ(out)

        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


#  ############## Models ################################
def resnet13_kn(num_classes=10, activation=nn.ReLU, knconv_dropout_p=0.1, kn_dropout_p=0.5):
    return ResNet13KN(num_classes, activation, knconv_dropout_p, kn_dropout_p)
