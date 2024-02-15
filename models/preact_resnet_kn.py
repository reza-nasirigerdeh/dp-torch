"""
The original implementation of PreactResNet for CIFAR10/100, by liukuang, is available at: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py,
and has been published under MIT license, available at:https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE
This implementation, by Reza NasiriGerdeh, is the kernel normalized version of the preact resnet, where
 (1) Conv2d layers are replaced with the KNConv2d layers
 (2) The BatchNorm layers are eliminated
 (3) kernel size in KNConv2d of shortcut is set to 2x2 instead of 1x1
 (4) (KernelNorm + ReLU) is inserted before the average-pooling layer

It is published under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from layers.knconv import KNConv2d
from layers.kernel_norm import KernelNorm2d1x1


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, knconv_dropout_p=0.1, activation=nn.ReLU):
        super(PreactBasicBlock, self).__init__()
        self.activ1 = activation()
        self.conv1 = KNConv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=(1, 1),
                              bias=True, dropout_p=knconv_dropout_p)
        self.activ2 = activation()
        self.conv2 = KNConv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                              bias=True, dropout_p=knconv_dropout_p)

        kernel_size = (1, 1) if stride != 1 else (1, 1)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                KNConv2d(in_planes, self.expansion*planes, kernel_size=kernel_size, stride=stride,
                         bias=True, dropout_p=knconv_dropout_p)
            )

    def forward(self, x):
        out = self.activ1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.activ2(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreactResNetKN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, knconv_dropout_p=0.1, kn_dropout_p=0.5, activation=nn.ReLU):
        super(PreactResNetKN, self).__init__()
        self.in_planes = 64

        self.conv1 = KNConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=knconv_dropout_p)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, knconv_dropout_p=knconv_dropout_p, activation=activation)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, knconv_dropout_p=knconv_dropout_p, activation=activation)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, knconv_dropout_p=knconv_dropout_p, activation=activation)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, knconv_dropout_p=knconv_dropout_p, activation=activation)

        self.knorm = KernelNorm2d1x1(dropout_p=kn_dropout_p)
        self.activ = activation()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, knconv_dropout_p, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, knconv_dropout_p, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.knorm(out)
        out = self.activ(out)

        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


def preact_resnet18_kn(num_classes=10, knconv_dropout_p=0.1, kn_dropout_p=0.5, activation=nn.ReLU):
    return PreactResNetKN(PreactBasicBlock, [2, 2, 2, 2], num_classes, knconv_dropout_p, kn_dropout_p, activation)
