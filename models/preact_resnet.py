"""
The original implementation of PreactResNet for CIFAR10/100, by liukuang, is available at: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py,
and has been published under MIT license, available at:https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE

This implementation, by Reza NasiriGerdeh, extends the original one by taking the normalization layer as additional argument
to have batch/layer/group normalized versions of PreactResNet.

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


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(PreactBasicBlock, self).__init__()
        self.norm1 = norm_layer(in_planes)
        self.activ1 = activation()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.norm2 = norm_layer(planes)
        self.activ2 = activation()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=(1, 1), stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.activ1(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.activ2(self.norm2(out))
        out = self.conv2(out)
        out += shortcut
        return out


class PreactResNet(nn.Module):
    """ Based on  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,
     "Identity Mappings in Deep Residual Networks". arXiv:1603.05027
    """
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(PreactResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_layer=norm_layer, activation=activation)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_layer=norm_layer, activation=activation)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_layer=norm_layer, activation=activation)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_layer=norm_layer, activation=activation)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


def preact_resnet18(num_classes=10, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
    return PreactResNet(PreactBasicBlock, [2, 2, 2, 2], num_classes, norm_layer, activation=activation)


def preact_resnet18_nn(num_classes=10, activation=nn.ReLU):
    return preact_resnet18(num_classes=num_classes, norm_layer=nn.Identity, activation=activation)


def preact_resnet18_ln(num_classes=10, activation=nn.ReLU):
    ln_layer = lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return preact_resnet18(num_classes=num_classes, norm_layer=ln_layer, activation=activation)


def preact_resnet18_gn(num_classes=10, group_size=32, activation=nn.ReLU):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return preact_resnet18(num_classes=num_classes, norm_layer=gn_layer, activation=activation)

