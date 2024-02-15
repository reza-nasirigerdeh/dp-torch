import torch
from torch import nn


#  ############## For GroupNorm, LayerNorm, and Identity() as normalization layers ################################
def conv_block(in_channels, out_channels, norm_layer=nn.Identity, activation=nn.ReLU, max_pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        norm_layer(out_channels),
        activation(),
    ]
    if max_pool:
        layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.Identity, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm1 = norm_layer(out_channels)
        self.activ1 = activation()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm2 = norm_layer(out_channels)
        self.activ2 = activation()

    def forward(self, input_t):
        identity = input_t

        out = self.conv1(input_t)
        out = self.norm1(out)
        out = self.activ1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activ2(out)

        out_f = out + identity
        return out_f


class ResNet8(nn.Module):
    def __init__(self, num_classes=10, norm_layer=nn.Identity, activation=nn.ReLU):
        """ 8-layer residual network """
        super(ResNet8, self).__init__()

        self.conv_block1 = conv_block(in_channels=3, out_channels=64, norm_layer=norm_layer, activation=activation, max_pool=False)
        self.conv_block2 = conv_block(in_channels=64, out_channels=128, norm_layer=norm_layer, activation=activation, max_pool=True)
        self.res_block1 = ResidualBlock(in_channels=128, out_channels=128, norm_layer=norm_layer, activation=activation)
        self.conv_block3 = conv_block(in_channels=128, out_channels=256, norm_layer=norm_layer, activation=activation, max_pool=True)
        self.res_block2 = ResidualBlock(in_channels=256, out_channels=256, norm_layer=norm_layer, activation=activation)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input_t):
        out = self.conv_block1(input_t)
        out = self.conv_block2(out)
        out = self.res_block1(out)
        out = self.conv_block3(out)
        out = self.res_block2(out)

        out = self.max_pool(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


#  ############## Models ################################
def resnet8_nn(num_classes=10, activation=nn.ReLU):
    return ResNet8(num_classes=num_classes, norm_layer=nn.Identity, activation=activation)


def resnet8_ln(num_classes=10, activation=nn.ReLU):
    ln_layer = lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return ResNet8(num_classes=num_classes, norm_layer=ln_layer, activation=activation)


def resnet8_gn(num_classes=10, group_size=32, activation=nn.ReLU):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return ResNet8(num_classes=num_classes, norm_layer=gn_layer, activation=activation)