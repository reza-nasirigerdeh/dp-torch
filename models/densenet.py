import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.norm = norm_layer(in_channels)
        self.activ = activation(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.conv(self.activ(self.norm(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate)

        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, dropout_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.norm = norm_layer(in_channels)
        self.activ = activation(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2))

        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.conv(self.activ(self.norm(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate)
        out = self.avg_pool(out)
        return out


class DenseNet20x16(nn.Module):
    def __init__(self, num_classes=10, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, dropout_rate=0.0):
        super(DenseNet20x16, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.dense1 = nn.Sequential(*[
            BasicBlock(in_channels=32, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=48, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=64, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=80, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=96, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
        ])

        self.transition1 = TransitionBlock(in_channels=112, out_channels=112, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate)

        self.dense2 = nn.Sequential(*[
            BasicBlock(in_channels=112, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=128, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=144, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=160, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=176, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
        ])

        self.transition2 = TransitionBlock(in_channels=192, out_channels=192, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate)

        self.dense3 = nn.Sequential(*[
            BasicBlock(in_channels=192, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=208, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=224, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=240, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
            BasicBlock(in_channels=256, out_channels=16, norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate),
        ])

        self.norm = norm_layer(272)
        self.activ = activation()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(272, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.transition1(out)
        out = self.dense2(out)
        out = self.transition2(out)
        out = self.dense3(out)
        out = self.norm(out)
        out = self.activ(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


def densenet20x16_nn(num_classes=10, activation=nn.ReLU, dropout_rate=0.0):
    return DenseNet20x16(num_classes=num_classes, norm_layer=nn.Identity, activation=activation, dropout_rate=dropout_rate)


def densenet20x16_gn(num_classes=10, group_size=16, activation=nn.ReLU, dropout_rate=0.0):
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_in_channels//group_size, num_channels=num_in_channels)
    return DenseNet20x16(num_classes=num_classes, norm_layer=gn_layer, activation=activation, dropout_rate=dropout_rate)


def densenet20x16_ln(num_classes=10, activation=nn.ReLU, dropout_rate=0.0):
    ln_layer = lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return DenseNet20x16(num_classes=num_classes, norm_layer=ln_layer, activation=activation, dropout_rate=dropout_rate)
