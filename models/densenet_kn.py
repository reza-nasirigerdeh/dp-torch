import torch
from torch import nn
from layers.knconv import KNConv2d
from layers.kernel_norm import KernelNorm2d1x1


class BasicBlockKN(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, knconv_dropout_p=0.1):
        super(BasicBlockKN, self).__init__()
        self.activ = activation(inplace=True)
        self.conv = KNConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                             bias=True, dropout_p=knconv_dropout_p)

    def forward(self, x):
        out = self.conv(self.activ(x))
        return torch.cat([x, out], 1)


class TransitionBlockKN(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, knconv_dropout_p=0.1):
        super(TransitionBlockKN, self).__init__()
        self.activ = activation(inplace=True)
        self.conv = KNConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                             bias=True, dropout_p=knconv_dropout_p)
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(self.activ(x))
        out = self.avg_pool(out)
        return out


class DenseNet20x16KN(nn.Module):
    def __init__(self, num_classes=10, activation=nn.ReLU, knconv_dropout_p=0.1, kn_dropout_p=0.5):
        super(DenseNet20x16KN, self).__init__()

        self.conv1 = KNConv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True, dropout_p=knconv_dropout_p)
        self.dense1 = nn.Sequential(*[
            BasicBlockKN(in_channels=32, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=48, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=64, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=80, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=96, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
        ])

        self.transition1 = TransitionBlockKN(in_channels=112, out_channels=112, activation=activation, knconv_dropout_p=knconv_dropout_p)

        self.dense2 = nn.Sequential(*[
            BasicBlockKN(in_channels=112, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=128, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=144, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=160, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=176, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
        ])

        self.transition2 = TransitionBlockKN(in_channels=192, out_channels=192, activation=activation, knconv_dropout_p=knconv_dropout_p)

        self.dense3 = nn.Sequential(*[
            BasicBlockKN(in_channels=192, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=208, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=224, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=240, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
            BasicBlockKN(in_channels=256, out_channels=16, activation=activation, knconv_dropout_p=knconv_dropout_p),
        ])

        self.knorm = KernelNorm2d1x1(dropout_p=kn_dropout_p)
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
        out = self.knorm(out)
        out = self.activ(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out


def densenet20x16_kn(num_classes=10, activation=nn.ReLU, knconv_dropout_p=0.1, kn_dropout_p=0.5):
    return DenseNet20x16KN(num_classes, activation, knconv_dropout_p, kn_dropout_p)
