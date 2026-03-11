import torch.nn.functional as F
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.batch_norm1(self.conv1(x))
        out = F.relu(out)
        out = self.batch_norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 50):
        super().__init__()
        self.n_classes = n_classes
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer_1 = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(3)])
        self.layer_2 = nn.Sequential(
            *[
                ResidualBlock(64 if _ == 0 else 128, 128, stride=2 if _ == 0 else 1)
                for _ in range(4)
            ]
        )
        self.layer_3 = nn.Sequential(
            *[
                ResidualBlock(128 if _ == 0 else 256, 256, stride=2 if _ == 0 else 1)
                for _ in range(6)
            ]
        )
        self.layer_4 = nn.Sequential(
            *[
                ResidualBlock(256 if _ == 0 else 512, 512, stride=2 if _ == 0 else 1)
                for _ in range(3)
            ]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512, self.n_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)

        return x
