import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = F.avg_pool2d(x, 2)
        return x


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(IdentityBlock, self).__init__()

        self.conv_block = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.batch = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor >= 2:
            x = F.avg_pool2d(x, self.scale_factor)
        x = self.conv_block(x)
        x = self.batch(x)
        x = self.relu(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=2, base_size=64,
                 dropout=0.2, ratio=16, kernel_size=7,
                 last_filters=8, last_fc=2):
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=base_size)
        self.iden1 = IdentityBlock(in_channels=base_size, out_channels=base_size*8,
                               scale_factor=8)

        self.conv_block2 = ConvBlock(in_channels=base_size, out_channels=base_size*2)
        self.iden2 = IdentityBlock(in_channels=base_size * 2, out_channels=base_size*8,
                               scale_factor=4)

        self.conv_block3 = ConvBlock(in_channels=base_size*2, out_channels=base_size*4)
        self.iden3 = IdentityBlock(in_channels=base_size*4, out_channels=base_size*8,
                               scale_factor=2)

        self.conv_block4 = ConvBlock(in_channels=base_size*4, out_channels=base_size*8)

        self.merge = IdentityBlock(base_size*8*4, base_size*last_filters, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.lin = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size*last_filters, base_size*last_fc),
            nn.PReLU(),
            nn.BatchNorm1d(base_size*last_fc),
            nn.Dropout(dropout/2),
            nn.Linear(base_size*last_fc, num_classes),
        )


    def forward(self, x):
        x = self.conv_block1(x)
        iden1 = self.iden1(x)

        x = self.conv_block2(x)
        iden2 = self.iden2(x)

        x = self.conv_block3(x)
        iden3 = self.iden3(x)

        x = self.conv_block4(x)

        x = torch.cat([x, iden1, iden2, iden3], dim=1)

        x = self.merge(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x
        