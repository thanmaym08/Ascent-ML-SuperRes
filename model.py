import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): return x + self.conv(x)

class SRResNet(nn.Module):
    def __init__(self, upscale_factor=4):
        super(SRResNet, self).__init__()
        self.initial = nn.Sequential(nn.Conv2d(3, 64, 9, padding=4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        self.mid_conv = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='nearest'),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        shortcut = torch.nn.functional.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        feat = self.initial(x)
        res = self.mid_conv(self.res_blocks(feat))
        return self.upsample(feat + res) + shortcut