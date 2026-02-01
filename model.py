import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.conv(x)

class SRResNet(nn.Module):
    def __init__(self, upscale_factor=4): 
        super(SRResNet, self).__init__()
        self.initial = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU())
        # 5 Residual blocks to learn urban patterns
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        # Upsampling logic to turn 10m/pixel into sharp images
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor), 
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.upsample(x)