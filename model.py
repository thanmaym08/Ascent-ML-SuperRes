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
        
        # 1. Feature Extraction (High-res start)
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), 
            nn.PReLU()
        )
        
        # 2. Deep Intelligence: 16 Blocks (Standard for 4K reconstruction)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # 3. 4K Upsampling Logic: Nearest-Neighbor avoids checkerboard grids
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Global Skip Connection (The "Bicubic" anchor)
        shortcut = torch.nn.functional.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        
        x_initial = self.initial(x)
        x_res = self.res_blocks(x_initial)
        x_mid = self.mid_conv(x_res)
        
        # Local Residual
        x_local = x_initial + x_mid
        
        # Reconstruct and add shortcut
        return self.upsample(x_local) + shortcut