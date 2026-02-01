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
        
        # Initial Extraction: 3 channels (RGB) -> 64 filters
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), 
            nn.PReLU()
        )
        
        # Deep Feature Extraction: 5 Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        
        # Mid-point correction (helps with Global Residual connection)
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling Logic
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor), # Channels drop from 256 -> 16
            nn.PReLU(),
            nn.Conv2d(16, 3, kernel_size=9, padding=4) # MUST BE 16, NOT 64
        )

    def forward(self, x):
        # 1. Store the original low-res input for the Global Skip Connection
        # We use bilinear interpolation to match the 4x target size
        shortcut = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 2. Main Processing path
        x_initial = self.initial(x)
        x_res = self.res_blocks(x_initial)
        x_mid = self.mid_conv(x_res)
        
        # 3. Add Local Residual Connection (from SRResNet paper)
        x_local = x_initial + x_mid
        
        # 4. Upsample and Add Global Residual Connection
        out = self.upsample(x_local)
        
        # Adding the bicubic-upsampled original input makes the model learn the "delta"
        return out + shortcut