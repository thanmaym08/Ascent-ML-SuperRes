import torch
import torch.nn as nn

class RRDB(nn.Module):
    """Residual in Residual Dense Block for high-fidelity 4K reconstruction."""
    def __init__(self, channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling (0.2) ensures training stability and prevents 'burst' pixels
        return x5 * 0.2 + x 

class AscentSR_Pro(nn.Module):
    def __init__(self, upscale_factor=4):
        super(AscentSR_Pro, self).__init__()
        self.initial = nn.Conv2d(3, 64, 3, 1, 1)
        
        # 16 RRDB Blocks: Deep enough for 4x upscale without quality loss
        self.body = nn.Sequential(*[RRDB(64) for _ in range(16)])
        
        self.conv_after_body = nn.Conv2d(64, 64, 3, 1, 1)
        
        # High-Performance Upsampling: Uses Bilinear + Conv to eliminate checkerboard grids
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        # Global Residual Anchor: Ensures structural similarity to the original Sentinel-2 data
        shortcut = torch.nn.functional.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        
        feat = self.initial(x)
        res = self.body(feat)
        res = self.conv_after_body(res)
        
        out = self.upsample(feat + res)
        # Clamping here prevents "burst" pixels during training and inference
        return torch.clamp(out + shortcut, 0, 1)