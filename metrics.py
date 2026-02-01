import torch
import numpy as np

def calculate_psnr(img1, img2):
    # Higher PSNR = Better reconstruction
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse.item()))