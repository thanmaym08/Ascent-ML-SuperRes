import torch
import torch.optim as optim
from model import SRResNet
from metrics import calculate_psnr

# Step 1: Initialize the Model
# We use the T4 GPU (.cuda()) because training is heavy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRResNet(upscale_factor=4).to(device)

# Step 2: Loss and Optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print(f"Model initialized on {device}. Ready for the Grind.")

# This is where the training loop will go when we move to Colab