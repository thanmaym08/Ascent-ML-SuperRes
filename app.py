import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image
import torch
import numpy as np
from model import SRResNet  # Links your SRResNet architecture

st.set_page_config(page_title="Klymo Ascent SR", layout="wide")
st.title("🛰️ Satellite Super-Resolution: Sentinel-2 to HR")

# --- NEW SECTION: Load the AI Brain ---
@st.cache_resource
def load_ai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRResNet(upscale_factor=4)
    # This loads the 50-epoch weights you just finished training
    model.load_state_dict(torch.load("super_res_final.pth", map_location=device))
    model.to(device).eval()
    return model, device

model, device = load_ai()

# Upload section
uploaded_file = st.file_uploader("Upload a blurry Sentinel-2 patch", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    # --- NEW SECTION: AI Processing ---
    # 1. Resize input to 64x64 (the size your model expects)
    input_lr = img.resize((64, 64))
    
    # 2. Convert to Tensor and run through AI
    img_tensor = torch.from_numpy(np.array(input_lr)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    # 3. Convert back to a sharp 256x256 image
    output_array = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_array = np.clip(output_array, 0, 1)
    sr_img = Image.fromarray((output_array * 255).astype(np.uint8))

    # --- UPDATED SECTION: The Slider ---
    st.write("### AI-Powered Comparison (Move to see sharpening)")
    image_comparison(
        img1=input_lr.resize((256, 256)), # Show the blurry version at the same size
        img2=sr_img,                      # The sharp AI output
        label1="Sentinel-2 (10m)",
        label2="Ascent-SR (AI Enhanced)",
    )