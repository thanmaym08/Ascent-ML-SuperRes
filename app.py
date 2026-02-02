import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image
import torch
import numpy as np
import os
from model import SRResNet 

st.set_page_config(page_title="Klymo Ascent SR", layout="wide")
st.title("🛰️ Satellite Super-Resolution: Sentinel-2 to HR")

# --- 🧠 Load the AI Brain ---
@st.cache_resource
def load_ai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRResNet(upscale_factor=4)
    # Ensure this file is in your F:\Ascent-ML folder!
    model.load_state_dict(torch.load("super_res_final.pth", map_location=device))
    model.to(device).eval()
    return model, device

model, device = load_ai()

# --- 🌍 Sidebar: City Selection & Upload ---
st.sidebar.title("🌍 Control Panel")
mode = st.sidebar.radio("Choose Mode:", ["City Gallery", "Manual Upload"])

target_image = None

if mode == "City Gallery":
    city = st.sidebar.selectbox("Select a City:", ["Delhi", "New York", "Bengaluru"])
    img_path = f"samples/{city.lower()}.png"
    if os.path.exists(img_path):
        target_image = Image.open(img_path).convert("RGB")
    else:
        st.sidebar.error(f"Please add {city.lower()}.png to your /samples folder!")

else:
    uploaded_file = st.sidebar.file_uploader("Upload Sentinel-2 patch", type=["jpg", "png"])
    if uploaded_file:
        target_image = Image.open(uploaded_file).convert("RGB")

# --- ⚡ AI Processing Engine ---
if target_image:
    # 1. Prepare input
    input_lr = target_image.resize((64, 64))
    
    # 2. Run Inference
    img_tensor = torch.from_numpy(np.array(input_lr)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    # 3. Prepare output
    output_array = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_array = np.clip(output_array, 0, 1)
    sr_img = Image.fromarray((output_array * 255).astype(np.uint8))

    # --- 🛰️ Display Result ---
    st.write(f"### AI Reconstruction: {city if mode == 'City Gallery' else 'Uploaded Image'}")
    image_comparison(
       img1=input_lr.resize((512, 512), resample=Image.BILINEAR), 
        img2=sr_img.resize((512, 512)), 
        label1="Original Sentinel-2",
        label2="Ascent-SR (AI Enhanced)",
    )
    
    st.success("AI Enhancement Complete! Notice the sharper building edges and road networks.")