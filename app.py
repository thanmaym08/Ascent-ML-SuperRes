import streamlit as st
import torch
import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from model import SRResNet
from streamlit_image_comparison import image_comparison
import time

# --- 1. Page Configuration & Professional Dark Theme ---
st.set_page_config(page_title="Ascent-ML Pro", layout="wide", page_icon="🛰️")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff; font-weight: 700; }
    .stMetric { background-color: #1c2128; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Efficient Model Loading (Cached) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    # Loading on CPU first then moving to device is faster for initialization
    model = SRResNet(upscale_factor=4)
    model.load_state_dict(torch.load("super_res_pro_final.pth", map_location='cpu'))
    model.to(device)
    model.eval()
    return model

# Initialize model only when needed
if 'model' not in st.session_state:
    with st.status("📡 Initializing Ascent-ML AI Engine...", expanded=False) as status:
        st.session_state.model = load_model()
        status.update(label="✅ Engine Online", state="complete")

model = st.session_state.model

# --- 3. Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092014.png", width=70)
    st.title("Ascent-ML Control")
    st.markdown("---")
    input_mode = st.radio("Select Data Source", ["Local Upload", "Remote URL"])
    st.info("Architecture: 16-Block SRResNet\nTarget: 4x Spatial Gain")

# --- 4. Main App Interface ---
st.title("🛰️ Satellite Super-Resolution")
st.write("Upscaling Sentinel-2 (10m) to Ultra-Sharp (2.5m) using deep residual blocks.")

img = None
if input_mode == "Local Upload":
    uploaded = st.file_uploader("Upload low-res scene", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
else:
    url = st.text_input("Paste Sentinel-2 Image URL:")
    if url:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("Unable to reach image URL. Please check the link.")

# --- 5. Optimized Inference with Progress Tracking ---
if img:
    # 1. Immediate Preview (User sees this instantly)
    st.divider()
    
    # 2. AI Processing with Feedback
    with st.status("🧠 AI Analysis in Progress...", expanded=True) as status:
        st.write("Preprocessing pixels...")
        lr_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        
        st.write("Running 16-Block Residual Inference...")
        progress_bar = st.progress(0)
        # Simulate progress for UI feel while PyTorch runs
        for i in range(1, 101, 20):
            time.sleep(0.1)
            progress_bar.progress(i)
            
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        progress_bar.progress(100)
        st.write("Post-processing reconstruction...")
        
        sr_output = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        sr_img = Image.fromarray((sr_output * 255).astype('uint8'))
        
        # Match sizes for comparison slider
        lr_resized = img.resize(sr_img.size, Image.Resampling.BICUBIC)
        status.update(label="✨ Enhancement Successful!", state="complete", expanded=False)

    # --- 6. Visual Results ---
    image_comparison(
        img1=lr_resized,
        img2=sr_img,
        label1="Original (10m)",
        label2="Ascent-ML 4K (2.5m)",
        width=1100,
        make_responsive=True
    )

    # Professional Download Option
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        buf = BytesIO()
        sr_img.save(buf, format="PNG")
        st.download_button("📥 Download Enhanced Image", buf.getvalue(), "ascent_4k.png", "image/png")