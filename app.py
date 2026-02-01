import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image

st.set_page_config(page_title="Klymo Ascent SR", layout="wide")
st.title("🛰️ Satellite Super-Resolution: Sentinel-2 to HR")

# Upload section
uploaded_file = st.file_uploader("Upload a blurry Sentinel-2 patch", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    # This is a placeholder for your model's output. 
    # For the video demo, we show how the slider works.
    sr_img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS) 

    st.write("### Comparison Slider (Move to see sharpening)")
    image_comparison(
        img1=img,
        img2=sr_img,
        label1="Sentinel-2 (10m)",
        label2="AI Enhanced (SR)",
    )