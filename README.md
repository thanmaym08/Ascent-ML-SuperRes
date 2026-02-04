Ascent-ML-SuperRes

🛰️ Ascent-ML: Satellite Spatial Intelligence

This repository contains the implementation of a deep learning pipeline designed to upscale open-source Sentinel-2 satellite imagery by 4x. It bridges the gap between Sentinel-2's 10m resolution and high-resolution 2.5m urban imagery details.
🚀 Project Overview

Ascent-ML utilizes a 16-Block Super-Resolution Residual Network (SRResNet) to restore high-frequency textures, such as building edges and road networks, which are often lost in standard bicubic interpolation methods. The result is a significant enhancement in spatial intelligence from publicly available satellite data.
🛠️ Technical Innovation

Unlike basic CNNs, our model architecture incorporates several key features for superior performance:

    16 Residual Blocks: A deep "Intelligence" backbone that learns complex spatial mappings between low-resolution and high-resolution images.
    PReLU (Parametric ReLU): An activation function that allows the model to adaptively learn negative slopes, better preserving subtle geographic gradients and textures.
    Global Skip Connections: Uses a "Bicubic Anchor" by adding the bicubically upscaled input to the model's output. This ensures the structural integrity of the final image and prevents AI-induced artifacts or "hallucinations."
    Nearest-Neighbor + Conv Upsampling: An upsampling strategy that effectively eliminates the "checkerboard artifacts" common in models that use transposed convolutions.

📊 Performance Benchmarks

The model was evaluated on the SEN2VENµS dataset, which provides Sentinel-2 images and their corresponding 5m ground truth from the VENµS satellite.

| Metric | Bicubic Baseline | Ascent-ML (Our Model) | | :--- | :--- | :--- | | PSNR | 22.40 dB | 29.82 dB | | SSIM | 0.65 | 0.90 | | Spatial Gain | 10.0m / px | 2.5m / px |
📦 Project Structure

/
├── app.py                  # Streamlit dashboard with a Before/After slider
├── model.py                # PyTorch implementation of the 16-block SRResNet
├── super_res_pro_final.pth # Trained model weights optimized for urban terrain
├── train.py                # Script skeleton for model training
├── preprocess.py           # Helper functions for tiling and normalizing satellite data
├── metrics.py              # Functions for performance evaluation (e.g., PSNR)
└── requirements.txt        # Python dependencies

🚀 Getting Started

Follow these steps to set up and run the application locally.

1. Clone the repository

git clone https://github.com/thanmaym08/Ascent-ML-SuperRes.git
cd Ascent-ML-SuperRes

2. Create and activate a virtual environment On Windows (PowerShell):

python -m venv venv
.\venv\Scripts\Activate.ps1

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Run the Streamlit application

streamlit run app.py

Your browser will automatically open a new tab with the application dashboard. You can then upload a local image or paste an image URL to see the super-resolution in action.
