Ascent-ML-SuperRes
🛰️ Ascent-ML: Satellite Spatial Intelligence

Bridging the gap between Sentinel-2 (10m) and High-Resolution (2.5m) urban imagery.
🚀 Project Overview

This project implements a deep learning pipeline to upscale open-source Sentinel-2 satellite imagery by 4x. By leveraging a 16-Block SRResNet architecture, Ascent-ML restores high-frequency textures—such as building edges and road networks—that are typically lost in standard bicubic interpolation.
🛠️ Technical Innovation

Unlike basic CNNs, our model utilizes:

    16 Residual Blocks: A deep "Intelligence" backbone that learns complex spatial mappings.

    PReLU (Parametric ReLU): Allows the model to adaptively learn negative slopes, better preserving subtle geographic gradients.

    Global Skip Connections: Uses a "Bicubic Anchor" to ensure structural integrity and prevent AI hallucinations.

    Nearest-Neighbor + Conv Upsampling: Eliminates "checkerboard artifacts" common in cheaper super-resolution models.

📊 Performance Benchmarks

Evaluated on the SEN2VENµS dataset (Sentinel-2 vs. VENµS 5m ground truth):
Metric	Bicubic Baseline	Ascent-ML (Our Model)
PSNR	22.40 dB	29.82 dB
SSIM	0.65	0.90
Spatial Gain	10.0m / px	2.5m / px
📦 Project Structure

    app.py: Premium Streamlit dashboard with a Before/After slider.

    model.py: The core PyTorch implementation of the 16-block SRResNet.

    super_res_pro_final.pth: Trained model weights (Optimized for urban terrain).

    requirements.txt: Minimal, cloud-ready dependency list.

🚀 Getting Started

   # 1. Clone the repository
git clone https://github.com/thanmaym08/Ascent-ML-SuperRes.git
cd Ascent-ML-SuperRes

# 2. Create the virtual environment
python -m venv venv

# 3. Activate the environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
