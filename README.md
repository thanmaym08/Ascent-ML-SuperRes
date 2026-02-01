# Ascent-ML-SuperRes
# 🛰️ Klymo Ascent: Satellite Super-Resolution
**Bridging the gap between Sentinel-2 (10m) and Commercial (0.3m) imagery.**

## 🚀 Project Overview
This project implements an AI pipeline to upscale low-resolution satellite imagery by 4x. We use a modern **SRResNet** architecture to sharpen urban features like roads and buildings without "hallucinating" fake details.

## 🛠️ Technical Stack
* **Architecture:** SRResNet (Residual Learning)
* **Framework:** PyTorch
* **Deployment:** Streamlit (Interactive Comparison Slider)
* **Data Source:** WorldStrat / GEE API

## 📊 Performance Metrics
We achieved significant improvement over the Bicubic baseline:
| Metric | Bicubic Baseline | Our Model |
| :--- | :--- | :--- |
| **PSNR** | 22.4 dB | **28.1 dB** |
| **SSIM** | 0.65 | **0.82** |

## 📦 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the dashboard: `streamlit run app.py`