# 🧠 Multimodal AI Systems for Brain Tumor Segmentation & Analysis

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.1-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art web application for automated brain tumor segmentation using multimodal MRI images with advanced CRF post-processing and 3D visualization capabilities.

## 🎯 Key Features

- **🔬 Deep Learning Segmentation**: SSUNet3D model for precise tumor delineation
- **🎨 CRF Post-Processing**: Enhanced boundary refinement using Conditional Random Fields
- **📊 3D Visualization**: Interactive brain and tumor mesh rendering
- **📈 Clinical Metrics**: Automated volume, surface area, and urgency assessment
- **🌐 Web Interface**: User-friendly drag-and-drop file upload
- **🚀 REST API**: Programmatic access for integration
- **📋 Multi-modal Support**: FLAIR, T1, T1ce, T2 MRI sequences

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/DeepakBolleddu/Multimodal-AI-Systems-for-Brain-Tumor-Segmentation-Analysis-.git
cd Multimodal-AI-Systems-for-Brain-Tumor-Segmentation-Analysis-

# Switch to enhanced CRF branch
git checkout crf

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
cd app
pip install -r requirements.txt
```

### 2. Launch Application

```bash
# Start the server
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Open browser to: http://127.0.0.1:8000
```

### 3. Upload & Analyze

1. **Upload NIfTI files** (.nii, .nii.gz) - supports 1-4 modalities
2. **Click "Run Inference"** - processing takes ~30-60 seconds  
3. **View Results** - 3D visualizations, metrics, and downloadable masks

## 📋 System Requirements

- **Python**: 3.12+ (recommended: 3.12.10)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB for models and dependencies
- **GPU**: CUDA-capable (optional, for faster inference)
- **OS**: Windows 10/11, Linux, macOS

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Pipeline   │
│                 │    │                 │    │                 │
│ • HTML/CSS/JS   │◄──►│ • FastAPI       │◄──►│ • SSUNet3D      │
│ • File Upload   │    │ • REST Endpoints│    │ • CRF Processing│
│ • 3D Rendering  │    │ • File Handling │    │ • 3D Meshing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

- **SSUNet3D Model**: State-of-the-art 3D U-Net architecture for medical segmentation
- **CRF Post-Processing**: 2D fully connected conditional random fields for boundary refinement
- **3D Visualization**: Marching cubes algorithm for surface mesh generation
- **Clinical Metrics**: Automated computation of medical indicators and urgency scoring

## 📊 Performance

| Metric | Value |
|--------|--------|
| **Model Architecture** | SSUNet3D (3D U-Net variant) |
| **Input Modalities** | FLAIR, T1, T1ce, T2 |
| **Processing Time** | 30-60 seconds per case |
| **Output Resolution** | Original input resolution maintained |
| **3D Mesh Quality** | Sub-voxel precision with smoothing |

## 🔬 Research Applications

- **Tumor Volume Quantification**: Precise measurement for treatment monitoring
- **Boundary Analysis**: Enhanced segmentation with CRF post-processing  
- **3D Spatial Analysis**: Geometric and morphological feature extraction
- **Clinical Decision Support**: Automated urgency and risk assessment
- **Dataset Processing**: Batch analysis capabilities for research studies

## 📁 Project Structure

```
├── app/                     # Main web application
│   ├── main.py             # FastAPI server & endpoints
│   ├── model_loader.py     # SSUNet3D model utilities
│   ├── crf.py              # CRF post-processing
│   ├── utils_nifti.py      # Medical image processing
│   ├── models/             # Pre-trained model weights
│   ├── public/             # Frontend (HTML/CSS/JS)
│   └── requirements.txt    # Python dependencies
│
├── crf/                    # Advanced CRF research tools
│   ├── 2DFCCRF.py         # Custom CRF implementation
│   ├── evaluate_crf.py    # Performance evaluation
│   └── run_crf_sweep.py   # Hyperparameter optimization
│
└── SETUP_GUIDE.md         # Detailed installation guide
```

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | System health check |
| `/dataset` | GET | Available dataset info |
| `/infer` | POST | Run tumor segmentation |

### Example API Usage

```python
import requests

# Upload files for segmentation
files = {
    'files': [
        ('files', open('flair.nii.gz', 'rb')),
        ('files', open('t1ce.nii.gz', 'rb')),
    ]
}

response = requests.post('http://localhost:8000/infer', files=files)
result = response.json()

print(f"Tumor Volume: {result['metrics']['volume_cm3']:.2f} cm³")
print(f"Urgency Level: {result['metrics']['urgency']['urgency_level']}")
```

## 🔧 Advanced Configuration

### CRF Parameters

Fine-tune post-processing in `main.py`:

```python
CRF_PARAMS = {
    'crf_iters': 5,      # Iteration count
    'sxy_g': 3,          # Gaussian spatial std
    'compat_g': 3,       # Gaussian compatibility  
    'sxy_b': 80,         # Bilateral spatial std
    'srgb_b': 13,        # Bilateral color std
    'compat_b': 10,      # Bilateral compatibility
}
```

### Environment Variables

```bash
# .env file configuration
DATASET_DIR=/path/to/datasets    # Optional: dataset location
OUTPUT_DIR=./static             # Output directory
MODEL_PATH=./models/best_model.pth  # Model file path
```

## 🧪 Sample Data

Includes BraTS2021 sample data for testing:
- Multi-modal MRI sequences (FLAIR, T1, T1ce, T2)
- Ground truth segmentation masks
- Pre-configured for immediate testing

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements  
- Documentation updates
- Issue reporting and feature requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Disclaimer

**Important**: This software is intended for research purposes only. It should not be used for clinical diagnosis or treatment decisions without proper validation and regulatory approval.

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{brain_tumor_segmentation_2025,
  title={Multimodal AI Systems for Brain Tumor Segmentation and Analysis},
  author={DeepakBolleddu},
  year={2025},
  url={https://github.com/DeepakBolleddu/Multimodal-AI-Systems-for-Brain-Tumor-Segmentation-Analysis-}
}
```

## 🔗 Related Work

- **BraTS Challenge**: Brain Tumor Segmentation benchmark datasets
- **SSUNet**: Spatial and Spectral U-Net for medical image segmentation  
- **Dense CRF**: Fully connected conditional random fields for semantic segmentation

---

**🎉 Ready to analyze brain tumors with state-of-the-art AI? Get started with the [Setup Guide](SETUP_GUIDE.md)!**