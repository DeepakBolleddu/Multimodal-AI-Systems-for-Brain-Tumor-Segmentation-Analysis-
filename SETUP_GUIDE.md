# Brain Tumor Segmentation System - Setup & Launch Guide

## 🚀 Quick Start Guide

This guide will help you set up and run the Brain Tumor Segmentation System with CRF post-processing capabilities.

## 📋 Prerequisites

- **Python 3.12+** (Recommended: Python 3.12.10)
- **Git** for version control
- **Windows 10/11** (tested environment)
- **At least 8GB RAM** for model inference
- **CUDA-capable GPU** (optional, for faster inference)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DeepakBolleddu/Multimodal-AI-Systems-for-Brain-Tumor-Segmentation-Analysis-.git
cd Multimodal-AI-Systems-for-Brain-Tumor-Segmentation-Analysis-
```

### 2. Switch to CRF Branch (Enhanced Version)

```bash
git checkout crf
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# For macOS/Linux
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
cd app
pip install -r requirements.txt
```

**Note**: If `pydensecrf` installation fails (common on Windows), use:
```bash
pip install https://github.com/lucasb-eyer/pydensecrf/archive/master.zip
```

## 🚀 Launch Application

### Method 1: Using Uvicorn (Recommended)

```bash
# Make sure you're in the app directory
cd app

# Start the server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Method 2: Using the Startup Script

```bash
cd app
python start_server.py
```

### Method 3: Development Mode (with auto-reload)

```bash
cd app
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## 🌐 Access the Application

Once the server is running, open your web browser and navigate to:

**http://127.0.0.1:8000**

You should see the Brain Tumor Segmentation web interface.

## 📊 Using the Application

### 1. Upload NIfTI Files

The system supports two input methods:

**Option A: Upload Files**
- Click "Choose Files" or drag & drop NIfTI files (.nii, .nii.gz)
- Supported modalities: FLAIR, T1ce, T1, T2
- Upload 1-4 modality files

**Option B: Use Dataset Samples**
- Select a patient ID from the dropdown (if dataset is configured)
- Uses pre-loaded sample data

### 2. Run Inference

- Click "Run Inference" button
- Wait for processing (typically 30-60 seconds)
- The system will:
  - Load and preprocess the images
  - Run the SSUNet3D model
  - Apply CRF post-processing (if enabled)
  - Generate 3D visualizations

### 3. View Results

The results page displays:
- **Tumor Mask**: Downloadable segmentation mask (.nii.gz)
- **3D Brain Mesh**: Interactive brain surface visualization
- **3D Tumor Mesh**: Interactive tumor visualization  
- **Medical Metrics**:
  - Tumor volume (cm³)
  - Surface area (mm²)
  - Centroid coordinates
  - Bounding box dimensions
  - **Urgency Assessment**: Automated risk scoring

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `app` directory:

```bash
# Optional: Dataset directory path
DATASET_DIR=/path/to/your/dataset

# Optional: Custom output directory
OUTPUT_DIR=./static

# Optional: Model path (uses default if not specified)
MODEL_PATH=./models/best_model.pth
```

### CRF Parameters

CRF post-processing can be configured in `main.py`:

```python
CRF_PARAMS = {
    'crf_iters': 5,      # Number of CRF iterations
    'sxy_g': 3,          # Spatial standard deviation (Gaussian)
    'compat_g': 3,       # Compatibility (Gaussian)
    'sxy_b': 80,         # Spatial standard deviation (Bilateral)
    'srgb_b': 13,        # Color standard deviation (Bilateral)
    'compat_b': 10,      # Compatibility (Bilateral)
}
```

## 🧪 Testing with Sample Data

The repository includes sample BraTS2021 data in `app/datasets/`:
- `BraTS2021_00003_flair.nii.gz`
- `BraTS2021_00003_t1.nii.gz`
- `BraTS2021_00003_t1ce.nii.gz`
- `BraTS2021_00003_t2.nii.gz`
- `BraTS2021_00003_seg.nii.gz` (ground truth)

## 🔍 Troubleshooting

### Common Issues

**1. CRF Import Error**
```
Warning: CRF module not available. CRF functionality will be disabled.
```
**Solution**: Install pydensecrf using the GitHub method:
```bash
pip install https://github.com/lucasb-eyer/pydensecrf/archive/master.zip
```

**2. Model Loading Error**
```
WARNING: Model path not found. The application will not produce valid segmentations.
```
**Solution**: Ensure `best_model.pth` is in the `app/models/` directory.

**3. Port Already in Use**
```
ERROR: [Errno 10048] Only one usage of each socket address
```
**Solution**: Change the port number:
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

**4. Memory Issues**
**Solution**: 
- Close other applications
- Use smaller input images
- Enable GPU if available

### Checking Installation

Verify CRF installation:
```bash
cd app
python -c "from crf import refine_volume_axial_binary; print('CRF OK')"
```

Verify model loading:
```bash
cd app
python -c "from model_loader import load_model; model = load_model('models/best_model.pth'); print('Model OK')"
```

## 📦 Project Structure

```
├── app/                          # Main application
│   ├── main.py                  # FastAPI application
│   ├── model_loader.py          # Model loading utilities
│   ├── utils_nifti.py          # NIfTI processing utilities
│   ├── crf.py                   # CRF post-processing
│   ├── start_server.py          # Startup script
│   ├── requirements.txt         # Python dependencies
│   ├── models/                  # Model files
│   │   └── best_model.pth       # Pre-trained SSUNet3D model
│   ├── public/                  # Frontend files
│   │   ├── index.html          # Web interface
│   │   ├── app.js              # JavaScript functionality
│   │   └── styles.css          # Styling
│   ├── static/                  # Output directory
│   └── datasets/                # Sample data
│
├── crf/                         # CRF research tools
│   ├── 2DFCCRF.py              # 2D fully connected CRF
│   ├── evaluate_crf.py         # CRF evaluation metrics
│   └── run_crf_sweep.py        # Parameter optimization
│
├── SETUP_GUIDE.md              # This file
└── README.md                   # Project overview
```

## 🎯 Features

- ✅ **Multi-modal Brain Tumor Segmentation** (FLAIR, T1, T1ce, T2)
- ✅ **SSUNet3D Deep Learning Model** (State-of-the-art architecture)
- ✅ **CRF Post-processing** (Enhanced boundary refinement)
- ✅ **3D Visualization** (Interactive brain and tumor meshes)
- ✅ **Medical Metrics** (Volume, surface area, spatial analysis)
- ✅ **Urgency Assessment** (Automated clinical risk scoring)
- ✅ **Web Interface** (User-friendly drag-and-drop)
- ✅ **REST API** (Programmatic access)

## 🏥 Clinical Use

This system is designed for research purposes. Key applications:

1. **Tumor Volume Quantification**
2. **Treatment Planning Support**
3. **Disease Progression Monitoring**
4. **Research Data Analysis**

**Important**: This tool is for research purposes only and should not be used for clinical diagnosis without proper validation.

## 🤝 Support

For technical issues:
1. Check this troubleshooting guide
2. Verify all dependencies are installed
3. Ensure Python 3.12+ is being used
4. Check GitHub issues for known problems

## 📚 Research Tools

The `crf/` directory contains advanced research tools:
- **2DFCCRF Implementation**: Custom 2D fully connected CRF
- **Evaluation Scripts**: Performance metrics and validation
- **Parameter Optimization**: Automated hyperparameter tuning

These tools were used to develop and optimize the CRF post-processing pipeline.

---

**🎉 You're all set! The Brain Tumor Segmentation System should now be running successfully.**

For the best experience, ensure CRF functionality is enabled (no warning messages on startup).