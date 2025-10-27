#!/usr/bin/env python3
"""
启动脚本 - 直接运行FastAPI应用
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("🚀 Starting Brain Tumor Segmentation API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)