#!/bin/bash

echo "=============================================="
echo "  Brain Tumor Segmentation System Launcher"
echo "=============================================="
echo

# Check if we're in the correct directory
if [ ! -f "app/main.py" ]; then
    echo "ERROR: Please run this script from the project root directory"
    echo "The directory should contain the 'app' folder"
    exit 1
fi

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run the following commands first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  cd app"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "✅ Starting Brain Tumor Segmentation API..."
echo
echo "📊 Server will be available at: http://127.0.0.1:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo

# Change to app directory and start the server
cd app
../.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000

echo
echo "Server stopped."