@echo off
echo ==============================================
echo   Brain Tumor Segmentation System Launcher
echo ==============================================
echo.

REM Check if we're in the correct directory
if not exist "app\main.py" (
    echo ERROR: Please run this script from the project root directory
    echo The directory should contain the 'app' folder
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please run the following commands first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   cd app
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Starting Brain Tumor Segmentation API...
echo.
echo 📊 Server will be available at: http://127.0.0.1:8000
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Change to app directory and start the server
cd app
"..\\.venv\\Scripts\\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped. Press any key to exit...
pause > nul