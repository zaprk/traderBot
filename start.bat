@echo off
REM Quick start script for DeepSeek Trader (Windows)

echo 🤖 DeepSeek Trader - Quick Start
echo ================================

REM Check if .env exists
if not exist "backend\.env" (
    echo ⚠️  No .env file found!
    echo Creating .env from example...
    copy backend\.env.example backend\.env
    echo ✅ Created backend\.env
    echo.
    echo ❗ IMPORTANT: Edit backend\.env and add your API keys!
    echo    - DEEPSEEK_API_KEY
    echo    - KRAKEN_API_KEY (optional for paper trading)
    echo    - KRAKEN_SECRET_KEY (optional for paper trading)
    echo.
    pause
)

REM Start backend
echo.
echo 🔧 Starting backend...
cd backend

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
pip install -q -r requirements.txt

REM Start backend in new window
echo Starting FastAPI server...
start "DeepSeek Trader Backend" cmd /k "venv\Scripts\activate.bat && uvicorn main:app --host 0.0.0.0 --port 8000"

cd ..

REM Start frontend
echo.
echo 🎨 Starting frontend...
cd frontend

REM Install dependencies if needed
if not exist "node_modules\" (
    echo Installing Node dependencies...
    call npm install
)

REM Start frontend in new window
echo Starting React dev server...
start "DeepSeek Trader Frontend" cmd /k "npm run dev"

cd ..

echo.
echo ✅ DeepSeek Trader is starting!
echo.
echo 📊 Backend API: http://127.0.0.1:8000
echo 📊 API Docs: http://127.0.0.1:8000/docs
echo 🖥️  Frontend: http://localhost:5173
echo.
echo Close the terminal windows to stop the servers
echo.
pause


