#!/bin/bash
# Quick start script for DeepSeek Trader

echo "🤖 DeepSeek Trader - Quick Start"
echo "================================"

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo "⚠️  No .env file found!"
    echo "Creating .env from example..."
    cp backend/.env.example backend/.env
    echo "✅ Created backend/.env"
    echo ""
    echo "❗ IMPORTANT: Edit backend/.env and add your API keys!"
    echo "   - DEEPSEEK_API_KEY"
    echo "   - KRAKEN_API_KEY (optional for paper trading)"
    echo "   - KRAKEN_SECRET_KEY (optional for paper trading)"
    echo ""
    read -p "Press Enter after you've configured .env..."
fi

# Start backend
echo ""
echo "🔧 Starting backend..."
cd backend

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

# Start backend in background
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

cd ..

# Start frontend
echo ""
echo "🎨 Starting frontend..."
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
fi

# Start frontend
echo "Starting React dev server..."
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "✅ DeepSeek Trader is starting!"
echo ""
echo "📊 Backend API: http://127.0.0.1:8000"
echo "📊 API Docs: http://127.0.0.1:8000/docs"
echo "🖥️  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait


