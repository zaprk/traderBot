# 🚀 Quick Start Guide

Get DeepSeek Trader running in 5 minutes!

## Prerequisites

- Python 3.10+ installed ([Download](https://www.python.org/downloads/))
- Node.js 18+ installed ([Download](https://nodejs.org/))
- DeepSeek API key ([Get one here](https://platform.deepseek.com))

## Option 1: Automated Start (Easiest)

### Windows:
```cmd
start.bat
```

### Mac/Linux:
```bash
chmod +x start.sh
./start.sh
```

This will:
1. Create a `.env` file if it doesn't exist
2. Install all dependencies
3. Start both backend and frontend

## Option 2: Manual Start

### Step 1: Configure API Keys

Create `backend/.env` file:

```bash
# Copy the example
cd backend
cp .env.example .env

# Edit with your favorite editor
# On Windows: notepad .env
# On Mac: open -e .env
# On Linux: nano .env
```

Add your DeepSeek API key (minimum required):
```env
DEEPSEEK_API_KEY=sk-your-key-here
PAPER_MODE=true
```

### Step 2: Start Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload
```

✅ Backend running at http://127.0.0.1:8000

### Step 3: Start Frontend (New Terminal)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

✅ Frontend running at http://localhost:5173

## Step 4: Access the Dashboard

Open your browser to: **http://localhost:5173**

You should see:
- 🤖 DeepSeek Trader dashboard
- 📄 PAPER MODE badge (confirming you're in safe mode)
- Balance widget showing $10,000 (simulated)
- Market tiles for BTC, ETH, SOL, DOGE, TON

## First Actions

### 1. Test the Backend API

Visit http://127.0.0.1:8000/docs to see the interactive API documentation.

Try:
- `GET /` - Health check
- `GET /balance` - Get balance
- `GET /symbols` - List trading pairs

### 2. Get an AI Decision

In the dashboard:
1. Click on any market tile (e.g., BTC/USDT)
2. Click **"🤖 Get AI Decision"**
3. Wait for DeepSeek to analyze the market
4. Review the AI's recommendation

### 3. Execute a Paper Trade (Optional)

If the AI suggests a trade:
1. Review the entry, stop loss, and take profit levels
2. Click **"Execute Trade"**
3. Confirm the trade
4. View it in the Trade Log section

**Note:** This is paper trading - no real money involved!

## Understanding Paper Mode

In paper mode (`PAPER_MODE=true`):
- ✅ All features work normally
- ✅ Trades are simulated and logged
- ✅ Perfect for testing and learning
- ❌ No real money at risk
- ❌ Orders don't hit exchanges

**Start here and run for 2-4 weeks before considering live trading.**

## What to Try Next

1. **Monitor Performance**: Check the Dashboard metrics
2. **View Trades**: Scroll down to see the Trade Log
3. **Test Controls**: Try Pause/Resume buttons
4. **Explore Indicators**: Click market tiles to see RSI, MACD, etc.
5. **Check Logs**: Look in `backend/logs/` for detailed logs

## Common Issues

### Port Already in Use

If port 8000 or 5173 is busy:

Backend:
```bash
uvicorn main:app --port 8001
```

Frontend - edit `vite.config.js`:
```js
server: { port: 5174 }
```

### Module Not Found

Make sure virtual environment is activated:
```bash
# You should see (venv) in your terminal prompt
# If not:
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### CORS Errors

Check that `FRONTEND_URL` in `.env` matches your frontend URL (default: http://localhost:5173)

### No Data Loading

1. Check backend is running (http://127.0.0.1:8000)
2. Check browser console for errors (F12)
3. Verify DeepSeek API key is valid

## Next Steps

Once comfortable with paper trading:

1. Read [SETUP.md](SETUP.md) for detailed configuration
2. Review [ROADMAP.md](ROADMAP.md) for system architecture
3. Check [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
4. Consider live trading (⚠️ only with small amounts!)

## Getting Help

- **Check logs**: `backend/logs/bot.log`
- **Review documentation**: See SETUP.md and ROADMAP.md
- **API docs**: http://127.0.0.1:8000/docs
- **Open an issue**: If you find bugs

## Safety Reminders

- ✅ Always start with paper trading
- ✅ Test thoroughly before live trading
- ✅ Use small amounts when going live
- ✅ Set conservative risk limits
- ✅ Never trade more than you can afford to lose
- ❌ Don't skip paper trading phase
- ❌ Don't use high leverage
- ❌ Don't leave bot unmonitored

## Success Checklist

- [ ] Backend running at port 8000
- [ ] Frontend running at port 5173
- [ ] Dashboard loads successfully
- [ ] Can see balance and market data
- [ ] Can get AI decisions
- [ ] Can execute paper trades
- [ ] Trades appear in Trade Log

## Have Fun! 🎉

You're now running an LLM-powered trading bot! Start exploring, testing strategies, and learning how algorithmic trading works.

**Remember:** This is a powerful tool for learning and experimentation. Always prioritize risk management and never trade with money you can't afford to lose.

Happy Trading! 📈🤖

