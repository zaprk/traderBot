# DeepSeek Trader - Setup Guide

Complete setup instructions for running the DeepSeek Trader bot locally and in production.

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- DeepSeek API key ([get one here](https://platform.deepseek.com))
- Kraken account with API credentials (optional for paper trading)

## Quick Start (Local Development)

### 1. Clone and Setup Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `backend` directory:

```bash
# Copy example file
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_SECRET_KEY=your_kraken_secret_key_here

# Start with paper trading
PAPER_MODE=true

# Risk settings (conservative defaults)
RISK_PER_TRADE=0.01
MAX_DAILY_LOSS=0.05
MAX_OPEN_POSITIONS=2
```

### 3. Run Backend

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### 4. Setup Frontend

In a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:5173`

## Testing

### Run Backend Tests

```bash
cd backend
pytest tests/test_indicators.py -v
```

### Test API Endpoints

```bash
# Health check
curl http://127.0.0.1:8000/

# Get balance
curl http://127.0.0.1:8000/balance

# Get symbols
curl http://127.0.0.1:8000/symbols
```

## Configuration Options

### Risk Management Settings

Edit these in `.env` to control risk:

- `RISK_PER_TRADE` (0.01 = 1%) - Amount of capital to risk per trade
- `MAX_DAILY_LOSS` (0.05 = 5%) - Maximum daily loss before bot pauses
- `MAX_OPEN_POSITIONS` (2) - Maximum number of concurrent positions
- `MAX_EXPOSURE` (0.20 = 20%) - Maximum % of capital per position
- `MIN_RR` (1.3) - Minimum risk:reward ratio required
- `MAX_ALLOWABLE_STOP_PCT` (0.10 = 10%) - Maximum stop loss distance

### Trading Configuration

- `COINS` - Comma-separated list of coins to trade (default: BTC,ETH,SOL,DOGE,TON)
- `TIMEFRAMES` - Analysis timeframes (default: 5m,15m,1h,4h)
- `BASE_CURRENCY` - Quote currency (default: USDT)

## Paper Trading

Always start with paper trading to test the system:

1. Set `PAPER_MODE=true` in `.env`
2. You can use any values for Kraken API keys (they won't be used)
3. The system will simulate trades and track performance
4. Run for at least 2-4 weeks before considering live trading

## Switching to Live Trading

⚠️ **WARNING: Live trading involves real money and risk!**

Before switching to live:

1. **Thoroughly test in paper mode** for several weeks
2. **Start with very small amounts** (e.g., $100-$500)
3. **Use conservative risk settings** (RISK_PER_TRADE=0.01 or less)
4. **Set up monitoring** and alerts
5. **Never trade more than you can afford to lose**

To enable live trading:

1. Set `PAPER_MODE=false` in `.env`
2. Ensure Kraken API keys have trading permissions
3. Verify API key permissions are correct on Kraken
4. Start with a small test trade to verify execution

## Monitoring & Maintenance

### Daily Checks

1. Check `/metrics` endpoint for performance
2. Review open positions
3. Verify no stuck orders
4. Check LLM API usage and costs

### Logs

Logs are stored in `backend/logs/`:
- `bot.log` - Main application log
- `trades.csv` - Trade history (CSV format)
- `decisions.jsonl` - LLM decisions log
- `errors.jsonl` - Error log

### Database

The SQLite database is stored at `backend/trades.db`

To backup:
```bash
cp backend/trades.db backend/trades_backup_$(date +%Y%m%d).db
```

## Troubleshooting

### "Module not found" errors

Make sure virtual environment is activated and dependencies installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### CORS errors in frontend

Check that `FRONTEND_URL` in backend `.env` matches your frontend URL.

### Exchange connection errors

- Verify API keys are correct
- Check Kraken API status
- Ensure API keys have necessary permissions
- Try paper mode first to isolate issues

### LLM returns invalid JSON

The bot has retry logic, but if persistent:
1. Check DeepSeek API status
2. Verify API key is valid
3. Check prompt in `bot/llm_agent.py`

### High API costs

- Reduce decision frequency
- Cache more aggressively
- Consider using smaller models (if available)
- Monitor usage in DeepSeek dashboard

## Production Deployment

See `DEPLOYMENT.md` for production deployment instructions.

## Support

For issues:
1. Check logs in `backend/logs/`
2. Review error messages
3. Test with paper mode
4. Check API status (DeepSeek, Kraken)

## Safety Reminders

- ✅ Always use paper mode first
- ✅ Start with tiny amounts in live trading
- ✅ Set conservative risk limits
- ✅ Monitor regularly
- ✅ Keep API keys secure
- ✅ Never commit `.env` to version control
- ❌ Don't trade more than you can afford to lose
- ❌ Don't leave bot unattended without alerts
- ❌ Don't increase risk without thorough testing


