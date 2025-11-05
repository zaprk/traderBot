DeepSeek Trader — LLM-Assisted Crypto Day-Trading Bot (Web Dashboard)

DeepSeek Trader is a full-stack, LLM-assisted crypto day-trading system.
It uses DeepSeek V3 (chat API) as an advisory engine and CCXT (Kraken) for market data and order execution. The system runs a Python backend (FastAPI) that performs market data collection, indicator calculation, LLM prompting, risk enforcement, order execution, logging, and exposes a REST API for a lightweight React web dashboard (works great on Android browser). The system supports paper trading, logging, backtesting and a production mode.

Main coins: BTC/USDT, ETH/USDT, SOL/USDT, DOGE/USDT, TON/USDT (swap any out for liquidity if needed).

Key features

Multi-timeframe indicator collection: 5m, 15m, 1h, 4h

Indicators: RSI, MACD, EMA(20/50/200), ATR, ATR-based stop sizing, volume comparisons

LLM decision engine: DeepSeek V3 (prompted with structured summaries across timeframes)

Execution: CCXT (Kraken) — supports paper and live trading

Risk management: configurable per-trade risk %, max daily loss cap, max open positions, position sizing enforced server-side

Logging: SQLite / SQLAlchemy + CSV export; full trade and decision logs including LLM reply

Frontend: React dashboard (balance, open trades, trade history, AI reasoning)

Deployment: instructions for local dev and cloud hosting (Render / Vercel / Cloud Run)

Safety: API keys stored in .env, backend enforces all trades, frontend only reads via REST endpoints

Extensible: add news sentiment, backtesting, Monte-Carlo simulations, alternate exchanges, and model replacements

Repo layout (final)
deepseek-trader/
├── backend/
│   ├── main.py                # FastAPI entry
│   ├── requirements.txt
│   ├── .env.example
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── market.py          # ohlcv fetching multi-timeframe
│   │   ├── indicators.py      # RSI, MACD, EMA, ATR, helpers
│   │   ├── llm_agent.py       # build prompts, call DeepSeek, parse JSON
│   │   ├── trade_manager.py   # risk logic, position sizing, execution (paper/live)
│   │   ├── db.py              # SQLAlchemy models, session, migrations
│   │   ├── logger.py          # append to CSV, export helpers
│   │   └── backtest.py        # basic backtesting runner
│   └── tests/
│       └── test_indicators.py
└── frontend/
    ├── package.json
    ├── src/
    │   ├── App.jsx
    │   ├── index.jsx
    │   ├── api.js             # axios clients to backend
    │   ├── components/
    │   │   ├── Dashboard.jsx
    │   │   ├── Balance.jsx
    │   │   ├── TradeLog.jsx
    │   │   └── Controls.jsx
    │   └── styles/            # tailwind config
    └── public/

Quick start (local dev)

Clone:

git clone <repo-url>
cd deepseek-trader/backend


Create virtualenv and install:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Create .env from .env.example and fill with keys:

DEEPSEEK_API_KEY=...
KRAKEN_API_KEY=...
KRAKEN_SECRET_KEY=...
PAPER_MODE=true
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.10


Run backend:

uvicorn main:app --reload


Open http://127.0.0.1:8000/docs for API docs.

Frontend:

cd ../frontend
npm install
npm run dev


Open the frontend in a browser (mobile friendly).

Safety & disclaimers

This is an experimental trading system. Always paper-trade first.

Never expose API keys in frontend. Keep .env secret and do not commit.

Configure stop-loss and risk settings conservatively for small balances.

This project does not promise profitability — markets are risky.