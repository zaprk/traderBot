DeepSeek Trader — Roadmap (Detailed, step-by-step)

This document explains everything to build from scratch — no steps skipped. Read and follow sequentially.

PHASE 0 — PLANNING & PREREQUISITES
0.1 Goals

Build day-trading bot for BTC/ETH/SOL/DOGE/TON

Use DeepSeek V3 as advisory LLM

Multi-timeframe (5m/15m/1h/4h) signals

Web dashboard to control and view bot

Paper trading first, then live mode with strict risk caps

0.2 Accounts / keys required

DeepSeek API key

Kraken API key & secret (with trading permissions if live)

Optional: Stripe/Payment (if you plan paid hosting)

GitHub account for repo

0.3 Tools & tech

Python 3.10+

FastAPI, Uvicorn

CCXT (Kraken exchange wrapper)

Requests (DeepSeek)

SQLAlchemy + SQLite (initial DB)

React + Tailwind for UI

Docker (optional for deployment)

Git and GitHub

PHASE 1 — PROJECT SCAFFOLDING & ENVIRONMENT
1.1 Repo scaffold

Create directories: backend/, frontend/, docs/

Initialize git repo and .gitignore (exclude .env, venv/, trades.db)

1.2 Backend env

Create requirements.txt with:

fastapi
uvicorn
ccxt
requests
python-dotenv
sqlalchemy
pydantic
pandas
ta
pytest


Create .env.example listing all env variables:

DEEPSEEK_API_KEY=
KRAKEN_API_KEY=
KRAKEN_SECRET_KEY=
PAPER_MODE=true
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.1
MAX_OPEN_POSITIONS=3
BASE_CURRENCY=USDT
COINS=BTC,ETH,SOL,DOGE,TON
TIMEFRAMES=5m,15m,1h,4h

1.3 Frontend env

Initialize React app (Vite or CRA). Add Tailwind config.

Ensure CORS allowed in backend for frontend dev URL (http://localhost:3000 or 5173).

PHASE 2 — CORE BACKEND: BASIC API + HEALTH CHECKS
2.1 main.py

Create FastAPI app with endpoints:

GET / → health

GET /balance → kraken.fetch_balance() (error-wrapped)

GET /symbols → list supported pairs

POST /decision → request LLM decision (for given symbol + timeframe)

POST /trade → execute trade (protected; will check PAPER_MODE)

GET /trades → fetch trade history from DB

GET /metrics → performance metrics

2.2 .env and load_dotenv

Use python-dotenv to safely load keys.

Make sure backend refuses to start if keys missing but allows paper mode without exchange keys.

PHASE 3 — MARKET DATA & INDICATORS
3.1 market.py

Implement fetch_ohlcv(exchange, symbol, timeframe, limit=200) with retry and rate limiting.

Implement fetch_multi_timeframes(symbol, timeframes=['5m','15m','1h','4h']) returning a dict of dataframes (pandas).

3.2 indicators.py

Implement functions:

rsi(series, period=14)

ema(series, period)

macd(series) → macd_line, signal_line, histogram

atr(df, period=14)

volume_change_pct(df, window=24)

market_structure(df, lookback=20) → return "HHHL" / "HLHH" / "range" etc.

Each function must be unit-tested (create backend/tests/test_indicators.py with known inputs).

3.3 Data caching

Create a simple in-memory TTL cache for fetched candles (e.g., cache for 30 seconds for 5m, 120s for 1h). Avoid over-requesting CCXT.

PHASE 4 — LLM AGENT (DeepSeek V3)
4.1 llm_agent.py responsibilities

Build structured prompts from indicators for each coin

Send to DeepSeek API with proper headers

Parse the response and extract JSON (we require structured JSON output — see template below)

Implement fallback / retry logic:

Retry up to 2 times on network errors

If model reply not valid JSON, call again with a stricter format request

4.2 Prompt design (MANDATORY format)

System message:

You are a professional crypto trading assistant. You analyze structured market data across multiple timeframes and return a single JSON object with action, entry, stop_loss, take_profit, confidence, reason. Do not output any text other than the JSON.


User message (example — pass formatted summary):

SYMBOL: BTC/USDT
BALANCE: 123.45
MAX_RISK_PER_TRADE: 2% (you must not suggest trades with more risk)
TIMEFRAMES:
5m: RSI=62, MACD=hist_pos, EMA20>EMA50, ATR=...
15m: ...
1h: ...
4h: ...
VOLUME: 1h +12% vs avg
STRATEGY: Day trading. Risk is capped. Use ATR to size stops. Output JSON:
{
 "action": "long"|"short"|"none",
 "entry_price": float,
 "stop_loss": float,
 "take_profit": float,
 "confidence": float (0.0-1.0),
 "reason": "brief explanation"
}


Note: Always include MAX_RISK_PER_TRADE and that suggestions must respect it. The backend will still validate.

4.3 Example parsing

Expect the model to return JSON inside choices[0].message.content. Parse with json.loads. If parsing fails, request reformat.

PHASE 5 — TRADE MANAGER & RISK LAYER
5.1 trade_manager.py responsibilities

Compute position size: size = (balance * RISK_PER_TRADE) / (abs(entry - stop_loss) / entry) for spot (convert to asset units)

Enforce:

size must not exceed MAX_EXPOSURE setting

stop_loss distance must not exceed preconfigured absolute (e.g., 10% max)

risk/reward >= MIN_RR (e.g., 1.3)

OPEN_POSITIONS < MAX_OPEN_POSITIONS

DAILY_LOSS not exceeded

Implement place_order(exchange, symbol, side, size, entry):

Use limit order at entry (or market if immediate)

Place stop-loss and take-profit orders as OCO where exchange supports; otherwise emulate by monitoring and placing opposite order on trigger

Implement paper_mode: if PAPER_MODE=true, simulate execution and write to DB only — do not call exchange.create_order

5.2 position monitoring

Poll open orders and positions every minute

On exit, compute PnL, update balance, persist trade to DB with full LLM response attached

PHASE 6 — DATABASE & LOGGING
6.1 DB schema (SQLAlchemy)

Table: trades

id, symbol, side, entry_price, stop_loss, take_profit, exit_price, entry_time, exit_time, units, pnl_usd, pnl_pct, confidence, llm_reason, llm_raw, paper_mode (bool)

Table: positions

current open positions (if any)

Table: metrics (daily snapshots)

date, balance, equity, max_drawdown, cumulative_return

Table: settings (runtime config overrides)

6.2 Logging

logger.py writes CSV rows for each trade and also logs to a rotating log file (python logging module).

Export helper: /export/trades.csv endpoint

PHASE 7 — BACKTEST ENGINE & SIMULATION
7.1 backtest.py

Accepts historical OHLCV data, applies indicator pipeline, simulates the LLM by using a deterministic rule set (so you can test workflow without actual DeepSeek queries)

Implement simple strategy engine to validate strategy logic and risk flows before enabling LLM in live simulation

Add Monte-Carlo runner to test different win rates, R:R ratios, and sequence risk to estimate probability of ruin

7.2 test dataset

Provide instruction to download historical candles for BTC/ETH/SOL during specific months (use CCXT fetching).

Validate backtest outputs: equity curve, win rate, profit factor, max drawdown

PHASE 8 — FRONTEND DASHBOARD
8.1 endpoints required for frontend

GET /balance → show balance & equity

GET /market/{symbol} → show last candles & indicators

GET /trades → trade log (paginated)

GET /open_positions → current positions

POST /control/pause → pause bot

POST /control/resume → resume bot

GET /metrics → weekly/monthly metrics for charts

GET /llm/sample → show last LLM decisions for debugging

8.2 UI components

Balance widget (current balance, realized/unrealized pnl)

Market tiles (BTC/ETH/SOL/DOGE/TON live price + 1h sparkline)

TradeLog table (sortable)

Controls: Toggle Paper/Live, Pause/Resume, Risk sliders (admin)

Chart: Equity curve (Chart.js or Recharts)

Modal: Show LLM reasoning and raw JSON for each trade

8.3 Mobile / responsive

Use Tailwind breakpoints. Ensure touch-friendly controls and large fonts for mobile.

PHASE 9 — TESTING & QA
9.1 Unit tests

Indicators functions (rsi, ema, macd)

Market fetcher (mock CCXT responses)

LLM prompt builder (validate generated prompt strings)

Trade manager risk checks (various fail/pass cases)

9.2 Integration tests

Simulate a full trade flow in PAPER_MODE: data fetch → llm decision (simulated) → trade execution → exit and logging

9.3 Manual QA

Run paper trading live for 2–4 weeks

Verify stop-loss execution, slippage assumptions, fees recorded

Confirm DB integrity and that frontend shows correct metrics

PHASE 10 — DEPLOYMENT
10.1 Dockerize backend

Create Dockerfile and docker-compose.yml for local dev (FastAPI + SQLite)

Environment variables passed securely in cloud provider

10.2 Deploy backend

Options: Render, Railway, Google Cloud Run, DigitalOcean App Platform

Use small VPS or cloud run for 24/7 availability

Attach domain and TLS

10.3 Deploy frontend

Vercel or Netlify for React app

Configure CORS backend URL and env vars

10.4 Monitoring & Alerts

Integrate Sentry for errors

Use simple alerting (email or Telegram) for:

Major exceptions

Bot paused automatically due to risk triggers

Daily PnL below threshold

Optional: Prometheus + Grafana for advanced metrics

PHASE 11 — OPERATIONAL PROCEDURES & MAINTENANCE
11.1 Daily checks

Review GET /metrics every morning

Confirm no stuck orders

Confirm LLM usage cost under budget

11.2 Emergency procedures

Expose /control/kill endpoint to immediately cancel all open trades and pause bot

Document manual steps to revoke API keys on Kraken

11.3 Updating the LLM prompt

Keep revision history of prompt templates

When you change prompt, run a 2-week paper test before switching live

PHASE 12 — EXTENSIONS (optional, later)

Add news sentiment scrapers (Twitter, CoinDesk, Google News) and include summarized sentiment in prompts

Add ensemble of LLMs (DeepSeek + OpenAI + local lightweight model) to increase robustness

Add per-asset tuning (different risk for BTC vs DOGE)

Multi-exchange support & arbitrage module

Portfolio rebalancing scheduler

Webhooks for Slack / Telegram

DETAILED INSTRUCTIONS & TEMPLATES (IMPLEMENTATION READY)
Prompt template (exact). Use verbatim in llm_agent.py system/user messaging:

System:

You are a professional crypto trading assistant. You will be given structured market data across multiple timeframes. You must ONLY output a single JSON object that fits the schema exactly. Do not output any extra text.

Schema:
{
 "action": "long" | "short" | "none",
 "entry_price": float | null,
 "stop_loss": float | null,
 "take_profit": float | null,
 "confidence": float (0.0 - 1.0),
 "reason": "short string"
}


User (example):

SYMBOL: BTC/USDT
BALANCE: 123.45
MAX_RISK_PER_TRADE: 2% (must not recommend trades violating this)
TIMEFRAMES:
5m: RSI=62, MACD=hist_pos, EMA20>EMA50, ATR=0.5, last_close=34000
15m: RSI=58, MACD=hist_pos, EMA20≈EMA50, ATR=1.0
1h: RSI=65, MACD=macd_crossover, EMA20>EMA50>EMA200, ATR=2.2
4h: RSI=70, MACD=hist_pos, EMA alignment bullish, ATR=4.5
VOLUME: 1h +12% vs avg
STRATEGY: Day trading. Risk per trade cap and ATR-based stop. Provide JSON.

Example expected reply:
{
 "action": "long",
 "entry_price": 34200.0,
 "stop_loss": 33700.0,
 "take_profit": 35000.0,
 "confidence": 0.78,
 "reason": "short-term momentum, volume confirming breakout; 4h RSI high so tight stop"
}

Trade sizing pseudo-code (in trade_manager.py):
def compute_position_size(balance_usd, risk_pct, entry, stop):
    risk_amount = balance_usd * risk_pct
    dollar_risk_per_unit = abs(entry - stop)
    if dollar_risk_per_unit == 0: return 0
    units = risk_amount / dollar_risk_per_unit
    return units

Risk validation checklist (backend must enforce before sending order):

units > 0

units * entry_price <= MAX_EXPOSURE (e.g., 20% of balance)

abs(entry - stop)/entry <= MAX_ALLOWABLE_STOP_PCT (e.g., 0.10)

projected R:R >= MIN_RR (e.g., 1.3)

Today’s cumulative loss + projected_loss <= MAX_DAILY_LOSS * starting_balance

CHECKLIST — MINIMUM VIABLE PRODUCT (MVP)

 Backend with health endpoint and basic DeepSeek test

 CCXT fetcher for 5m/15m/1h/4h candles for BTC/ETH/SOL/DOGE/TON

 Indicators module and unit tests

 LLM prompt builder + call + JSON parsing with retries

 Trade manager with position sizing and paper execution

 SQLite DB + logging of trades including LLM raw outputs

 Frontend dashboard showing balance, last LLM decision, and trades

 Paper trading mode fully functional for 2+ weeks of live simulation

 Backtesting runner for quick validation of strategy logic

 Safety endpoints (pause/resume/kill)

Operational Notes & Costs

DeepSeek V3 token costs: follow your DeepSeek dashboard — estimate small per-call costs during dev; scale checks for frequent calls

Kraken fees: include trading fees and maker/taker assumptions in PnL calc

Hosting: small instance for backend is sufficient for MVP (e.g., $5–$10/month). Frontend can be hosted on Vercel free tier.

Monitoring: plan an alert channel (Telegram/Email) for critical bot events

Troubleshooting & FAQ (common issues)

Invalid address on card / payment — the billing address must match card's AVS data. Use exact bank record.

DeepSeek returns malformed JSON — re-request with a stricter "Output JSON only" system message and set retries.

Exchange rate limits — add exponential backoff and CCXT throttle. Use data caching.

Bot not placing orders — check PAPER_MODE flag, then verify API key permissions and system clocks.

High latency on 5m decisions — move 5m data caching to local and reduce prompt size by summarizing candles.