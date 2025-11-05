"""
FastAPI main application
DeepSeek Trader Bot REST API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from bot.market import create_exchange, fetch_multi_timeframes, get_current_price, format_symbol
from bot.indicators import calculate_all_indicators
from bot.llm_agent import LLMAgent
from bot.trade_manager import TradeManager
from bot.db import db
from bot.logger import log_trade_to_csv, log_decision, log_llm_reasoning, export_trades_csv
from bot.backtest import Backtest, simple_rsi_strategy, monte_carlo_simulation
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DeepSeek Trader API",
    description="LLM-Assisted Crypto Day-Trading Bot",
    version="1.0.0"
)

# CORS middleware - Allow all origins for development and production
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app$",  # Allow all Vercel URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
exchange = None
llm_agent = None
trade_manager = None
bot_paused = False


# Pydantic models for requests
class BatchDecisionRequest(BaseModel):
    symbols: List[str]
    balance: Optional[float] = None

class TradeRequest(BaseModel):
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str


class DecisionRequest(BaseModel):
    symbol: str
    balance: Optional[float] = None


class ControlRequest(BaseModel):
    action: str  # 'pause', 'resume', 'kill'


# Initialize exchange and agents
def initialize_system():
    """Initialize exchange, LLM agent, and trade manager"""
    global exchange, llm_agent, trade_manager
    
    logger.info("Initializing trading system...")
    
    # Create exchange
    exchange = create_exchange(
        exchange_id="kraken",
        api_key=settings.kraken_api_key,
        secret=settings.kraken_secret_key,
        paper_mode=settings.paper_mode
    )
    logger.info(f"Exchange initialized: {'PAPER MODE' if settings.paper_mode else 'LIVE MODE'}")
    
    # Create LLM agent
    if settings.deepseek_api_key:
        llm_agent = LLMAgent(api_key=settings.deepseek_api_key)
        logger.info("LLM agent initialized")
    else:
        logger.warning("DeepSeek API key not provided - LLM agent disabled")
    
    # Create trade manager
    trade_manager_config = {
        'paper_mode': settings.paper_mode,
        'risk_per_trade': settings.risk_per_trade,
        'max_exposure': settings.max_exposure,
        'min_rr': settings.min_rr,
        'max_allowable_stop_pct': settings.max_allowable_stop_pct,
        'max_open_positions': settings.max_open_positions,
        'max_daily_loss': settings.max_daily_loss
    }
    
    trade_manager = TradeManager(exchange, trade_manager_config)
    logger.info("Trade manager initialized")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    initialize_system()
    logger.info("DeepSeek Trader API started successfully")


# Health check
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "1.0.0",
        "paper_mode": settings.paper_mode,
        "paused": bot_paused
    }


# Get balance
@app.get("/balance")
async def get_balance():
    """Get account balance"""
    try:
        if settings.paper_mode:
            # Return simulated balance
            return {
                "balance": 10000.0,
                "currency": settings.base_currency,
                "paper_mode": True
            }
        else:
            balance_data = exchange.fetch_balance()
            usdt_balance = balance_data.get(settings.base_currency, {}).get('free', 0)
            return {
                "balance": usdt_balance,
                "currency": settings.base_currency,
                "paper_mode": False
            }
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get supported symbols
@app.get("/symbols")
async def get_symbols():
    """Get list of supported trading pairs"""
    coins = settings.get_coins_list()
    symbols = [format_symbol(coin, settings.base_currency) for coin in coins]
    return {
        "symbols": symbols,
        "base_currency": settings.base_currency
    }


# Get market data for a symbol
@app.get("/market")
async def get_market_data(symbol: str):
    """Get current market data and indicators for a symbol"""
    try:
        # Fetch multi-timeframe data
        timeframes = settings.get_timeframes_list()
        data_multi_tf = fetch_multi_timeframes(exchange, symbol, timeframes)
        
        # Calculate indicators for each timeframe
        indicators_multi_tf = {}
        for tf, df in data_multi_tf.items():
            if not df.empty:
                indicators_multi_tf[tf] = calculate_all_indicators(df)
        
        # Get current price
        current_price = get_current_price(exchange, symbol)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "indicators": indicators_multi_tf,
            "timeframes": timeframes
        }
    
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get LLM decision
@app.post("/decision")
async def get_decision(request: DecisionRequest):
    """Get trading decision from LLM"""
    if not llm_agent:
        raise HTTPException(status_code=503, detail="LLM agent not available")
    
    if bot_paused:
        raise HTTPException(status_code=503, detail="Bot is paused")
    
    try:
        # Get balance
        if request.balance:
            balance = request.balance
        else:
            balance_data = await get_balance()
            balance = balance_data['balance']
        
        # Fetch market data and indicators
        timeframes = settings.get_timeframes_list()
        data_multi_tf = fetch_multi_timeframes(exchange, request.symbol, timeframes)
        
        indicators_multi_tf = {}
        for tf, df in data_multi_tf.items():
            if not df.empty:
                indicators_multi_tf[tf] = calculate_all_indicators(df)
        
        # Get LLM decision
        decision = llm_agent.get_decision(
            symbol=request.symbol,
            balance=balance,
            risk_pct=settings.risk_per_trade,
            indicators_multi_tf=indicators_multi_tf
        )
        
        if not decision:
            raise HTTPException(status_code=500, detail="Failed to get LLM decision")
        
        # Log decision
        log_decision(request.symbol, decision, indicators_multi_tf)
        
        return decision
    
    except Exception as e:
        logger.error(f"Error getting decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get batch LLM decisions for all symbols
@app.post("/decision/batch")
async def get_batch_decisions(request: BatchDecisionRequest):
    """Get trading decisions for multiple symbols in one API call"""
    if not llm_agent:
        raise HTTPException(status_code=503, detail="LLM agent not available")
    
    if bot_paused:
        raise HTTPException(status_code=503, detail="Bot is paused")
    
    try:
        # Get balance
        if request.balance:
            balance = request.balance
        else:
            balance_data = await get_balance()
            balance = balance_data['balance']
        
        # Fetch market data for all symbols
        timeframes = settings.get_timeframes_list()
        symbols_data = {}
        
        for symbol in request.symbols:
            data_multi_tf = fetch_multi_timeframes(exchange, symbol, timeframes)
            
            indicators_multi_tf = {}
            for tf, df in data_multi_tf.items():
                if not df.empty:
                    indicators_multi_tf[tf] = calculate_all_indicators(df)
            
            symbols_data[symbol] = indicators_multi_tf
        
        # Get batch decisions from LLM
        response = llm_agent.get_batch_decisions(
            symbols_data=symbols_data,
            balance=balance,
            risk_pct=settings.risk_per_trade
        )
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to get batch LLM decisions")
        
        # Log full response including reasoning
        log_llm_reasoning(request.symbols, response)
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting batch decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Execute trade
@app.post("/trade")
async def execute_trade(request: TradeRequest):
    """Execute a trade based on LLM decision"""
    if bot_paused:
        raise HTTPException(status_code=503, detail="Bot is paused")
    
    try:
        # Get balance
        balance_data = await get_balance()
        balance = balance_data['balance']
        
        # Compute position size
        units = trade_manager.compute_position_size(
            balance_usd=balance,
            risk_pct=settings.risk_per_trade,
            entry=request.entry_price,
            stop=request.stop_loss
        )
        
        # Prepare LLM response for logging
        llm_response = {
            'action': request.side,
            'entry_price': request.entry_price,
            'stop_loss': request.stop_loss,
            'take_profit': request.take_profit,
            'confidence': request.confidence,
            'reason': request.reason
        }
        
        # Execute trade
        result = trade_manager.execute_trade(
            symbol=request.symbol,
            side=request.side,
            entry=request.entry_price,
            stop=request.stop_loss,
            take_profit=request.take_profit,
            units=units,
            balance=balance,
            llm_response=llm_response
        )
        
        # If trade executed, save to database
        if result['executed']:
            position = result['position']
            trade_id = db.add_trade(position)
            log_trade_to_csv(position)
            result['trade_id'] = trade_id
        
        return result
    
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get all trades
@app.get("/trades")
async def get_trades(limit: int = 100):
    """Get trade history"""
    try:
        trades = db.get_all_trades(limit=limit)
        return {
            "trades": trades,
            "count": len(trades)
        }
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get open positions
@app.get("/positions")
async def get_open_positions():
    """Get currently open positions"""
    try:
        positions = trade_manager.get_open_positions()
        return {
            "positions": positions,
            "count": len(positions)
        }
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get metrics
@app.get("/metrics")
async def get_metrics(days: int = 30):
    """Get performance metrics"""
    try:
        metrics = db.get_metrics(days=days)
        
        # Calculate summary stats from trades
        trades = db.get_all_trades(limit=1000)
        closed_trades = [t for t in trades if t['status'] == 'closed']
        
        total_pnl = sum([t['pnl_usd'] for t in closed_trades if t['pnl_usd']])
        winning_trades = [t for t in closed_trades if t['pnl_usd'] and t['pnl_usd'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl_usd'] and t['pnl_usd'] < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        return {
            "metrics": metrics,
            "summary": {
                "total_trades": len(closed_trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "total_pnl": total_pnl
            }
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Control bot
@app.post("/control")
async def control_bot(request: ControlRequest):
    """Control bot (pause/resume/kill)"""
    global bot_paused
    
    if request.action == "pause":
        bot_paused = True
        logger.info("Bot paused by user")
        return {"status": "paused", "message": "Bot paused successfully"}
    
    elif request.action == "resume":
        bot_paused = False
        logger.info("Bot resumed by user")
        return {"status": "active", "message": "Bot resumed successfully"}
    
    elif request.action == "kill":
        # Close all open positions
        bot_paused = True
        positions = trade_manager.get_open_positions()
        
        for pos_id in list(positions.keys()):
            symbol = positions[pos_id]['symbol']
            current_price = get_current_price(exchange, symbol)
            trade_manager.close_position(pos_id, current_price, 'manual_kill')
        
        logger.warning("Bot killed - all positions closed")
        return {"status": "killed", "message": f"Bot killed, {len(positions)} positions closed"}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")


# Export trades
@app.get("/export/trades")
async def export_trades():
    """Export all trades to CSV"""
    try:
        trades = db.get_all_trades(limit=10000)
        output_file = export_trades_csv(trades)
        return {
            "success": True,
            "file": output_file,
            "count": len(trades)
        }
    except Exception as e:
        logger.error(f"Error exporting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run backtest
@app.get("/backtest/simple")
async def run_simple_backtest(symbol: str = "BTC/USDT", days: int = 30):
    """Run simple RSI backtest"""
    try:
        # Fetch historical data
        from bot.market import fetch_ohlcv
        from bot.indicators import rsi, atr
        
        df = fetch_ohlcv(exchange, symbol, '1h', limit=days*24)
        
        # Add indicators
        df['rsi'] = rsi(df['close'], 14)
        df['atr'] = atr(df, 14)
        
        # Run backtest
        bt = Backtest(initial_balance=10000, risk_per_trade=0.02)
        results = bt.run_simple_strategy(df, simple_rsi_strategy)
        
        return {
            "symbol": symbol,
            "period": f"{days} days",
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monte Carlo simulation
@app.get("/backtest/montecarlo")
async def run_monte_carlo(win_rate: float = 0.55, avg_win: float = 100, 
                         avg_loss: float = 50, num_simulations: int = 1000):
    """Run Monte Carlo risk simulation"""
    try:
        results = monte_carlo_simulation(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_simulations=num_simulations
        )
        return results
    except Exception as e:
        logger.error(f"Error running Monte Carlo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

