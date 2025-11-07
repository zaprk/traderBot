"""
FastAPI main application
DeepSeek Trader Bot REST API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import json
import os
import asyncio
from datetime import datetime, timedelta

from bot.market import create_exchange, fetch_multi_timeframes, get_current_price, format_symbol
from bot.indicators import calculate_all_indicators
from bot.llm_agent import LLMAgent
from bot.trade_manager import TradeManager
from bot.db import db
from bot.logger import log_trade_to_csv, log_decision, log_llm_reasoning, export_trades_csv, LOG_DIR
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
    allow_origins=["*"],  # Allow ALL origins
    allow_credentials=False,  # Can't use credentials with wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
exchange = None
llm_agent = None
trade_manager = None
bot_paused = False
auto_trading_event = None  # Will be initialized at startup


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


# Background task for monitoring positions (checks SL/TP)
async def position_monitor_loop():
    """Monitor open positions every 30 seconds for stop-loss/take-profit"""
    # Wait for system to initialize
    await asyncio.sleep(15)
    logger.info("Position monitor started - checking every 30 seconds")
    
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Check if exchange and trade_manager are initialized
            if not exchange or not trade_manager:
                continue
            
            # Get all open positions from database
            open_positions_dict = trade_manager.get_open_positions()
            
            if not open_positions_dict:
                continue  # No positions to monitor
            
            # Fetch current prices for all open positions
            current_prices = {}
            for position_id, position in open_positions_dict.items():
                symbol = position['symbol']
                try:
                    formatted_symbol = format_symbol(symbol)
                    ticker = exchange.fetch_ticker(formatted_symbol)
                    current_prices[symbol] = ticker['last']
                except Exception as e:
                    logger.error(f"Error fetching price for {symbol}: {e}")
            
            # Monitor positions and close if SL/TP hit
            closed_positions = trade_manager.monitor_positions(current_prices)
            
            # Update database for closed positions
            for closed_pos in closed_positions:
                try:
                    db.update_trade(closed_pos['id'], {
                        'status': 'closed',
                        'exit_price': closed_pos['exit_price'],
                        'exit_time': closed_pos['exit_time'],
                        'pnl_usd': closed_pos['pnl_usd'],
                        'pnl_pct': closed_pos['pnl_pct'],
                        'exit_reason': closed_pos['exit_reason']
                    })
                    log_trade_to_csv(closed_pos)
                    logger.info(f"✅ Position closed: {closed_pos['symbol']} {closed_pos['exit_reason']} - P&L: ${closed_pos['pnl_usd']:.2f}")
                except Exception as e:
                    logger.error(f"Error updating closed position: {e}")
                    
        except Exception as e:
            logger.error(f"Error in position monitor loop: {e}")
            await asyncio.sleep(30)


# Background task for auto-trading
async def auto_trading_loop():
    """Background task that runs hourly auto-trading analysis"""
    # Wait for system to initialize
    await asyncio.sleep(10)
    logger.info("Auto-trading scheduler started")
    
    while True:
        try:
            # Check if auto-trading is enabled (check BEFORE sleep so it runs immediately when enabled)
            auto_trading = db.get_setting('auto_trading')
            if auto_trading != 'true':
                logger.info("Auto-trading disabled, waiting for enable signal...")
                # Wait for auto-trading to be enabled (interruptible sleep)
                try:
                    await asyncio.wait_for(auto_trading_event.wait(), timeout=60)
                    auto_trading_event.clear()  # Reset event
                    logger.info("🚀 Auto-trading enabled signal received! Starting analysis...")
                except asyncio.TimeoutError:
                    pass  # Timeout, check again
                continue
            
            # Check if bot is paused
            if bot_paused:
                logger.info("Bot paused, skipping auto-trading")
                continue
            
            # Check if exchange and llm_agent are initialized
            if not exchange or not llm_agent:
                logger.error("Exchange or LLM agent not initialized, skipping auto-trading")
                continue
            
            logger.info("🤖 AUTO-TRADING: Starting hourly analysis...")
            
            # Get all symbols
            coins = settings.get_coins_list()
            symbols = [f"{coin}/USDT" for coin in coins]
            logger.info(f"📊 Analyzing {len(symbols)} symbols: {symbols}")
            
            # Fetch market data for all symbols
            market_data_batch = {}
            for symbol in symbols:
                try:
                    # Symbol is already formatted (e.g., 'BTC/USDT'), don't format again!
                    data = fetch_multi_timeframes(exchange, symbol)
                    if data:
                        indicators_5m = calculate_all_indicators(data['5m'])
                        indicators_15m = calculate_all_indicators(data['15m'])
                        indicators_1h = calculate_all_indicators(data['1h'])
                        
                        market_data_batch[symbol] = {
                            'indicators': {
                                '5m': indicators_5m,
                                '15m': indicators_15m,
                                '1h': indicators_1h
                            },
                            'current_price': get_current_price(exchange, symbol)
                        }
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            
            if not market_data_batch:
                logger.error("No market data available for auto-trading")
                continue
            
            # Get current balance for decision-making
            balance_data = await get_balance()
            balance = balance_data['balance']
            logger.info(f"💰 Current balance: ${balance:.2f}")
            
            # Extract just the indicators for each symbol (remove 'current_price')
            symbols_indicators = {
                symbol: data['indicators']
                for symbol, data in market_data_batch.items()
            }
            
            # Get batch decisions from LLM
            try:
                logger.info("🧠 Calling DeepSeek AI for batch analysis...")
                decisions = llm_agent.get_batch_decisions(
                    symbols_data=symbols_indicators,
                    balance=balance,
                    risk_pct=settings.risk_per_trade
                )
                logger.info(f"✅ Auto-trading: Received decisions for {len(decisions) if decisions else 0} symbols")
                
                if not decisions:
                    logger.warning("No decisions returned from AI")
                    continue
                
                # Log AI reasoning for persistence
                log_llm_reasoning(
                    symbol=list(market_data_batch.keys()),
                    decision=decisions,
                    full_response=decisions  # Full response for logging
                )
                logger.info("💾 AI reasoning logged to persistent storage")
                
                # Execute high-confidence trades
                high_confidence_count = 0
                for symbol, decision in decisions.items():
                    action = decision.get('action')
                    confidence = decision.get('confidence', 0)
                    
                    if action in ['long', 'short'] and confidence > 0.7:
                        high_confidence_count += 1
                        logger.info(f"🎯 High confidence trade opportunity: {symbol} {action.upper()} (confidence={confidence})")
                        try:
                            # Get balance
                            balance_data = await get_balance()
                            balance = balance_data['balance']
                            
                            # Compute position size
                            units = trade_manager.compute_position_size(
                                balance_usd=balance,
                                risk_pct=settings.risk_per_trade,
                                entry=decision['entry_price'],
                                stop=decision['stop_loss']
                            )
                            
                            # Execute trade
                            result = trade_manager.execute_trade(
                                symbol=symbol,
                                side=action,
                                entry=decision['entry_price'],
                                stop=decision['stop_loss'],
                                take_profit=decision['take_profit'],
                                units=units,
                                balance=balance,
                                llm_response=decision
                            )
                            
                            if result['executed']:
                                position = result['position']
                                trade_id = db.add_trade(position)
                                log_trade_to_csv(position)
                                logger.info(f"✅ AUTO-TRADE EXECUTED: {symbol} {action.upper()} @ ${decision['entry_price']}, SL=${decision['stop_loss']}, TP=${decision['take_profit']}, confidence={confidence}")
                        except Exception as e:
                            logger.error(f"Error executing auto-trade for {symbol}: {e}")
                
                logger.info(f"📈 Auto-trading cycle complete: Found {high_confidence_count} high-confidence opportunities")
                
            except Exception as e:
                logger.error(f"Error in auto-trading batch analysis: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error in auto-trading loop: {e}", exc_info=True)  # Log full traceback
        
        # Wait 1 hour before next analysis (interruptible)
        logger.info("⏰ Auto-trading: Waiting 1 hour until next analysis (or until re-enabled)...")
        try:
            await asyncio.wait_for(auto_trading_event.wait(), timeout=3600)
            auto_trading_event.clear()  # Reset event
            logger.info("🔄 Auto-trading re-triggered early!")
        except asyncio.TimeoutError:
            logger.info("⏰ 1 hour passed, starting next analysis cycle...")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    global auto_trading_event
    
    # Initialize event for auto-trading signaling
    auto_trading_event = asyncio.Event()
    logger.info("Auto-trading event initialized")
    
    initialize_system()
    
    # Initialize auto_trading setting if not exists
    if db.get_setting('auto_trading') is None:
        db.set_setting('auto_trading', 'false')
    
    # Start background tasks
    asyncio.create_task(auto_trading_loop())
    asyncio.create_task(position_monitor_loop())
    
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
            # Calculate balance from trades in database
            trades = db.get_all_trades(limit=10000)  # Get all trades
            
            # Start with initial balance
            balance = float(settings.initial_balance)
            
            # Add/subtract PnL from closed trades
            for trade in trades:
                if trade['status'] == 'closed' and trade['pnl_usd'] is not None:
                    balance += trade['pnl_usd']
            
            return {
                "balance": balance,
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


# Get/Set auto-trading state
@app.get("/auto-trading")
async def get_auto_trading():
    """Get auto-trading state"""
    try:
        auto_trading = db.get_setting('auto_trading')
        return {
            "enabled": auto_trading == 'true',
            "auto_trading": auto_trading == 'true'  # backward compat
        }
    except Exception as e:
        logger.error(f"Error getting auto-trading state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auto-trading")
async def set_auto_trading(enabled: bool = True):
    """Enable or disable auto-trading"""
    try:
        db.set_setting('auto_trading', 'true' if enabled else 'false')
        logger.info(f"Auto-trading {'enabled' if enabled else 'disabled'}")
        
        # Signal the background task to wake up immediately if enabled
        if enabled:
            logger.info("🚀 Signaling auto-trading loop to start immediately...")
            auto_trading_event.set()
        
        return {
            "success": True,
            "enabled": enabled,
            "message": f"Auto-trading {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        logger.error(f"Error setting auto-trading state: {e}")
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
        response = llm_agent.get_decision(
            symbol=request.symbol,
            balance=balance,
            risk_pct=settings.risk_per_trade,
            indicators_multi_tf=indicators_multi_tf
        )
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to get LLM decision")
        
        # Extract single decision from batch format
        decisions = response.get('decisions', {})
        decision = decisions.get(request.symbol, {
            "action": "none",
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "confidence": 0,
            "reason": "No decision found for symbol"
        })
        
        # Log decision (CSV)
        log_decision(request.symbol, decision, indicators_multi_tf)
        
        # Log full LLM reasoning (for AI Logs tab)
        log_llm_reasoning(request.symbol, decision, response)
        
        # Return in expected format
        return {
            "decision": decision,
            "_raw_response": response.get('_raw_response'),
            "summary": response.get('summary')
        }
    
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
        
        # Auto-execute trades with high confidence (>0.7)
        executed_trades = []
        if response.get('decisions'):
            for symbol, decision in response['decisions'].items():
                if decision['action'] in ['long', 'short'] and decision.get('confidence', 0) > 0.7:
                    try:
                        # Calculate position size
                        units = trade_manager.compute_position_size(
                            balance_usd=balance,
                            risk_pct=settings.risk_per_trade,
                            entry=decision['entry_price'],
                            stop=decision['stop_loss']
                        )
                        
                        # Execute trade
                        result = trade_manager.execute_trade(
                            symbol=symbol,
                            side=decision['action'],
                            entry=decision['entry_price'],
                            stop=decision['stop_loss'],
                            take_profit=decision['take_profit'],
                            units=units,
                            balance=balance,
                            llm_response=decision
                        )
                        
                        # Save to database if executed
                        if result['executed']:
                            position = result['position']
                            trade_id = db.add_trade(position)
                            log_trade_to_csv(position)
                            executed_trades.append({
                                'symbol': symbol,
                                'trade_id': trade_id,
                                'action': decision['action'],
                                'confidence': decision['confidence']
                            })
                            logger.info(f"Auto-executed {decision['action'].upper()} on {symbol} (confidence: {decision['confidence']})")
                    except Exception as e:
                        logger.error(f"Failed to auto-execute trade for {symbol}: {e}")
        
        # Add execution info to response
        response['auto_executed'] = executed_trades
        
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


# Get AI reasoning logs
@app.get("/ai-logs")
async def get_ai_logs(days: int = 7):
    """Get AI reasoning logs from the past N days"""
    try:
        logs = []
        today = datetime.utcnow()
        
        # Ensure logs directory exists
        if not os.path.exists(LOG_DIR):
            return {"logs": [], "count": 0}
        
        # Check logs for the past N days
        for day_offset in range(days):
            date = today - timedelta(days=day_offset)
            date_str = date.strftime('%Y%m%d')
            log_file = os.path.join(LOG_DIR, f"llm_reasoning_{date_str}.jsonl")
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    entry = json.loads(line)
                                    logs.append(entry)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")
                    continue
        
        # Sort by timestamp descending (newest first)
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            "logs": logs[:100],  # Return last 100 entries
            "count": len(logs[:100])
        }
    except Exception as e:
        logger.error(f"Error fetching AI logs: {e}")
        return {"logs": [], "count": 0, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

