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
from bot.sentiment import get_batch_sentiment
from bot.volatility_monitor import volatility_monitor
from bot.position_monitor import position_monitor
from bot.breakout_detector import breakout_detector
from bot.convergence_scorer import convergence_scorer
from bot.order_flow import order_flow_analyzer
from bot.regime_filter import regime_filter
from bot.correlation_filter import correlation_filter
from bot.market_memory import initialize_market_memory, market_memory
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
    """🎯 ADVANCED Position Monitor - Trailing stops, partial exits, dynamic risk management"""
    # Wait for system to initialize
    await asyncio.sleep(15)
    logger.info("🎯 Advanced position monitor started - checking every 30 seconds")
    
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
            
            # 🎯 ADVANCED POSITION MANAGEMENT
            for position_id, position in open_positions_dict.items():
                symbol = position['symbol']
                
                try:
                    # Fetch current price and ATR
                    formatted_symbol = format_symbol(symbol)
                    ticker = exchange.fetch_ticker(formatted_symbol)
                    current_price = ticker['last']
                    
                    # Get current ATR from 1h timeframe
                    data_1h = fetch_multi_timeframes(exchange, formatted_symbol, timeframes=['1h'])
                    if data_1h and '1h' in data_1h:
                        indicators_1h = calculate_all_indicators(data_1h['1h'])
                        atr = indicators_1h.get('atr', 0)
                        
                        # 🚀 CHECK FOR BREAKOUT AGAINST POSITION (reversal opportunity)
                        from bot.realtime_breakout_monitor import realtime_breakout_monitor
                        
                        # Calculate volume ratio
                        if len(data_1h['1h']) >= 2:
                            current_volume = data_1h['1h'].iloc[-1]['volume']
                            avg_volume = data_1h['1h'].iloc[-20:-1]['volume'].mean() if len(data_1h['1h']) >= 20 else current_volume
                        else:
                            current_volume = 0
                            avg_volume = 1
                        
                        # Check for reversal opportunity
                        reversal_signal = realtime_breakout_monitor.check_position_breakout(
                            position, current_price, current_volume, avg_volume
                        )
                        
                        if reversal_signal:
                            logger.warning(
                                f"🔄 {symbol}: {reversal_signal['recommendation']} "
                                f"(Volume: {reversal_signal['volume_ratio']:.1f}x)"
                            )
                            # Note: In production, you could trigger immediate analysis here
                            # For now, we just log the opportunity
                    else:
                        atr = 0
                    
                    # 🎯 APPLY ADVANCED POSITION MANAGEMENT
                    action = position_monitor.manage_position(symbol, position, current_price, atr)
                    
                    # Handle different actions
                    if action['action'] == 'close':
                        # Close entire position
                        db.update_trade(position_id, {
                            'status': 'closed',
                            'exit_price': action['price'],
                            'exit_time': datetime.now().isoformat(),
                            'exit_reason': action['reason']
                        })
                        # Calculate P&L
                        if position['side'] == 'long':
                            pnl_usd = (action['price'] - position['entry_price']) * position['units']
                        else:
                            pnl_usd = (position['entry_price'] - action['price']) * position['units']
                        
                        log_trade_to_csv({**position, 'exit_price': action['price'], 'pnl_usd': pnl_usd})
                        logger.info(f"🛑 Position closed: {symbol} {action['reason']} @ ${action['price']:.2f} - P&L: ${pnl_usd:.2f}")
                        
                        # Reset tracking
                        position_monitor.reset_position_tracking(symbol)
                    
                    elif action['action'] == 'partial_exit':
                        # Execute partial exits (in paper trading, just log it)
                        for exit in action['exits']:
                            logger.info(f"💰 {symbol}: {exit['reason']}")
                    
                    elif action['action'] == 'update_stop':
                        # Update trailing stop in database
                        db.update_trade(position_id, {
                            'stop_loss': action['new_stop']
                        })
                        logger.info(f"📈 {symbol}: Stop updated to ${action['new_stop']:.2f}")
                    
                    elif action['action'] == 'hold':
                        # Just monitoring
                        pass
                
                except Exception as e:
                    logger.error(f"Error managing position for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in position monitor loop: {e}")
            await asyncio.sleep(30)


# Background task for auto-trading
async def auto_trading_loop():
    """Background task that runs hourly auto-trading analysis"""
    # Wait for system to initialize and Railway health check to pass
    await asyncio.sleep(30)
    logger.info("Auto-trading scheduler started")
    
    while True:
        try:
            # Check if auto-trading is enabled (check BEFORE sleep so it runs immediately when enabled)
            if not db:
                logger.error("Database not available, auto-trading disabled")
                await asyncio.sleep(60)
                continue
            
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
            
            # Fetch market data for all symbols with ADVANCED ANALYSIS
            market_data_batch = {}
            for symbol in symbols:
                try:
                    # Symbol is already formatted (e.g., 'BTC/USDT'), don't format again!
                    data = fetch_multi_timeframes(exchange, symbol, timeframes=['5m', '15m', '30m', '1h'])
                    if data:
                        indicators_5m = calculate_all_indicators(data['5m'])
                        indicators_15m = calculate_all_indicators(data['15m'])
                        indicators_30m = calculate_all_indicators(data['30m'])
                        indicators_1h = calculate_all_indicators(data['1h'])
                        
                        # 🔥 BREAKOUT DETECTION
                        breakout_15m = breakout_detector.calculate_breakout_score(data['15m'], indicators_15m)
                        
                        # 🎯 MULTI-TIMEFRAME CONVERGENCE SCORING
                        indicators_multi_tf = {
                            '5m': indicators_5m,
                            '15m': indicators_15m,
                            '30m': indicators_30m,
                            '1h': indicators_1h
                        }
                        convergence_analysis = convergence_scorer.compare_directions(indicators_multi_tf)
                        
                        # 💧 ORDER FLOW & LIQUIDITY ANALYSIS
                        order_flow = order_flow_analyzer.analyze_order_flow(data['1h'])
                        
                        # 🚨 FIX #2: RECENT MOMENTUM ANALYSIS (last 3 candles on 5m)
                        from bot.indicators import analyze_recent_momentum
                        recent_momentum_5m = analyze_recent_momentum(data['5m'], lookback=3)
                        recent_momentum_15m = analyze_recent_momentum(data['15m'], lookback=3)
                        
                        # Log warnings if volume spike detected
                        if recent_momentum_5m['warning']:
                            logger.warning(f"⚡ {symbol} (5m): {recent_momentum_5m['warning']}")
                        if recent_momentum_15m['warning']:
                            logger.warning(f"⚡ {symbol} (15m): {recent_momentum_15m['warning']}")
                        
                        market_data_batch[symbol] = {
                            'indicators': indicators_multi_tf,
                            'current_price': get_current_price(exchange, symbol),
                            'breakout': breakout_15m,
                            'convergence': convergence_analysis,
                            'order_flow': order_flow,
                            'recent_momentum': {
                                '5m': recent_momentum_5m,
                                '15m': recent_momentum_15m
                            },
                            'raw_data': data  # Keep raw data for position monitoring
                        }
                        
                        # 🚀 UPDATE REALTIME BREAKOUT TRACKING
                        from bot.realtime_breakout_monitor import realtime_breakout_monitor
                        
                        # Update tracked range for this symbol (uses 15m high/low from last 20 candles)
                        recent_15m = data['15m'].tail(20)
                        tracked_high = recent_15m['high'].max()
                        tracked_low = recent_15m['low'].min()
                        realtime_breakout_monitor.update_tracking(symbol, tracked_high, tracked_low)
                        
                        # Check if current price represents a breakout
                        current_candle = data['15m'].iloc[-1]
                        breakout_signal = realtime_breakout_monitor.check_breakout(
                            symbol,
                            current_price=current_candle['close'],
                            current_volume=current_candle['volume'],
                            avg_volume=data['15m'].iloc[-20:-1]['volume'].mean(),
                            indicators=indicators_15m
                        )
                        
                        if breakout_signal:
                            logger.warning(
                                f"🚀 {symbol}: {breakout_signal['type'].upper().replace('_', ' ')} "
                                f"at ${breakout_signal['current_price']:.2f} "
                                f"(broke ${breakout_signal['breakout_level']:.2f}, "
                                f"volume {breakout_signal['volume_ratio']:.1f}x, "
                                f"strength: {breakout_signal['strength']})"
                            )
                        
                        # Log high-conviction signals
                        if convergence_analysis['has_clear_edge'] and convergence_analysis['best_direction'] != 'none':
                            logger.info(f"🎯 {symbol}: {convergence_analysis['recommendation']}")
                        
                        if breakout_15m['is_breakout']:
                            logger.warning(f"🔥 {symbol}: {breakout_15m['strength'].upper()} breakout detected!")
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            
            if not market_data_batch:
                logger.error("No market data available for auto-trading")
                continue
            
            # 🎯 REGIME FILTER: Skip symbols in ranging markets (ADX < 20)
            if settings.enable_regime_filter:
                regime_analysis = regime_filter.analyze_market_conditions(market_data_batch)
                logger.info(f"📊 Market Regimes: {regime_analysis['regimes']}, Avg ADX: {regime_analysis['avg_adx']}, Overall: {regime_analysis['overall_state']}")
                
                market_data_batch = regime_filter.filter_symbols(market_data_batch)
                
                if not market_data_batch:
                    logger.warning("⚠️ REGIME FILTER: All symbols are ranging (ADX < 20) - Skipping cycle to avoid chop")
                    # Still wait before next cycle
                    wait_interval = 900  # 15 min default
                    wait_minutes = wait_interval // 60
                    logger.info(f"⏰ Waiting {wait_minutes} minutes for market to trend...")
                    try:
                        await asyncio.wait_for(auto_trading_event.wait(), timeout=wait_interval)
                        auto_trading_event.clear()
                        logger.info("🔄 Auto-trading re-triggered early!")
                    except asyncio.TimeoutError:
                        logger.info(f"⏰ {wait_minutes} minutes passed, starting next cycle...")
                    continue
            
            # ⚡ ADAPTIVE VOLATILITY TIMING
            optimal_interval, volatility_regime = volatility_monitor.get_optimal_interval(market_data_batch)
            logger.info(f"⚡ Market volatility: {volatility_regime} → Next analysis in {optimal_interval//60} minutes")
            
            # Get current balance for decision-making
            balance_data = await get_balance()
            balance = balance_data['balance']
            logger.info(f"💰 Current balance: ${balance:.2f}")
            
            # Fetch sentiment data for all symbols
            logger.info("📊 Fetching sentiment data...")
            sentiment_data = get_batch_sentiment(list(market_data_batch.keys()))
            logger.info(f"✅ Sentiment data fetched for {len(sentiment_data)} symbols")
            
            # 🎯 FILTER SYMBOLS BY CONVERGENCE SCORE (only pass high-quality setups to AI)
            filtered_symbols = {}
            for symbol, data in market_data_batch.items():
                convergence = data['convergence']
                breakout = data['breakout']
                
                # Include if: HIGH convergence OR strong breakout
                if (convergence['has_clear_edge'] and 
                    convergence[convergence['best_direction']]['total_score'] >= 60) or \
                   (breakout['is_breakout'] and breakout['strength'] in ['strong', 'explosive']):
                    filtered_symbols[symbol] = data
                    logger.info(f"✅ {symbol} passed filters (convergence: {convergence[convergence['best_direction']]['total_score']:.1f}, breakout: {breakout['strength']})")
            
            if not filtered_symbols:
                logger.warning("⚠️ No symbols passed quality filters this cycle")
                # Still wait before next cycle (don't hammer the system!)
                wait_interval = optimal_interval if 'optimal_interval' in locals() else 900  # Default 15 min
                wait_minutes = wait_interval // 60
                logger.info(f"⏰ Waiting {wait_minutes} minutes until next analysis...")
                try:
                    await asyncio.wait_for(auto_trading_event.wait(), timeout=wait_interval)
                    auto_trading_event.clear()
                    logger.info("🔄 Auto-trading re-triggered early!")
                except asyncio.TimeoutError:
                    logger.info(f"⏰ {wait_minutes} minutes passed, starting next cycle...")
                continue
            
            # Extract just the indicators for each symbol (use FILTERED symbols only)
            symbols_indicators = {
                symbol: data['indicators']
                for symbol, data in filtered_symbols.items()
            }
            
            # Extract order flow data for each symbol
            order_flow_batch = {
                symbol: data['order_flow']
                for symbol, data in filtered_symbols.items()
            }
            
            # 🧠 FETCH HISTORICAL CONTEXT (Market Memory)
            logger.info("🧠 Fetching historical context from Market Memory...")
            historical_context_batch = {}
            
            for symbol, data in filtered_symbols.items():
                current_price = data['current_price']
                indicators = data['indicators']
                order_flow = data['order_flow']
                
                # Update market memory with current data (if initialized)
                if market_memory:
                    try:
                        market_memory.update_from_market_data(symbol, current_price, indicators, order_flow)
                        
                        # Get historical context for LLM
                        historical_context_batch[symbol] = market_memory.get_historical_context(symbol, current_price)
                    except Exception as e:
                        logger.warning(f"Market memory error for {symbol}: {e}")
                        historical_context_batch[symbol] = None
            
            # Filter out None values from historical context
            historical_context_batch = {k: v for k, v in historical_context_batch.items() if v is not None}
            logger.info(f"✅ Historical context fetched for {len(historical_context_batch)} symbols")
            
            # Get batch decisions from LLM
            try:
                logger.info("🧠 Calling DeepSeek AI for batch analysis...")
                response = llm_agent.get_batch_decisions(
                    symbols_data=symbols_indicators,
                    balance=balance,
                    risk_pct=settings.risk_per_trade,
                    sentiment_data=sentiment_data,
                    order_flow_data=order_flow_batch,
                    historical_context=historical_context_batch if historical_context_batch else None
                )
                
                # Extract decisions from response
                if not response:
                    logger.warning("No response returned from AI")
                    continue
                
                decisions = response.get('decisions', {})
                logger.info(f"✅ Auto-trading: Received decisions for {len(decisions)} symbols")
                
                if not decisions:
                    logger.warning("No decisions in response from AI")
                    continue
                
                # Log AI reasoning for persistence
                log_llm_reasoning(
                    symbol=list(market_data_batch.keys()),
                    decision=decisions,
                    full_response=response  # Full response with reasoning
                )
                logger.info("💾 AI reasoning logged to persistent storage")
                
                # 🧠 SAVE ANALYSIS SNAPSHOTS (Market Memory)
                for symbol, decision in decisions.items():
                    if symbol in filtered_symbols:
                        current_price = filtered_symbols[symbol]['current_price']
                        convergence_data = filtered_symbols[symbol].get('convergence_data', {})
                        
                        if market_memory:
                            try:
                                market_memory.save_analysis_snapshot(
                                    symbol=symbol,
                                    price=current_price,
                                    decision=decision,
                                    convergence_data=convergence_data
                                )
                            except Exception as e:
                                logger.warning(f"Failed to save analysis snapshot for {symbol}: {e}")
                
                logger.info(f"🧠 Analysis snapshots saved for {len(decisions)} symbols")
                
                # Execute high-confidence trades with ORDER FLOW QUALITY CHECK
                high_confidence_count = 0
                for symbol, decision in decisions.items():
                    action = decision.get('action')
                    confidence = decision.get('confidence', 0)
                    
                    if action in ['long', 'short'] and confidence > 0.7:
                        # 🚨 FIX #1: MOMENTUM CONFLICT CHECK (prevent trading against recent momentum)
                        if symbol in filtered_symbols:
                            indicators_multi_tf = filtered_symbols[symbol]['indicators']
                            momentum_check = convergence_scorer.check_momentum_conflict(
                                indicators_multi_tf, action
                            )
                            
                            if momentum_check['has_conflict']:
                                logger.warning(
                                    f"⚠️ {symbol}: SKIPPING {action.upper()} - "
                                    f"Momentum conflict: {', '.join(momentum_check['conflicts'])}"
                                )
                                logger.warning(
                                    f"   Lower TF momentum: {momentum_check['lower_tf_momentum']}"
                                )
                                continue
                        
                        # 💧 ORDER FLOW QUALITY CHECK
                        if symbol in filtered_symbols:
                            order_flow = filtered_symbols[symbol]['order_flow']
                            current_price = filtered_symbols[symbol]['current_price']
                            entry_quality = order_flow_analyzer.get_entry_quality(
                                current_price, action, order_flow
                            )
                            
                            logger.info(f"💧 {symbol}: Entry quality: {entry_quality['quality']} (score: {entry_quality['score']}/100)")
                            logger.info(f"   Reason: {entry_quality['reason']}")
                            
                            # Require at least "acceptable" entry quality
                            if entry_quality['score'] < 50:
                                logger.warning(f"⚠️ {symbol}: Skipping trade due to poor entry quality")
                                continue
                        
                        # 🔗 CORRELATION FILTER: Check if new position correlates with existing
                        if settings.enable_correlation_filter:
                            # Get current open positions
                            open_positions = db.get_open_positions()
                            
                            # Check correlation
                            allowed, reason = correlation_filter.check_new_signal(symbol, action, open_positions)
                            logger.info(f"🔗 {symbol}: {reason}")
                            
                            if not allowed:
                                logger.warning(f"⚠️ {symbol}: Skipping trade - correlation limit reached")
                                continue
                            
                            # Log portfolio risk analysis
                            if open_positions:
                                risk_analysis = correlation_filter.analyze_portfolio_risk(open_positions)
                                if risk_analysis['warning']:
                                    logger.warning(risk_analysis['warning'])
                        
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
        
        # ⚡ ADAPTIVE WAIT based on market volatility (interruptible)
        wait_interval = optimal_interval if 'optimal_interval' in locals() else 900  # Default to 15 min
        wait_minutes = wait_interval // 60
        logger.info(f"⏰ Auto-trading: Waiting {wait_minutes} minutes until next analysis (or until re-enabled)...")
        try:
            await asyncio.wait_for(auto_trading_event.wait(), timeout=wait_interval)
            auto_trading_event.clear()  # Reset event
            logger.info("🔄 Auto-trading re-triggered early!")
        except asyncio.TimeoutError:
            logger.info(f"⏰ {wait_minutes} minutes passed, starting next analysis cycle...")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    global auto_trading_event, market_memory
    
    # Initialize event for auto-trading signaling
    auto_trading_event = asyncio.Event()
    logger.info("Auto-trading event initialized")
    
    # Initialize system with error handling (non-blocking)
    try:
        initialize_system()
        logger.info("✅ Trading system initialized successfully")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize trading system: {e}")
        logger.warning("App will start but trading features may be unavailable")
    
    # Initialize Market Memory System (with error handling)
    if db:
        try:
            market_memory = initialize_market_memory(db)
            logger.info("🧠 Market Memory System initialized")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize Market Memory: {e}")
            market_memory = None
            logger.warning("Continuing without Market Memory (historical context disabled)")
    else:
        logger.warning("⚠️ Database not available, skipping Market Memory initialization")
        market_memory = None
    
    # Initialize auto_trading setting if not exists
    if db and db.get_setting('auto_trading') is None:
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
        if not db:
            return {"enabled": False, "auto_trading": False, "error": "Database not available"}
        
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

