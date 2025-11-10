"""
Market Memory System
Tracks historical context for institutional-grade trading decisions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketMemory:
    """
    Maintains institutional memory of:
    - Historical support/resistance levels
    - Order block mitigation status
    - Market regime transitions
    - Trading performance per symbol
    - Previous analysis comparisons
    """
    
    def __init__(self, db):
        self.db = db
        self.cache = {}  # In-memory cache for fast lookups
    
    def update_from_market_data(self, symbol: str, current_price: float, 
                                indicators: Dict, order_flow: Dict):
        """
        Update historical memory from current market analysis
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            indicators: Technical indicators from all timeframes
            order_flow: Order flow analysis data
        """
        try:
            # 1. Track key levels from order flow
            self._track_key_levels(symbol, current_price, order_flow)
            
            # 2. Track order blocks
            self._track_order_blocks(symbol, current_price, order_flow)
            
            # 3. Check for level breaks
            self._check_level_breaks(symbol, current_price, order_flow)
            
            # 4. Check for order block mitigation
            self._check_order_block_mitigation(symbol, current_price)
            
            # 5. Track market regime
            self._track_regime(symbol, indicators)
            
        except Exception as e:
            logger.error(f"Error updating market memory for {symbol}: {e}")
    
    def _track_key_levels(self, symbol: str, current_price: float, order_flow: Dict):
        """Extract and store key support/resistance levels"""
        if not order_flow or 'key_levels' not in order_flow:
            return
        
        for level in order_flow['key_levels']:
            level_type = level.get('type')
            price = level.get('price')
            strength = level.get('strength', 'moderate')
            
            if price is None or level_type is None:
                continue
            
            # Determine timeframe based on distance from current price
            distance_pct = abs(price - current_price) / current_price * 100
            
            if distance_pct < 2:
                timeframe = 'intraday'
            elif distance_pct < 5:
                timeframe = 'daily'
            elif distance_pct < 10:
                timeframe = 'weekly'
            else:
                timeframe = 'monthly'
            
            self.db.add_historical_level(
                symbol=symbol,
                level_type=level_type,
                price=price,
                strength=strength,
                timeframe=timeframe,
                metadata={'distance_pct': distance_pct}
            )
    
    def _track_order_blocks(self, symbol: str, current_price: float, order_flow: Dict):
        """Store detected order blocks in database"""
        if not order_flow or 'order_blocks' not in order_flow:
            return
        
        for block in order_flow['order_blocks']:
            block_type = block.get('category', '').replace('_order_block', '')
            
            if 'bullish' in block_type or 'bearish' in block_type:
                price_low = block.get('price_low')
                price_high = block.get('price_high')
                strength_score = block.get('strength_score', 50)
                volume_ratio = block.get('volume_ratio', 1.0)
                
                if price_low and price_high:
                    self.db.add_order_block(
                        symbol=symbol,
                        block_type=block_type,
                        price_low=price_low,
                        price_high=price_high,
                        strength_score=strength_score,
                        volume_ratio=volume_ratio
                    )
    
    def _check_level_breaks(self, symbol: str, current_price: float, order_flow: Dict):
        """Check if price has broken through any historical levels"""
        # Get nearby support levels
        supports = self.db.get_historical_levels(symbol, current_price, max_distance_pct=0.02)
        
        for level_data in supports:
            level_price = level_data['price']
            level_type = level_data['level_type']
            
            # Check if price crossed the level
            if level_type == 'support' and current_price < level_price * 0.995:
                # Support broken downward
                self.db.mark_level_broken(symbol, level_price, 'support')
                logger.info(f"📉 {symbol}: Support broken at ${level_price:.2f}")
            
            elif level_type == 'resistance' and current_price > level_price * 1.005:
                # Resistance broken upward
                self.db.mark_level_broken(symbol, level_price, 'resistance')
                logger.info(f"📈 {symbol}: Resistance broken at ${level_price:.2f}")
    
    def _check_order_block_mitigation(self, symbol: str, current_price: float):
        """Check if price has returned to fill any order blocks"""
        valid_blocks = self.db.get_valid_order_blocks(symbol, current_price, max_age_days=30)
        
        for block in valid_blocks:
            price_low = block['price_low']
            price_high = block['price_high']
            block_type = block['block_type']
            block_id = block['id']
            
            # Check if price is inside the order block
            if price_low <= current_price <= price_high:
                self.db.mark_order_block_mitigated(block_id)
                logger.info(
                    f"💰 {symbol}: {block_type} order block mitigated "
                    f"(${price_low:.2f}-${price_high:.2f})"
                )
    
    def _track_regime(self, symbol: str, indicators: Dict):
        """Track market regime changes (trending/ranging)"""
        if '1h' not in indicators:
            return
        
        ind_1h = indicators['1h']
        adx = ind_1h.get('adx')
        
        if adx is None:
            return
        
        # Determine regime
        if adx > 25:
            regime = 'trending'
        elif adx < 20:
            regime = 'ranging'
        else:
            regime = 'transitioning'
        
        # Determine trend direction
        ema_20_above_50 = ind_1h.get('ema_20_above_50')
        if ema_20_above_50 is True:
            trend_direction = 'up'
        elif ema_20_above_50 is False:
            trend_direction = 'down'
        else:
            trend_direction = 'neutral'
        
        # Determine volatility (from indicators if available)
        volatility = 'normal'  # Default
        
        self.db.add_regime_snapshot(
            symbol=symbol,
            regime=regime,
            adx=adx,
            trend_direction=trend_direction,
            volatility=volatility
        )
    
    def save_analysis_snapshot(self, symbol: str, price: float, 
                               decision: Dict, convergence_data: Dict = None):
        """
        Save current analysis for future comparison
        
        Args:
            symbol: Trading symbol
            price: Current price
            decision: LLM decision dict (action, confidence, reason)
            convergence_data: Multi-timeframe convergence data
        """
        action = decision.get('action', 'none')
        confidence = decision.get('confidence')
        key_reason = decision.get('reason', '')
        
        convergence_score = None
        trend = None
        
        if convergence_data:
            convergence_score = convergence_data.get('convergence_score')
            
            # Determine trend from convergence data
            bullish_score = convergence_data.get('bullish_score', 0)
            bearish_score = convergence_data.get('bearish_score', 0)
            
            if bullish_score > bearish_score + 20:
                trend = 'uptrend'
            elif bearish_score > bullish_score + 20:
                trend = 'downtrend'
            else:
                trend = 'neutral'
        
        # Create indicators snapshot (key indicators only)
        indicators_snapshot = {}
        if convergence_data and 'indicators' in convergence_data:
            indicators_snapshot = {
                'rsi_1h': convergence_data['indicators'].get('1h', {}).get('rsi'),
                'macd_1h': convergence_data['indicators'].get('1h', {}).get('macd_interpretation'),
                'volume_change': convergence_data['indicators'].get('1h', {}).get('volume_change')
            }
        
        self.db.add_analysis_snapshot(
            symbol=symbol,
            price=price,
            action=action,
            confidence=confidence,
            convergence_score=convergence_score,
            trend=trend,
            key_reason=key_reason[:500] if key_reason else None,  # Truncate long reasons
            indicators_snapshot=indicators_snapshot
        )
    
    def get_historical_context(self, symbol: str, current_price: float) -> Dict:
        """
        Get comprehensive historical context for a symbol
        
        Returns:
            Dict with historical levels, order blocks, regime, performance, previous analysis
        """
        # Get historical levels near current price (within 5%)
        historical_levels = self.db.get_historical_levels(symbol, current_price, max_distance_pct=0.05)
        
        # Get valid order blocks
        order_blocks = self.db.get_valid_order_blocks(symbol, current_price, max_age_days=30)
        
        # Get regime history (last 48 hours)
        regime_history = self.db.get_regime_history(symbol, hours=48)
        
        # Get recent performance (last 7 days)
        performance = self.db.get_recent_performance(symbol, days=7)
        
        # Get recent analysis (last 24 hours)
        recent_analysis = self.db.get_recent_analysis(symbol, hours=24)
        
        # Analyze regime stability
        regime_summary = self._summarize_regime_history(regime_history)
        
        # Analyze performance trend
        performance_summary = self._summarize_performance(performance)
        
        # Compare with previous analysis
        analysis_diff = self._compare_recent_analysis(recent_analysis, current_price)
        
        return {
            'historical_levels': historical_levels,
            'order_blocks': order_blocks,
            'regime_history': regime_history,
            'regime_summary': regime_summary,
            'performance': performance,
            'performance_summary': performance_summary,
            'recent_analysis': recent_analysis,
            'analysis_diff': analysis_diff
        }
    
    def _summarize_regime_history(self, regime_history: List[Dict]) -> str:
        """Create human-readable regime summary"""
        if not regime_history:
            return "No regime history available"
        
        # Count regime types
        trending_count = sum(1 for r in regime_history if r['regime'] == 'trending')
        ranging_count = sum(1 for r in regime_history if r['regime'] == 'ranging')
        
        total = len(regime_history)
        
        if trending_count > ranging_count * 2:
            return f"Consistently TRENDING ({trending_count}/{total} snapshots, avg ADX: {np.mean([r['adx'] for r in regime_history if r['adx']]):.1f})"
        elif ranging_count > trending_count * 2:
            return f"Consistently RANGING ({ranging_count}/{total} snapshots, avg ADX: {np.mean([r['adx'] for r in regime_history if r['adx']]):.1f})"
        else:
            return f"MIXED regime (trending: {trending_count}, ranging: {ranging_count})"
    
    def _summarize_performance(self, performance: Dict) -> str:
        """Create human-readable performance summary"""
        total = performance['total_trades']
        
        if total == 0:
            return "No recent trades"
        
        win_rate = performance['win_rate'] * 100
        avg_pnl = performance['avg_pnl']
        total_pnl = performance['total_pnl']
        
        if win_rate >= 60 and total_pnl > 0:
            return f"✅ Strong performance: {win_rate:.0f}% win rate, ${total_pnl:+.2f} profit ({total} trades)"
        elif win_rate >= 40 and total_pnl > 0:
            return f"📊 Moderate performance: {win_rate:.0f}% win rate, ${total_pnl:+.2f} profit ({total} trades)"
        else:
            return f"⚠️ Poor performance: {win_rate:.0f}% win rate, ${total_pnl:+.2f} loss ({total} trades)"
    
    def _compare_recent_analysis(self, recent_analysis: List[Dict], current_price: float) -> str:
        """Compare current analysis with recent history"""
        if not recent_analysis or len(recent_analysis) < 2:
            return "No previous analysis for comparison"
        
        # Get most recent analysis (not current one)
        prev = recent_analysis[0]
        
        prev_action = prev['action']
        prev_price = prev['price']
        prev_trend = prev['trend']
        
        # Calculate price change since last analysis
        price_change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Check for trend reversal
        # Count recent actions
        recent_actions = [a['action'] for a in recent_analysis[:5]]
        long_count = recent_actions.count('long')
        short_count = recent_actions.count('short')
        
        if long_count > 3:
            recent_bias = "bullish"
        elif short_count > 3:
            recent_bias = "bearish"
        else:
            recent_bias = "neutral"
        
        time_diff = (datetime.utcnow() - datetime.fromisoformat(prev['timestamp'].replace('Z', '+00:00'))).seconds // 3600
        
        summary = f"Last analysis {time_diff}h ago: {prev_action.upper()} at ${prev_price:.2f} ({prev_trend or 'unknown trend'}). "
        summary += f"Price moved {price_change_pct:+.2f}% since then. "
        summary += f"Recent bias: {recent_bias} ({long_count} LONG, {short_count} SHORT signals in last 5 analyses)."
        
        return summary
    
    def get_performance_feedback(self, symbol: str) -> Dict:
        """
        Get performance feedback for LLM context
        
        Returns:
            Dict with performance metrics and recommendations
        """
        performance = self.db.get_recent_performance(symbol, days=7)
        
        # Generate feedback
        feedback = {
            'total_trades': performance['total_trades'],
            'win_rate': performance['win_rate'],
            'avg_pnl': performance['avg_pnl'],
            'total_pnl': performance['total_pnl'],
            'recommendation': ''
        }
        
        if performance['total_trades'] == 0:
            feedback['recommendation'] = "No trade history for this symbol"
        elif performance['win_rate'] < 0.4 and performance['total_trades'] >= 3:
            feedback['recommendation'] = "⚠️ CAUTION: Poor recent performance on this symbol (win rate < 40%)"
        elif performance['total_pnl'] < -100 and performance['total_trades'] >= 3:
            feedback['recommendation'] = f"⚠️ WARNING: Significant losses on this symbol (${performance['total_pnl']:.2f})"
        elif performance['win_rate'] >= 0.6 and performance['total_pnl'] > 50:
            feedback['recommendation'] = "✅ Strong recent performance - good track record"
        else:
            feedback['recommendation'] = "📊 Neutral performance - no strong bias"
        
        return feedback


# Global instance
market_memory = None  # Will be initialized in main.py with db instance


def initialize_market_memory(db):
    """Initialize global market memory instance"""
    global market_memory
    market_memory = MarketMemory(db)
    logger.info("Market Memory System initialized")
    return market_memory

