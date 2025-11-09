"""
ENHANCED Position Monitoring with REAL-TIME Stop Loss Enforcement
🚨 CRITICAL: This ensures positions are closed IMMEDIATELY when SL/TP hit
"""
from typing import Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EnhancedPositionMonitor:
    """
    Enhanced position monitor with stricter SL enforcement
    """
    
    def __init__(self):
        self.trailing_stops = {}  # symbol -> trailing stop price
        self.partial_exit_levels = {}  # symbol -> list of hit exit levels
        self.last_check_prices = {}  # symbol -> last price checked
        self.sl_warnings_sent = {}  # symbol -> count of warnings near SL
    
    def check_position_health(self, symbol: str, position: Dict, 
                             current_price: float) -> Dict:
        """
        🚨 CRITICAL: Check if position is in danger or should be closed
        
        Returns:
            Dict with:
                - status: 'safe', 'danger', 'stop_hit', 'target_hit'
                - action: 'hold', 'close', 'warn'
                - reason: explanation
        """
        side = position.get('side')
        entry_price = position.get('entry_price')
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit')
        
        # Use trailing stop if available
        if symbol in self.trailing_stops:
            stop_loss = self.trailing_stops[symbol]
        
        # Calculate current P&L
        if side == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            distance_to_sl_pct = ((current_price - stop_loss) / entry_price) * 100 if stop_loss else 999
        else:  # short
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            distance_to_sl_pct = ((stop_loss - current_price) / entry_price) * 100 if stop_loss else 999
        
        # 🛑 CHECK 1: STOP LOSS HIT (CRITICAL)
        if side == 'long' and current_price <= stop_loss:
            logger.error(
                f"🚨 {symbol} LONG STOP LOSS HIT! "
                f"Price: ${current_price:.2f} <= SL: ${stop_loss:.2f} "
                f"P&L: {pnl_pct:.2f}%"
            )
            return {
                'status': 'stop_hit',
                'action': 'close',
                'reason': f'Stop loss hit (price: ${current_price:.2f}, SL: ${stop_loss:.2f}, P&L: {pnl_pct:.2f}%)',
                'price': current_price,
                'pnl_pct': pnl_pct
            }
        
        if side == 'short' and current_price >= stop_loss:
            logger.error(
                f"🚨 {symbol} SHORT STOP LOSS HIT! "
                f"Price: ${current_price:.2f} >= SL: ${stop_loss:.2f} "
                f"P&L: {pnl_pct:.2f}%"
            )
            return {
                'status': 'stop_hit',
                'action': 'close',
                'reason': f'Stop loss hit (price: ${current_price:.2f}, SL: ${stop_loss:.2f}, P&L: {pnl_pct:.2f}%)',
                'price': current_price,
                'pnl_pct': pnl_pct
            }
        
        # 🎯 CHECK 2: TAKE PROFIT HIT
        if take_profit:
            if (side == 'long' and current_price >= take_profit) or \
               (side == 'short' and current_price <= take_profit):
                logger.info(
                    f"🎯 {symbol} TAKE PROFIT HIT! "
                    f"Price: ${current_price:.2f}, TP: ${take_profit:.2f}, "
                    f"P&L: {pnl_pct:.2f}%"
                )
                return {
                    'status': 'target_hit',
                    'action': 'close',
                    'reason': f'Take profit hit (price: ${current_price:.2f}, TP: ${take_profit:.2f}, P&L: {pnl_pct:.2f}%)',
                    'price': current_price,
                    'pnl_pct': pnl_pct
                }
        
        # ⚠️ CHECK 3: DANGER ZONE (within 0.5% of stop loss)
        if distance_to_sl_pct < 0.5 and distance_to_sl_pct > 0:
            # Increment warning counter
            self.sl_warnings_sent[symbol] = self.sl_warnings_sent.get(symbol, 0) + 1
            
            if self.sl_warnings_sent[symbol] % 3 == 1:  # Log every 3rd check to avoid spam
                logger.warning(
                    f"⚠️ {symbol} IN DANGER ZONE! "
                    f"Only {distance_to_sl_pct:.2f}% from stop loss. "
                    f"Current: ${current_price:.2f}, SL: ${stop_loss:.2f}"
                )
            
            return {
                'status': 'danger',
                'action': 'warn',
                'reason': f'Near stop loss ({distance_to_sl_pct:.2f}% away)',
                'price': current_price,
                'pnl_pct': pnl_pct,
                'distance_to_sl_pct': distance_to_sl_pct
            }
        
        # ✅ CHECK 4: POSITION IS SAFE
        # Store last check price for trend analysis
        self.last_check_prices[symbol] = current_price
        
        return {
            'status': 'safe',
            'action': 'hold',
            'reason': f'Position healthy (P&L: {pnl_pct:.2f}%, SL distance: {distance_to_sl_pct:.2f}%)',
            'price': current_price,
            'pnl_pct': pnl_pct,
            'distance_to_sl_pct': distance_to_sl_pct
        }
    
    def update_trailing_stop(self, symbol: str, position: Dict, 
                            current_price: float, atr: float) -> Optional[float]:
        """
        Update trailing stop for a position (only move in profitable direction)
        """
        side = position.get('side')
        entry_price = position.get('entry_price')
        current_stop = position.get('stop_loss')
        
        if not all([side, entry_price, current_stop, atr]):
            return None
        
        # Calculate profit %
        if side == 'long':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            # Trail stop: current_price - 2*ATR (but never lower than current stop)
            new_stop = current_price - (atr * 2)
            
            # Only move stop UP (never down) and only if in profit
            if profit_pct > 2 and new_stop > current_stop:
                logger.info(
                    f"📈 {symbol} LONG: Trailing stop {current_stop:.2f} → {new_stop:.2f} "
                    f"(profit: {profit_pct:+.2f}%)"
                )
                self.trailing_stops[symbol] = new_stop
                return new_stop
        
        elif side == 'short':
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            # Trail stop: current_price + 2*ATR (but never higher than current stop)
            new_stop = current_price + (atr * 2)
            
            # Only move stop DOWN (never up) and only if in profit
            if profit_pct > 2 and new_stop < current_stop:
                logger.info(
                    f"📉 {symbol} SHORT: Trailing stop {current_stop:.2f} → {new_stop:.2f} "
                    f"(profit: {profit_pct:+.2f}%)"
                )
                self.trailing_stops[symbol] = new_stop
                return new_stop
        
        return None
    
    def reset_position_tracking(self, symbol: str):
        """Reset all tracking for a closed position"""
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]
        if symbol in self.partial_exit_levels:
            del self.partial_exit_levels[symbol]
        if symbol in self.last_check_prices:
            del self.last_check_prices[symbol]
        if symbol in self.sl_warnings_sent:
            del self.sl_warnings_sent[symbol]
        
        logger.info(f"🔄 {symbol}: Position tracking reset")


# Global instance
enhanced_position_monitor = EnhancedPositionMonitor()

