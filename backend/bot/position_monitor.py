"""
Position Monitoring & Dynamic Risk Management Module
Implements trailing stops, partial exits, and adaptive risk management
"""
from typing import Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Monitors open positions and manages dynamic risk
    """
    
    def __init__(self):
        self.trailing_stops = {}  # symbol -> trailing stop price
        self.partial_exit_levels = {}  # symbol -> list of hit exit levels
    
    def update_trailing_stop(self, symbol: str, position: Dict, 
                            current_price: float, atr: float) -> Optional[float]:
        """
        Update trailing stop for a position
        
        Args:
            symbol: Trading pair
            position: Position dict with entry, side, stop_loss
            current_price: Current market price
            atr: Current ATR value
        
        Returns:
            New stop price if updated, None otherwise
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
            
            # Only move stop UP (never down)
            if new_stop > current_stop:
                logger.info(f"📈 {symbol} LONG: Trailing stop {current_stop:.2f} → {new_stop:.2f} (profit: {profit_pct:+.2f}%)")
                self.trailing_stops[symbol] = new_stop
                return new_stop
        
        elif side == 'short':
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            # Trail stop: current_price + 2*ATR (but never higher than current stop)
            new_stop = current_price + (atr * 2)
            
            # Only move stop DOWN (never up)
            if new_stop < current_stop:
                logger.info(f"📉 {symbol} SHORT: Trailing stop {current_stop:.2f} → {new_stop:.2f} (profit: {profit_pct:+.2f}%)")
                self.trailing_stops[symbol] = new_stop
                return new_stop
        
        return None
    
    def check_partial_exits(self, symbol: str, position: Dict, 
                           current_price: float) -> List[Dict]:
        """
        Check if position should take partial profits
        
        Args:
            symbol: Trading pair
            position: Position dict
            current_price: Current market price
        
        Returns:
            List of partial exit instructions
        """
        side = position.get('side')
        entry_price = position.get('entry_price')
        units = position.get('units', 0)
        
        if not all([side, entry_price]) or units <= 0:
            return []
        
        # Track which levels we've already hit
        if symbol not in self.partial_exit_levels:
            self.partial_exit_levels[symbol] = []
        
        hit_levels = self.partial_exit_levels[symbol]
        exits = []
        
        # Calculate profit %
        if side == 'long':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Partial exit levels
        # Level 1: 3% profit → close 30%
        if profit_pct >= 3 and '3pct' not in hit_levels:
            exits.append({
                'level': '3pct',
                'percentage': 30,
                'units': units * 0.3,
                'reason': f'Taking 30% profit at {profit_pct:.2f}%'
            })
            hit_levels.append('3pct')
            logger.info(f"💰 {symbol}: Partial exit triggered at {profit_pct:.2f}% (closing 30%)")
        
        # Level 2: 6% profit → close another 30% (60% total closed)
        if profit_pct >= 6 and '6pct' not in hit_levels:
            exits.append({
                'level': '6pct',
                'percentage': 30,
                'units': units * 0.3,
                'reason': f'Taking another 30% profit at {profit_pct:.2f}%'
            })
            hit_levels.append('6pct')
            logger.info(f"💰💰 {symbol}: Second partial exit at {profit_pct:.2f}% (closing another 30%)")
        
        # Level 3: 10% profit → close another 20% (80% total closed, let 20% run)
        if profit_pct >= 10 and '10pct' not in hit_levels:
            exits.append({
                'level': '10pct',
                'percentage': 20,
                'units': units * 0.2,
                'reason': f'Taking third partial at {profit_pct:.2f}%, letting 20% runner ride'
            })
            hit_levels.append('10pct')
            logger.info(f"🚀 {symbol}: Third partial exit at {profit_pct:.2f}% (20% runner remains)")
        
        return exits
    
    def check_stop_hit(self, symbol: str, position: Dict, 
                      current_price: float) -> bool:
        """
        Check if stop loss has been hit
        
        Args:
            symbol: Trading pair
            position: Position dict
            current_price: Current market price
        
        Returns:
            True if stop hit, False otherwise
        """
        side = position.get('side')
        stop_loss = position.get('stop_loss')
        
        # Use trailing stop if available
        if symbol in self.trailing_stops:
            stop_loss = self.trailing_stops[symbol]
        
        if not all([side, stop_loss]):
            return False
        
        # Check if stop hit
        if side == 'long' and current_price <= stop_loss:
            logger.warning(f"🛑 {symbol} LONG: Stop loss HIT at {current_price:.2f} (stop: {stop_loss:.2f})")
            return True
        
        elif side == 'short' and current_price >= stop_loss:
            logger.warning(f"🛑 {symbol} SHORT: Stop loss HIT at {current_price:.2f} (stop: {stop_loss:.2f})")
            return True
        
        return False
    
    def check_target_hit(self, symbol: str, position: Dict, 
                        current_price: float) -> bool:
        """
        Check if take profit target has been hit
        
        Args:
            symbol: Trading pair
            position: Position dict
            current_price: Current market price
        
        Returns:
            True if target hit, False otherwise
        """
        side = position.get('side')
        take_profit = position.get('take_profit')
        
        if not all([side, take_profit]):
            return False
        
        # Check if target hit
        if side == 'long' and current_price >= take_profit:
            logger.info(f"🎯 {symbol} LONG: Take profit HIT at {current_price:.2f} (target: {take_profit:.2f})")
            return True
        
        elif side == 'short' and current_price <= take_profit:
            logger.info(f"🎯 {symbol} SHORT: Take profit HIT at {current_price:.2f} (target: {take_profit:.2f})")
            return True
        
        return False
    
    def manage_position(self, symbol: str, position: Dict, 
                       current_price: float, atr: float) -> Dict:
        """
        Full position management logic
        
        Returns:
            Dict with management actions: {
                'action': 'hold'|'close'|'partial_exit'|'update_stop',
                'data': relevant action data
            }
        """
        # 1. Check if stop hit (highest priority)
        if self.check_stop_hit(symbol, position, current_price):
            return {
                'action': 'close',
                'reason': 'stop_loss_hit',
                'price': current_price
            }
        
        # 2. Check if target hit
        if self.check_target_hit(symbol, position, current_price):
            return {
                'action': 'close',
                'reason': 'take_profit_hit',
                'price': current_price
            }
        
        # 3. Check for partial exits
        partial_exits = self.check_partial_exits(symbol, position, current_price)
        if partial_exits:
            return {
                'action': 'partial_exit',
                'exits': partial_exits,
                'price': current_price
            }
        
        # 4. Update trailing stop
        new_stop = self.update_trailing_stop(symbol, position, current_price, atr)
        if new_stop:
            return {
                'action': 'update_stop',
                'new_stop': new_stop,
                'price': current_price
            }
        
        # 5. Hold position
        return {
            'action': 'hold',
            'price': current_price
        }
    
    def reset_position_tracking(self, symbol: str):
        """Reset tracking data when position is closed"""
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]
        if symbol in self.partial_exit_levels:
            del self.partial_exit_levels[symbol]
        logger.debug(f"Reset position tracking for {symbol}")


# Global instance
position_monitor = PositionMonitor()

