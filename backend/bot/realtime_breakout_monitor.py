"""
Real-Time Breakout Monitor
Continuously monitors for breakouts and triggers immediate analysis
🚨 CRITICAL: Catches breakouts within 30-60 seconds, not 5-15 minutes
"""
import asyncio
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class RealtimeBreakoutMonitor:
    """
    Monitors market for breakouts in real-time and triggers immediate analysis
    """
    
    def __init__(self):
        self.tracking = {}  # symbol -> {high, low, last_check_time}
        self.breakout_cooldown = {}  # symbol -> last_breakout_time (prevent spam)
        self.check_interval = 30  # Check every 30 seconds
        
    def update_tracking(self, symbol: str, high: float, low: float):
        """Update tracked range for a symbol"""
        if symbol not in self.tracking:
            self.tracking[symbol] = {
                'high': high,
                'low': low,
                'last_update': asyncio.get_event_loop().time()
            }
        else:
            # Expand range if needed
            self.tracking[symbol]['high'] = max(self.tracking[symbol]['high'], high)
            self.tracking[symbol]['low'] = min(self.tracking[symbol]['low'], low)
            self.tracking[symbol]['last_update'] = asyncio.get_event_loop().time()
    
    def reset_tracking(self, symbol: str):
        """Reset tracking for a symbol (after analysis)"""
        if symbol in self.tracking:
            del self.tracking[symbol]
    
    def check_breakout(self, symbol: str, current_price: float, 
                      current_volume: float, avg_volume: float,
                      indicators: Dict) -> Optional[Dict]:
        """
        Check if current price represents a breakout
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            current_volume: Current volume
            avg_volume: Average volume (recent)
            indicators: Current indicators
        
        Returns:
            Breakout info if detected, None otherwise
        """
        if symbol not in self.tracking:
            return None
        
        tracked = self.tracking[symbol]
        
        # Check cooldown (don't spam breakout signals)
        current_time = asyncio.get_event_loop().time()
        if symbol in self.breakout_cooldown:
            time_since_last = current_time - self.breakout_cooldown[symbol]
            if time_since_last < 300:  # 5 minute cooldown
                return None
        
        # Calculate volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # BULLISH BREAKOUT: Price breaks above tracked high with volume
        if current_price > tracked['high'] * 1.001:  # 0.1% above high
            # Require strong volume confirmation
            if volume_ratio > 1.5:  # 1.5x average volume
                # Check momentum confirmation
                macd_interp = indicators.get('macd_interpretation', '').lower()
                rsi = indicators.get('rsi', 50)
                
                # Additional confirmation
                is_strong_breakout = (
                    volume_ratio > 2.0 or  # Very high volume
                    (volume_ratio > 1.5 and rsi > 50 and 'bullish' in macd_interp)
                )
                
                if is_strong_breakout:
                    logger.warning(
                        f"🚀 {symbol}: BULLISH BREAKOUT DETECTED! "
                        f"Price ${current_price:.2f} broke ${tracked['high']:.2f} "
                        f"with {volume_ratio:.1f}x volume"
                    )
                    
                    self.breakout_cooldown[symbol] = current_time
                    
                    return {
                        'type': 'bullish_breakout',
                        'symbol': symbol,
                        'current_price': current_price,
                        'breakout_level': tracked['high'],
                        'volume_ratio': volume_ratio,
                        'strength': 'strong' if volume_ratio > 2.5 else 'moderate',
                        'trigger_immediate_analysis': True
                    }
        
        # BEARISH BREAKDOWN: Price breaks below tracked low with volume
        elif current_price < tracked['low'] * 0.999:  # 0.1% below low
            if volume_ratio > 1.5:
                macd_interp = indicators.get('macd_interpretation', '').lower()
                rsi = indicators.get('rsi', 50)
                
                is_strong_breakdown = (
                    volume_ratio > 2.0 or
                    (volume_ratio > 1.5 and rsi < 50 and 'bearish' in macd_interp)
                )
                
                if is_strong_breakdown:
                    logger.warning(
                        f"🔻 {symbol}: BEARISH BREAKDOWN DETECTED! "
                        f"Price ${current_price:.2f} broke ${tracked['low']:.2f} "
                        f"with {volume_ratio:.1f}x volume"
                    )
                    
                    self.breakout_cooldown[symbol] = current_time
                    
                    return {
                        'type': 'bearish_breakdown',
                        'symbol': symbol,
                        'current_price': current_price,
                        'breakout_level': tracked['low'],
                        'volume_ratio': volume_ratio,
                        'strength': 'strong' if volume_ratio > 2.5 else 'moderate',
                        'trigger_immediate_analysis': True
                    }
        
        return None
    
    def check_position_breakout(self, position: Dict, current_price: float,
                                current_volume: float, avg_volume: float) -> Optional[Dict]:
        """
        Check if price is breaking out AGAINST an open position
        🚨 CRITICAL: This catches reversals like SOL SHORT -> should LONG
        
        Args:
            position: Open position data
            current_price: Current price
            current_volume: Current candle volume
            avg_volume: Average recent volume
        
        Returns:
            Reversal signal if detected, None otherwise
        """
        symbol = position['symbol']
        side = position['side']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # SHORT POSITION: Price breaking UP past stop loss
        if side == 'short':
            # Price above SL with strong volume = strong bullish breakout
            if current_price > stop_loss:
                price_above_sl_pct = ((current_price - stop_loss) / stop_loss) * 100
                
                # If price is significantly above SL with volume = consider reversal
                if price_above_sl_pct > 0.2 and volume_ratio > 1.5:
                    logger.warning(
                        f"🔄 {symbol}: REVERSAL OPPORTUNITY! "
                        f"SHORT position being broken out against "
                        f"(price ${current_price:.2f} vs SL ${stop_loss:.2f}, "
                        f"volume {volume_ratio:.1f}x)"
                    )
                    
                    return {
                        'type': 'reversal_long',
                        'symbol': symbol,
                        'reason': 'Short position broken out against with strong volume',
                        'current_price': current_price,
                        'volume_ratio': volume_ratio,
                        'recommendation': 'Consider LONG entry after closing SHORT'
                    }
        
        # LONG POSITION: Price breaking DOWN past stop loss
        elif side == 'long':
            if current_price < stop_loss:
                price_below_sl_pct = ((stop_loss - current_price) / stop_loss) * 100
                
                if price_below_sl_pct > 0.2 and volume_ratio > 1.5:
                    logger.warning(
                        f"🔄 {symbol}: REVERSAL OPPORTUNITY! "
                        f"LONG position being broken out against "
                        f"(price ${current_price:.2f} vs SL ${stop_loss:.2f}, "
                        f"volume {volume_ratio:.1f}x)"
                    )
                    
                    return {
                        'type': 'reversal_short',
                        'symbol': symbol,
                        'reason': 'Long position broken out against with strong volume',
                        'current_price': current_price,
                        'volume_ratio': volume_ratio,
                        'recommendation': 'Consider SHORT entry after closing LONG'
                    }
        
        return None


# Global instance
realtime_breakout_monitor = RealtimeBreakoutMonitor()

