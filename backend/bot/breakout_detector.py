"""
Breakout Momentum Confirmation Module
Detects and scores explosive breakout moves
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BreakoutDetector:
    """
    Detects breakout conditions and scores momentum strength
    """
    
    def __init__(self):
        self.lookback_period = 20  # Candles to look back for range
    
    def calculate_breakout_score(self, df: pd.DataFrame, 
                                 indicators: Dict) -> Dict:
        """
        Calculate comprehensive breakout score
        
        Args:
            df: OHLCV DataFrame
            indicators: Calculated indicators
        
        Returns:
            Dict with breakout analysis
        """
        if len(df) < self.lookback_period:
            return {
                'breakout_score': 0,
                'is_breakout': False,
                'direction': 'none',
                'strength': 'insufficient_data'
            }
        
        recent = df.tail(self.lookback_period)
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # 1. VOLUME RATIO (40% weight)
        volume_ratio = self._calculate_volume_ratio(recent, current)
        volume_score = min(volume_ratio / 2.5, 1.0) * 0.4  # Max 0.4
        
        # 2. RANGE EXPANSION (30% weight)
        range_expansion = self._calculate_range_expansion(recent, current)
        range_score = min(range_expansion / 2.0, 1.0) * 0.3  # Max 0.3
        
        # 3. CLOSE STRENGTH (30% weight)
        close_strength = self._calculate_close_strength(current)
        close_score = close_strength * 0.3  # Max 0.3
        
        # Total breakout score (0-1 scale)
        breakout_score = volume_score + range_score + close_score
        
        # Direction detection
        direction = self._detect_breakout_direction(df, current, indicators)
        
        # Key level check
        above_key_level = self._check_key_level_break(df, current, direction)
        
        # Determine if this is a valid breakout
        is_breakout = (
            breakout_score > 0.8 and 
            above_key_level and
            direction != 'none'
        )
        
        # Strength classification
        if breakout_score > 0.9:
            strength = 'explosive'
        elif breakout_score > 0.8:
            strength = 'strong'
        elif breakout_score > 0.6:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        result = {
            'breakout_score': round(breakout_score, 2),
            'is_breakout': is_breakout,
            'direction': direction,
            'strength': strength,
            'volume_ratio': round(volume_ratio, 2),
            'range_expansion': round(range_expansion, 2),
            'close_strength': round(close_strength, 2),
            'above_key_level': above_key_level
        }
        
        if is_breakout:
            logger.warning(
                f"🔥 BREAKOUT DETECTED! Direction: {direction.upper()}, "
                f"Score: {breakout_score:.2f}, Strength: {strength}, "
                f"Volume: {volume_ratio:.1f}x, Range: {range_expansion:.1f}x"
            )
        
        return result
    
    def _calculate_volume_ratio(self, recent: pd.DataFrame, 
                                current: pd.Series) -> float:
        """
        Calculate current volume vs average volume
        
        Returns:
            Ratio (e.g., 2.5 = 2.5x average volume)
        """
        avg_volume = recent['volume'].iloc[:-1].mean()  # Exclude current
        current_volume = current['volume']
        
        if avg_volume == 0:
            return 1.0
        
        return current_volume / avg_volume
    
    def _calculate_range_expansion(self, recent: pd.DataFrame, 
                                   current: pd.Series) -> float:
        """
        Calculate current candle range vs average range
        
        Returns:
            Ratio (e.g., 1.8 = 1.8x average range)
        """
        # Average range of recent candles
        recent_ranges = recent['high'] - recent['low']
        avg_range = recent_ranges.iloc[:-1].mean()  # Exclude current
        
        # Current candle range
        current_range = current['high'] - current['low']
        
        if avg_range == 0:
            return 1.0
        
        return current_range / avg_range
    
    def _calculate_close_strength(self, current: pd.Series) -> float:
        """
        Calculate where close is within the candle range
        
        Returns:
            Score 0-1 (1 = closed at high for bullish, at low for bearish)
        """
        open_price = current['open']
        close_price = current['close']
        high_price = current['high']
        low_price = current['low']
        
        range_size = high_price - low_price
        
        if range_size == 0:
            return 0.5  # Doji
        
        # For bullish candles, close near high is strong
        if close_price >= open_price:
            # How close to high? (1.0 = exactly at high)
            close_position = (close_price - low_price) / range_size
            return close_position
        
        # For bearish candles, close near low is strong
        else:
            # How close to low? (1.0 = exactly at low)
            close_position = (high_price - close_price) / range_size
            return close_position
    
    def _detect_breakout_direction(self, df: pd.DataFrame, 
                                   current: pd.Series, 
                                   indicators: Dict) -> str:
        """
        Detect breakout direction based on price action and indicators
        
        Returns:
            'bullish', 'bearish', or 'none'
        """
        # Recent highs and lows
        recent = df.tail(self.lookback_period)
        resistance = recent['high'].iloc[:-1].max()  # Exclude current
        support = recent['low'].iloc[:-1].min()  # Exclude current
        
        current_close = current['close']
        current_high = current['high']
        current_low = current['low']
        
        # Trend alignment from indicators
        trend_interp = indicators.get('trend_interpretation', '')
        macd_interp = indicators.get('macd_interpretation', '')
        
        # Bullish breakout conditions
        if (current_high > resistance and 
            current_close > resistance and
            'uptrend' in trend_interp.lower()):
            return 'bullish'
        
        # Bearish breakout conditions
        elif (current_low < support and 
              current_close < support and
              'downtrend' in trend_interp.lower()):
            return 'bearish'
        
        return 'none'
    
    def _check_key_level_break(self, df: pd.DataFrame, 
                               current: pd.Series, 
                               direction: str) -> bool:
        """
        Check if breakout is above/below a significant key level
        
        Returns:
            True if breaking a key level
        """
        if direction == 'none':
            return False
        
        recent = df.tail(self.lookback_period * 2)  # Look back further for key levels
        
        if direction == 'bullish':
            # Check if we're above recent resistance
            resistance_zone = recent['high'].quantile(0.90)  # 90th percentile high
            return current['close'] > resistance_zone
        
        elif direction == 'bearish':
            # Check if we're below recent support
            support_zone = recent['low'].quantile(0.10)  # 10th percentile low
            return current['close'] < support_zone
        
        return False
    
    def detect_consolidation_breakout(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect breakout from consolidation/range
        
        Returns:
            Breakout info if detected, None otherwise
        """
        if len(df) < self.lookback_period + 5:
            return None
        
        # Look for consolidation period (tight range)
        consolidation_period = df.iloc[-(self.lookback_period + 5):-5]
        high_range = consolidation_period['high'].max() - consolidation_period['low'].min()
        avg_price = consolidation_period['close'].mean()
        
        if avg_price == 0:
            return None
        
        range_pct = (high_range / avg_price) * 100
        
        # Is it consolidating? (range < 5%)
        if range_pct < 5:
            # Check if recent candles broke out
            recent = df.tail(5)
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            
            consolidation_high = consolidation_period['high'].max()
            consolidation_low = consolidation_period['low'].min()
            
            # Bullish breakout from consolidation
            if recent_high > consolidation_high:
                return {
                    'type': 'consolidation_breakout',
                    'direction': 'bullish',
                    'range_pct': round(range_pct, 2),
                    'breakout_level': consolidation_high
                }
            
            # Bearish breakdown from consolidation
            elif recent_low < consolidation_low:
                return {
                    'type': 'consolidation_breakout',
                    'direction': 'bearish',
                    'range_pct': round(range_pct, 2),
                    'breakout_level': consolidation_low
                }
        
        return None


# Global instance
breakout_detector = BreakoutDetector()

