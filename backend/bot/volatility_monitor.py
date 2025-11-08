"""
Adaptive Volatility Timing Module
Dynamically adjusts analysis frequency based on market conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class VolatilityMonitor:
    """
    Monitors market volatility and determines optimal analysis frequency
    """
    
    def __init__(self):
        self.normal_interval = 900  # 15 minutes
        self.high_volatility_interval = 300  # 5 minutes
        self.low_volatility_interval = 1800  # 30 minutes
        
        # Historical ATR tracking for spike detection
        self.atr_history = {}
    
    def update_atr_history(self, symbol: str, atr: float):
        """Track ATR history for each symbol"""
        if symbol not in self.atr_history:
            self.atr_history[symbol] = []
        
        self.atr_history[symbol].append(atr)
        
        # Keep only last 20 readings (5 hours at 15-min intervals)
        if len(self.atr_history[symbol]) > 20:
            self.atr_history[symbol] = self.atr_history[symbol][-20:]
    
    def detect_volatility_regime(self, symbol: str, current_atr: float, 
                                 indicators: Dict) -> str:
        """
        Detect current volatility regime for a symbol
        
        Returns:
            'high', 'normal', or 'low'
        """
        # Update history
        self.update_atr_history(symbol, current_atr)
        
        if len(self.atr_history[symbol]) < 5:
            return 'normal'  # Not enough data
        
        # Calculate average ATR over lookback period
        avg_atr = np.mean(self.atr_history[symbol])
        
        # ATR spike detection (current vs average)
        if avg_atr > 0:
            atr_ratio = (current_atr / avg_atr) - 1
        else:
            atr_ratio = 0
        
        # Volume confirmation
        volume_change = indicators.get('volume_change', 0)
        
        # Range expansion check
        last_close = indicators.get('last_close', 0)
        if last_close > 0:
            atr_pct = (current_atr / last_close) * 100
        else:
            atr_pct = 0
        
        # HIGH VOLATILITY: ATR spike > 50% OR very high volume + big range
        if atr_ratio > 0.5 or (volume_change > 50 and atr_pct > 3):
            logger.info(f"🔥 {symbol}: HIGH volatility detected (ATR spike: {atr_ratio*100:+.1f}%, volume: {volume_change:+.1f}%)")
            return 'high'
        
        # LOW VOLATILITY: ATR declining + low volume + tight range
        elif atr_ratio < -0.3 and volume_change < -20 and atr_pct < 1:
            logger.info(f"😴 {symbol}: LOW volatility detected (consolidation)")
            return 'low'
        
        # NORMAL VOLATILITY
        else:
            return 'normal'
    
    def get_optimal_interval(self, market_data: Dict[str, Dict]) -> Tuple[int, str]:
        """
        Determine optimal analysis interval based on all symbols
        
        Args:
            market_data: Dict mapping symbol to indicator data
        
        Returns:
            Tuple of (interval_seconds, regime_description)
        """
        regimes = []
        
        for symbol, data in market_data.items():
            indicators = data.get('indicators', {}).get('1h', {})
            current_atr = indicators.get('atr')
            
            if current_atr is None:
                continue
            
            regime = self.detect_volatility_regime(symbol, current_atr, indicators)
            regimes.append(regime)
        
        if not regimes:
            return self.normal_interval, "normal (insufficient data)"
        
        # Count regime occurrences
        high_count = regimes.count('high')
        low_count = regimes.count('low')
        normal_count = regimes.count('normal')
        
        # Decision logic: If ANY symbol is high volatility, analyze more frequently
        if high_count > 0:
            return self.high_volatility_interval, f"high ({high_count}/{len(regimes)} symbols)"
        
        # If majority are low volatility, reduce frequency
        elif low_count > len(regimes) / 2:
            return self.low_volatility_interval, f"low ({low_count}/{len(regimes)} symbols consolidating)"
        
        # Otherwise, maintain normal schedule
        else:
            return self.normal_interval, f"normal ({normal_count}/{len(regimes)} symbols)"
    
    def detect_consolidation(self, df: pd.DataFrame, period: int = 20) -> bool:
        """
        Detect if market is consolidating (low volatility, range-bound)
        
        Args:
            df: OHLCV DataFrame
            period: Lookback period
        
        Returns:
            True if consolidating, False otherwise
        """
        if len(df) < period:
            return False
        
        recent = df.tail(period)
        
        # Calculate range as % of price
        high_max = recent['high'].max()
        low_min = recent['low'].min()
        avg_price = recent['close'].mean()
        
        if avg_price == 0:
            return False
        
        range_pct = ((high_max - low_min) / avg_price) * 100
        
        # Volume declining
        volume_trend = recent['volume'].iloc[-5:].mean() / recent['volume'].iloc[:5].mean()
        
        # Consolidation = tight range (<3%) + declining volume
        return range_pct < 3 and volume_trend < 0.8
    
    def should_trigger_immediate_analysis(self, symbol: str, current_atr: float, 
                                         previous_atr: float) -> bool:
        """
        Check if conditions warrant immediate analysis (outside normal schedule)
        
        Returns:
            True if should analyze immediately
        """
        if previous_atr == 0:
            return False
        
        # Sudden ATR spike (>80% increase in single interval)
        atr_change = (current_atr / previous_atr) - 1
        
        if atr_change > 0.8:
            logger.warning(f"⚡ {symbol}: IMMEDIATE analysis triggered (ATR spike: {atr_change*100:+.1f}%)")
            return True
        
        return False


# Global instance
volatility_monitor = VolatilityMonitor()

