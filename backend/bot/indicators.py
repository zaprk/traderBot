"""
Technical indicators calculation module
Implements RSI, MACD, EMA, ATR, and market structure analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        series: Price series (typically close prices)
        period: RSI period (default 14)
    
    Returns:
        RSI values as Series
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        series: Price series
        period: EMA period
    
    Returns:
        EMA values as Series
    """
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        series: Price series (typically close prices)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default 14)
    
    Returns:
        ATR values as Series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = true_range.rolling(window=period).mean()
    
    return atr_values


def volume_change_pct(df: pd.DataFrame, window: int = 24) -> float:
    """
    Calculate volume change percentage compared to average
    
    Args:
        df: DataFrame with volume column
        window: Lookback window for average (default 24)
    
    Returns:
        Volume change percentage
    """
    if len(df) < window + 1:
        return 0.0
    
    recent_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].iloc[-(window+1):-1].mean()
    
    if avg_volume == 0:
        return 0.0
    
    change_pct = ((recent_volume - avg_volume) / avg_volume) * 100
    return round(change_pct, 2)


def market_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Analyze market structure (higher highs/higher lows, etc.)
    
    Args:
        df: DataFrame with high, low columns
        lookback: Number of candles to analyze
    
    Returns:
        Market structure description: 'HHHL' (uptrend), 'LHLL' (downtrend), 'range', 'insufficient_data'
    """
    if len(df) < lookback:
        return "insufficient_data"
    
    recent_data = df.iloc[-lookback:]
    highs = recent_data['high']
    lows = recent_data['low']
    
    # Find peaks and troughs
    high_peak = highs.max()
    low_trough = lows.min()
    
    # Compare recent half with earlier half
    mid_point = lookback // 2
    first_half_high = highs.iloc[:mid_point].max()
    second_half_high = highs.iloc[mid_point:].max()
    first_half_low = lows.iloc[:mid_point].min()
    second_half_low = lows.iloc[mid_point:].min()
    
    # Determine structure
    higher_highs = second_half_high > first_half_high
    higher_lows = second_half_low > first_half_low
    lower_highs = second_half_high < first_half_high
    lower_lows = second_half_low < first_half_low
    
    if higher_highs and higher_lows:
        return "HHHL"  # Uptrend
    elif lower_highs and lower_lows:
        return "LHLL"  # Downtrend
    else:
        return "range"  # Ranging/consolidation


def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate all indicators for a given DataFrame
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with all indicator values
    """
    if len(df) < 50:
        logger.warning("Insufficient data for indicator calculation")
        return {
            'rsi': None,
            'macd_hist': None,
            'ema_20': None,
            'ema_50': None,
            'ema_200': None,
            'atr': None,
            'last_close': None,
            'volume_change': 0.0,
            'market_structure': 'insufficient_data'
        }
    
    close = df['close']
    
    # Calculate indicators
    rsi_values = rsi(close, 14)
    macd_line, signal_line, histogram = macd(close)
    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    ema_200 = ema(close, 200)
    atr_values = atr(df, 14)
    vol_change = volume_change_pct(df, 24)
    structure = market_structure(df, 20)
    
    # Get latest values
    latest_idx = -1
    
    result = {
        'rsi': round(float(rsi_values.iloc[latest_idx]), 2) if not pd.isna(rsi_values.iloc[latest_idx]) else None,
        'macd_hist': round(float(histogram.iloc[latest_idx]), 4) if not pd.isna(histogram.iloc[latest_idx]) else None,
        'macd_positive': bool(histogram.iloc[latest_idx] > 0) if not pd.isna(histogram.iloc[latest_idx]) else False,
        'ema_20': round(float(ema_20.iloc[latest_idx]), 2) if not pd.isna(ema_20.iloc[latest_idx]) else None,
        'ema_50': round(float(ema_50.iloc[latest_idx]), 2) if not pd.isna(ema_50.iloc[latest_idx]) else None,
        'ema_200': round(float(ema_200.iloc[latest_idx]), 2) if not pd.isna(ema_200.iloc[latest_idx]) else None,
        'atr': round(float(atr_values.iloc[latest_idx]), 2) if not pd.isna(atr_values.iloc[latest_idx]) else None,
        'last_close': round(float(close.iloc[latest_idx]), 2),
        'volume_change': float(vol_change),
        'market_structure': structure
    }
    
    # EMA alignment - convert to Python bool
    if result['ema_20'] and result['ema_50']:
        result['ema_20_above_50'] = bool(result['ema_20'] > result['ema_50'])
    
    if result['ema_50'] and result['ema_200']:
        result['ema_50_above_200'] = bool(result['ema_50'] > result['ema_200'])
    
    return result

