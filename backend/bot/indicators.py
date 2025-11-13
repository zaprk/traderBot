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


def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, str]:
    """
    Calculate Average Directional Index (ADX) - Trend Strength Indicator
    
    ADX measures trend strength on a scale of 0-100:
    - ADX < 20: Weak trend / Ranging market (avoid trading)
    - ADX 20-25: Developing trend (cautious)
    - ADX > 25: Strong trend (ideal for trading)
    - ADX > 40: Very strong trend
    
    Args:
        df: DataFrame with high, low, close columns
        period: ADX period (default 14)
    
    Returns:
        Tuple of (adx_values, regime_classification)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate +DM and -DM (Directional Movement)
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth the values
    atr_smoothed = true_range.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_smoothed)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_smoothed)
    
    # Calculate DX (Directional Movement Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX (smoothed DX)
    adx_values = dx.rolling(window=period).mean()
    
    # Classify market regime based on latest ADX
    latest_adx = adx_values.iloc[-1] if len(adx_values) > 0 and not pd.isna(adx_values.iloc[-1]) else 0
    
    if latest_adx < 20:
        regime = "RANGING"  # Weak trend, choppy market
    elif latest_adx < 25:
        regime = "TRANSITION"  # Developing trend
    elif latest_adx < 40:
        regime = "TRENDING"  # Strong trend
    else:
        regime = "STRONG_TREND"  # Very strong trend
    
    return adx_values, regime


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


def analyze_recent_momentum(df: pd.DataFrame, lookback: int = 3) -> Dict[str, any]:
    """
    🚨 FIX #2: Analyze recent price momentum from last N candles (most recent data)
    CRITICAL: This catches momentum shifts that historical averages miss
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of recent candles to analyze (default 3)
    
    Returns:
        Dictionary with recent momentum analysis
    """
    if df.empty or len(df) < lookback:
        return {
            'recent_momentum_pct': 0.0,
            'recent_direction': 'neutral',
            'volume_spike': False,
            'volume_direction': 'neutral',
            'volume_ratio': 1.0,
            'warning': None
        }
    
    # Last N candles
    recent = df.tail(lookback)
    
    # Calculate momentum from first to last close in recent window
    first_close = recent.iloc[0]['close']
    last_close = recent.iloc[-1]['close']
    momentum_pct = ((last_close - first_close) / first_close) * 100
    
    # Direction
    direction = 'bullish' if momentum_pct > 0.3 else 'bearish' if momentum_pct < -0.3 else 'neutral'
    
    # Volume spike detection
    current_volume = df.iloc[-1]['volume']
    # Compare to previous 5 candles (excluding current)
    if len(df) >= 6:
        recent_avg_volume = df.iloc[-6:-1]['volume'].mean()
        volume_ratio = current_volume / recent_avg_volume if recent_avg_volume > 0 else 1.0
        volume_spike = volume_ratio > 2.0  # 2x or more is a spike
    else:
        volume_ratio = 1.0
        volume_spike = False
    
    # Volume direction (is the current candle bullish or bearish?)
    current_candle = df.iloc[-1]
    volume_direction = 'bullish' if current_candle['close'] >= current_candle['open'] else 'bearish'
    
    # Generate warning if volume spike contradicts historical trend
    warning = None
    if volume_spike:
        if volume_direction == 'bullish' and abs(momentum_pct) > 0.3:
            warning = f"🔥 BULLISH volume spike ({volume_ratio:.1f}x) - Recent momentum: {momentum_pct:+.2f}%"
        elif volume_direction == 'bearish' and abs(momentum_pct) > 0.3:
            warning = f"🔻 BEARISH volume spike ({volume_ratio:.1f}x) - Recent momentum: {momentum_pct:+.2f}%"
    
    return {
        'recent_momentum_pct': round(momentum_pct, 2),
        'recent_direction': direction,
        'volume_spike': volume_spike,
        'volume_direction': volume_direction,
        'volume_ratio': round(volume_ratio, 2),
        'warning': warning
    }


def analyze_candle(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze the most recent candle pattern and momentum
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with candle analysis
    """
    if len(df) < 2:
        return {
            'candle_type': 'unknown',
            'momentum_pct': 0.0,
            'upper_wick_ratio': 0.0,
            'lower_wick_ratio': 0.0,
            'body_size_pct': 0.0,
            'summary': 'Insufficient data'
        }
    
    last = df.iloc[-1]
    open_price = last['open']
    close_price = last['close']
    high_price = last['high']
    low_price = last['low']
    
    # Momentum calculation
    momentum_pct = ((close_price - open_price) / open_price) * 100
    
    # Candle type
    candle_type = 'green' if close_price >= open_price else 'red'
    
    # Body and wick calculations
    body_size = abs(close_price - open_price)
    total_range = high_price - low_price
    
    if total_range > 0:
        body_pct = (body_size / total_range) * 100
        
        # Wick sizes
        if candle_type == 'green':
            upper_wick = high_price - close_price
            lower_wick = open_price - low_price
        else:
            upper_wick = high_price - open_price
            lower_wick = close_price - low_price
        
        upper_wick_ratio = (upper_wick / total_range) * 100
        lower_wick_ratio = (lower_wick / total_range) * 100
    else:
        body_pct = 0
        upper_wick_ratio = 0
        lower_wick_ratio = 0
    
    # Generate summary
    momentum_str = "strong" if abs(momentum_pct) > 1 else "moderate" if abs(momentum_pct) > 0.3 else "weak"
    direction = "bullish" if momentum_pct > 0 else "bearish"
    
    wick_note = ""
    if upper_wick_ratio > 50:
        wick_note = ", rejection at highs"
    elif lower_wick_ratio > 50:
        wick_note = ", rejection at lows"
    elif body_pct > 70:
        wick_note = ", decisive move"
    
    summary = f"{momentum_str.capitalize()} {direction} candle{wick_note}"
    
    return {
        'candle_type': candle_type,
        'momentum_pct': round(momentum_pct, 2),
        'upper_wick_ratio': round(upper_wick_ratio, 1),
        'lower_wick_ratio': round(lower_wick_ratio, 1),
        'body_size_pct': round(body_pct, 1),
        'summary': summary
    }


def normalize_indicators(indicators: Dict[str, any]) -> Dict[str, any]:
    """
    Normalize raw indicators to 0-10 scale and add interpretations
    
    Args:
        indicators: Raw indicator values
    
    Returns:
        Dictionary with normalized values and interpretations
    """
    normalized = {}
    
    # RSI normalization (0-100 -> 0-10, with zones)
    rsi = indicators.get('rsi')
    if rsi is not None:
        normalized['rsi_score'] = round((rsi / 10), 1)
        if rsi > 70:
            normalized['rsi_interpretation'] = "Overbought"
        elif rsi > 60:
            normalized['rsi_interpretation'] = "Strong"
        elif rsi > 40:
            normalized['rsi_interpretation'] = "Neutral"
        elif rsi > 30:
            normalized['rsi_interpretation'] = "Weak"
        else:
            normalized['rsi_interpretation'] = "Oversold"
    else:
        normalized['rsi_score'] = None
        normalized['rsi_interpretation'] = "Unknown"
    
    # MACD interpretation
    macd_hist = indicators.get('macd_hist', 0)
    if macd_hist > 0:
        normalized['macd_strength'] = min(10, round(abs(macd_hist) * 100, 1))
        normalized['macd_interpretation'] = "Bullish momentum"
    elif macd_hist < 0:
        normalized['macd_strength'] = min(10, round(abs(macd_hist) * 100, 1))
        normalized['macd_interpretation'] = "Bearish momentum"
    else:
        normalized['macd_strength'] = 0
        normalized['macd_interpretation'] = "Neutral"
    
    # EMA trend strength
    ema_20_above_50 = indicators.get('ema_20_above_50')
    ema_50_above_200 = indicators.get('ema_50_above_200')
    
    if ema_20_above_50 and ema_50_above_200:
        normalized['trend_strength'] = 9
        normalized['trend_interpretation'] = "Strong uptrend"
    elif ema_20_above_50:
        normalized['trend_strength'] = 7
        normalized['trend_interpretation'] = "Moderate uptrend"
    elif ema_20_above_50 is False and ema_50_above_200 is False:
        normalized['trend_strength'] = 1
        normalized['trend_interpretation'] = "Strong downtrend"
    elif ema_20_above_50 is False:
        normalized['trend_strength'] = 3
        normalized['trend_interpretation'] = "Moderate downtrend"
    else:
        normalized['trend_strength'] = 5
        normalized['trend_interpretation'] = "Neutral/Ranging"
    
    # Volume strength (0-10 scale)
    vol_change = indicators.get('volume_change', 0)
    normalized['volume_score'] = min(10, max(0, round(5 + (vol_change / 20), 1)))
    if vol_change > 50:
        normalized['volume_interpretation'] = "Very high volume"
    elif vol_change > 20:
        normalized['volume_interpretation'] = "High volume"
    elif vol_change > -20:
        normalized['volume_interpretation'] = "Normal volume"
    else:
        normalized['volume_interpretation'] = "Low volume"
    
    return normalized


def obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    OBV measures buying and selling pressure by accumulating volume on up days
    and subtracting volume on down days.
    
    Args:
        df: DataFrame with 'close' and 'volume' columns
    
    Returns:
        Series of OBV values
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate price direction
    price_direction = close.diff()
    
    # Calculate signed volume
    signed_volume = volume.copy()
    signed_volume[price_direction < 0] = -volume[price_direction < 0]
    signed_volume[price_direction == 0] = 0
    
    # Cumulative sum
    obv_values = signed_volume.cumsum()
    
    return obv_values


def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Detect swing highs and swing lows for structure-based stops
    
    A swing high is a peak where the high is higher than 'lookback' periods
    before and after it. Swing low is the opposite.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of periods to look back/forward
    
    Returns:
        Dictionary with last swing high and low prices
    """
    if len(df) < lookback * 2 + 1:
        return {
            'swing_high': None,
            'swing_low': None,
            'swing_high_idx': None,
            'swing_low_idx': None
        }
    
    highs = df['high'].values
    lows = df['low'].values
    
    swing_highs = []
    swing_lows = []
    
    # Scan for swing points (skip first and last 'lookback' periods)
    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        is_swing_high = all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
                       all(highs[i] > highs[i+j] for j in range(1, lookback+1))
        
        # Check for swing low
        is_swing_low = all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
                      all(lows[i] < lows[i+j] for j in range(1, lookback+1))
        
        if is_swing_high:
            swing_highs.append((i, highs[i]))
        
        if is_swing_low:
            swing_lows.append((i, lows[i]))
    
    # Get most recent swing points
    last_swing_high = swing_highs[-1] if swing_highs else (None, None)
    last_swing_low = swing_lows[-1] if swing_lows else (None, None)
    
    return {
        'swing_high': last_swing_high[1],
        'swing_low': last_swing_low[1],
        'swing_high_idx': last_swing_high[0],
        'swing_low_idx': last_swing_low[0]
    }


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
            'adx': 0,
            'adx_regime': 'UNKNOWN',
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
    adx_values, regime = adx(df, 14)
    vol_change = volume_change_pct(df, 24)
    structure = market_structure(df, 20)
    obv_values = obv(df)
    swing_points = detect_swing_points(df, lookback=5)
    
    # Get latest values
    latest_idx = -1
    
    # Calculate volume metrics
    volume = df['volume'].iloc[latest_idx] if 'volume' in df else 0
    volume_sma = df['volume'].rolling(window=20).mean().iloc[latest_idx] if 'volume' in df and len(df) >= 20 else volume
    
    result = {
        'rsi': round(float(rsi_values.iloc[latest_idx]), 2) if not pd.isna(rsi_values.iloc[latest_idx]) else None,
        'macd_hist': round(float(histogram.iloc[latest_idx]), 4) if not pd.isna(histogram.iloc[latest_idx]) else None,
        'macd_positive': bool(histogram.iloc[latest_idx] > 0) if not pd.isna(histogram.iloc[latest_idx]) else False,
        'ema_20': round(float(ema_20.iloc[latest_idx]), 2) if not pd.isna(ema_20.iloc[latest_idx]) else None,
        'ema_50': round(float(ema_50.iloc[latest_idx]), 2) if not pd.isna(ema_50.iloc[latest_idx]) else None,
        'ema_200': round(float(ema_200.iloc[latest_idx]), 2) if not pd.isna(ema_200.iloc[latest_idx]) else None,
        'atr': round(float(atr_values.iloc[latest_idx]), 2) if not pd.isna(atr_values.iloc[latest_idx]) else None,
        'adx': round(float(adx_values.iloc[latest_idx]), 2) if not pd.isna(adx_values.iloc[latest_idx]) else 0,
        'adx_regime': regime,
        'last_close': round(float(close.iloc[latest_idx]), 2),
        'volume': float(volume),
        'volume_sma': float(volume_sma),
        'volume_change': float(vol_change),
        'market_structure': structure,
        'obv': round(float(obv_values.iloc[latest_idx]), 2) if not pd.isna(obv_values.iloc[latest_idx]) else None,
        'swing_high': swing_points['swing_high'],
        'swing_low': swing_points['swing_low']
    }
    
    # EMA alignment - convert to Python bool
    if result['ema_20'] and result['ema_50']:
        result['ema_20_above_50'] = bool(result['ema_20'] > result['ema_50'])
    
    if result['ema_50'] and result['ema_200']:
        result['ema_50_above_200'] = bool(result['ema_50'] > result['ema_200'])
    
    # Add candle analysis
    candle_info = analyze_candle(df)
    result.update(candle_info)
    
    # Add normalized indicators and interpretations
    normalized = normalize_indicators(result)
    result.update(normalized)
    
    return result

