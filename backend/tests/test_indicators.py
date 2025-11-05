"""
Unit tests for indicators module
"""
import pytest
import pandas as pd
import numpy as np
from bot.indicators import rsi, ema, macd, atr, volume_change_pct, market_structure, calculate_all_indicators


def create_sample_df(length=100):
    """Create sample OHLCV dataframe for testing"""
    np.random.seed(42)
    
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(length) * 2)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(length) * 2,
        'low': prices - np.random.rand(length) * 2,
        'close': prices,
        'volume': np.random.randint(1000, 10000, length)
    })
    
    return df


def test_rsi_calculation():
    """Test RSI calculation"""
    df = create_sample_df(50)
    rsi_values = rsi(df['close'], period=14)
    
    # RSI should be between 0 and 100
    valid_rsi = rsi_values.dropna()
    assert all(valid_rsi >= 0)
    assert all(valid_rsi <= 100)
    
    # Should have NaN values for first period-1 entries
    assert rsi_values.iloc[:13].isna().all()


def test_ema_calculation():
    """Test EMA calculation"""
    df = create_sample_df(50)
    ema_values = ema(df['close'], period=20)
    
    # EMA should not be NaN after warmup
    assert not ema_values.iloc[-1] != ema_values.iloc[-1]  # Not NaN
    
    # EMA should be positive for positive prices
    assert all(ema_values.dropna() > 0)


def test_macd_calculation():
    """Test MACD calculation"""
    df = create_sample_df(100)
    macd_line, signal_line, histogram = macd(df['close'])
    
    # All should have same length
    assert len(macd_line) == len(signal_line) == len(histogram)
    
    # Histogram should be macd - signal
    diff = macd_line - signal_line
    assert np.allclose(diff.dropna(), histogram.dropna(), rtol=1e-5)


def test_atr_calculation():
    """Test ATR calculation"""
    df = create_sample_df(50)
    atr_values = atr(df, period=14)
    
    # ATR should be positive
    assert all(atr_values.dropna() > 0)
    
    # ATR should not be NaN after warmup
    assert not atr_values.iloc[-1] != atr_values.iloc[-1]


def test_volume_change_pct():
    """Test volume change percentage"""
    df = create_sample_df(50)
    vol_change = volume_change_pct(df, window=10)
    
    # Should return a number
    assert isinstance(vol_change, float)


def test_market_structure():
    """Test market structure analysis"""
    df = create_sample_df(50)
    structure = market_structure(df, lookback=20)
    
    # Should return one of expected values
    assert structure in ['HHHL', 'LHLL', 'range', 'insufficient_data']


def test_market_structure_insufficient_data():
    """Test market structure with insufficient data"""
    df = create_sample_df(10)
    structure = market_structure(df, lookback=20)
    
    assert structure == 'insufficient_data'


def test_calculate_all_indicators():
    """Test calculating all indicators at once"""
    df = create_sample_df(100)
    indicators = calculate_all_indicators(df)
    
    # Check all expected keys present
    expected_keys = ['rsi', 'macd_hist', 'ema_20', 'ema_50', 'ema_200', 
                    'atr', 'last_close', 'volume_change', 'market_structure']
    
    for key in expected_keys:
        assert key in indicators
    
    # Check values are reasonable
    if indicators['rsi'] is not None:
        assert 0 <= indicators['rsi'] <= 100
    
    if indicators['atr'] is not None:
        assert indicators['atr'] > 0
    
    assert indicators['last_close'] > 0


def test_calculate_all_indicators_insufficient_data():
    """Test indicators with insufficient data"""
    df = create_sample_df(30)  # Less than 50 required
    indicators = calculate_all_indicators(df)
    
    # Should return None for most indicators
    assert indicators['rsi'] is None
    assert indicators['market_structure'] == 'insufficient_data'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


