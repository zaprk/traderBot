"""
Market data fetching module using CCXT
Handles multi-timeframe OHLCV data with caching and rate limiting
"""
import ccxt
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketDataCache:
    """Simple in-memory TTL cache for market data"""
    
    def __init__(self):
        self.cache = {}
        self.ttl_map = {
            '1m': 30,
            '5m': 30,
            '15m': 60,
            '1h': 120,
            '4h': 240,
            '1d': 600
        }
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self._get_ttl(key):
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: pd.DataFrame):
        self.cache[key] = (data, time.time())
    
    def _get_ttl(self, key: str) -> int:
        for tf in self.ttl_map:
            if tf in key:
                return self.ttl_map[tf]
        return 60


# Global cache instance
cache = MarketDataCache()


def create_exchange(exchange_id: str = "kraken", api_key: str = "", secret: str = "", 
                   paper_mode: bool = True) -> ccxt.Exchange:
    """
    Create and configure CCXT exchange instance
    """
    exchange_class = getattr(ccxt, exchange_id)
    
    config = {
        'enableRateLimit': True,
        'rateLimit': 1000,
    }
    
    if not paper_mode and api_key and secret:
        config['apiKey'] = api_key
        config['secret'] = secret
    
    exchange = exchange_class(config)
    return exchange


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str = '5m', 
                limit: int = 200, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV data with retry logic and caching
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe ('5m', '15m', '1h', '4h')
        limit: Number of candles to fetch
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    cache_key = f"{symbol}_{timeframe}_{limit}"
    
    # Check cache first
    if use_cache:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_data
    
    # Fetch from exchange with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {symbol} {timeframe} (limit={limit})")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                raise ValueError(f"No data returned for {symbol} {timeframe}")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the result
            cache.set(cache_key, df)
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except ccxt.NetworkError as e:
            logger.warning(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error fetching OHLCV: {e}")
            raise
    
    raise Exception(f"Failed to fetch data after {max_retries} attempts")


def fetch_multi_timeframes(exchange: ccxt.Exchange, symbol: str, 
                          timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple timeframes
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframes: List of timeframes to fetch (default: ['5m', '15m', '1h', '4h'])
    
    Returns:
        Dictionary mapping timeframe to DataFrame
    """
    if timeframes is None:
        timeframes = ['5m', '15m', '1h', '4h']
    
    result = {}
    for tf in timeframes:
        try:
            df = fetch_ohlcv(exchange, symbol, tf)
            result[tf] = df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {tf}: {e}")
            # Continue with other timeframes even if one fails
            result[tf] = pd.DataFrame()
    
    return result


def get_current_price(exchange: ccxt.Exchange, symbol: str) -> float:
    """
    Get current market price for a symbol
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
    
    Returns:
        Current price as float
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logger.error(f"Failed to fetch current price for {symbol}: {e}")
        raise


def format_symbol(coin: str, base_currency: str = "USDT") -> str:
    """
    Format symbol for CCXT (e.g., 'BTC' -> 'BTC/USDT')
    """
    return f"{coin}/{base_currency}"


