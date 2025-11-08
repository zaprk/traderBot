"""
Sentiment and news analysis module
Integrates with CoinGecko API for trending coins and market sentiment
"""
import requests
import time
from typing import Dict, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# CoinGecko API (no key required)
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Simple cache with TTL
_sentiment_cache = {}
_cache_ttl = 300  # 5 minutes


def _get_coin_id(symbol: str) -> Optional[str]:
    """
    Map trading symbol to CoinGecko coin ID
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
    
    Returns:
        CoinGecko coin ID (e.g., 'bitcoin')
    """
    # Extract base currency
    base = symbol.split('/')[0].upper()
    
    # Common mappings
    mapping = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BNB': 'binancecoin',
        'XRP': 'ripple',
        'ADA': 'cardano',
        'DOGE': 'dogecoin',
        'MATIC': 'matic-network',
        'DOT': 'polkadot',
        'AVAX': 'avalanche-2',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'TON': 'the-open-network',
        'LTC': 'litecoin'
    }
    
    return mapping.get(base)


def get_sentiment_score(symbol: str) -> Dict[str, any]:
    """
    Fetch sentiment data for a symbol from CoinGecko
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
    
    Returns:
        Dictionary with sentiment data:
        {
            'sentiment_score': float (-1 to +1),
            'buzz': str ('Low', 'Medium', 'High'),
            'trend': str ('Trending' or 'Normal'),
            'summary': str (human-readable summary)
        }
    """
    # Check cache
    cache_key = f"sentiment_{symbol}"
    if cache_key in _sentiment_cache:
        data, timestamp = _sentiment_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            logger.debug(f"Sentiment cache hit for {symbol}")
            return data
    
    coin_id = _get_coin_id(symbol)
    if not coin_id:
        logger.warning(f"No CoinGecko mapping for {symbol}")
        return _default_sentiment()
    
    try:
        # Fetch coin data with market sentiment
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'community_data': 'true',
            'developer_data': 'false',
            'sparkline': 'false'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract sentiment indicators
        sentiment_votes_up_percentage = data.get('sentiment_votes_up_percentage', 50)
        sentiment_votes_down_percentage = data.get('sentiment_votes_down_percentage', 50)
        
        # Calculate sentiment score (-1 to +1)
        sentiment_score = (sentiment_votes_up_percentage - 50) / 50
        
        # Community data for buzz
        community_data = data.get('community_data', {})
        twitter_followers = community_data.get('twitter_followers', 0)
        reddit_subscribers = community_data.get('reddit_subscribers', 0)
        
        # Market cap rank for buzz assessment
        market_cap_rank = data.get('market_cap_rank', 999)
        
        # Determine buzz level
        if market_cap_rank <= 10 or twitter_followers > 1000000:
            buzz = "High"
        elif market_cap_rank <= 50 or twitter_followers > 100000:
            buzz = "Medium"
        else:
            buzz = "Low"
        
        # Check trending status
        trend = "Normal"
        try:
            trending_response = requests.get(f"{COINGECKO_BASE_URL}/search/trending", timeout=10)
            if trending_response.status_code == 200:
                trending_data = trending_response.json()
                trending_coins = [item['item']['id'] for item in trending_data.get('coins', [])]
                if coin_id in trending_coins:
                    trend = "Trending"
        except Exception as e:
            logger.debug(f"Could not fetch trending data: {e}")
        
        # Generate summary
        sentiment_str = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
        summary = f"{sentiment_str.capitalize()} sentiment (score: {sentiment_score:+.2f}), {buzz.lower()} buzz"
        if trend == "Trending":
            summary += ", currently trending"
        
        result = {
            'sentiment_score': round(sentiment_score, 2),
            'buzz': buzz,
            'trend': trend,
            'summary': summary,
            'sentiment_votes_up_pct': sentiment_votes_up_percentage,
            'sentiment_votes_down_pct': sentiment_votes_down_percentage
        }
        
        # Cache the result
        _sentiment_cache[cache_key] = (result, time.time())
        
        logger.info(f"Fetched sentiment for {symbol}: {summary}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch sentiment for {symbol}: {e}")
        return _default_sentiment()
    except Exception as e:
        logger.error(f"Error processing sentiment for {symbol}: {e}")
        return _default_sentiment()


def _default_sentiment() -> Dict[str, any]:
    """
    Return default neutral sentiment when data is unavailable
    """
    return {
        'sentiment_score': 0.0,
        'buzz': 'Unknown',
        'trend': 'Normal',
        'summary': 'No sentiment data available',
        'sentiment_votes_up_pct': 50,
        'sentiment_votes_down_pct': 50
    }


def get_batch_sentiment(symbols: list) -> Dict[str, Dict]:
    """
    Fetch sentiment for multiple symbols
    
    Args:
        symbols: List of trading pairs
    
    Returns:
        Dictionary mapping symbol to sentiment data
    """
    result = {}
    for symbol in symbols:
        result[symbol] = get_sentiment_score(symbol)
        time.sleep(0.5)  # Rate limiting - CoinGecko allows ~10-50 req/min on free tier
    
    return result

