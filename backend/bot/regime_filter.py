"""
Market Regime Filter Module
Determines if market conditions are suitable for trading based on ADX
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    Filters symbols based on market regime (trending vs ranging)
    
    Uses ADX (Average Directional Index) to determine trend strength:
    - ADX < 20: RANGING (skip - chop/noise)
    - ADX 20-25: TRANSITION (cautious)
    - ADX >= 25: TRENDING (ideal for trading)
    """
    
    def __init__(self, min_adx: float = 20):
        """
        Args:
            min_adx: Minimum ADX value required for trading (default 20)
        """
        self.min_adx = min_adx
    
    def is_tradeable(self, indicators: Dict) -> tuple:
        """
        Check if a single symbol is tradeable based on ADX regime
        
        Args:
            indicators: Dictionary with 'adx' and 'adx_regime' keys
        
        Returns:
            Tuple of (is_tradeable: bool, reason: str)
        """
        adx_value = indicators.get('adx', 0)
        regime = indicators.get('adx_regime', 'UNKNOWN')
        
        if adx_value == 0 or regime == 'UNKNOWN':
            return False, "ADX data unavailable"
        
        if regime == 'RANGING':
            return False, f"Market ranging (ADX: {adx_value:.1f} < 20) - Choppy, avoid trading"
        
        if regime == 'TRANSITION':
            return True, f"Market in transition (ADX: {adx_value:.1f}) - Proceed cautiously"
        
        # TRENDING or STRONG_TREND
        return True, f"Market trending (ADX: {adx_value:.1f}) - Ideal conditions"
    
    def filter_symbols(self, market_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Filter multiple symbols by market regime
        
        Args:
            market_data: Dict mapping symbol to data dict with 'indicators' key
        
        Returns:
            Dict with only tradeable symbols
        """
        tradeable = {}
        skipped = []
        
        for symbol, data in market_data.items():
            # Get 1H indicators for regime detection (primary timeframe)
            indicators_1h = data.get('indicators', {}).get('1h', {})
            
            is_ok, reason = self.is_tradeable(indicators_1h)
            
            if is_ok:
                tradeable[symbol] = data
                logger.info(f"✅ {symbol}: {reason}")
            else:
                skipped.append(symbol)
                logger.info(f"❌ {symbol}: {reason}")
        
        if skipped:
            logger.warning(f"⚠️ Regime filter removed {len(skipped)}/{len(market_data)} symbols: {', '.join(skipped)}")
        
        return tradeable
    
    def analyze_market_conditions(self, market_data: Dict[str, Dict]) -> Dict:
        """
        Analyze overall market conditions across all symbols
        
        Returns:
            Dict with statistics about market regimes
        """
        regimes = {'RANGING': 0, 'TRANSITION': 0, 'TRENDING': 0, 'STRONG_TREND': 0, 'UNKNOWN': 0}
        adx_values = []
        
        for symbol, data in market_data.items():
            indicators_1h = data.get('indicators', {}).get('1h', {})
            regime = indicators_1h.get('adx_regime', 'UNKNOWN')
            adx = indicators_1h.get('adx', 0)
            
            regimes[regime] += 1
            if adx > 0:
                adx_values.append(adx)
        
        avg_adx = sum(adx_values) / len(adx_values) if adx_values else 0
        
        # Determine overall market state
        total = len(market_data)
        trending_pct = (regimes['TRENDING'] + regimes['STRONG_TREND']) / total * 100 if total > 0 else 0
        
        if trending_pct >= 60:
            overall = "STRONG_TRENDING_MARKET"
        elif trending_pct >= 40:
            overall = "MIXED_MARKET"
        elif regimes['RANGING'] / total >= 0.6 if total > 0 else False:
            overall = "RANGING_MARKET"
        else:
            overall = "UNCERTAIN_MARKET"
        
        return {
            'overall_state': overall,
            'regimes': regimes,
            'avg_adx': round(avg_adx, 1),
            'trending_percentage': round(trending_pct, 1),
            'tradeable_count': regimes['TRENDING'] + regimes['STRONG_TREND'] + regimes['TRANSITION']
        }


# Global instance
regime_filter = RegimeFilter(min_adx=20)

