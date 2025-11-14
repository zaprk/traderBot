"""
Trend Filter Module
Hard blocks countertrend trades when higher timeframe is in strong trend

CRITICAL: Prevents mean-reversion bias that loses money consistently
"""
import logging
from typing import Dict, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class TrendFilter:
    """
    Hard blocks countertrend trades based on higher timeframe trend
    
    Rules:
    - If 1h/30m in STRONG downtrend → BLOCK long entries (regardless of RSI)
    - If 1h/30m in STRONG uptrend → BLOCK short entries (regardless of RSI)
    - Only allow countertrend trades if structure break confirmed (higher high/lower low)
    """
    
    def __init__(self):
        # Required timeframes for trend analysis
        self.primary_timeframes = ['1h', '30m']
        self.confirmation_timeframes = ['15m', '5m']
    
    def check_trend_alignment(self, indicators_multi_tf: Dict[str, Dict], 
                             proposed_action: str) -> Dict:
        """
        Check if proposed trade aligns with higher timeframe trend
        
        Args:
            indicators_multi_tf: Multi-timeframe indicators
            proposed_action: 'long' or 'short'
        
        Returns:
            Dict with alignment status, trend info, and recommendation
        """
        # Get higher timeframe trend
        higher_tf_trend = self._get_higher_tf_trend(indicators_multi_tf)
        
        if not higher_tf_trend:
            return {
                'allowed': True,
                'reason': 'Insufficient data for trend analysis',
                'higher_tf_trend': None,
                'strength': None,
                'recommendation': 'PROCEED_WITH_CAUTION'
            }
        
        trend_direction = higher_tf_trend['direction']
        trend_strength = higher_tf_trend['strength']
        trend_tf = higher_tf_trend['timeframe']
        
        # Determine if trade is countertrend
        is_countertrend = False
        if proposed_action == 'long' and trend_direction == 'bearish':
            is_countertrend = True
        elif proposed_action == 'short' and trend_direction == 'bullish':
            is_countertrend = True
        
        # STRONG trend (ADX > 30) → HARD BLOCK countertrend trades
        if is_countertrend and trend_strength == 'STRONG':
            return {
                'allowed': False,
                'reason': (
                    f"🚨 HARD BLOCK: {proposed_action.upper()} in STRONG {trend_direction.upper()} trend "
                    f"on {trend_tf} (ADX > 30). Countertrend trades blocked to prevent mean-reversion losses."
                ),
                'higher_tf_trend': trend_direction,
                'strength': trend_strength,
                'timeframe': trend_tf,
                'recommendation': 'BLOCK_TRADE',
                'is_countertrend': True
            }
        
        # MODERATE trend (ADX 25-30) → Require structure confirmation for countertrend
        if is_countertrend and trend_strength == 'MODERATE':
            # Check for structure break confirmation
            has_confirmation = self._check_structure_confirmation(
                indicators_multi_tf, proposed_action
            )
            
            if not has_confirmation:
                return {
                    'allowed': False,
                    'reason': (
                        f"⚠️ BLOCK: {proposed_action.upper()} in {trend_direction.upper()} trend on {trend_tf} "
                        f"(ADX 25-30). No structure break confirmation. Wait for higher high (long) or lower low (short)."
                    ),
                    'higher_tf_trend': trend_direction,
                    'strength': trend_strength,
                    'timeframe': trend_tf,
                    'recommendation': 'REQUIRE_CONFIRMATION',
                    'is_countertrend': True,
                    'has_confirmation': False
                }
            else:
                return {
                    'allowed': True,
                    'reason': (
                        f"✅ ALLOWED: {proposed_action.upper()} in {trend_direction.upper()} trend on {trend_tf} "
                        f"with structure break confirmation (higher high/lower low detected)."
                    ),
                    'higher_tf_trend': trend_direction,
                    'strength': trend_strength,
                    'timeframe': trend_tf,
                    'recommendation': 'PROCEED_WITH_CONFIRMATION',
                    'is_countertrend': True,
                    'has_confirmation': True
                }
        
        # WEAK trend or ALIGNED trade → Allow
        if not is_countertrend:
            alignment = 'ALIGNED'
        elif trend_strength == 'WEAK':
            alignment = 'WEAK_TREND'
        else:
            alignment = 'UNKNOWN'
        
        return {
            'allowed': True,
            'reason': (
                f"✅ ALLOWED: {proposed_action.upper()} {'aligns with' if not is_countertrend else 'in weak'} "
                f"{trend_direction.upper()} trend on {trend_tf} (ADX < 25)."
            ),
            'higher_tf_trend': trend_direction,
            'strength': trend_strength,
            'timeframe': trend_tf,
            'recommendation': 'PROCEED',
            'is_countertrend': is_countertrend if trend_strength != 'WEAK' else False,
            'alignment': alignment
        }
    
    def _get_higher_tf_trend(self, indicators_multi_tf: Dict[str, Dict]) -> Optional[Dict]:
        """
        Get higher timeframe trend direction and strength
        
        Returns:
            Dict with direction, strength, and timeframe, or None
        """
        # Check primary timeframes (1h, 30m) for trend
        for tf in self.primary_timeframes:
            if tf not in indicators_multi_tf:
                continue
            
            indicators = indicators_multi_tf[tf]
            trend_interp = indicators.get('trend_interpretation', '').lower()
            adx = indicators.get('adx', 0)
            adx_regime = indicators.get('adx_regime', 'UNKNOWN')
            
            # Determine trend direction
            if 'uptrend' in trend_interp:
                direction = 'bullish'
            elif 'downtrend' in trend_interp:
                direction = 'bearish'
            else:
                continue  # Skip neutral/ranging
            
            # Determine trend strength based on ADX
            if adx >= 30:
                strength = 'STRONG'
            elif adx >= 25:
                strength = 'MODERATE'
            elif adx >= 20:
                strength = 'WEAK'
            else:
                strength = 'WEAK'  # Treat as weak if ADX < 20
            
            return {
                'direction': direction,
                'strength': strength,
                'timeframe': tf,
                'adx': adx,
                'regime': adx_regime
            }
        
        return None
    
    def _check_structure_confirmation(self, indicators_multi_tf: Dict[str, Dict],
                                     proposed_action: str) -> bool:
        """
        Check if structure break confirmation exists for countertrend trade
        
        For LONG in downtrend: Require higher high on 15m or 5m
        For SHORT in uptrend: Require lower low on 15m or 5m
        
        Returns:
            True if structure break confirmed, False otherwise
        """
        # Check confirmation timeframes (15m, 5m) for structure break
        for tf in self.confirmation_timeframes:
            if tf not in indicators_multi_tf:
                continue
            
            indicators = indicators_multi_tf[tf]
            market_structure = indicators.get('market_structure', '')
            trend_interp = indicators.get('trend_interpretation', '').lower()
            
            # For LONG in downtrend: Look for higher high (HHHL or uptrend structure)
            if proposed_action == 'long':
                if 'HHHL' in market_structure or 'uptrend' in trend_interp:
                    logger.info(f"✅ Structure confirmation: Higher high detected on {tf}")
                    return True
            
            # For SHORT in uptrend: Look for lower low (LHLL or downtrend structure)
            elif proposed_action == 'short':
                if 'LHLL' in market_structure or 'downtrend' in trend_interp:
                    logger.info(f"✅ Structure confirmation: Lower low detected on {tf}")
                    return True
        
        logger.warning(f"❌ No structure confirmation: No higher high (long) or lower low (short) detected")
        return False
    
    def get_trend_summary(self, indicators_multi_tf: Dict[str, Dict]) -> Dict:
        """
        Get summary of trend conditions across timeframes
        
        Returns:
            Dict with trend summary
        """
        summary = {
            'primary_trend': None,
            'trend_strength': None,
            'trend_timeframe': None,
            'all_timeframes': {}
        }
        
        # Analyze all timeframes
        for tf in ['1h', '30m', '15m', '5m']:
            if tf not in indicators_multi_tf:
                continue
            
            indicators = indicators_multi_tf[tf]
            trend_interp = indicators.get('trend_interpretation', '').lower()
            adx = indicators.get('adx', 0)
            market_structure = indicators.get('market_structure', '')
            
            # Determine trend
            if 'uptrend' in trend_interp:
                direction = 'bullish'
            elif 'downtrend' in trend_interp:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            # Determine strength
            if adx >= 30:
                strength = 'STRONG'
            elif adx >= 25:
                strength = 'MODERATE'
            elif adx >= 20:
                strength = 'WEAK'
            else:
                strength = 'RANGING'
            
            summary['all_timeframes'][tf] = {
                'direction': direction,
                'strength': strength,
                'adx': adx,
                'structure': market_structure
            }
        
        # Get primary trend (1h or 30m)
        primary_trend = self._get_higher_tf_trend(indicators_multi_tf)
        if primary_trend:
            summary['primary_trend'] = primary_trend['direction']
            summary['trend_strength'] = primary_trend['strength']
            summary['trend_timeframe'] = primary_trend['timeframe']
        
        return summary


# Global instance
trend_filter = TrendFilter()

