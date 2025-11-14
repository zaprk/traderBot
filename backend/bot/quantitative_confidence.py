"""
Quantitative Confidence Scoring System
Replaces subjective LLM confidence with measurable, calibrated metrics
"""
import logging
from typing import Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class QuantitativeConfidence:
    """
    Calculate quantitative confidence scores based on technical indicators
    
    Confidence = weighted composite of:
    - Trend strength (40%): ADX normalized
    - Volume score (30%): Current volume vs 30-day average
    - Momentum score (30%): RSI distance from neutral
    """
    
    def __init__(self):
        self.weights = {
            'trend_strength': 0.40,
            'volume_score': 0.30,
            'momentum_score': 0.30
        }
    
    def calculate_confidence(self, indicators_1h: Dict, action: str, 
                            indicators_multi_tf: Dict[str, Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate quantitative confidence score (0.0 - 1.0)
        
        Args:
            indicators_1h: 1-hour timeframe indicators
            action: 'long', 'short', or 'none'
            indicators_multi_tf: Multi-timeframe indicators (optional, for trend alignment check)
        
        Returns:
            (confidence_score, components_breakdown)
        """
        if action == 'none':
            return 0.0, {'reason': 'No trade action'}
        
        # Extract required indicators
        adx = indicators_1h.get('adx')
        rsi = indicators_1h.get('rsi')
        volume = indicators_1h.get('volume')
        volume_sma = indicators_1h.get('volume_sma')
        trend_interp = indicators_1h.get('trend_interpretation', '').lower()
        
        # Validate data availability
        if any(x is None for x in [adx, rsi, volume, volume_sma]):
            logger.warning("Missing indicators for confidence calculation")
            return 0.5, {'reason': 'Insufficient data'}
        
        # 1. TREND STRENGTH (0-1): ADX normalized
        trend_strength = self._calculate_trend_strength(adx)
        
        # 2. VOLUME SCORE (0-1): Current volume vs average
        volume_score = self._calculate_volume_score(volume, volume_sma)
        
        # 3. MOMENTUM SCORE (0-1): RSI alignment with direction
        momentum_score = self._calculate_momentum_score(rsi, action)
        
        # 4. 🚨 TREND ALIGNMENT PENALTY: Heavily penalize countertrend trades
        trend_alignment_penalty = self._calculate_trend_alignment_penalty(
            indicators_1h, action, indicators_multi_tf
        )
        
        # Base weighted composite
        base_confidence = (
            self.weights['trend_strength'] * trend_strength +
            self.weights['volume_score'] * volume_score +
            self.weights['momentum_score'] * momentum_score
        )
        
        # Apply trend alignment penalty (multiplier)
        # If countertrend in strong trend → penalty reduces confidence by 50-80%
        confidence = base_confidence * trend_alignment_penalty
        
        # Ensure bounds [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        components = {
            'trend_strength': round(trend_strength, 3),
            'volume_score': round(volume_score, 3),
            'momentum_score': round(momentum_score, 3),
            'trend_alignment_penalty': round(trend_alignment_penalty, 3),
            'base_confidence': round(base_confidence, 3),
            'confidence': round(confidence, 3),
            'adx': round(adx, 2),
            'rsi': round(rsi, 2),
            'volume_ratio': round(volume / volume_sma, 2) if volume_sma > 0 else 0
        }
        
        if trend_alignment_penalty < 1.0:
            logger.warning(
                f"⚠️ Countertrend penalty applied: {trend_alignment_penalty:.2f}x "
                f"(confidence reduced from {base_confidence:.3f} to {confidence:.3f})"
            )
        
        logger.info(f"📊 Quantitative Confidence: {confidence:.3f} | Components: {components}")
        
        return confidence, components
    
    def _calculate_trend_strength(self, adx: float) -> float:
        """
        ADX-based trend strength (0-1)
        
        ADX Scale:
        - < 20: No trend (0.0 - 0.2)
        - 20-25: Weak trend (0.2 - 0.4)
        - 25-40: Strong trend (0.4 - 0.8)
        - 40+: Very strong trend (0.8 - 1.0)
        """
        if adx < 20:
            return adx / 100  # 0.0 - 0.2
        elif adx < 25:
            return 0.2 + (adx - 20) * 0.04  # 0.2 - 0.4
        elif adx < 40:
            return 0.4 + (adx - 25) * 0.027  # 0.4 - 0.8
        else:
            return min(1.0, 0.8 + (adx - 40) * 0.01)  # 0.8 - 1.0 (capped)
    
    def _calculate_volume_score(self, current_volume: float, avg_volume: float) -> float:
        """
        Volume score based on current vs average (0-1)
        
        Volume Ratio:
        - < 0.5x: Very low (0.0)
        - 0.5-1.0x: Below average (0.3)
        - 1.0-1.5x: Normal (0.6)
        - 1.5-2.0x: Above average (0.8)
        - > 2.0x: Spike (1.0)
        """
        if avg_volume == 0:
            return 0.5  # Neutral if no data
        
        ratio = current_volume / avg_volume
        
        if ratio < 0.5:
            return 0.0
        elif ratio < 1.0:
            return 0.3 + (ratio - 0.5) * 0.6  # 0.3 - 0.6
        elif ratio < 1.5:
            return 0.6 + (ratio - 1.0) * 0.4  # 0.6 - 0.8
        elif ratio < 2.0:
            return 0.8 + (ratio - 1.5) * 0.4  # 0.8 - 1.0
        else:
            return 1.0
    
    def _calculate_momentum_score(self, rsi: float, action: str) -> float:
        """
        RSI-based momentum score with directional alignment (0-1)
        
        For LONG:
        - RSI > 50: Bullish momentum (scaled by distance from 50)
        - RSI < 50: Bearish momentum (penalty)
        
        For SHORT:
        - RSI < 50: Bearish momentum (scaled by distance from 50)
        - RSI > 50: Bullish momentum (penalty)
        """
        neutral = 50.0
        distance = abs(rsi - neutral)
        
        if action == 'long':
            if rsi > neutral:
                # Bullish: score increases with RSI distance from 50
                # RSI 50-70 is ideal (score 0.5-1.0)
                score = 0.5 + min(0.5, (rsi - neutral) / 40)
            else:
                # Bearish: penalty for going long in oversold
                # RSI 30-50 gives diminishing scores (0.2-0.5)
                score = max(0.0, rsi / 100)
        
        elif action == 'short':
            if rsi < neutral:
                # Bearish: score increases as RSI falls below 50
                # RSI 30-50 is ideal (score 0.5-1.0)
                score = 0.5 + min(0.5, (neutral - rsi) / 40)
            else:
                # Bullish: penalty for going short in overbought
                # RSI 50-70 gives diminishing scores (0.5-0.2)
                score = max(0.0, (100 - rsi) / 100)
        else:
            score = 0.0
        
        return score
    
    def _calculate_trend_alignment_penalty(self, indicators_1h: Dict, action: str,
                                          indicators_multi_tf: Dict[str, Dict] = None) -> float:
        """
        Calculate trend alignment penalty for countertrend trades
        
        Returns:
            Multiplier (0.0 - 1.0):
            - 1.0 = No penalty (aligned trade)
            - 0.5-0.8 = Moderate penalty (weak countertrend)
            - 0.2-0.5 = Strong penalty (moderate countertrend)
            - 0.0-0.2 = Severe penalty (strong countertrend)
        """
        # Check higher timeframe trend (1h or 30m)
        higher_tf_trend = None
        if indicators_multi_tf:
            for tf in ['1h', '30m']:
                if tf in indicators_multi_tf:
                    indicators = indicators_multi_tf[tf]
                    trend_interp = indicators.get('trend_interpretation', '').lower()
                    adx = indicators.get('adx', 0)
                    
                    if 'uptrend' in trend_interp:
                        higher_tf_trend = {'direction': 'bullish', 'adx': adx}
                        break
                    elif 'downtrend' in trend_interp:
                        higher_tf_trend = {'direction': 'bearish', 'adx': adx}
                        break
        
        # If no higher TF data, use 1h indicators
        if not higher_tf_trend:
            trend_interp = indicators_1h.get('trend_interpretation', '').lower()
            adx = indicators_1h.get('adx', 0)
            
            if 'uptrend' in trend_interp:
                higher_tf_trend = {'direction': 'bullish', 'adx': adx}
            elif 'downtrend' in trend_interp:
                higher_tf_trend = {'direction': 'bearish', 'adx': adx}
        
        # No trend data → no penalty
        if not higher_tf_trend:
            return 1.0
        
        # Check if trade is countertrend
        is_countertrend = False
        if action == 'long' and higher_tf_trend['direction'] == 'bearish':
            is_countertrend = True
        elif action == 'short' and higher_tf_trend['direction'] == 'bullish':
            is_countertrend = True
        
        # No countertrend → no penalty
        if not is_countertrend:
            return 1.0
        
        # Calculate penalty based on ADX (trend strength)
        adx_value = higher_tf_trend['adx']
        
        if adx_value >= 30:
            # STRONG trend → Severe penalty (0.0-0.2)
            # Countertrend in strong trend is VERY dangerous
            penalty = max(0.0, 0.2 - (adx_value - 30) * 0.01)
            logger.warning(
                f"🚨 SEVERE COUNTERTREND PENALTY: {action.upper()} in STRONG "
                f"{higher_tf_trend['direction'].upper()} trend (ADX: {adx_value:.1f}) → "
                f"Penalty: {penalty:.2f}x"
            )
        elif adx_value >= 25:
            # MODERATE trend → Strong penalty (0.2-0.5)
            penalty = 0.5 - (adx_value - 25) * 0.06
            logger.warning(
                f"⚠️ STRONG COUNTERTREND PENALTY: {action.upper()} in MODERATE "
                f"{higher_tf_trend['direction'].upper()} trend (ADX: {adx_value:.1f}) → "
                f"Penalty: {penalty:.2f}x"
            )
        elif adx_value >= 20:
            # WEAK trend → Moderate penalty (0.5-0.8)
            penalty = 0.8 - (adx_value - 20) * 0.06
            logger.info(
                f"⚠️ MODERATE COUNTERTREND PENALTY: {action.upper()} in WEAK "
                f"{higher_tf_trend['direction'].upper()} trend (ADX: {adx_value:.1f}) → "
                f"Penalty: {penalty:.2f}x"
            )
        else:
            # RANGING (ADX < 20) → No penalty (trend is weak anyway)
            penalty = 1.0
        
        return max(0.0, min(1.0, penalty))
    
    def adjust_confidence_for_convergence(self, base_confidence: float, 
                                         convergence_score: int) -> float:
        """
        Adjust quantitative confidence based on multi-timeframe convergence
        
        Args:
            base_confidence: Base confidence (0-1)
            convergence_score: Convergence score (0-100)
        
        Returns:
            Adjusted confidence (0-1)
        """
        # Convergence score acts as a multiplier/penalty
        # High convergence (>80) boosts confidence
        # Low convergence (<40) reduces confidence
        
        convergence_factor = convergence_score / 100.0
        
        # Weighted blend: 70% base confidence, 30% convergence
        adjusted = 0.7 * base_confidence + 0.3 * convergence_factor
        
        return max(0.0, min(1.0, adjusted))


# Global instance
quantitative_confidence = QuantitativeConfidence()

