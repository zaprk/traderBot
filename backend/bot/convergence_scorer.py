"""
Multi-Timeframe Convergence Scoring Module
Quantifies signal strength across timeframes
"""
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ConvergenceScorer:
    """
    Scores trading signals based on multi-timeframe alignment
    """
    
    def __init__(self):
        # 🚨 FIX #3: Adjusted timeframe weights to favor RECENT action more
        # Old: 1h=40%, 30m=30%, 15m=20%, 5m=10% (too backward-looking)
        # New: More balanced, recent action has more influence
        self.weights = {
            '1h': 0.25,   # Primary trend - reduced from 40%
            '30m': 0.25,  # Context timeframe - reduced from 30%
            '15m': 0.30,  # Signal timeframe - increased from 20%
            '5m': 0.20    # Trigger timeframe - increased from 10%
        }
    
    def calculate_convergence_score(self, indicators_multi_tf: Dict[str, Dict],
                                    target_direction: str = 'bullish') -> Dict:
        """
        Calculate convergence score across all timeframes
        
        Args:
            indicators_multi_tf: Dict mapping timeframe to indicators
            target_direction: 'bullish' or 'bearish'
        
        Returns:
            Dict with score breakdown and analysis
        """
        total_score = 0
        timeframe_scores = {}
        alignment_details = []
        
        for tf, weight in self.weights.items():
            if tf not in indicators_multi_tf:
                continue
            
            indicators = indicators_multi_tf[tf]
            
            # Score this timeframe (-100 to +100)
            tf_score = self._score_timeframe(indicators, target_direction)
            weighted_score = (tf_score / 100) * weight * 100  # Convert to weighted points
            
            total_score += weighted_score
            timeframe_scores[tf] = {
                'raw_score': tf_score,
                'weighted_score': round(weighted_score, 1),
                'weight': weight * 100
            }
            
            # Alignment status
            if tf_score > 50:
                status = f"✅ {target_direction.capitalize()}"
            elif tf_score < -50:
                opposite = 'bearish' if target_direction == 'bullish' else 'bullish'
                status = f"❌ {opposite.capitalize()}"
            else:
                status = "⚠️ Neutral"
            
            alignment_details.append(f"{tf}: {status} (score: {tf_score:+.0f}/100)")
        
        # Overall conviction level
        conviction = self._determine_conviction(total_score)
        
        result = {
            'total_score': round(total_score, 1),
            'max_score': 100,
            'conviction': conviction,
            'timeframe_scores': timeframe_scores,
            'alignment_details': alignment_details,
            'is_aligned': total_score >= 70  # High threshold for alignment
        }
        
        if result['is_aligned']:
            logger.info(
                f"🎯 HIGH CONVICTION {target_direction.upper()} SIGNAL: "
                f"Score {total_score:.1f}/100 ({conviction})"
            )
            for detail in alignment_details:
                logger.info(f"   {detail}")
        
        return result
    
    def _score_timeframe(self, indicators: Dict, target_direction: str) -> float:
        """
        Score a single timeframe (-100 to +100)
        
        Positive = aligned with target direction
        Negative = contrary to target direction
        """
        score = 0
        
        # 1. TREND SCORE (40 points)
        trend_interp = indicators.get('trend_interpretation', '').lower()
        
        if target_direction == 'bullish':
            if 'strong uptrend' in trend_interp:
                score += 40
            elif 'moderate uptrend' in trend_interp:
                score += 25
            elif 'neutral' in trend_interp or 'ranging' in trend_interp:
                score += 0
            elif 'moderate downtrend' in trend_interp:
                score -= 25
            elif 'strong downtrend' in trend_interp:
                score -= 40
        
        else:  # bearish
            if 'strong downtrend' in trend_interp:
                score += 40
            elif 'moderate downtrend' in trend_interp:
                score += 25
            elif 'neutral' in trend_interp or 'ranging' in trend_interp:
                score += 0
            elif 'moderate uptrend' in trend_interp:
                score -= 25
            elif 'strong uptrend' in trend_interp:
                score -= 40
        
        # 2. RSI SCORE (25 points)
        rsi_interp = indicators.get('rsi_interpretation', '').lower()
        rsi = indicators.get('rsi') or 50  # Default to 50 if None or missing
        
        if target_direction == 'bullish':
            if 'oversold' in rsi_interp:
                score += 25  # Bullish reversal opportunity
            elif rsi and rsi > 50:
                score += 15  # Momentum aligned
            elif rsi and rsi < 30:
                score += 20  # Very oversold
            else:
                score += 0
        
        else:  # bearish
            if 'overbought' in rsi_interp:
                score += 25  # Bearish reversal opportunity
            elif rsi and rsi < 50:
                score += 15  # Momentum aligned
            elif rsi and rsi > 70:
                score += 20  # Very overbought
            else:
                score += 0
        
        # 3. MACD SCORE (20 points)
        macd_interp = indicators.get('macd_interpretation', '').lower()
        
        if target_direction == 'bullish':
            if 'bullish momentum' in macd_interp:
                score += 20
            elif 'neutral' in macd_interp:
                score += 0
            else:
                score -= 10
        
        else:  # bearish
            if 'bearish momentum' in macd_interp:
                score += 20
            elif 'neutral' in macd_interp:
                score += 0
            else:
                score -= 10
        
        # 4. VOLUME SCORE (15 points)
        volume_interp = indicators.get('volume_interpretation', '').lower()
        
        if 'very high' in volume_interp:
            score += 15  # Strong conviction
        elif 'high' in volume_interp:
            score += 10  # Good confirmation
        elif 'normal' in volume_interp:
            score += 5   # Acceptable
        else:
            score += 0   # Low volume = weak signal
        
        # Cap at -100 to +100
        return max(-100, min(100, score))
    
    def _determine_conviction(self, total_score: float) -> str:
        """
        Determine conviction level based on total score
        
        Returns:
            Conviction level string
        """
        if total_score >= 85:
            return "VERY HIGH"
        elif total_score >= 70:
            return "HIGH"
        elif total_score >= 50:
            return "MODERATE"
        elif total_score >= 30:
            return "LOW"
        else:
            return "VERY LOW"
    
    def compare_directions(self, indicators_multi_tf: Dict[str, Dict]) -> Dict:
        """
        Compare bullish vs bearish conviction
        
        Returns:
            Dict with best direction and comparative scores
        """
        bullish_score = self.calculate_convergence_score(indicators_multi_tf, 'bullish')
        bearish_score = self.calculate_convergence_score(indicators_multi_tf, 'bearish')
        
        # Determine winner
        if bullish_score['total_score'] > bearish_score['total_score']:
            best_direction = 'bullish'
            best_score = bullish_score
            edge = bullish_score['total_score'] - bearish_score['total_score']
        else:
            best_direction = 'bearish'
            best_score = bearish_score
            edge = bearish_score['total_score'] - bullish_score['total_score']
        
        # Require significant edge (>20 points difference)
        has_clear_edge = edge > 20
        
        return {
            'best_direction': best_direction if has_clear_edge else 'none',
            'bullish': bullish_score,
            'bearish': bearish_score,
            'edge': round(edge, 1),
            'has_clear_edge': has_clear_edge,
            'recommendation': self._generate_recommendation(best_direction, best_score, has_clear_edge)
        }
    
    def _generate_recommendation(self, direction: str, score_data: Dict, 
                                has_edge: bool) -> str:
        """Generate human-readable recommendation"""
        if not has_edge:
            return "NO TRADE - Conflicting signals across timeframes"
        
        conviction = score_data['conviction']
        total_score = score_data['total_score']
        
        if conviction in ['VERY HIGH', 'HIGH']:
            return f"✅ STRONG {direction.upper()} SETUP - Score: {total_score:.1f}/100, Conviction: {conviction}"
        elif conviction == 'MODERATE':
            return f"⚠️ MODERATE {direction.upper()} SETUP - Score: {total_score:.1f}/100, Conviction: {conviction}"
        else:
            return f"❌ WEAK SIGNAL - Score: {total_score:.1f}/100, Conviction: {conviction}"
    
    def check_momentum_conflict(self, indicators_multi_tf: Dict[str, Dict], 
                               proposed_direction: str) -> Dict:
        """
        🚨 FIX #1: Detect when lower timeframe momentum contradicts higher timeframe trend
        CRITICAL: This prevents trading against recent momentum (the SOL SHORT mistake)
        
        Args:
            indicators_multi_tf: Multi-timeframe indicators
            proposed_direction: 'long' or 'short'
        
        Returns:
            Dict with conflict status and details
        """
        conflicts = []
        
        # Get higher TF trend (1h, 30m)
        higher_tf_trend = None
        for tf in ['1h', '30m']:
            if tf in indicators_multi_tf:
                trend_interp = indicators_multi_tf[tf].get('trend_interpretation', '').lower()
                if 'uptrend' in trend_interp:
                    higher_tf_trend = 'bullish'
                    break
                elif 'downtrend' in trend_interp:
                    higher_tf_trend = 'bearish'
                    break
        
        # Get lower TF momentum (5m, 15m)
        lower_tf_momentum = {}
        for tf in ['5m', '15m']:
            if tf in indicators_multi_tf:
                macd_interp = indicators_multi_tf[tf].get('macd_interpretation', '').lower()
                if 'bullish' in macd_interp:
                    lower_tf_momentum[tf] = 'bullish'
                elif 'bearish' in macd_interp:
                    lower_tf_momentum[tf] = 'bearish'
                else:
                    lower_tf_momentum[tf] = 'neutral'
        
        # Check for conflicts
        if higher_tf_trend and lower_tf_momentum:
            target_direction = 'bullish' if proposed_direction == 'long' else 'bearish'
            
            # Case 1: Trying to SHORT but 5m/15m showing BULLISH momentum
            if target_direction == 'bearish':
                for tf, momentum in lower_tf_momentum.items():
                    if momentum == 'bullish':
                        conflicts.append(f"{tf} shows BULLISH momentum (contradicts SHORT)")
            
            # Case 2: Trying to LONG but 5m/15m showing BEARISH momentum
            elif target_direction == 'bullish':
                for tf, momentum in lower_tf_momentum.items():
                    if momentum == 'bearish':
                        conflicts.append(f"{tf} shows BEARISH momentum (contradicts LONG)")
        
        has_conflict = len(conflicts) > 0
        
        if has_conflict:
            logger.warning(
                f"⚠️ MOMENTUM CONFLICT DETECTED: "
                f"Proposed {proposed_direction.upper()} but {', '.join(conflicts)}"
            )
        
        return {
            'has_conflict': has_conflict,
            'conflicts': conflicts,
            'higher_tf_trend': higher_tf_trend,
            'lower_tf_momentum': lower_tf_momentum,
            'recommendation': 'SKIP_TRADE' if has_conflict else 'PROCEED'
        }


# Global instance
convergence_scorer = ConvergenceScorer()

