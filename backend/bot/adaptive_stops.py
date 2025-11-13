"""
Adaptive Stop-Loss System
Dynamic stop placement based on volatility regime and market structure
"""
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class AdaptiveStops:
    """
    Calculate adaptive stop-loss and take-profit levels
    
    Features:
    - Volatility-aware ATR multipliers
    - Structure-based stops (swing highs/lows)
    - Regime-adaptive positioning
    """
    
    def __init__(self):
        # ATR multipliers by volatility regime
        self.atr_multipliers = {
            'expanding': {  # High volatility
                'stop': 1.2,
                'take_profit': 2.4  # 2:1 RR
            },
            'normal': {  # Normal volatility
                'stop': 1.5,
                'take_profit': 3.0  # 2:1 RR
            },
            'contracting': {  # Low volatility
                'stop': 2.0,
                'take_profit': 4.0  # 2:1 RR
            }
        }
    
    def calculate_stops(self, 
                       entry_price: float, 
                       action: str,
                       indicators_1h: Dict,
                       atr_history: Optional[list] = None) -> Dict:
        """
        Calculate adaptive stop-loss and take-profit levels
        
        Args:
            entry_price: Proposed entry price
            action: 'long' or 'short'
            indicators_1h: 1-hour indicators (must include atr, adx, swing_high/low)
            atr_history: Optional list of recent ATR values for regime detection
        
        Returns:
            Dictionary with stop_loss, take_profit, risk_reward, method
        """
        if action not in ['long', 'short']:
            return {
                'stop_loss': None,
                'take_profit': None,
                'risk_reward': None,
                'method': 'none'
            }
        
        # Extract required data
        atr = indicators_1h.get('atr')
        adx = indicators_1h.get('adx')
        swing_high = indicators_1h.get('swing_high')
        swing_low = indicators_1h.get('swing_low')
        
        if not atr or atr == 0:
            logger.warning("ATR not available for stop calculation")
            return self._fallback_stops(entry_price, action)
        
        # 1. DETECT VOLATILITY REGIME
        vol_regime = self._detect_volatility_regime(atr, atr_history)
        logger.info(f"📊 Volatility Regime: {vol_regime} (ATR: {atr:.2f})")
        
        # 2. CALCULATE ATR-BASED STOPS
        atr_stops = self._calculate_atr_stops(entry_price, atr, action, vol_regime)
        
        # 3. CALCULATE STRUCTURE-BASED STOPS
        structure_stops = self._calculate_structure_stops(
            entry_price, action, swing_high, swing_low
        )
        
        # 4. CHOOSE BEST METHOD
        final_stops = self._select_best_stops(
            entry_price, action, atr_stops, structure_stops, adx
        )
        
        logger.info(
            f"🎯 Adaptive Stops ({action.upper()}): "
            f"Entry={entry_price:.2f} | SL={final_stops['stop_loss']:.2f} | "
            f"TP={final_stops['take_profit']:.2f} | RR={final_stops['risk_reward']:.2f} | "
            f"Method={final_stops['method']}"
        )
        
        return final_stops
    
    def _detect_volatility_regime(self, current_atr: float, 
                                  atr_history: Optional[list] = None) -> str:
        """
        Detect if volatility is expanding, contracting, or normal
        
        Args:
            current_atr: Current ATR value
            atr_history: List of recent ATR values (e.g., last 20 periods)
        
        Returns:
            'expanding', 'contracting', or 'normal'
        """
        if not atr_history or len(atr_history) < 10:
            return 'normal'  # Default if no history
        
        # Calculate trend in ATR
        recent_atr = atr_history[-10:]  # Last 10 periods
        avg_atr = sum(recent_atr) / len(recent_atr)
        
        # Compare current ATR to recent average
        change_pct = (current_atr - avg_atr) / avg_atr if avg_atr > 0 else 0
        
        if change_pct > 0.15:  # ATR increasing by >15%
            return 'expanding'
        elif change_pct < -0.15:  # ATR decreasing by >15%
            return 'contracting'
        else:
            return 'normal'
    
    def _calculate_atr_stops(self, entry: float, atr: float, 
                            action: str, vol_regime: str) -> Dict:
        """
        Calculate ATR-based stops with regime-adaptive multipliers
        """
        multipliers = self.atr_multipliers.get(vol_regime, self.atr_multipliers['normal'])
        
        if action == 'long':
            stop_loss = entry - (atr * multipliers['stop'])
            take_profit = entry + (atr * multipliers['take_profit'])
        else:  # short
            stop_loss = entry + (atr * multipliers['stop'])
            take_profit = entry - (atr * multipliers['take_profit'])
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'method': f'atr_{vol_regime}'
        }
    
    def _calculate_structure_stops(self, entry: float, action: str,
                                   swing_high: Optional[float],
                                   swing_low: Optional[float]) -> Optional[Dict]:
        """
        Calculate structure-based stops using swing highs/lows
        """
        if action == 'long' and swing_low:
            # For longs, place stop below recent swing low
            stop_loss = swing_low * 0.995  # 0.5% buffer below swing
            
            # Take profit based on 2:1 risk-reward
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.0)
            
            # Validate: stop should be below entry
            if stop_loss >= entry:
                return None
            
            risk_reward = (take_profit - entry) / (entry - stop_loss)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward,
                'method': 'structure_swing_low'
            }
        
        elif action == 'short' and swing_high:
            # For shorts, place stop above recent swing high
            stop_loss = swing_high * 1.005  # 0.5% buffer above swing
            
            # Take profit based on 2:1 risk-reward
            risk = stop_loss - entry
            take_profit = entry - (risk * 2.0)
            
            # Validate: stop should be above entry
            if stop_loss <= entry:
                return None
            
            risk_reward = (entry - take_profit) / (stop_loss - entry)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward,
                'method': 'structure_swing_high'
            }
        
        return None
    
    def _select_best_stops(self, entry: float, action: str,
                          atr_stops: Dict, structure_stops: Optional[Dict],
                          adx: float) -> Dict:
        """
        Select best stop method based on market conditions
        
        Logic:
        - Strong trend (ADX > 30): Prefer structure stops (ride trend)
        - Weak trend (ADX < 25): Prefer tighter ATR stops (protect capital)
        - Validate risk-reward ratios (must be >= 1.3)
        """
        candidates = [atr_stops]
        
        if structure_stops:
            candidates.append(structure_stops)
        
        # Filter by minimum risk-reward
        valid_candidates = [c for c in candidates if c['risk_reward'] >= 1.3]
        
        if not valid_candidates:
            # If no valid candidates, use ATR with minimum RR adjustment
            logger.warning("No stops met minimum RR=1.3, adjusting take-profit")
            atr_stops['take_profit'] = self._adjust_take_profit_for_min_rr(
                entry, atr_stops['stop_loss'], action, min_rr=1.3
            )
            atr_stops['risk_reward'] = 1.3
            return atr_stops
        
        # Decision logic based on ADX
        if adx > 30 and structure_stops and structure_stops in valid_candidates:
            # Strong trend: use structure stops (wider, let winners run)
            logger.info(f"Strong trend (ADX={adx:.1f}): Using structure-based stops")
            return structure_stops
        else:
            # Weak/normal trend: use ATR stops (tighter, protect capital)
            logger.info(f"Normal trend (ADX={adx:.1f}): Using ATR-based stops")
            return atr_stops
    
    def _adjust_take_profit_for_min_rr(self, entry: float, stop_loss: float,
                                       action: str, min_rr: float = 1.3) -> float:
        """
        Adjust take-profit to meet minimum risk-reward ratio
        """
        risk = abs(entry - stop_loss)
        reward = risk * min_rr
        
        if action == 'long':
            return entry + reward
        else:
            return entry - reward
    
    def _fallback_stops(self, entry: float, action: str) -> Dict:
        """
        Fallback stops when ATR is unavailable (use 2% fixed)
        """
        logger.warning("Using fallback stops (2% fixed)")
        
        if action == 'long':
            stop_loss = entry * 0.98
            take_profit = entry * 1.04  # 2:1 RR
        else:
            stop_loss = entry * 1.02
            take_profit = entry * 0.96
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': 2.0,
            'method': 'fallback_fixed_2pct'
        }


# Global instance
adaptive_stops = AdaptiveStops()

