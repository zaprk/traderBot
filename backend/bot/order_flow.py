"""
Order Flow & Liquidity Detection Module
Identifies key levels, liquidity pools, and order blocks
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """
    Analyzes order flow and detects liquidity zones
    """
    
    def __init__(self):
        self.lookback_period = 50  # Candles to analyze
    
    def analyze_order_flow(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive order flow analysis
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Dict with liquidity pools, order blocks, key levels, and advanced volume metrics
        """
        if len(df) < self.lookback_period:
            return {
                'liquidity_pools': [],
                'order_blocks': [],
                'key_levels': [],
                'volume_delta': 0,
                'volume_profile': {},
                'absorption_detected': False,
                'analysis': 'Insufficient data'
            }
        
        # 1. Find liquidity pools (stop hunt zones)
        liquidity_pools = self._find_liquidity_pools(df)
        
        # 2. Identify order blocks (institutional zones) - ENHANCED
        order_blocks = self._detect_systematic_order_blocks(df)
        
        # 3. Detect key support/resistance levels
        key_levels = self._find_key_levels(df)
        
        # 4. Advanced volume metrics
        volume_delta = self._calculate_volume_delta(df)
        volume_profile = self._build_volume_profile(df)
        absorption_detected = self._detect_absorption(df)
        
        # 5. Current price context
        current_price = df.iloc[-1]['close']
        nearest_support, nearest_resistance = self._find_nearest_levels(
            current_price, liquidity_pools, order_blocks, key_levels
        )
        
        return {
            'liquidity_pools': liquidity_pools,
            'order_blocks': order_blocks,
            'key_levels': key_levels,
            'volume_delta': volume_delta,
            'volume_profile': volume_profile,
            'absorption_detected': absorption_detected,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'analysis': self._generate_analysis(
                current_price, nearest_support, nearest_resistance,
                volume_delta, absorption_detected
            )
        }
    
    def _find_liquidity_pools(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find liquidity pools (clusters of stops above highs / below lows)
        
        These are areas where stop losses are likely clustered
        """
        pools = []
        recent = df.tail(self.lookback_period)
        
        # Find swing highs (local peaks)
        for i in range(2, len(recent) - 2):
            current_high = recent.iloc[i]['high']
            
            # Is it higher than 2 candles before and after?
            if (current_high > recent.iloc[i-1]['high'] and 
                current_high > recent.iloc[i-2]['high'] and
                current_high > recent.iloc[i+1]['high'] and
                current_high > recent.iloc[i+2]['high']):
                
                pools.append({
                    'type': 'resistance',
                    'price': current_high,
                    'category': 'liquidity_pool',
                    'description': f'Stop hunt zone above ${current_high:.2f}'
                })
        
        # Find swing lows (local troughs)
        for i in range(2, len(recent) - 2):
            current_low = recent.iloc[i]['low']
            
            # Is it lower than 2 candles before and after?
            if (current_low < recent.iloc[i-1]['low'] and 
                current_low < recent.iloc[i-2]['low'] and
                current_low < recent.iloc[i+1]['low'] and
                current_low < recent.iloc[i+2]['low']):
                
                pools.append({
                    'type': 'support',
                    'price': current_low,
                    'category': 'liquidity_pool',
                    'description': f'Stop hunt zone below ${current_low:.2f}'
                })
        
        return pools
    
    def _detect_systematic_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """
        🔥 SYSTEMATIC ORDER BLOCK DETECTION
        Detects smart money accumulation/distribution zones
        
        Requirements:
        - Bullish OB: Strong green candle (2%+ body) after red candle with volume spike
        - Bearish OB: Strong red candle (2%+ body) after green candle with volume spike
        """
        blocks = []
        recent = df.tail(self.lookback_period)
        
        for i in range(2, len(recent)):
            current = recent.iloc[i]
            previous = recent.iloc[i-1]
            prev_prev = recent.iloc[i-2] if i >= 2 else None
            
            current_open = current['open']
            current_close = current['close']
            current_high = current['high']
            current_low = current['low']
            current_volume = current['volume']
            
            prev_close = previous['close']
            prev_open = previous['open']
            prev_volume = previous['volume']
            
            # Calculate body percentage
            body_pct = abs(current_close - current_open) / current_open
            
            # 🟢 BULLISH ORDER BLOCK DETECTION
            if (current_close > current_open * 1.02 and  # 2%+ green body
                prev_close < prev_open and  # Previous candle is red
                current_volume > prev_volume * 1.5):  # Volume spike (50%+)
                
                # Calculate strength
                strength_score = 0
                if body_pct > 0.03: strength_score += 30  # 3%+ body
                if current_volume > prev_volume * 2: strength_score += 30  # 2x volume
                if current_close == current_high: strength_score += 20  # Closed at high
                if i >= 2 and prev_prev is not None:
                    # Check if this breaks downtrend
                    if prev_prev['close'] < prev_prev['open']:
                        strength_score += 20
                
                strength = 'strong' if strength_score >= 60 else 'moderate'
                
                blocks.append({
                    'type': 'support',
                    'price_low': current_low,
                    'price_high': current_open,
                    'category': 'bullish_order_block',
                    'description': f'💰 Bullish OB: ${current_low:.2f}-${current_open:.2f} (vol: {(current_volume/prev_volume):.1f}x)',
                    'strength': strength,
                    'strength_score': strength_score,
                    'volume_ratio': current_volume / prev_volume,
                    'body_pct': body_pct * 100,
                    'index': i
                })
            
            # 🔴 BEARISH ORDER BLOCK DETECTION
            elif (current_close < current_open * 0.98 and  # 2%+ red body
                  prev_close > prev_open and  # Previous candle is green
                  current_volume > prev_volume * 1.5):  # Volume spike (50%+)
                
                # Calculate strength
                strength_score = 0
                if body_pct > 0.03: strength_score += 30  # 3%+ body
                if current_volume > prev_volume * 2: strength_score += 30  # 2x volume
                if current_close == current_low: strength_score += 20  # Closed at low
                if i >= 2 and prev_prev is not None:
                    # Check if this breaks uptrend
                    if prev_prev['close'] > prev_prev['open']:
                        strength_score += 20
                
                strength = 'strong' if strength_score >= 60 else 'moderate'
                
                blocks.append({
                    'type': 'resistance',
                    'price_low': current_open,
                    'price_high': current_high,
                    'category': 'bearish_order_block',
                    'description': f'💰 Bearish OB: ${current_open:.2f}-${current_high:.2f} (vol: {(current_volume/prev_volume):.1f}x)',
                    'strength': strength,
                    'strength_score': strength_score,
                    'volume_ratio': current_volume / prev_volume,
                    'body_pct': body_pct * 100,
                    'index': i
                })
        
        # Keep only most recent 5 of each type (strongest first)
        bullish_blocks = sorted(
            [b for b in blocks if 'bullish' in b['category']], 
            key=lambda x: x['strength_score'], 
            reverse=True
        )[:5]
        
        bearish_blocks = sorted(
            [b for b in blocks if 'bearish' in b['category']], 
            key=lambda x: x['strength_score'], 
            reverse=True
        )[:5]
        
        return bullish_blocks + bearish_blocks
    
    def _find_key_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find key support/resistance levels based on price clusters
        """
        levels = []
        recent = df.tail(self.lookback_period)
        
        # Use quantiles to find significant levels
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Resistance levels (upper quantiles)
        r1 = np.quantile(highs, 0.75)
        r2 = np.quantile(highs, 0.90)
        r3 = np.quantile(highs, 0.95)
        
        # Support levels (lower quantiles)
        s1 = np.quantile(lows, 0.25)
        s2 = np.quantile(lows, 0.10)
        s3 = np.quantile(lows, 0.05)
        
        # Add to levels
        levels.extend([
            {'type': 'resistance', 'price': r3, 'strength': 'strong', 'description': f'Strong R: ${r3:.2f}'},
            {'type': 'resistance', 'price': r2, 'strength': 'moderate', 'description': f'Moderate R: ${r2:.2f}'},
            {'type': 'resistance', 'price': r1, 'strength': 'weak', 'description': f'Weak R: ${r1:.2f}'},
            {'type': 'support', 'price': s1, 'strength': 'weak', 'description': f'Weak S: ${s1:.2f}'},
            {'type': 'support', 'price': s2, 'strength': 'moderate', 'description': f'Moderate S: ${s2:.2f}'},
            {'type': 'support', 'price': s3, 'strength': 'strong', 'description': f'Strong S: ${s3:.2f}'}
        ])
        
        return levels
    
    def _find_nearest_levels(self, current_price: float, 
                            liquidity_pools: List[Dict],
                            order_blocks: List[Dict],
                            key_levels: List[Dict]) -> tuple:
        """
        Find nearest support and resistance to current price
        
        Returns:
            Tuple of (nearest_support, nearest_resistance)
        """
        # Collect all supports and resistances
        supports = []
        resistances = []
        
        # From liquidity pools
        for pool in liquidity_pools:
            price = pool['price']
            if pool['type'] == 'support' and price < current_price:
                supports.append(price)
            elif pool['type'] == 'resistance' and price > current_price:
                resistances.append(price)
        
        # From order blocks
        for block in order_blocks:
            if 'price_low' in block and 'price_high' in block:
                mid_price = (block['price_low'] + block['price_high']) / 2
            else:
                continue
            
            if block['type'] == 'support' and mid_price < current_price:
                supports.append(mid_price)
            elif block['type'] == 'resistance' and mid_price > current_price:
                resistances.append(mid_price)
        
        # From key levels
        for level in key_levels:
            price = level['price']
            if level['type'] == 'support' and price < current_price:
                supports.append(price)
            elif level['type'] == 'resistance' and price > current_price:
                resistances.append(price)
        
        # Find nearest
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None
        
        return nearest_support, nearest_resistance
    
    def _calculate_volume_delta(self, df: pd.DataFrame) -> float:
        """
        📊 VOLUME DELTA: Calculate buying vs selling pressure
        
        Returns:
            Positive = buying pressure, Negative = selling pressure
        """
        recent = df.tail(20)  # Last 20 candles
        total_delta = 0
        
        for idx, candle in recent.iterrows():
            if candle['close'] >= candle['open']:
                # Green candle: majority volume is buying
                buy_volume = candle['volume'] * 0.7
                sell_volume = candle['volume'] * 0.3
            else:
                # Red candle: majority volume is selling
                buy_volume = candle['volume'] * 0.3
                sell_volume = candle['volume'] * 0.7
            
            total_delta += (buy_volume - sell_volume)
        
        return total_delta
    
    def _build_volume_profile(self, df: pd.DataFrame, price_levels: int = 20) -> Dict:
        """
        📊 VOLUME PROFILE: Identify high-volume nodes
        
        Returns:
            Dict of price bands with their volume
        """
        recent = df.tail(100)  # Last 100 candles
        
        price_min = recent['low'].min()
        price_max = recent['high'].max()
        level_size = (price_max - price_min) / price_levels
        
        profile = {}
        high_volume_nodes = []
        
        for level in range(price_levels):
            price_band_start = price_min + (level * level_size)
            price_band_end = price_band_start + level_size
            
            # Sum volume for candles touching this price band
            band_volume = 0
            for idx, candle in recent.iterrows():
                if candle['low'] <= price_band_end and candle['high'] >= price_band_start:
                    band_volume += candle['volume']
            
            band_key = f"{price_band_start:.2f}-{price_band_end:.2f}"
            profile[band_key] = band_volume
            
            # Track high volume nodes (top 20%)
            if band_volume > 0:
                high_volume_nodes.append((band_key, band_volume))
        
        # Sort and keep top 5 high volume nodes
        high_volume_nodes = sorted(high_volume_nodes, key=lambda x: x[1], reverse=True)[:5]
        profile['high_volume_nodes'] = [node[0] for node in high_volume_nodes]
        
        return profile
    
    def _detect_absorption(self, df: pd.DataFrame) -> bool:
        """
        📊 ABSORPTION VOLUME: Detect institutional accumulation/distribution
        
        Large volume + small price movement = absorption
        Returns True if absorption detected in last 10 candles
        """
        recent = df.tail(20)
        avg_volume = recent['volume'].mean()
        
        # Check last 10 candles for absorption
        for idx, candle in recent.tail(10).iterrows():
            body_size = abs(candle['close'] - candle['open']) / candle['open']
            
            # Large volume (2x average) + small body (<0.5%)
            if candle['volume'] > avg_volume * 2.0 and body_size < 0.005:
                logger.info(f"🔍 Absorption detected: Volume {candle['volume']:.0f} vs avg {avg_volume:.0f}, body {body_size*100:.2f}%")
                return True
        
        return False
    
    def _generate_analysis(self, current_price: float, 
                          nearest_support: Optional[float],
                          nearest_resistance: Optional[float],
                          volume_delta: float = 0,
                          absorption_detected: bool = False) -> str:
        """Generate human-readable analysis with volume context"""
        base_analysis = ""
        
        if nearest_support and nearest_resistance:
            support_distance = ((current_price - nearest_support) / current_price) * 100
            resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
            
            base_analysis = (
                f"Price at ${current_price:.2f}. "
                f"Support: ${nearest_support:.2f} (-{support_distance:.2f}%), "
                f"Resistance: ${nearest_resistance:.2f} (+{resistance_distance:.2f}%)"
            )
        elif nearest_support:
            support_distance = ((current_price - nearest_support) / current_price) * 100
            base_analysis = f"Price at ${current_price:.2f}. Support: ${nearest_support:.2f} (-{support_distance:.2f}%)"
        elif nearest_resistance:
            resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
            base_analysis = f"Price at ${current_price:.2f}. Resistance: ${nearest_resistance:.2f} (+{resistance_distance:.2f}%)"
        else:
            base_analysis = f"Price at ${current_price:.2f}. No clear support/resistance nearby."
        
        # Add volume context
        if volume_delta > 0:
            base_analysis += f" 📊 Buying pressure (Δ: +{volume_delta:.0f})"
        elif volume_delta < 0:
            base_analysis += f" 📊 Selling pressure (Δ: {volume_delta:.0f})"
        
        if absorption_detected:
            base_analysis += " 🔍 Institutional absorption detected"
        
        return base_analysis
    
    def get_entry_quality(self, current_price: float, direction: str,
                         order_flow_data: Dict) -> Dict:
        """
        Assess entry quality based on proximity to order flow zones
        
        Returns:
            Dict with quality score and reasoning
        """
        nearest_support = order_flow_data.get('nearest_support')
        nearest_resistance = order_flow_data.get('nearest_resistance')
        
        if direction == 'long':
            # For longs, entering near support is better
            if nearest_support:
                distance_pct = ((current_price - nearest_support) / current_price) * 100
                
                if distance_pct < 1:
                    return {
                        'quality': 'excellent',
                        'score': 95,
                        'reason': f'Entering very close to support (${nearest_support:.2f}), optimal risk/reward'
                    }
                elif distance_pct < 2:
                    return {
                        'quality': 'good',
                        'score': 75,
                        'reason': f'Entering near support (${nearest_support:.2f}), good risk/reward'
                    }
                elif distance_pct < 5:
                    return {
                        'quality': 'acceptable',
                        'score': 55,
                        'reason': f'Reasonable distance from support (${nearest_support:.2f})'
                    }
                else:
                    return {
                        'quality': 'poor',
                        'score': 30,
                        'reason': f'Too far from support (${nearest_support:.2f}), poor risk/reward'
                    }
        
        elif direction == 'short':
            # For shorts, entering near resistance is better
            if nearest_resistance:
                distance_pct = ((nearest_resistance - current_price) / current_price) * 100
                
                if distance_pct < 1:
                    return {
                        'quality': 'excellent',
                        'score': 95,
                        'reason': f'Entering very close to resistance (${nearest_resistance:.2f}), optimal risk/reward'
                    }
                elif distance_pct < 2:
                    return {
                        'quality': 'good',
                        'score': 75,
                        'reason': f'Entering near resistance (${nearest_resistance:.2f}), good risk/reward'
                    }
                elif distance_pct < 5:
                    return {
                        'quality': 'acceptable',
                        'score': 55,
                        'reason': f'Reasonable distance from resistance (${nearest_resistance:.2f})'
                    }
                else:
                    return {
                        'quality': 'poor',
                        'score': 30,
                        'reason': f'Too far from resistance (${nearest_resistance:.2f}), poor risk/reward'
                    }
        
        return {
            'quality': 'unknown',
            'score': 50,
            'reason': 'Insufficient order flow data'
        }


# Global instance
order_flow_analyzer = OrderFlowAnalyzer()

