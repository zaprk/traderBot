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
            Dict with liquidity pools, order blocks, and key levels
        """
        if len(df) < self.lookback_period:
            return {
                'liquidity_pools': [],
                'order_blocks': [],
                'key_levels': [],
                'analysis': 'Insufficient data'
            }
        
        # 1. Find liquidity pools (stop hunt zones)
        liquidity_pools = self._find_liquidity_pools(df)
        
        # 2. Identify order blocks (institutional zones)
        order_blocks = self._identify_order_blocks(df)
        
        # 3. Detect key support/resistance levels
        key_levels = self._find_key_levels(df)
        
        # 4. Current price context
        current_price = df.iloc[-1]['close']
        nearest_support, nearest_resistance = self._find_nearest_levels(
            current_price, liquidity_pools, order_blocks, key_levels
        )
        
        return {
            'liquidity_pools': liquidity_pools,
            'order_blocks': order_blocks,
            'key_levels': key_levels,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'analysis': self._generate_analysis(
                current_price, nearest_support, nearest_resistance
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
    
    def _identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify order blocks (institutional entry/exit zones)
        
        Order blocks are strong impulse moves followed by retracement
        """
        blocks = []
        recent = df.tail(self.lookback_period)
        
        for i in range(1, len(recent) - 1):
            current = recent.iloc[i]
            previous = recent.iloc[i-1]
            next_candle = recent.iloc[i+1]
            
            # Calculate move size
            move_size = abs(current['close'] - current['open'])
            avg_size = (recent['high'] - recent['low']).mean()
            
            # Is this an impulse candle? (large move with volume)
            if move_size > avg_size * 1.5:
                # Bullish order block (strong up move)
                if current['close'] > current['open']:
                    # Zone is from open to low of impulse candle
                    blocks.append({
                        'type': 'support',
                        'price_low': current['low'],
                        'price_high': current['open'],
                        'category': 'bullish_order_block',
                        'description': f'Bullish OB: ${current["low"]:.2f}-${current["open"]:.2f}',
                        'strength': 'strong' if move_size > avg_size * 2 else 'moderate'
                    })
                
                # Bearish order block (strong down move)
                else:
                    # Zone is from open to high of impulse candle
                    blocks.append({
                        'type': 'resistance',
                        'price_low': current['open'],
                        'price_high': current['high'],
                        'category': 'bearish_order_block',
                        'description': f'Bearish OB: ${current["open"]:.2f}-${current["high"]:.2f}',
                        'strength': 'strong' if move_size > avg_size * 2 else 'moderate'
                    })
        
        # Keep only most recent 5 of each type
        bullish_blocks = [b for b in blocks if 'bullish' in b['category']][-5:]
        bearish_blocks = [b for b in blocks if 'bearish' in b['category']][-5:]
        
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
    
    def _generate_analysis(self, current_price: float, 
                          nearest_support: Optional[float],
                          nearest_resistance: Optional[float]) -> str:
        """Generate human-readable analysis"""
        if nearest_support and nearest_resistance:
            support_distance = ((current_price - nearest_support) / current_price) * 100
            resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
            
            return (
                f"Price at ${current_price:.2f}. "
                f"Nearest support: ${nearest_support:.2f} (-{support_distance:.2f}%), "
                f"Nearest resistance: ${nearest_resistance:.2f} (+{resistance_distance:.2f}%)"
            )
        elif nearest_support:
            support_distance = ((current_price - nearest_support) / current_price) * 100
            return f"Price at ${current_price:.2f}. Nearest support: ${nearest_support:.2f} (-{support_distance:.2f}%)"
        elif nearest_resistance:
            resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
            return f"Price at ${current_price:.2f}. Nearest resistance: ${nearest_resistance:.2f} (+{resistance_distance:.2f}%)"
        else:
            return f"Price at ${current_price:.2f}. No clear support/resistance nearby."
    
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

