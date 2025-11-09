"""
Correlation Filter Module
Prevents opening multiple positions in highly correlated assets
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class CorrelationFilter:
    """
    Filters trade signals to avoid concentrated risk in correlated assets
    
    Known correlation groups:
    - BTC/ETH: High correlation (~0.85)
    - BTC/SOL: High correlation (~0.75)
    - ETH/SOL: High correlation (~0.70)
    - DOGE/SHIB: Very high correlation (~0.90)
    """
    
    def __init__(self, max_correlated_positions: int = 2):
        """
        Args:
            max_correlated_positions: Maximum positions allowed in same correlation group
        """
        self.max_correlated_positions = max_correlated_positions
        
        # Define correlation groups (assets that tend to move together)
        self.correlation_groups = {
            'major_alts': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT'],
            'meme_coins': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT'],
            'defi_tokens': ['UNI/USDT', 'AAVE/USDT', 'LINK/USDT'],
            'layer1': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'DOT/USDT']
        }
    
    def get_correlation_group(self, symbol: str) -> str:
        """
        Identify which correlation group a symbol belongs to
        
        Returns:
            Group name or 'uncorrelated'
        """
        for group_name, members in self.correlation_groups.items():
            if symbol in members:
                return group_name
        return 'uncorrelated'
    
    def check_new_signal(self, symbol: str, direction: str, open_positions: List[Dict]) -> tuple:
        """
        Check if a new trade signal should be executed given existing positions
        
        Args:
            symbol: New symbol to trade
            direction: 'long' or 'short'
            open_positions: List of currently open positions
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if not open_positions:
            return True, "No existing positions"
        
        # Get correlation group of new signal
        new_group = self.get_correlation_group(symbol)
        
        if new_group == 'uncorrelated':
            return True, f"{symbol} is uncorrelated with existing positions"
        
        # Count existing positions in same correlation group and direction
        same_group_count = 0
        same_group_symbols = []
        
        for pos in open_positions:
            pos_symbol = pos.get('symbol')
            pos_side = pos.get('side')
            pos_group = self.get_correlation_group(pos_symbol)
            
            # Check if same group AND same direction
            if pos_group == new_group and pos_side == direction:
                same_group_count += 1
                same_group_symbols.append(pos_symbol)
        
        # Check limit
        if same_group_count >= self.max_correlated_positions:
            return False, (
                f"❌ CORRELATION LIMIT: {same_group_count} {direction} positions already open in {new_group} group "
                f"({', '.join(same_group_symbols)}). Max allowed: {self.max_correlated_positions}"
            )
        
        return True, (
            f"✅ {symbol} allowed: {same_group_count}/{self.max_correlated_positions} {direction} positions "
            f"in {new_group} group"
        )
    
    def analyze_portfolio_risk(self, open_positions: List[Dict]) -> Dict:
        """
        Analyze concentration risk in current portfolio
        
        Returns:
            Dict with risk metrics
        """
        if not open_positions:
            return {
                'total_positions': 0,
                'correlation_groups': {},
                'concentration_risk': 'LOW',
                'warning': None
            }
        
        # Group positions by correlation group
        group_positions = {}
        for pos in open_positions:
            symbol = pos.get('symbol')
            side = pos.get('side')
            group = self.get_correlation_group(symbol)
            
            key = f"{group}_{side}"
            if key not in group_positions:
                group_positions[key] = []
            group_positions[key].append(symbol)
        
        # Determine concentration risk
        max_in_group = max(len(symbols) for symbols in group_positions.values()) if group_positions else 0
        
        if max_in_group >= 3:
            risk_level = 'HIGH'
            warning = f"⚠️ {max_in_group} correlated positions detected - diversify!"
        elif max_in_group == 2:
            risk_level = 'MODERATE'
            warning = f"⚠️ {max_in_group} correlated positions - at limit"
        else:
            risk_level = 'LOW'
            warning = None
        
        return {
            'total_positions': len(open_positions),
            'correlation_groups': group_positions,
            'max_in_group': max_in_group,
            'concentration_risk': risk_level,
            'warning': warning
        }


# Global instance
correlation_filter = CorrelationFilter(max_correlated_positions=2)

