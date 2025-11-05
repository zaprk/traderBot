"""
Backtesting engine for strategy validation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Backtest:
    """Simple backtesting engine"""
    
    def __init__(self, initial_balance: float = 10000, risk_per_trade: float = 0.02):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.trades = []
        self.equity_curve = []
        
    def run_simple_strategy(self, df: pd.DataFrame, strategy_func) -> Dict:
        """
        Run a simple strategy on historical data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            strategy_func: Function that takes a row and returns (action, entry, stop, tp)
        
        Returns:
            Backtest results dictionary
        """
        position = None
        
        for idx, row in df.iterrows():
            # Check if we have an open position
            if position:
                # Check for exit
                if position['side'] == 'long':
                    if row['low'] <= position['stop_loss']:
                        # Stop hit
                        self._close_position(position, position['stop_loss'], 'stop_loss')
                        position = None
                    elif row['high'] >= position['take_profit']:
                        # Take profit hit
                        self._close_position(position, position['take_profit'], 'take_profit')
                        position = None
                else:  # short
                    if row['high'] >= position['stop_loss']:
                        # Stop hit
                        self._close_position(position, position['stop_loss'], 'stop_loss')
                        position = None
                    elif row['low'] <= position['take_profit']:
                        # Take profit hit
                        self._close_position(position, position['take_profit'], 'take_profit')
                        position = None
            
            # Look for new entry if no position
            if not position:
                action, entry, stop, tp = strategy_func(row)
                
                if action in ['long', 'short']:
                    # Calculate position size
                    risk_amount = self.balance * self.risk_per_trade
                    stop_distance = abs(entry - stop)
                    if stop_distance > 0:
                        units = risk_amount / stop_distance
                        
                        position = {
                            'side': action,
                            'entry_price': entry,
                            'stop_loss': stop,
                            'take_profit': tp,
                            'units': units,
                            'entry_time': row['timestamp'] if 'timestamp' in df.columns else idx
                        }
            
            # Record equity
            equity = self.balance
            if position:
                # Add unrealized PnL
                current_price = row['close']
                if position['side'] == 'long':
                    unrealized = (current_price - position['entry_price']) * position['units']
                else:
                    unrealized = (position['entry_price'] - current_price) * position['units']
                equity += unrealized
            
            self.equity_curve.append({
                'timestamp': row['timestamp'] if 'timestamp' in df.columns else idx,
                'equity': equity
            })
        
        # Close any remaining position at last price
        if position:
            last_price = df.iloc[-1]['close']
            self._close_position(position, last_price, 'backtest_end')
        
        return self.get_results()
    
    def _close_position(self, position: Dict, exit_price: float, reason: str):
        """Close a position and update balance"""
        entry = position['entry_price']
        units = position['units']
        side = position['side']
        
        if side == 'long':
            pnl = (exit_price - entry) * units
        else:
            pnl = (entry - exit_price) * units
        
        self.balance += pnl
        
        trade = {
            **position,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry * units)) * 100,
            'exit_reason': reason
        }
        
        self.trades.append(trade)
        logger.debug(f"Backtest trade closed: {side} @ {entry} -> {exit_price}, PnL: ${pnl:.2f}")
    
    def get_results(self) -> Dict:
        """Calculate and return backtest statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_balance': self.balance
            }
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        total_wins = sum([t['pnl'] for t in winning_trades])
        total_losses = abs(sum([t['pnl'] for t in losing_trades]))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate max drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # Calculate Sharpe ratio (simplified)
        returns = [(self.equity_curve[i]['equity'] - self.equity_curve[i-1]['equity']) / 
                  self.equity_curve[i-1]['equity'] 
                  for i in range(1, len(self.equity_curve))]
        
        sharpe_ratio = 0
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe_ratio = (avg_return / std_return) * np.sqrt(252)  # Annualized
        
        total_pnl = self.balance - self.initial_balance
        total_return_pct = (total_pnl / self.initial_balance) * 100
        
        results = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0,
        }
        
        return results
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not equity_values:
            return 0
        
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd


def simple_rsi_strategy(row) -> Tuple[str, float, float, float]:
    """
    Example: Simple RSI-based strategy
    
    Returns:
        (action, entry, stop, take_profit)
    """
    if 'rsi' not in row or pd.isna(row['rsi']):
        return 'none', 0, 0, 0
    
    close = row['close']
    atr = row.get('atr', close * 0.02)  # Default to 2% if no ATR
    
    # Long when RSI oversold
    if row['rsi'] < 30:
        entry = close
        stop = entry - (2 * atr)
        tp = entry + (3 * atr)
        return 'long', entry, stop, tp
    
    # Short when RSI overbought
    elif row['rsi'] > 70:
        entry = close
        stop = entry + (2 * atr)
        tp = entry - (3 * atr)
        return 'short', entry, stop, tp
    
    return 'none', 0, 0, 0


def monte_carlo_simulation(win_rate: float, avg_win: float, avg_loss: float, 
                          num_trades: int = 100, num_simulations: int = 1000,
                          initial_balance: float = 10000) -> Dict:
    """
    Run Monte Carlo simulation to estimate risk of ruin
    
    Args:
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)
        num_trades: Number of trades to simulate
        num_simulations: Number of simulation runs
        initial_balance: Starting balance
    
    Returns:
        Simulation results dictionary
    """
    logger.info(f"Running Monte Carlo: {num_simulations} simulations, {num_trades} trades each")
    
    final_balances = []
    ruin_count = 0
    
    for _ in range(num_simulations):
        balance = initial_balance
        
        for _ in range(num_trades):
            # Random outcome based on win rate
            if np.random.random() < win_rate:
                balance += avg_win
            else:
                balance -= avg_loss
            
            # Check for ruin (balance drops below 20% of initial)
            if balance < initial_balance * 0.2:
                ruin_count += 1
                break
        
        final_balances.append(balance)
    
    final_balances = np.array(final_balances)
    
    results = {
        'mean_final_balance': float(np.mean(final_balances)),
        'median_final_balance': float(np.median(final_balances)),
        'std_final_balance': float(np.std(final_balances)),
        'min_final_balance': float(np.min(final_balances)),
        'max_final_balance': float(np.max(final_balances)),
        'risk_of_ruin': ruin_count / num_simulations,
        'profit_probability': np.sum(final_balances > initial_balance) / num_simulations
    }
    
    logger.info(f"Monte Carlo results: Mean=${results['mean_final_balance']:.2f}, "
               f"Risk of Ruin={results['risk_of_ruin']*100:.2f}%")
    
    return results


