"""
Trade Manager module
Handles position sizing, risk validation, and order execution (paper/live)
"""
import ccxt
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TradeManager:
    """Manages trade execution with risk controls"""
    
    def __init__(self, exchange: ccxt.Exchange, config: Dict):
        self.exchange = exchange
        self.paper_mode = config.get('paper_mode', True)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.max_exposure = config.get('max_exposure', 0.20)
        self.min_rr = config.get('min_rr', 1.3)
        self.max_allowable_stop_pct = config.get('max_allowable_stop_pct', 0.10)
        self.max_open_positions = config.get('max_open_positions', 3)
        self.max_daily_loss = config.get('max_daily_loss', 0.10)
        
        # Track positions and daily loss
        self.open_positions = {}
        self.daily_loss = 0.0
        self.starting_balance = 0.0
    
    def compute_position_size(self, balance_usd: float, risk_pct: float, 
                             entry: float, stop: float) -> float:
        """
        Compute position size based on risk
        
        Args:
            balance_usd: Current balance in USD
            risk_pct: Risk percentage per trade (e.g., 0.02 for 2%)
            entry: Entry price
            stop: Stop loss price
        
        Returns:
            Position size in base asset units
        """
        risk_amount = balance_usd * risk_pct
        dollar_risk_per_unit = abs(entry - stop)
        
        if dollar_risk_per_unit == 0:
            logger.error("Stop loss equals entry price, cannot size position")
            return 0.0
        
        units = risk_amount / dollar_risk_per_unit
        return units
    
    def validate_trade(self, balance: float, side: str, entry: float, 
                      stop: float, take_profit: float, units: float) -> Tuple[bool, str]:
        """
        Validate trade against risk parameters
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check position size
        if units <= 0:
            return False, "Position size must be positive"
        
        # Check exposure
        position_value = units * entry
        if position_value > balance * self.max_exposure:
            return False, f"Position value ${position_value:.2f} exceeds max exposure {self.max_exposure*100}%"
        
        # Check stop distance
        stop_distance_pct = abs(entry - stop) / entry
        if stop_distance_pct > self.max_allowable_stop_pct:
            return False, f"Stop distance {stop_distance_pct*100:.2f}% exceeds max {self.max_allowable_stop_pct*100}%"
        
        # Check risk:reward ratio
        risk = abs(entry - stop)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return False, "Risk is zero"
        
        rr_ratio = reward / risk
        if rr_ratio < self.min_rr:
            return False, f"R:R ratio {rr_ratio:.2f} below minimum {self.min_rr}"
        
        # Check open positions limit
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"
        
        # Check daily loss limit
        max_loss = balance * self.risk_per_trade
        if self.daily_loss + max_loss > self.starting_balance * self.max_daily_loss:
            return False, f"Daily loss limit would be exceeded"
        
        return True, "Trade validated"
    
    def execute_trade(self, symbol: str, side: str, entry: float, stop: float, 
                     take_profit: float, units: float, balance: float,
                     llm_response: Dict) -> Dict:
        """
        Execute trade (paper or live)
        
        Args:
            symbol: Trading pair
            side: 'long' or 'short'
            entry: Entry price
            stop: Stop loss price
            take_profit: Take profit price
            units: Position size in units
            balance: Current balance
            llm_response: Full LLM decision for logging
        
        Returns:
            Trade execution result dict
        """
        # Validate first
        is_valid, reason = self.validate_trade(balance, side, entry, stop, take_profit, units)
        
        if not is_valid:
            logger.warning(f"Trade validation failed: {reason}")
            return {
                'success': False,
                'reason': reason,
                'executed': False
            }
        
        # Execute based on mode
        if self.paper_mode:
            return self._execute_paper_trade(symbol, side, entry, stop, take_profit, 
                                            units, balance, llm_response)
        else:
            return self._execute_live_trade(symbol, side, entry, stop, take_profit, 
                                           units, balance, llm_response)
    
    def _execute_paper_trade(self, symbol: str, side: str, entry: float, 
                            stop: float, take_profit: float, units: float,
                            balance: float, llm_response: Dict) -> Dict:
        """Execute simulated paper trade"""
        logger.info(f"[PAPER] Executing {side} {units:.6f} {symbol} @ {entry}")
        
        # Create position record
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit': take_profit,
            'units': units,
            'entry_time': datetime.utcnow().isoformat(),
            'llm_response': llm_response,
            'paper_mode': True,
            'status': 'open'
        }
        
        # Store position
        position_id = f"{symbol}_{datetime.utcnow().timestamp()}"
        self.open_positions[position_id] = position
        
        logger.info(f"[PAPER] Position opened: {position_id}")
        
        return {
            'success': True,
            'executed': True,
            'position_id': position_id,
            'position': position,
            'reason': 'Paper trade executed successfully'
        }
    
    def _execute_live_trade(self, symbol: str, side: str, entry: float, 
                           stop: float, take_profit: float, units: float,
                           balance: float, llm_response: Dict) -> Dict:
        """Execute real live trade"""
        logger.info(f"[LIVE] Executing {side} {units:.6f} {symbol} @ {entry}")
        
        try:
            # Determine order side for exchange
            order_side = 'buy' if side == 'long' else 'sell'
            
            # Place entry order (limit order)
            logger.info(f"Placing {order_side} limit order: {units} @ {entry}")
            entry_order = self.exchange.create_limit_order(
                symbol=symbol,
                side=order_side,
                amount=units,
                price=entry
            )
            
            logger.info(f"Entry order placed: {entry_order['id']}")
            
            # Note: Stop-loss and take-profit order placement depends on exchange support
            # For Kraken, you might need to use conditional orders or monitor manually
            # This is a simplified version
            
            position = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry,
                'stop_loss': stop,
                'take_profit': take_profit,
                'units': units,
                'entry_time': datetime.utcnow().isoformat(),
                'entry_order_id': entry_order['id'],
                'llm_response': llm_response,
                'paper_mode': False,
                'status': 'open'
            }
            
            position_id = f"{symbol}_{entry_order['id']}"
            self.open_positions[position_id] = position
            
            return {
                'success': True,
                'executed': True,
                'position_id': position_id,
                'entry_order': entry_order,
                'position': position,
                'reason': 'Live trade executed successfully'
            }
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            return {
                'success': False,
                'executed': False,
                'reason': f'Insufficient funds: {str(e)}'
            }
        
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            return {
                'success': False,
                'executed': False,
                'reason': f'Exchange error: {str(e)}'
            }
        
        except Exception as e:
            logger.error(f"Unexpected error executing trade: {e}")
            return {
                'success': False,
                'executed': False,
                'reason': f'Error: {str(e)}'
            }
    
    def monitor_positions(self, current_prices: Dict[str, float]) -> list:
        """
        Monitor open positions and check for stop/take profit triggers
        
        Args:
            current_prices: Dict mapping symbol to current price
        
        Returns:
            List of closed positions
        """
        closed_positions = []
        
        for position_id, position in list(self.open_positions.items()):
            symbol = position['symbol']
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            side = position['side']
            stop = position['stop_loss']
            tp = position['take_profit']
            
            # Check if stop or take profit hit
            hit_stop = False
            hit_tp = False
            
            if side == 'long':
                if current_price <= stop:
                    hit_stop = True
                elif current_price >= tp:
                    hit_tp = True
            else:  # short
                if current_price >= stop:
                    hit_stop = True
                elif current_price <= tp:
                    hit_tp = True
            
            if hit_stop or hit_tp:
                exit_reason = 'stop_loss' if hit_stop else 'take_profit'
                closed = self.close_position(position_id, current_price, exit_reason)
                if closed:
                    closed_positions.append(closed)
        
        return closed_positions
    
    def close_position(self, position_id: str, exit_price: float, 
                      reason: str = 'manual') -> Optional[Dict]:
        """
        Close a position and calculate PnL
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            reason: Reason for closing (stop_loss, take_profit, manual)
        
        Returns:
            Closed position dict with PnL
        """
        if position_id not in self.open_positions:
            logger.error(f"Position {position_id} not found")
            return None
        
        position = self.open_positions[position_id]
        entry_price = position['entry_price']
        units = position['units']
        side = position['side']
        
        # Calculate PnL
        if side == 'long':
            pnl_per_unit = exit_price - entry_price
        else:  # short
            pnl_per_unit = entry_price - exit_price
        
        pnl_usd = pnl_per_unit * units
        pnl_pct = (pnl_per_unit / entry_price) * 100
        
        # Update daily loss
        if pnl_usd < 0:
            self.daily_loss += abs(pnl_usd)
        
        # Create closed position record
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.utcnow().isoformat()
        position['pnl_usd'] = pnl_usd
        position['pnl_pct'] = pnl_pct
        position['exit_reason'] = reason
        position['status'] = 'closed'
        
        # Remove from open positions
        del self.open_positions[position_id]
        
        logger.info(f"Position closed: {position_id}, PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")
        
        return position
    
    def get_open_positions(self) -> Dict:
        """Get all open positions"""
        return self.open_positions
    
    def reset_daily_loss(self, starting_balance: float):
        """Reset daily loss counter (call at start of each day)"""
        self.daily_loss = 0.0
        self.starting_balance = starting_balance
        logger.info(f"Daily loss reset. Starting balance: ${starting_balance:.2f}")


