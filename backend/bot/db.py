"""
Database models and session management using SQLAlchemy
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone, timedelta
import json
import logging
import os

logger = logging.getLogger(__name__)

Base = declarative_base()


class Trade(Base):
    """Trade history table"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    units = Column(Float, nullable=False)
    pnl_usd = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    llm_reason = Column(Text, nullable=True)
    llm_raw = Column(Text, nullable=True)  # Full JSON response
    paper_mode = Column(Boolean, default=True)
    exit_reason = Column(String(20), nullable=True)  # 'stop_loss', 'take_profit', 'manual'
    status = Column(String(20), default='open')  # 'open', 'closed'
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'units': self.units,
            'pnl_usd': self.pnl_usd,
            'pnl_pct': self.pnl_pct,
            'confidence': self.confidence,
            'llm_reason': self.llm_reason,
            'paper_mode': self.paper_mode,
            'exit_reason': self.exit_reason,
            'status': self.status
        }


class MetricSnapshot(Base):
    """Daily metrics snapshots"""
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=True)
    cumulative_return = Column(Float, nullable=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'balance': self.balance,
            'equity': self.equity,
            'max_drawdown': self.max_drawdown,
            'cumulative_return': self.cumulative_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        }


class SystemSettings(Base):
    """Runtime configuration settings"""
    __tablename__ = 'settings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(50), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class HistoricalLevel(Base):
    """Historical support/resistance levels across timeframes"""
    __tablename__ = 'historical_levels'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    level_type = Column(String(20), nullable=False)  # 'support', 'resistance', 'order_block'
    price = Column(Float, nullable=False)
    strength = Column(String(20), nullable=False)  # 'strong', 'moderate', 'weak'
    timeframe = Column(String(10), nullable=False)  # 'daily', 'weekly', 'monthly', 'intraday'
    first_detected = Column(DateTime, nullable=False)
    last_tested = Column(DateTime, nullable=True)
    test_count = Column(Integer, default=0)  # How many times price tested this level
    broken = Column(Boolean, default=False)  # Has this level been broken?
    broken_at = Column(DateTime, nullable=True)
    meta_info = Column(Text, nullable=True)  # JSON: volume, distance from current price, etc. (renamed from metadata to avoid SQLAlchemy conflict)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'level_type': self.level_type,
            'price': self.price,
            'strength': self.strength,
            'timeframe': self.timeframe,
            'first_detected': self.first_detected.isoformat() if self.first_detected else None,
            'last_tested': self.last_tested.isoformat() if self.last_tested else None,
            'test_count': self.test_count,
            'broken': self.broken,
            'broken_at': self.broken_at.isoformat() if self.broken_at else None,
            'metadata': json.loads(self.meta_info) if self.meta_info else {}
        }


class OrderBlockHistory(Base):
    """Historical order blocks with mitigation tracking"""
    __tablename__ = 'order_blocks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    block_type = Column(String(20), nullable=False)  # 'bullish', 'bearish'
    price_low = Column(Float, nullable=False)
    price_high = Column(Float, nullable=False)
    strength_score = Column(Integer, nullable=False)  # 0-100
    volume_ratio = Column(Float, nullable=False)
    detected_at = Column(DateTime, nullable=False)
    mitigated = Column(Boolean, default=False)  # Has price returned to fill this block?
    mitigated_at = Column(DateTime, nullable=True)
    still_valid = Column(Boolean, default=True)  # False if broken or too old
    candle_index = Column(Integer, nullable=True)  # Position in historical data
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'block_type': self.block_type,
            'price_low': self.price_low,
            'price_high': self.price_high,
            'strength_score': self.strength_score,
            'volume_ratio': self.volume_ratio,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
            'mitigated': self.mitigated,
            'mitigated_at': self.mitigated_at.isoformat() if self.mitigated_at else None,
            'still_valid': self.still_valid
        }


class MarketRegimeHistory(Base):
    """Historical market regime snapshots (trending vs ranging)"""
    __tablename__ = 'market_regime_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    regime = Column(String(20), nullable=False)  # 'trending', 'ranging', 'volatile'
    adx = Column(Float, nullable=True)
    trend_direction = Column(String(10), nullable=True)  # 'up', 'down', 'neutral'
    volatility = Column(String(20), nullable=True)  # 'high', 'normal', 'low'
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'regime': self.regime,
            'adx': self.adx,
            'trend_direction': self.trend_direction,
            'volatility': self.volatility
        }


class AnalysisSnapshot(Base):
    """Historical analysis snapshots for comparison"""
    __tablename__ = 'analysis_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    action = Column(String(10), nullable=False)  # 'long', 'short', 'none'
    confidence = Column(Float, nullable=True)
    convergence_score = Column(Float, nullable=True)
    trend = Column(String(20), nullable=True)  # 'uptrend', 'downtrend', 'neutral'
    key_reason = Column(Text, nullable=True)  # Main reasoning
    indicators_snapshot = Column(Text, nullable=True)  # JSON of key indicators
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'price': self.price,
            'action': self.action,
            'confidence': self.confidence,
            'convergence_score': self.convergence_score,
            'trend': self.trend,
            'key_reason': self.key_reason,
            'indicators_snapshot': json.loads(self.indicators_snapshot) if self.indicators_snapshot else {}
        }


class Database:
    """Database manager"""
    
    def __init__(self, db_url: str = None):
        # Use persistent storage when deployed
        if db_url is None:
            # Try multiple locations in order of preference
            db_locations = [
                os.environ.get('DB_DIR'),  # User-specified
                '/data',  # Railway volume mount
                '/app/data',  # App directory
                '/tmp',  # Fallback to temp (not persistent but works)
                '.'  # Current directory (last resort)
            ]
            
            db_dir = None
            for loc in db_locations:
                if loc is None:
                    continue
                try:
                    os.makedirs(loc, exist_ok=True)
                    # Test write permissions
                    test_file = os.path.join(loc, '.write_test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    db_dir = loc
                    logger.info(f"Using database directory: {db_dir}")
                    break
                except Exception as e:
                    logger.warning(f"Cannot use {loc}: {e}")
                    continue
            
            if db_dir is None:
                raise Exception("No writable directory found for database")
            
            db_url = f"sqlite:///{db_dir}/trades.db"
        
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def add_trade(self, position: dict) -> int:
        """
        Add a trade to database
        
        Args:
            position: Position dictionary from trade_manager
        
        Returns:
            Trade ID
        """
        session = self.get_session()
        try:
            # Parse timestamps
            entry_time = datetime.fromisoformat(position['entry_time'].replace('Z', '+00:00'))
            exit_time = None
            if position.get('exit_time'):
                exit_time = datetime.fromisoformat(position['exit_time'].replace('Z', '+00:00'))
            
            # Extract LLM info
            llm_response = position.get('llm_response', {})
            llm_reason = llm_response.get('reason', '')
            llm_raw = json.dumps(llm_response) if llm_response else ''
            confidence = llm_response.get('confidence', None)
            
            trade = Trade(
                symbol=position['symbol'],
                side=position['side'],
                entry_price=position['entry_price'],
                stop_loss=position['stop_loss'],
                take_profit=position['take_profit'],
                exit_price=position.get('exit_price'),
                entry_time=entry_time,
                exit_time=exit_time,
                units=position['units'],
                pnl_usd=position.get('pnl_usd'),
                pnl_pct=position.get('pnl_pct'),
                confidence=confidence,
                llm_reason=llm_reason,
                llm_raw=llm_raw,
                paper_mode=position.get('paper_mode', True),
                exit_reason=position.get('exit_reason'),
                status=position.get('status', 'closed')
            )
            
            session.add(trade)
            session.commit()
            trade_id = trade.id
            logger.info(f"Trade saved to database: ID={trade_id}")
            return trade_id
            
        except Exception as e:
            logger.error(f"Error adding trade to database: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_all_trades(self, limit: int = 100) -> list:
        """Get all trades (most recent first)"""
        session = self.get_session()
        try:
            trades = session.query(Trade).order_by(Trade.entry_time.desc()).limit(limit).all()
            return [trade.to_dict() for trade in trades]
        finally:
            session.close()
    
    def get_open_trades(self) -> list:
        """Get all open trades"""
        session = self.get_session()
        try:
            trades = session.query(Trade).filter(Trade.status == 'open').all()
            return [trade.to_dict() for trade in trades]
        finally:
            session.close()
    
    def update_trade(self, trade_id: int, updates: dict):
        """Update a trade record"""
        session = self.get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                for key, value in updates.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                session.commit()
                logger.info(f"Trade {trade_id} updated")
            else:
                logger.warning(f"Trade {trade_id} not found")
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def add_metric_snapshot(self, balance: float, equity: float, 
                           max_drawdown: float = None, cumulative_return: float = None):
        """Add daily metric snapshot"""
        session = self.get_session()
        try:
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate trade stats
            trades = session.query(Trade).filter(Trade.status == 'closed').all()
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl_usd and t.pnl_usd > 0])
            losing_trades = len([t for t in trades if t.pnl_usd and t.pnl_usd < 0])
            
            # Check if snapshot exists for today
            existing = session.query(MetricSnapshot).filter(MetricSnapshot.date == today).first()
            
            if existing:
                existing.balance = balance
                existing.equity = equity
                existing.max_drawdown = max_drawdown
                existing.cumulative_return = cumulative_return
                existing.total_trades = total_trades
                existing.winning_trades = winning_trades
                existing.losing_trades = losing_trades
            else:
                snapshot = MetricSnapshot(
                    date=today,
                    balance=balance,
                    equity=equity,
                    max_drawdown=max_drawdown,
                    cumulative_return=cumulative_return,
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades
                )
                session.add(snapshot)
            
            session.commit()
            logger.info("Metric snapshot saved")
            
        except Exception as e:
            logger.error(f"Error adding metric snapshot: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_metrics(self, days: int = 30) -> list:
        """Get metric snapshots for last N days"""
        session = self.get_session()
        try:
            snapshots = session.query(MetricSnapshot).order_by(
                MetricSnapshot.date.desc()
            ).limit(days).all()
            return [s.to_dict() for s in snapshots]
        finally:
            session.close()
    
    def get_setting(self, key: str) -> str:
        """Get a setting value"""
        session = self.get_session()
        try:
            setting = session.query(SystemSettings).filter(SystemSettings.key == key).first()
            return setting.value if setting else None
        finally:
            session.close()
    
    def set_setting(self, key: str, value: str):
        """Set a setting value"""
        session = self.get_session()
        try:
            setting = session.query(SystemSettings).filter(SystemSettings.key == key).first()
            if setting:
                setting.value = value
                setting.updated_at = datetime.now(timezone.utc)
            else:
                setting = SystemSettings(key=key, value=value)
                session.add(setting)
            session.commit()
        except Exception as e:
            logger.error(f"Error setting value: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    # ========================================
    # MARKET MEMORY METHODS
    # ========================================
    
    def add_historical_level(self, symbol: str, level_type: str, price: float, 
                            strength: str, timeframe: str, metadata: dict = None):
        """Add or update a historical support/resistance level"""
        session = self.get_session()
        try:
            # Check if similar level exists (within 0.5%)
            tolerance = price * 0.005
            existing = session.query(HistoricalLevel).filter(
                HistoricalLevel.symbol == symbol,
                HistoricalLevel.level_type == level_type,
                HistoricalLevel.timeframe == timeframe,
                HistoricalLevel.price >= price - tolerance,
                HistoricalLevel.price <= price + tolerance,
                HistoricalLevel.broken == False
            ).first()
            
            if existing:
                # Update existing level
                existing.last_tested = datetime.now(timezone.utc)
                existing.test_count += 1
                existing.strength = strength  # Update strength
                if metadata:
                    existing.meta_info = json.dumps(metadata)
            else:
                # Create new level
                level = HistoricalLevel(
                    symbol=symbol,
                    level_type=level_type,
                    price=price,
                    strength=strength,
                    timeframe=timeframe,
                    first_detected=datetime.now(timezone.utc),
                    test_count=0,
                    meta_info=json.dumps(metadata) if metadata else None
                )
                session.add(level)
            
            session.commit()
        except Exception as e:
            logger.error(f"Error adding historical level: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_historical_levels(self, symbol: str, current_price: float, 
                             max_distance_pct: float = 0.05, limit: int = 10):
        """Get relevant historical levels near current price"""
        session = self.get_session()
        try:
            tolerance = current_price * max_distance_pct
            levels = session.query(HistoricalLevel).filter(
                HistoricalLevel.symbol == symbol,
                HistoricalLevel.broken == False,
                HistoricalLevel.price >= current_price - tolerance,
                HistoricalLevel.price <= current_price + tolerance
            ).order_by(HistoricalLevel.test_count.desc()).limit(limit).all()
            
            return [level.to_dict() for level in levels]
        finally:
            session.close()
    
    def mark_level_broken(self, symbol: str, price: float, level_type: str):
        """Mark a level as broken when price passes through"""
        session = self.get_session()
        try:
            tolerance = price * 0.005
            levels = session.query(HistoricalLevel).filter(
                HistoricalLevel.symbol == symbol,
                HistoricalLevel.level_type == level_type,
                HistoricalLevel.price >= price - tolerance,
                HistoricalLevel.price <= price + tolerance,
                HistoricalLevel.broken == False
            ).all()
            
            for level in levels:
                level.broken = True
                level.broken_at = datetime.now(timezone.utc)
            
            session.commit()
            return len(levels)
        except Exception as e:
            logger.error(f"Error marking level as broken: {e}")
            session.rollback()
            return 0
        finally:
            session.close()
    
    def add_order_block(self, symbol: str, block_type: str, price_low: float, 
                       price_high: float, strength_score: int, volume_ratio: float):
        """Add a new order block to history"""
        session = self.get_session()
        try:
            block = OrderBlockHistory(
                symbol=symbol,
                block_type=block_type,
                price_low=price_low,
                price_high=price_high,
                strength_score=strength_score,
                volume_ratio=volume_ratio,
                detected_at=datetime.now(timezone.utc),
                mitigated=False,
                still_valid=True
            )
            session.add(block)
            session.commit()
            return block.id
        except Exception as e:
            logger.error(f"Error adding order block: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def get_valid_order_blocks(self, symbol: str, current_price: float, 
                               max_age_days: int = 30):
        """Get valid (unfilled) order blocks near current price"""
        session = self.get_session()
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            
            blocks = session.query(OrderBlockHistory).filter(
                OrderBlockHistory.symbol == symbol,
                OrderBlockHistory.still_valid == True,
                OrderBlockHistory.mitigated == False,
                OrderBlockHistory.detected_at >= cutoff_date
            ).order_by(OrderBlockHistory.strength_score.desc()).all()
            
            return [block.to_dict() for block in blocks]
        finally:
            session.close()
    
    def mark_order_block_mitigated(self, block_id: int):
        """Mark an order block as mitigated (filled)"""
        session = self.get_session()
        try:
            block = session.query(OrderBlockHistory).filter(OrderBlockHistory.id == block_id).first()
            if block:
                block.mitigated = True
                block.mitigated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error marking order block as mitigated: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def add_regime_snapshot(self, symbol: str, regime: str, adx: float = None,
                           trend_direction: str = None, volatility: str = None):
        """Add a market regime snapshot"""
        session = self.get_session()
        try:
            snapshot = MarketRegimeHistory(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                regime=regime,
                adx=adx,
                trend_direction=trend_direction,
                volatility=volatility
            )
            session.add(snapshot)
            session.commit()
        except Exception as e:
            logger.error(f"Error adding regime snapshot: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_regime_history(self, symbol: str, hours: int = 48):
        """Get recent regime history for a symbol"""
        session = self.get_session()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            snapshots = session.query(MarketRegimeHistory).filter(
                MarketRegimeHistory.symbol == symbol,
                MarketRegimeHistory.timestamp >= cutoff
            ).order_by(MarketRegimeHistory.timestamp.desc()).all()
            
            return [s.to_dict() for s in snapshots]
        finally:
            session.close()
    
    def add_analysis_snapshot(self, symbol: str, price: float, action: str, 
                             confidence: float = None, convergence_score: float = None,
                             trend: str = None, key_reason: str = None, 
                             indicators_snapshot: dict = None):
        """Add an analysis snapshot for historical comparison"""
        session = self.get_session()
        try:
            snapshot = AnalysisSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=price,
                action=action,
                confidence=confidence,
                convergence_score=convergence_score,
                trend=trend,
                key_reason=key_reason,
                indicators_snapshot=json.dumps(indicators_snapshot) if indicators_snapshot else None
            )
            session.add(snapshot)
            session.commit()
        except Exception as e:
            logger.error(f"Error adding analysis snapshot: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_recent_analysis(self, symbol: str, hours: int = 24):
        """Get recent analysis snapshots for comparison"""
        session = self.get_session()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            snapshots = session.query(AnalysisSnapshot).filter(
                AnalysisSnapshot.symbol == symbol,
                AnalysisSnapshot.timestamp >= cutoff
            ).order_by(AnalysisSnapshot.timestamp.desc()).all()
            
            return [s.to_dict() for s in snapshots]
        finally:
            session.close()
    
    def get_recent_performance(self, symbol: str, days: int = 7):
        """Get recent trade performance for a symbol"""
        session = self.get_session()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            trades = session.query(Trade).filter(
                Trade.symbol == symbol,
                Trade.entry_time >= cutoff,
                Trade.status == 'closed'
            ).all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0
                }
            
            wins = [t for t in trades if t.pnl_usd and t.pnl_usd > 0]
            losses = [t for t in trades if t.pnl_usd and t.pnl_usd < 0]
            total_pnl = sum(t.pnl_usd for t in trades if t.pnl_usd)
            
            return {
                'total_trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'avg_pnl': total_pnl / len(trades) if trades else 0,
                'total_pnl': total_pnl
            }
        finally:
            session.close()


# Global database instance (with error handling)
try:
    db = Database()
    logger.info("✅ Database initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize database: {e}")
    logger.error("App will start but database features will be unavailable")
    # Create a dummy db object that won't crash the app
    db = None


