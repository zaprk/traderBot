"""
Database models and session management using SQLAlchemy
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
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


class Database:
    """Database manager"""
    
    def __init__(self, db_url: str = None):
        # Use persistent storage when deployed
        if db_url is None:
            db_dir = os.environ.get('DB_DIR', '/app/data')
            os.makedirs(db_dir, exist_ok=True)
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
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
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
                setting.updated_at = datetime.utcnow()
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


# Global database instance
db = Database()


