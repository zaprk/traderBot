"""
Configuration management using Pydantic settings
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # DeepSeek API
    deepseek_api_key: str = ""
    
    # Kraken Exchange
    kraken_api_key: str = ""
    kraken_secret_key: str = ""
    
    # Trading Mode
    paper_mode: bool = True
    initial_balance: float = 10000.0
    
    # Risk Management
    risk_per_trade: float = 0.02  # 2% default (PROFESSIONAL STANDARD)
    # ⚠️ Advanced users can set higher (up to 7%), but understand the risks:
    # - 2% risk = Professional standard, stable growth
    # - 7% risk = Aggressive, 3 losses = 21% drawdown
    max_daily_loss: float = 0.10
    max_open_positions: int = 3
    # Correlation & Regime Filters
    enable_regime_filter: bool = True  # Skip ranging markets (ADX < 20)
    enable_correlation_filter: bool = True  # Max 2 correlated positions
    min_adx_threshold: float = 20.0  # Minimum ADX for trading
    max_exposure: float = 0.20
    min_rr: float = 1.3
    max_allowable_stop_pct: float = 0.10
    
    # Trading Configuration
    base_currency: str = "USDT"
    coins: str = "BTC,ETH,SOL,DOGE,TON"
    timeframes: str = "5m,15m,1h,4h"
    
    # API Configuration
    backend_url: str = "http://127.0.0.1:8000"
    frontend_url: str = "http://localhost:5173"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_coins_list(self) -> List[str]:
        return [coin.strip() for coin in self.coins.split(",")]
    
    def get_timeframes_list(self) -> List[str]:
        return [tf.strip() for tf in self.timeframes.split(",")]


settings = Settings()


