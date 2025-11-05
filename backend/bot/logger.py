"""
Logging utilities for trades and system events
"""
import logging
import csv
import os
from datetime import datetime
from typing import Dict, List
import json

# Configure logging
LOG_DIR = "logs"
CSV_FILE = "logs/trades.csv"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def setup_logging(log_level=logging.INFO):
    """Configure Python logging"""
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler (rotating)
    log_file = f"{LOG_DIR}/bot.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    
    logging.info("Logging configured")


def init_csv_log():
    """Initialize CSV log file if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'symbol',
                'side',
                'entry_price',
                'exit_price',
                'units',
                'pnl_usd',
                'pnl_pct',
                'entry_time',
                'exit_time',
                'stop_loss',
                'take_profit',
                'confidence',
                'reason',
                'paper_mode',
                'exit_reason'
            ])
        logging.info(f"CSV log file created: {CSV_FILE}")


def log_trade_to_csv(trade: Dict):
    """
    Append trade to CSV log
    
    Args:
        trade: Trade dictionary from trade_manager or database
    """
    try:
        init_csv_log()  # Ensure file exists
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                trade.get('symbol', ''),
                trade.get('side', ''),
                trade.get('entry_price', ''),
                trade.get('exit_price', ''),
                trade.get('units', ''),
                trade.get('pnl_usd', ''),
                trade.get('pnl_pct', ''),
                trade.get('entry_time', ''),
                trade.get('exit_time', ''),
                trade.get('stop_loss', ''),
                trade.get('take_profit', ''),
                trade.get('confidence', ''),
                trade.get('llm_reason', ''),
                trade.get('paper_mode', True),
                trade.get('exit_reason', '')
            ])
        
        logging.info(f"Trade logged to CSV: {trade.get('symbol')}")
        
    except Exception as e:
        logging.error(f"Error logging trade to CSV: {e}")


def export_trades_csv(trades: List[Dict], output_file: str = None) -> str:
    """
    Export trades to CSV file
    
    Args:
        trades: List of trade dictionaries
        output_file: Output file path (optional)
    
    Returns:
        Path to exported file
    """
    if output_file is None:
        output_file = f"{LOG_DIR}/trades_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        with open(output_file, 'w', newline='') as f:
            if not trades:
                return output_file
            
            # Use first trade keys as headers
            fieldnames = list(trades[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(trades)
        
        logging.info(f"Exported {len(trades)} trades to {output_file}")
        return output_file
        
    except Exception as e:
        logging.error(f"Error exporting trades: {e}")
        raise


def log_decision(symbol: str, decision: Dict, indicators: Dict):
    """
    Log LLM decision for debugging
    
    Args:
        symbol: Trading pair
        decision: LLM decision dictionary
        indicators: Indicator values
    """
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'symbol': symbol,
        'action': decision.get('action'),
        'confidence': decision.get('confidence'),
        'reason': decision.get('reason'),
        'entry_price': decision.get('entry_price'),
        'stop_loss': decision.get('stop_loss'),
        'take_profit': decision.get('take_profit'),
        'indicators': indicators
    }
    
    # Log to dedicated decisions file
    decisions_file = f"{LOG_DIR}/decisions.jsonl"
    try:
        with open(decisions_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logging.error(f"Error logging decision: {e}")


def log_llm_reasoning(symbols: List[str], full_response: Dict):
    """
    Log full LLM response including reasoning to separate file
    
    Args:
        symbols: List of symbols analyzed
        full_response: Complete response from LLM including _raw_response
    """
    timestamp = datetime.utcnow().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'symbols': symbols,
        'response': full_response
    }
    
    # Log JSON response
    filename = f"{LOG_DIR}/llm_reasoning_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    try:
        with open(filename, 'a') as f:
            f.write(json.dumps(log_entry, indent=2) + '\n')
    except Exception as e:
        logging.error(f"Error logging LLM reasoning (JSON): {e}")
    
    # Also log reasoning to a human-readable file
    if full_response.get('_raw_response', {}).get('reasoning'):
        readable_file = f"{LOG_DIR}/llm_reasoning_{datetime.utcnow().strftime('%Y%m%d')}.txt"
        try:
            with open(readable_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Symbols: {', '.join(symbols)}\n")
                f.write(f"{'='*80}\n")
                f.write("REASONING:\n")
                f.write(full_response['_raw_response']['reasoning'])
                f.write(f"\n{'='*80}\n\n")
            
            logging.info(f"Logged full LLM reasoning for {len(symbols)} symbols")
        except Exception as e:
            logging.error(f"Error logging LLM reasoning (TXT): {e}")


def log_error(error_type: str, message: str, details: Dict = None):
    """
    Log error with details
    
    Args:
        error_type: Type of error
        message: Error message
        details: Additional details dictionary
    """
    error_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'type': error_type,
        'message': message,
        'details': details or {}
    }
    
    # Log to dedicated errors file
    errors_file = f"{LOG_DIR}/errors.jsonl"
    try:
        with open(errors_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
        
        logging.error(f"{error_type}: {message}")
        
    except Exception as e:
        logging.error(f"Error logging error (meta!): {e}")


# Initialize logging on import
setup_logging()

