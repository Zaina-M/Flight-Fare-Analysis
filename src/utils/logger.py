"""
Logging Utility Module.

Provides centralized logging configuration for the entire project.
Following production standards for observability and debugging.

Industry Standard Justification:
- Centralized logging enables better debugging and monitoring
- Structured logs facilitate log aggregation in production
- Consistent format across all modules improves maintainability
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Import config
try:
    from config.config import logging_config, LOGS_DIR
except ImportError:
    from ..config.config import logging_config, LOGS_DIR


def setup_logger(
    name: str,
    level: str = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up and return a logger with both file and console handlers.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    level = level or logging_config.level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=logging_config.format,
        datefmt=logging_config.date_format
    )
    
    # File handler
    if log_to_file:
        log_file = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d')}_{logging_config.log_file}"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Logs to file only (no console output) to keep terminal clean.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return setup_logger(name, log_to_console=False)
