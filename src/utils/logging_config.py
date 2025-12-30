"""
Logging configuration with rotating file handler.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = "speaker_id.log"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

_logger_configured = False


def setup_logging(level=logging.INFO, console=True):
    """
    Configure application-wide logging.
    """
    global _logger_configured
    if _logger_configured:
        return logging.getLogger("speaker_id")
        
    # Create logs directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("speaker_id")
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File Handler (rotating)
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, LOG_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    # Console Handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    _logger_configured = True
    logger.info("Logging initialized.")
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a child logger."""
    if name:
        return logging.getLogger(f"speaker_id.{name}")
    return logging.getLogger("speaker_id")
