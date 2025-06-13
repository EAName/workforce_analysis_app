import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from config.config import config

def setup_logger(name: str, log_level: str = None) -> logging.Logger:
    """
    Set up and configure logger with rotating file handler
    
    Args:
        name (str): Name of the logger
        log_level (str): Logging level (defaults to config setting)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logging configuration
    log_config = config.logging
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_config.file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level from config or parameter
    level = getattr(logging, log_level.upper()) if log_level else getattr(logging, log_config.level.upper())
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_config.file,
        maxBytes=log_config.max_size,
        backupCount=log_config.backup_count
    )
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_config.format)
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger('workforce_analysis')

# Add convenience functions for common logging operations
def log_error(message: str, exc_info: bool = True):
    """Log error message with optional exception info"""
    logger.error(message, exc_info=exc_info)

def log_warning(message: str):
    """Log warning message"""
    logger.warning(message)

def log_info(message: str):
    """Log info message"""
    logger.info(message)

def log_debug(message: str):
    """Log debug message"""
    logger.debug(message) 