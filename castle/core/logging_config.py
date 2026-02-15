"""
castle/core/logging_config.py
Unified logging configuration for Castle AI modules.
"""

import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup standardized logger for Castle modules.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger
