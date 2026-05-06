"""
Logging Utility
"""
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(config) -> logging.Logger:
    """Setup logger"""
    logger = logging.getLogger('Experiment')
    logger.setLevel(logging.INFO)

    # Create log directory
    log_dir = Path(config.setup.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
