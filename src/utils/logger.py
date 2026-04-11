"""
Logging Configuration for Healthcare Intelligence System
==========================================================
Provides a reusable factory function that returns a configured
Python ``logging.Logger`` with console and optional file handlers.
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    log_file : str, optional
        Path to a log file.  If provided a ``FileHandler`` is attached.
        Parent directories are created automatically.
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Fully configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
