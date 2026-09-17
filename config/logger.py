"""Structured JSON Lines Logger for TripMate."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings


class JSONLinesFormatter(logging.Formatter):
    """Formats log records as valid JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom extra fields if provided
        if hasattr(record, "event_type"):
            log_entry["event"] = record.event_type
        if hasattr(record, "tool_name"):
            log_entry["tool"] = record.tool_name
        if hasattr(record, "tool_args"):
            log_entry["args"] = record.tool_args
        if hasattr(record, "tool_result"):
            log_entry["result"] = record.tool_result
        if hasattr(record, "reasoning_trace"):
            log_entry["trace"] = record.reasoning_trace

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(name: str = "tripmate", verbose: bool = False) -> logging.Logger:
    """Configures structured JSON lines logging to file and standard stream."""
    logger = logging.getLogger(name)
    log_level = getattr(logging, settings.TRIPMATE_LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    formatter = JSONLinesFormatter()

    # Ensure logs directory exists
    log_file_path: Path = settings.LOG_FILE
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler: always JSON lines
    file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler: JSON lines if verbose, or concise clean log
    console_handler = logging.StreamHandler(sys.stdout)
    if verbose or settings.TRIPMATE_VERBOSE:
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
    else:
        console_handler.setLevel(logging.WARNING)
        plain_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(plain_formatter)

    logger.addHandler(console_handler)
    return logger


tripmate_logger = setup_logger()
