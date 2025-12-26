"""Shared logging utilities for financial data feeds."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

LOG_PREFIX = "financial_feeds"


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for better log parsing and monitoring."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            }:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class PerformanceTracker:
    """Context manager and decorator for performance tracking."""

    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            "Starting operation: %s",
            self.operation_name,
            extra={
                "operation": self.operation_name,
                "event": "start",
                "performance_tracking": True,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        log_data = {
            "operation": self.operation_name,
            "event": "end",
            "duration_seconds": duration,
            "performance_tracking": True,
        }

        if exc_type:
            log_data["error"] = str(exc_val)
            self.logger.error("Operation failed: %s", self.operation_name, extra=log_data)
        else:
            self.logger.info("Operation completed: %s", self.operation_name, extra=log_data)

    @classmethod
    def track(cls, operation_name: str | None = None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                with cls(name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


def setup_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    root_logger_name: str = LOG_PREFIX,
) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger(root_logger_name)
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    if format_type == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        from logging.handlers import RotatingFileHandler

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configure_specific_loggers(numeric_level)

    root_logger.info(
        "Logging configured",
        extra={
            "level": level,
            "format_type": format_type,
            "log_file": log_file,
            "console_output": console_output,
            "configuration": True,
        },
    )


def _configure_specific_loggers(level: int) -> None:
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str, prefix: str = LOG_PREFIX) -> logging.Logger:
    if prefix and not name.startswith(prefix):
        name = f"{prefix}.{name}"
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for adding contextual information to logs."""

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.adapter = None

    def __enter__(self):
        self.adapter = logging.LoggerAdapter(self.logger, self.context)
        return self.adapter

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def log_operation_start(logger: logging.Logger, operation: str, **kwargs):
    logger.info(
        "Starting: %s",
        operation,
        extra={"operation": operation, "event": "start", **kwargs},
    )


def log_operation_end(
    logger: logging.Logger,
    operation: str,
    success: bool = True,
    **kwargs,
):
    level = logging.INFO if success else logging.ERROR
    status = "completed" if success else "failed"
    logger.log(
        level,
        "Operation %s: %s",
        status,
        operation,
        extra={"operation": operation, "event": "end", "success": success, **kwargs},
    )


def log_api_call(
    logger: logging.Logger,
    provider: str,
    endpoint: str,
    success: bool,
    duration: float,
    response_count: int = 0,
    error: Optional[str] = None,
):
    log_data = {
        "api_call": True,
        "provider": provider,
        "endpoint": endpoint,
        "success": success,
        "duration_seconds": duration,
        "response_count": response_count,
    }

    if error:
        log_data["error"] = error

    if success:
        logger.info("API call successful: %s", provider, extra=log_data)
    else:
        logger.error("API call failed: %s", provider, extra=log_data)


def log_data_processing(
    logger: logging.Logger,
    operation: str,
    input_count: int,
    output_count: int,
    duration: float,
):
    logger.info(
        "Data processing: %s",
        operation,
        extra={
            "data_processing": True,
            "operation": operation,
            "input_count": input_count,
            "output_count": output_count,
            "duration_seconds": duration,
            "throughput": input_count / duration if duration > 0 else 0,
        },
    )


def configure_default_logging(log_level: str = "INFO", root_logger_name: str = LOG_PREFIX):
    setup_logging(
        level=log_level,
        format_type="standard",
        console_output=True,
        log_file=None,
        root_logger_name=root_logger_name,
    )


__all__ = [
    "StructuredFormatter",
    "PerformanceTracker",
    "setup_logging",
    "get_logger",
    "log_operation_start",
    "log_operation_end",
    "log_api_call",
    "log_data_processing",
    "configure_default_logging",
    "LoggingContext",
]
