from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


_STANDARD_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, sort_keys=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_blueknight_configured", False):
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger._blueknight_configured = True  # type: ignore[attr-defined]
    return logger


def log_stage(
    logger: logging.Logger,
    *,
    trace_id: str,
    stage: str,
    duration_ms: int,
    item_count: int,
    **extra: object,
) -> None:
    logger.info(
        "stage boundary",
        extra={
            "trace_id": trace_id,
            "stage": stage,
            "duration_ms": duration_ms,
            "item_count": item_count,
            **extra,
        },
    )
