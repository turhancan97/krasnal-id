"""Small standard-library JSON logging setup used by every command."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from krasnal_id.config import LoggingConfig


class JsonFormatter(logging.Formatter):
    """Render stable one-record-per-line JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record without relying on optional dependencies."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: LoggingConfig) -> None:
    """Replace root handlers with the configured deterministic console handler."""
    handler = logging.StreamHandler()
    if config.json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(config.level)
