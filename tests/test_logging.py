"""Tests for structured and human-readable logging modes."""

import json
import logging

from krasnal_id.config import LoggingConfig
from krasnal_id.logging import JsonFormatter, configure_logging


def test_json_formatter_emits_structured_record() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello %s", ("world",), None)

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_logging_configuration_supports_both_formats() -> None:
    configure_logging(LoggingConfig(level="DEBUG", json_output=True))
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)

    configure_logging(LoggingConfig(level="INFO", json_output=False))
    assert isinstance(logging.getLogger().handlers[0].formatter, logging.Formatter)
    assert logging.getLogger().level == logging.INFO
