"""Structured logging setup for the eval toolchain.

Why this exists: the toolchain historically used `print` for everything —
mixing two very different things on the same stream:

1. **CLI product output** — verdict reports, leaderboard tables, "Wrote output
   to <path>", `--list-tasks` rows. This is the command's actual result and
   MUST stay on stdout so pipes/redirects keep working. These remain `print`.

2. **Diagnostics** — progress chatter ("Running task X on provider Y"),
   retries, warnings, timing. These are *about* the run, not its result, and
   belong on a logger that can be levelled, silenced, timestamped, or shipped
   as JSON without touching the product output.

This module owns category 2. Call `setup_logging()` once at a CLI entry point;
elsewhere call `get_logger(__name__)` and log. The default is quiet (WARNING)
so existing stdout behavior is unchanged unless a caller opts into -v/--verbose
or sets EVAL_LOG_LEVEL / EVAL_LOG_FORMAT.

Env knobs (read at setup):
  EVAL_LOG_LEVEL   one of DEBUG/INFO/WARNING/ERROR (default WARNING)
  EVAL_LOG_FORMAT  "text" (default) or "json"

Logs go to stderr so they never pollute stdout product output.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

LOGGER_ROOT = "eval"
_configured = False


class _JsonFormatter(logging.Formatter):
    """Render each record as one compact JSON object (one line)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any structured extras the caller passed via `extra={...}`.
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_KEYS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# Standard LogRecord attributes we never want to echo as "extras".
_RESERVED_LOGRECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message",
}


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    name = str(level or os.environ.get("EVAL_LOG_LEVEL") or "WARNING").upper()
    return getattr(logging, name, logging.WARNING)


def setup_logging(
    *,
    level: str | int | None = None,
    fmt: str | None = None,
    stream: Any = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the `eval` logger tree once. Idempotent unless force=True.

    level: explicit level, else EVAL_LOG_LEVEL, else WARNING.
    fmt:   "text" or "json", else EVAL_LOG_FORMAT, else "text".
    stream: defaults to stderr (keeps stdout clean for product output).
    """
    global _configured
    logger = logging.getLogger(LOGGER_ROOT)
    if _configured and not force:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    chosen_fmt = (fmt or os.environ.get("EVAL_LOG_FORMAT") or "text").lower()
    handler = logging.StreamHandler(stream or sys.stderr)
    if chosen_fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    logger.addHandler(handler)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False
    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the `eval` logger. Pass __name__ from the caller."""
    if not name or name == "__main__":
        return logging.getLogger(LOGGER_ROOT)
    # Normalize "eval.foo" regardless of the module's dotted path.
    leaf = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{LOGGER_ROOT}.{leaf}")


def verbosity_to_level(verbose_count: int) -> int:
    """Map a -v count to a level: 0->WARNING, 1->INFO, 2+->DEBUG."""
    if verbose_count <= 0:
        return logging.WARNING
    if verbose_count == 1:
        return logging.INFO
    return logging.DEBUG


def _self_test() -> int:
    import io

    # Text format: a WARNING is emitted, an INFO below level is not.
    buf = io.StringIO()
    setup_logging(level="WARNING", fmt="text", stream=buf, force=True)
    log = get_logger("eval.selftest")
    log.info("this should be filtered")
    log.warning("this should appear")
    text = buf.getvalue()
    assert "this should appear" in text, text
    assert "this should be filtered" not in text, text

    # JSON format: each line parses; extras are carried through.
    buf2 = io.StringIO()
    setup_logging(level="INFO", fmt="json", stream=buf2, force=True)
    log2 = get_logger("eval.selftest")
    log2.info("hello", extra={"provider_id": "tested", "attempt": 2})
    line = buf2.getvalue().strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["msg"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["provider_id"] == "tested"
    assert parsed["attempt"] == 2

    # verbosity mapping.
    assert verbosity_to_level(0) == logging.WARNING
    assert verbosity_to_level(1) == logging.INFO
    assert verbosity_to_level(5) == logging.DEBUG

    # get_logger normalizes any dotted module path to an `eval.<leaf>` child.
    assert get_logger("a.b.campaigns").name == "eval.campaigns"
    assert get_logger("__main__").name == "eval"

    # Reset module state so other tests/CLIs reconfigure cleanly.
    global _configured
    _configured = False

    print("logging_setup self-test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
