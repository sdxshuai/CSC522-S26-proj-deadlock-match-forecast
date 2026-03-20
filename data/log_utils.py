"""
log_utils.py
=============
Shared logging setup for data collection scripts.

Each script calls ``get_logger(name)`` once at the top of ``main()``.
The returned logger writes to **both** stdout and a timestamped log file:

    data/logs/{name}_{YYYYMMDD_HHMMSS}.log

Log format (both destinations):
    2026-03-20 12:00:00 [INFO ] [Phase 1] Starting …

Usage in each script::

    from log_utils import get_logger
    log = logging.getLogger(__name__)   # module-level placeholder

    def main():
        global log
        log = get_logger("fetch_matches")
        ...
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Log files are written relative to the project root (one level above this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = _PROJECT_ROOT / "data" / "logs"


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to stdout AND a timestamped file.

    Args:
        name: Script identifier used as logger name and in the filename,
              e.g. ``"fetch_matches"`` → ``data/logs/fetch_matches_20260320_120000.log``

    Returns:
        Configured :class:`logging.Logger`.  Handlers are added only once;
        safe to call multiple times.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Guard: avoid duplicate handlers if main() is called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (stdout) ──────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler ──────────────────────────────────────────────────────
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger
