"""
utils/logger.py
---------------
Structured, coloured console logger + JSON experiment recorder.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# ── ANSI colour helpers ───────────────────────────────────────────────────

_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    """Adds colour to levelname in terminal output."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = _COLOURS.get(record.levelname, "")
        reset  = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        formatter = logging.Formatter(self.FMT, datefmt=self.DATE_FMT)
        return formatter.format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a coloured console handler.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColourFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ── Experiment recorder ───────────────────────────────────────────────────

class ExperimentLogger:
    """
    Appends one JSON record per run to a newline-delimited log file.

    Usage
    -----
    >>> exp = ExperimentLogger("experiments_log.json")
    >>> exp.log({"run_id": "...", "val_f1": 0.88})
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append *record* (enriched with timestamp) to the log file."""
        record["timestamp"] = datetime.utcnow().isoformat()
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
