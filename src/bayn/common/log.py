"""One consistent log format for every Bayn process."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO", name: str = "bayn") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
