"""Единая настройка логирования вместо print()."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Настраивает корневой логгер. Повторный вызов перенастраивает уровень."""
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, force=True)
