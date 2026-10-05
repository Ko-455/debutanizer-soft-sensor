"""Загрузка, проверка и разбиение данных."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from soft_sensor.config import DataConfig

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Данные не соответствуют ожидаемой схеме."""


@dataclass(frozen=True)
class TimeSplit:
    """Разбиение ряда по времени без перемешивания."""

    train: pd.DataFrame
    test: pd.DataFrame


def validate_frame(frame: pd.DataFrame, cfg: DataConfig, *, require_target: bool) -> pd.DataFrame:
    """Проверяет схему и качество данных, возвращает очищенную копию.

    Проверки: все колонки на месте, значения числовые, доля пропусков
    не выше порога. Оставшиеся одиночные пропуски заполняются
    предыдущим значением — для тегов АСУ ТП это стандартная практика.

    Args:
        frame: исходная таблица.
        cfg: описание ожидаемых колонок.
        require_target: нужна ли целевая колонка (при инференсе её нет).

    Raises:
        DataValidationError: если данные не проходят проверку.
    """
    required = list(cfg.input_columns)
    if require_target:
        required.append(cfg.target_column)

    missing = [col for col in required if col not in frame.columns]
    if missing:
        msg = f"нет колонок: {', '.join(missing)}"
        raise DataValidationError(msg)

    subset = frame[required].copy()
    non_numeric = [col for col in required if not pd.api.types.is_numeric_dtype(subset[col])]
    if non_numeric:
        msg = f"нечисловые колонки: {', '.join(non_numeric)}"
        raise DataValidationError(msg)

    missing_share = subset.isna().mean()
    too_sparse = missing_share[missing_share > cfg.max_missing_fraction]
    if not too_sparse.empty:
        details = ", ".join(f"{col}={share:.1%}" for col, share in too_sparse.items())
        msg = f"слишком много пропусков: {details}"
        raise DataValidationError(msg)

    n_missing = int(subset.isna().sum().sum())
    if n_missing:
        logger.warning("Заполняю %d пропусков предыдущим значением", n_missing)
        subset = subset.ffill().bfill()

    return subset.reset_index(drop=True)


def load_dataset(path: Path, cfg: DataConfig, *, require_target: bool = True) -> pd.DataFrame:
    """Читает CSV и проверяет его через :func:`validate_frame`."""
    if not path.is_file():
        msg = f"файл не найден: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    logger.info("Загружено %d строк из %s", len(frame), path)
    return validate_frame(frame, cfg, require_target=require_target)


def time_split(frame: pd.DataFrame, test_fraction: float) -> TimeSplit:
    """Отрезает хвост ряда под тест. Перемешивание запрещено — это временной ряд."""
    if not 0.0 < test_fraction < 1.0:
        msg = "test_fraction должен быть в (0, 1)"
        raise ValueError(msg)
    cut = int(len(frame) * (1.0 - test_fraction))
    if cut == 0 or cut == len(frame):
        msg = "слишком мало данных для разбиения"
        raise DataValidationError(msg)
    return TimeSplit(train=frame.iloc[:cut], test=frame.iloc[cut:])
