"""Построение признаков.

В прототипе этот код был скопирован дважды (обучение и применение) и уже
разошёлся бы при первой правке. Здесь — одна функция для обоих путей.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from soft_sensor.config import FeatureConfig

BOTTOM_TEMPERATURE_SENSORS = ("u6", "u7")
BOTTOM_TEMPERATURE_MEAN = "u6u7_mean"


def feature_names(input_columns: Sequence[str], cfg: FeatureConfig) -> list[str]:
    """Имена признаков в том порядке, в котором их ждёт модель."""
    names = list(input_columns)
    names += [f"{col}_lag{lag}" for col in input_columns for lag in cfg.lags]
    if cfg.add_bottom_temperature_mean:
        names.append(BOTTOM_TEMPERATURE_MEAN)
    return names


def build_features(
    frame: pd.DataFrame, input_columns: Sequence[str], cfg: FeatureConfig
) -> pd.DataFrame:
    """Строит матрицу признаков по текущим и прошлым значениям тегов.

    Признак в момент ``t`` использует только данные до ``t`` включительно —
    утечки будущего нет. Первые ``cfg.max_lag`` строк не имеют полной
    истории и отбрасываются; индекс результата совпадает с индексом ``frame``,
    поэтому целевую переменную выравнивают через ``.loc``.

    Args:
        frame: таблица с тегами, упорядоченная по времени.
        input_columns: какие колонки считать входами.
        cfg: параметры признаков.

    Returns:
        DataFrame признаков без пропусков.
    """
    inputs = frame[list(input_columns)]
    parts: dict[str, pd.Series] = {col: inputs[col] for col in input_columns}
    for col in input_columns:
        for lag in cfg.lags:
            parts[f"{col}_lag{lag}"] = inputs[col].shift(lag)
    if cfg.add_bottom_temperature_mean:
        sensors = [s for s in BOTTOM_TEMPERATURE_SENSORS if s in inputs.columns]
        parts[BOTTOM_TEMPERATURE_MEAN] = inputs[sensors].mean(axis=1)

    features = pd.DataFrame(parts, index=frame.index)
    return features.iloc[cfg.max_lag :][feature_names(input_columns, cfg)]
