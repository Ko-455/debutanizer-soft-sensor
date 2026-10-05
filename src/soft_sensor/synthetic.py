"""Генератор синтетических данных, похожих на эталонный датасет дебутанизатора.

Нужен, пока нет реальной выгрузки из АСУ ТП/LIMS. Структура — как у
Fortuna et al. (2007): 7 нормированных тегов и содержание бутана в кубе.
Целевая переменная зависит от входов с транспортным запаздыванием,
поэтому лаговые признаки действительно несут информацию.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soft_sensor.config import INPUT_COLUMNS, TARGET_COLUMN

# (коэффициент авторегрессии, СКО шума) для u1..u6; u7 — дублирующий датчик u6
_AR_PARAMS: tuple[tuple[float, float], ...] = (
    (0.98, 0.020),
    (0.97, 0.015),
    (0.95, 0.030),
    (0.96, 0.025),
    (0.985, 0.020),
    (0.99, 0.015),
)
_DUPLICATE_SENSOR_NOISE = 0.01
_TARGET_NOISE = 0.015
_WARMUP = 6  # первые шаги без полной истории запаздываний


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=n)
    series = np.empty(n)
    series[0] = 0.0
    for i in range(1, n):
        series[i] = phi * series[i - 1] + noise[i]
    return series


def generate_debutanizer(n_samples: int = 2394, seed: int = 42) -> pd.DataFrame:
    """Генерирует ряд «теги колонны → содержание бутана».

    Args:
        n_samples: длина ряда (в эталонном датасете 2394 точки).
        seed: зерно генератора для воспроизводимости.

    Returns:
        DataFrame с колонками ``u1..u7`` и ``y``; все значения в [0, 1].
    """
    if n_samples <= _WARMUP:
        msg = f"n_samples должен быть больше {_WARMUP}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    centered = [_ar1(rng, n_samples, phi, sigma) for phi, sigma in _AR_PARAMS]
    u1, u2, u3, _u4, u5, u6 = centered
    u7 = u6 + rng.normal(0.0, _DUPLICATE_SENSOR_NOISE, size=n_samples)

    # Запаздывания: 6-я тарелка — 4 шага, низ колонны — 2, флегма — 1.
    t = np.arange(_WARMUP, n_samples)
    y = np.zeros(n_samples)
    y[t] = (
        0.3
        - 0.8 * u5[t - 4]
        - 0.6 * u6[t - 2]
        + 0.35 * u3[t - 1]
        + 0.9 * u5[t - 4] ** 2
        + 0.8 * u2[t] * u1[t]
        + rng.normal(0.0, _TARGET_NOISE, size=t.size)
    )

    values = np.column_stack([*centered, u7]) + 0.5
    frame = pd.DataFrame(np.clip(values, 0.0, 1.0), columns=list(INPUT_COLUMNS))
    frame[TARGET_COLUMN] = np.clip(y, 0.0, 1.0)
    return frame
