"""Конфигурация обучения: схема на pydantic и загрузка из YAML.

Все «магические числа» прототипа (лаги, доля теста, гиперпараметры, пути)
вынесены сюда и проверяются при загрузке, а не падают посреди обучения.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

ModelName = Literal["ridge", "hgb"]

INPUT_COLUMNS: tuple[str, ...] = ("u1", "u2", "u3", "u4", "u5", "u6", "u7")
"""Технологические теги дебутанизатора (порядок как в датасете Fortuna et al.)."""

TARGET_COLUMN = "y"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    """Где лежат данные и какие колонны в них ожидаются."""

    path: Path = Path("data/raw/debutanizer.csv")
    input_columns: tuple[str, ...] = INPUT_COLUMNS
    target_column: str = TARGET_COLUMN
    max_missing_fraction: float = Field(default=0.01, ge=0.0, lt=1.0)


class FeatureConfig(_Strict):
    """Параметры построения признаков."""

    lags: tuple[PositiveInt, ...] = (1, 2, 3, 4, 5, 6)
    add_bottom_temperature_mean: bool = True

    @field_validator("lags")
    @classmethod
    def _sorted_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            msg = "лаги не должны повторяться"
            raise ValueError(msg)
        return tuple(sorted(value))

    @property
    def max_lag(self) -> int:
        """Сколько строк истории нужно, чтобы посчитать признаки для одной точки."""
        return max(self.lags, default=0)


class SplitConfig(_Strict):
    """Разбиение по времени: хвост ряда — тест, на остальном — кросс-валидация."""

    test_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    cv_splits: int = Field(default=5, ge=2)


class RidgeParams(_Strict):
    """Гиперпараметры гребневой регрессии."""

    alpha: float = Field(default=1.0, gt=0.0)


class HgbParams(_Strict):
    """Гиперпараметры градиентного бустинга."""

    max_iter: PositiveInt = 300
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    max_leaf_nodes: PositiveInt = 31


class ModelConfig(_Strict):
    """Модели-кандидаты: побеждает лучшая по RMSE на кросс-валидации."""

    candidates: tuple[ModelName, ...] = ("ridge", "hgb")
    ridge: RidgeParams = RidgeParams()
    hgb: HgbParams = HgbParams()
    random_state: int = 42

    @field_validator("candidates")
    @classmethod
    def _non_empty(cls, value: tuple[ModelName, ...]) -> tuple[ModelName, ...]:
        if not value:
            msg = "нужна хотя бы одна модель-кандидат"
            raise ValueError(msg)
        return value


class TrainConfig(_Strict):
    """Корневой конфиг обучения."""

    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    artifacts_dir: Path = Path("models")


def load_config(path: Path) -> TrainConfig:
    """Читает YAML и валидирует его по схеме.

    Args:
        path: путь к YAML-файлу.

    Returns:
        Проверенный конфиг. Отсутствующие ключи получают значения по умолчанию.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TrainConfig.model_validate(raw)
