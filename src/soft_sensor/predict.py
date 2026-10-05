"""Применение обученной модели к новым данным."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from soft_sensor.config import DataConfig, FeatureConfig
from soft_sensor.data import DataValidationError, validate_frame
from soft_sensor.features import build_features
from soft_sensor.model import ModelMetadata, load_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PREDICTION_COLUMN = "y_pred"


class SoftSensor:
    """Обёртка над артефактом: принимает сырые теги, возвращает прогноз.

    Признаки строятся той же функцией, что и при обучении, по параметрам
    из метаданных артефакта, — расхождение «обучали так, применяем иначе»
    исключено.
    """

    def __init__(self, pipeline: Pipeline, metadata: ModelMetadata) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self._data_cfg = DataConfig(input_columns=tuple(metadata.input_columns))
        self._feature_cfg = FeatureConfig(
            lags=tuple(metadata.lags),
            add_bottom_temperature_mean=metadata.add_bottom_temperature_mean,
        )

    @classmethod
    def load(cls, model_dir: Path) -> SoftSensor:
        """Загружает модель из каталога артефакта."""
        return cls(*load_artifact(model_dir))

    @property
    def history_required(self) -> int:
        """Сколько строк истории нужно перед первой точкой прогноза."""
        return self._feature_cfg.max_lag

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Прогноз содержания бутана для каждой строки, у которой хватает истории.

        Args:
            frame: теги колонны, упорядоченные по времени; целевая колонка не нужна.

        Returns:
            Серия прогнозов с индексом исходных строк (первые
            ``history_required`` строк уходят на построение лагов).

        Raises:
            DataValidationError: если не хватает колонок или истории.
        """
        clean = validate_frame(frame, self._data_cfg, require_target=False)
        clean.index = frame.index
        if len(clean) <= self.history_required:
            msg = f"нужно минимум {self.history_required + 1} строк, получено {len(clean)}"
            raise DataValidationError(msg)

        features = build_features(clean, self._data_cfg.input_columns, self._feature_cfg)
        if list(features.columns) != self.metadata.feature_names:
            msg = "набор признаков не совпадает с тем, на котором обучалась модель"
            raise DataValidationError(msg)

        values = self.pipeline.predict(features)
        logger.info("Сделано %d прогнозов", len(values))
        return pd.Series(values, index=features.index, name=PREDICTION_COLUMN)
