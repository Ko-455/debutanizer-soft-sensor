"""Сценарий обучения: данные → признаки → выбор модели → оценка → артефакт."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from soft_sensor.data import load_dataset, time_split
from soft_sensor.features import build_features, feature_names
from soft_sensor.model import Metrics, ModelMetadata, build_pipeline, save_artifact, select_model

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from soft_sensor.config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Итог обучения."""

    model_dir: Path
    metadata: ModelMetadata


def _xy(frame: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.Series]:
    x = build_features(frame, cfg.data.input_columns, cfg.features)
    y = frame.loc[x.index, cfg.data.target_column]
    return x, y


def run_training(cfg: TrainConfig) -> TrainingResult:
    """Обучает модель по конфигу и сохраняет артефакт в ``cfg.artifacts_dir``."""
    frame = load_dataset(cfg.data.path, cfg.data)
    split = time_split(frame, cfg.split.test_fraction)

    x_train, y_train = _xy(split.train, cfg)
    # Тест получает хвост истории из train, чтобы первые точки теста
    # имели полные лаги и ни одна лабораторная проба не пропала.
    test_with_history = frame.iloc[len(split.train) - cfg.features.max_lag :]
    x_test, y_test = _xy(test_with_history, cfg)

    best, cv_rmse = select_model(x_train, y_train, cfg.model, cfg.split.cv_splits)
    logger.info("Выбрана модель: %s", best)

    pipeline = build_pipeline(best, cfg.model).fit(x_train, y_train)
    test_metrics = Metrics.compute(y_test, pipeline.predict(x_test))
    logger.info(
        "Тест: RMSE=%.4f MAE=%.4f R2=%.3f", test_metrics.rmse, test_metrics.mae, test_metrics.r2
    )

    metadata = ModelMetadata(
        model_name=best,
        feature_names=feature_names(cfg.data.input_columns, cfg.features),
        input_columns=list(cfg.data.input_columns),
        lags=list(cfg.features.lags),
        add_bottom_temperature_mean=cfg.features.add_bottom_temperature_mean,
        test_metrics=test_metrics,
        cv_rmse=cv_rmse,
    )
    model_dir = save_artifact(pipeline, metadata, cfg.artifacts_dir)
    return TrainingResult(model_dir=model_dir, metadata=metadata)
