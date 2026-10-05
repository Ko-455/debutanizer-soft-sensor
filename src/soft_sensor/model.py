"""Модели, метрики и сохранение артефакта.

Скейлер и модель живут в одном sklearn-``Pipeline``: в прототипе они
сохранялись кортежем в pickle, и перепутать их порядок при загрузке
было делом одной опечатки.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from soft_sensor import __version__

if TYPE_CHECKING:
    import pandas as pd

    from soft_sensor.config import ModelConfig, ModelName

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class Metrics:
    """Метрики качества регрессии."""

    rmse: float
    mae: float
    r2: float

    @classmethod
    def compute(cls, y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Metrics:
        """Считает RMSE, MAE и R² по факту и прогнозу."""
        return cls(
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2=float(r2_score(y_true, y_pred)),
        )


@dataclass(frozen=True)
class ModelMetadata:
    """Всё, что нужно знать о модели, не загружая её."""

    model_name: str
    feature_names: list[str]
    input_columns: list[str]
    lags: list[int]
    add_bottom_temperature_mean: bool
    test_metrics: Metrics
    cv_rmse: dict[str, float]
    package_version: str = __version__
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelMetadata:
        """Восстанавливает метаданные из JSON."""
        return cls(**{**raw, "test_metrics": Metrics(**raw["test_metrics"])})


def build_pipeline(name: ModelName, cfg: ModelConfig) -> Pipeline:
    """Собирает пайплайн «масштабирование → регрессор» по имени модели."""
    estimator: Ridge | HistGradientBoostingRegressor
    if name == "ridge":
        estimator = Ridge(alpha=cfg.ridge.alpha)
    elif name == "hgb":
        estimator = HistGradientBoostingRegressor(
            max_iter=cfg.hgb.max_iter,
            learning_rate=cfg.hgb.learning_rate,
            max_leaf_nodes=cfg.hgb.max_leaf_nodes,
            random_state=cfg.random_state,
        )
    else:  # pragma: no cover — защищено Literal-типом в конфиге
        msg = f"неизвестная модель: {name}"
        raise ValueError(msg)
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def select_model(
    x_train: pd.DataFrame, y_train: pd.Series, cfg: ModelConfig, cv_splits: int
) -> tuple[ModelName, dict[str, float]]:
    """Выбирает лучшую модель по RMSE на скользящей кросс-валидации.

    Тестовая выборка в выборе не участвует — иначе оценка на тесте
    перестаёт быть честной.

    Returns:
        Имя победителя и средний RMSE на CV для каждого кандидата.
    """
    splitter = TimeSeriesSplit(n_splits=cv_splits)
    scores: dict[str, float] = {}
    for name in cfg.candidates:
        neg_rmse = cross_val_score(
            build_pipeline(name, cfg),
            x_train,
            y_train,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
        )
        scores[name] = float(-neg_rmse.mean())
        logger.info("CV RMSE %-5s = %.4f", name, scores[name])
    best = min(cfg.candidates, key=lambda n: scores[n])
    return best, scores


def save_artifact(pipeline: Pipeline, metadata: ModelMetadata, out_dir: Path) -> Path:
    """Сохраняет модель и метаданные в каталог, возвращает путь к нему."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Модель сохранена в %s", out_dir)
    return out_dir


def load_artifact(model_dir: Path) -> tuple[Pipeline, ModelMetadata]:
    """Загружает модель и метаданные, сохранённые :func:`save_artifact`.

    Внимание: joblib/pickle исполняет код при загрузке — открывайте
    только артефакты из доверенного источника.
    """
    model_path = model_dir / MODEL_FILE
    meta_path = model_dir / METADATA_FILE
    for path in (model_path, meta_path):
        if not path.is_file():
            msg = f"нет файла артефакта: {path}"
            raise FileNotFoundError(msg)
    metadata = ModelMetadata.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
    pipeline: Pipeline = joblib.load(model_path)
    return pipeline, metadata
