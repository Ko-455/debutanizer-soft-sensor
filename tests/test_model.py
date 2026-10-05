from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from soft_sensor.config import INPUT_COLUMNS, FeatureConfig, HgbParams, ModelConfig
from soft_sensor.features import build_features
from soft_sensor.model import (
    Metrics,
    ModelMetadata,
    build_pipeline,
    load_artifact,
    save_artifact,
    select_model,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


@pytest.fixture
def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = build_features(frame, INPUT_COLUMNS, FeatureConfig())
    return x, frame.loc[x.index, "y"]


def test_metrics_perfect_prediction() -> None:
    y = np.array([0.1, 0.2, 0.3])
    m = Metrics.compute(y, y)
    assert m.rmse == 0.0
    assert m.mae == 0.0
    assert m.r2 == 1.0


@pytest.mark.parametrize("name", ["ridge", "hgb"])
def test_pipeline_fits(name: str, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = xy
    cfg = ModelConfig(hgb=HgbParams(max_iter=20))
    pipe = build_pipeline(name, cfg).fit(x, y)  # type: ignore[arg-type]
    assert pipe.predict(x).shape == (len(x),)


def test_select_model_returns_lowest_cv_rmse(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = xy
    best, scores = select_model(x, y, ModelConfig(hgb=HgbParams(max_iter=30)), cv_splits=3)
    assert set(scores) == {"ridge", "hgb"}
    assert scores[best] == min(scores.values())


def test_artifact_roundtrip(tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = xy
    pipe = build_pipeline("ridge", ModelConfig()).fit(x, y)
    meta = ModelMetadata(
        model_name="ridge",
        feature_names=list(x.columns),
        input_columns=list(INPUT_COLUMNS),
        lags=[1, 2, 3, 4, 5, 6],
        add_bottom_temperature_mean=True,
        test_metrics=Metrics(rmse=0.1, mae=0.05, r2=0.9),
        cv_rmse={"ridge": 0.1},
    )
    save_artifact(pipe, meta, tmp_path / "art")
    loaded_pipe, loaded_meta = load_artifact(tmp_path / "art")
    assert loaded_meta == meta
    np.testing.assert_allclose(loaded_pipe.predict(x), pipe.predict(x))


def test_load_artifact_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path)
