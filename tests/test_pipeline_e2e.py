"""Сквозные тесты: обучение → артефакт → прогноз, в т.ч. через CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from typer.testing import CliRunner

from soft_sensor import __version__
from soft_sensor.cli import app
from soft_sensor.data import DataValidationError
from soft_sensor.predict import SoftSensor
from soft_sensor.train import run_training

if TYPE_CHECKING:
    from pathlib import Path

    from soft_sensor.config import TrainConfig

runner = CliRunner()


def test_training_produces_useful_model(train_cfg: TrainConfig) -> None:
    result = run_training(train_cfg)
    assert (result.model_dir / "model.joblib").is_file()
    assert result.metadata.model_name in {"ridge", "hgb"}
    # Синтетика хорошо объясняется лагами — модель должна быть заметно лучше среднего.
    assert result.metadata.test_metrics.r2 > 0.6


def test_predict_matches_training_features(train_cfg: TrainConfig, frame: pd.DataFrame) -> None:
    run_training(train_cfg)
    sensor = SoftSensor.load(train_cfg.artifacts_dir)
    tail = frame.tail(50).drop(columns="y")
    preds = sensor.predict(tail)
    assert len(preds) == 50 - sensor.history_required
    assert preds.index[0] == tail.index[sensor.history_required]


def test_predict_needs_history(train_cfg: TrainConfig, frame: pd.DataFrame) -> None:
    run_training(train_cfg)
    sensor = SoftSensor.load(train_cfg.artifacts_dir)
    with pytest.raises(DataValidationError, match="минимум"):
        sensor.predict(frame.head(3))


def test_predict_rejects_feature_drift(train_cfg: TrainConfig, frame: pd.DataFrame) -> None:
    run_training(train_cfg)
    sensor = SoftSensor.load(train_cfg.artifacts_dir)
    sensor.metadata.feature_names.reverse()
    with pytest.raises(DataValidationError, match="признаков"):
        sensor.predict(frame.tail(20))


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "raw.csv"
    models = tmp_path / "models"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data:\n  path: {data}\nsplit:\n  cv_splits: 3\n"
        f"model:\n  hgb:\n    max_iter: 30\nartifacts_dir: {models}\n",
        encoding="utf-8",
    )

    gen = runner.invoke(app, ["generate-data", "--out", str(data), "--n-samples", "500"])
    assert gen.exit_code == 0, gen.output

    tr = runner.invoke(app, ["--log-level", "WARNING", "train", "--config", str(cfg)])
    assert tr.exit_code == 0, tr.output
    assert "RMSE=" in tr.stdout

    preds_path = tmp_path / "preds.csv"
    pr = runner.invoke(
        app, ["predict", str(data), "--model-dir", str(models), "--out", str(preds_path)]
    )
    assert pr.exit_code == 0, pr.output
    assert len(pd.read_csv(preds_path)) == 500 - 6

    stdout = runner.invoke(app, ["predict", str(data), "--model-dir", str(models)])
    assert stdout.exit_code == 0
    assert stdout.stdout.startswith("row,y_pred")
