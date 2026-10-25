from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from soft_sensor.config import INPUT_COLUMNS, FeatureConfig, ModelConfig
from soft_sensor.features import build_features
from soft_sensor.model import Metrics, ModelMetadata, build_pipeline, load_artifact
from soft_sensor.registry import ModelRegistry, RegistryError

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from sklearn.pipeline import Pipeline


@pytest.fixture
def fitted(frame: pd.DataFrame) -> tuple[Pipeline, ModelMetadata]:
    x = build_features(frame, INPUT_COLUMNS, FeatureConfig())
    pipe = build_pipeline("ridge", ModelConfig()).fit(x, frame.loc[x.index, "y"])
    meta = ModelMetadata(
        model_name="ridge",
        feature_names=list(x.columns),
        input_columns=list(INPUT_COLUMNS),
        lags=[1, 2, 3, 4, 5, 6],
        add_bottom_temperature_mean=True,
        test_metrics=Metrics(rmse=0.03, mae=0.02, r2=0.95),
        cv_rmse={"ridge": 0.03},
    )
    return pipe, meta


def test_empty_registry(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg")
    assert reg.versions() == []
    assert reg.production_version() is None
    with pytest.raises(RegistryError):
        reg.path("v0001")


def test_versions_increment(tmp_path: Path, fitted: tuple[Pipeline, ModelMetadata]) -> None:
    reg = ModelRegistry(tmp_path / "reg")
    assert reg.register(*fitted) == "v0001"
    assert reg.register(*fitted) == "v0002"
    assert reg.versions() == ["v0001", "v0002"]
    assert reg.production_version() is None  # регистрация не означает выкладку


def test_promote_and_rollback(tmp_path: Path, fitted: tuple[Pipeline, ModelMetadata]) -> None:
    reg = ModelRegistry(tmp_path / "reg")
    deploy = tmp_path / "prod"
    v1, v2 = reg.register(*fitted), reg.register(*fitted)

    reg.promote(v1, deploy, reason="initial")
    assert reg.production_version() == v1
    load_artifact(deploy)  # рабочая модель загружается
    assert not (deploy / ".staging").exists()

    reg.promote(v2, deploy, reason="drift")
    assert reg.rollback(deploy) == v1
    assert reg.production_version() == v1
    assert [h["reason"] for h in reg.history()] == ["initial", "drift", "rollback"]


def test_rollback_without_previous(tmp_path: Path, fitted: tuple[Pipeline, ModelMetadata]) -> None:
    reg = ModelRegistry(tmp_path / "reg")
    reg.promote(reg.register(*fitted), tmp_path / "prod", reason="initial")
    with pytest.raises(RegistryError, match="некуда"):
        reg.rollback(tmp_path / "prod")


def test_retire(tmp_path: Path, fitted: tuple[Pipeline, ModelMetadata]) -> None:
    reg = ModelRegistry(tmp_path / "reg")
    deploy = tmp_path / "prod"
    reg.promote(reg.register(*fitted), deploy, reason="initial")
    reg.retire(deploy, reason="broken")
    assert reg.production_version() is None
    assert not (deploy / "model.joblib").exists()


def test_report_roundtrip(tmp_path: Path, fitted: tuple[Pipeline, ModelMetadata]) -> None:
    reg = ModelRegistry(tmp_path / "reg")
    version = reg.register(*fitted)
    reg.write_report(version, {"decision": "promoted", "сообщение": "ок"})
    assert reg.read_report(version)["сообщение"] == "ок"
