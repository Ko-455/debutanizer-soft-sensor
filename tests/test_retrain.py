"""Все ветки процесса переобучения из Model_retrain_BPMN.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from soft_sensor import retrain as retrain_module
from soft_sensor.cli import EXIT_DATA_ERROR, EXIT_DRIFT, EXIT_REJECTED, EXIT_ROLLED_BACK, app
from soft_sensor.config import (
    DataConfig,
    GateConfig,
    HgbParams,
    ModelConfig,
    RetrainConfig,
    SplitConfig,
    TrainConfig,
)
from soft_sensor.data import DataValidationError
from soft_sensor.model import Metrics
from soft_sensor.predict import SoftSensor
from soft_sensor.registry import ModelRegistry
from soft_sensor.retrain import (
    Decision,
    InsufficientDataError,
    Trigger,
    check_drift,
    evaluate_gates,
    run_retrain,
    smoke_test,
)
from soft_sensor.synthetic import generate_debutanizer

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _cfg(tmp_path: Path, gates: GateConfig | None = None) -> TrainConfig:
    return TrainConfig(
        data=DataConfig(path=tmp_path / "fresh.csv"),
        split=SplitConfig(cv_splits=3),
        model=ModelConfig(candidates=("ridge",), hgb=HgbParams(max_iter=20)),
        retrain=RetrainConfig(gates=gates or GateConfig(), registry_dir=tmp_path / "reg"),
        artifacts_dir=tmp_path / "prod",
    )


def _write(cfg: TrainConfig, **kwargs: float) -> pd.DataFrame:
    frame = generate_debutanizer(**kwargs)  # type: ignore[arg-type]
    frame.to_csv(cfg.data.path, index=False)
    return frame


# ---------------------------------------------------------------- проверки


def test_gates_without_champion() -> None:
    results = evaluate_gates(Metrics(0.03, 0.02, 0.95, 0.001), None, GateConfig())
    assert all(g.passed for g in results)


@pytest.mark.parametrize(
    ("challenger", "failed"),
    [
        (Metrics(0.06, 0.05, 0.95, 0.0), {"rmse", "vs_champion"}),
        (Metrics(0.03, 0.02, 0.70, 0.0), {"r2"}),
        (Metrics(0.03, 0.02, 0.95, 0.05), {"bias"}),
        (Metrics(0.04, 0.03, 0.95, 0.0), {"vs_champion"}),
    ],
)
def test_each_gate_can_fail(challenger: Metrics, failed: set[str]) -> None:
    champion = Metrics(0.03, 0.02, 0.95, 0.0)
    results = evaluate_gates(challenger, champion, GateConfig())
    assert {g.name for g in results if not g.passed} == failed


def test_small_degradation_is_tolerated() -> None:
    champion = Metrics(0.030, 0.02, 0.95, 0.0)
    challenger = Metrics(0.0305, 0.02, 0.95, 0.0)  # хуже на 1.7 % < 2 %
    assert all(g.passed for g in evaluate_gates(challenger, champion, GateConfig()))


# ---------------------------------------------------------------- сценарий дрейфа


def test_full_drift_scenario(tmp_path: Path) -> None:
    """Первая версия → дрейф → переобучение → новая версия в работе."""
    cfg = _cfg(tmp_path)

    _write(cfg, seed=42)
    first = run_retrain(cfg, Trigger.INITIAL)
    assert first.decision == Decision.PROMOTED
    assert first.champion is None

    drifted = _write(cfg, seed=7, regime_shift=0.5)
    sensor = SoftSensor.load(cfg.artifacts_dir)
    drift = check_drift(sensor, drifted, cfg.retrain.monitor)
    assert drift.triggered
    assert drift.rmse > 1.5 * drift.reference_rmse

    second = run_retrain(cfg, Trigger.DRIFT)
    assert second.decision == Decision.PROMOTED
    assert second.champion is not None
    assert second.challenger["rmse"] < second.champion["rmse"]
    assert ModelRegistry(cfg.retrain.registry_dir).production_version() == second.version

    after = check_drift(SoftSensor.load(cfg.artifacts_dir), drifted, cfg.retrain.monitor)
    assert not after.triggered


def test_rejected_keeps_production(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, seed=42)
    run_retrain(cfg, Trigger.INITIAL)

    strict = _cfg(tmp_path, GateConfig(max_rmse=0.001))
    report = run_retrain(strict, Trigger.SCHEDULE)
    registry = ModelRegistry(cfg.retrain.registry_dir)
    assert report.decision == Decision.REJECTED
    assert "rmse" in report.message
    assert registry.production_version() == "v0001"
    assert registry.versions() == ["v0001", "v0002"]  # отклонённая тоже сохранена
    assert registry.read_report("v0002")["decision"] == "rejected"


def test_bad_data_changes_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, seed=42)
    run_retrain(cfg, Trigger.INITIAL)

    pd.read_csv(cfg.data.path).drop(columns="u5").to_csv(cfg.data.path, index=False)
    with pytest.raises(DataValidationError):
        run_retrain(cfg, Trigger.MANUAL)
    assert ModelRegistry(cfg.retrain.registry_dir).versions() == ["v0001"]


def test_too_little_data(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, GateConfig(min_train_samples=5000))
    _write(cfg, seed=42)
    with pytest.raises(InsufficientDataError):
        run_retrain(cfg, Trigger.MANUAL)


def test_smoke_failure_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, seed=42)
    run_retrain(cfg, Trigger.INITIAL)

    monkeypatch.setattr(retrain_module, "smoke_test", lambda *_: "прогнозы вне диапазона")
    report = run_retrain(cfg, Trigger.MANUAL)
    assert report.decision == Decision.ROLLED_BACK
    assert report.production_version == "v0001"
    assert (
        SoftSensor.load(cfg.artifacts_dir).metadata
        == SoftSensor.load(tmp_path / "reg" / "v0001").metadata
    )


def test_smoke_failure_without_previous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, seed=42)
    monkeypatch.setattr(retrain_module, "smoke_test", lambda *_: "сломано")
    report = run_retrain(cfg, Trigger.INITIAL)
    assert report.decision == Decision.ROLLED_BACK
    assert report.production_version is None
    assert not (cfg.artifacts_dir / "model.joblib").exists()


def test_smoke_test_checks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _write(cfg, seed=42)
    run_retrain(cfg, Trigger.INITIAL)
    sensor = SoftSensor.load(cfg.artifacts_dir)
    gates = GateConfig()

    assert smoke_test(sensor, frame, gates) is None
    assert "диапазона" in str(smoke_test(sensor, frame, GateConfig(smoke_output_range=(5, 6))))
    assert "не запускается" in str(smoke_test(sensor, frame.drop(columns="u1"), gates))
    sensor.pipeline.predict = lambda x: np.full(len(x), np.nan)
    assert "NaN" in str(smoke_test(sensor, frame, gates))


# ---------------------------------------------------------------- CLI и коды завершения


def test_cli_exit_codes(tmp_path: Path) -> None:
    data, reg, prod = tmp_path / "fresh.csv", tmp_path / "reg", tmp_path / "prod"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data: {{path: {data}}}\nartifacts_dir: {prod}\nsplit: {{cv_splits: 3}}\n"
        f"model: {{candidates: [ridge]}}\nretrain: {{registry_dir: {reg}}}\n",
        encoding="utf-8",
    )
    c = ["--config", str(cfg)]

    def run(*args: str) -> int:
        return runner.invoke(app, ["--log-level", "ERROR", *args]).exit_code

    assert run("generate-data", "--out", str(data)) == 0
    assert run("retrain", "--trigger", "initial", *c) == 0
    assert run("monitor", str(data), *c) == 0
    assert run("rollback", *c) == EXIT_DATA_ERROR  # откатываться некуда

    assert run("generate-data", "--out", str(data), "--seed", "7", "--regime-shift", "0.5") == 0
    assert run("monitor", str(data), *c) == EXIT_DRIFT
    assert run("retrain", "--trigger", "drift", *c) == 0
    assert run("rollback", *c) == 0

    strict = tmp_path / "strict.yaml"
    strict.write_text(
        cfg.read_text(encoding="utf-8").replace(
            f"registry_dir: {reg}", f"registry_dir: {reg}, gates: {{max_rmse: 0.001}}"
        ),
        encoding="utf-8",
    )
    assert run("retrain", "--config", str(strict)) == EXIT_REJECTED

    pd.read_csv(data).head(100).to_csv(data, index=False)
    assert run("retrain", *c) == EXIT_DATA_ERROR


def test_cli_rolled_back_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_obj = _cfg(tmp_path)
    _write(cfg_obj, seed=42)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data: {{path: {cfg_obj.data.path}}}\nartifacts_dir: {cfg_obj.artifacts_dir}\n"
        f"split: {{cv_splits: 3}}\nretrain: {{registry_dir: {tmp_path / 'reg'}}}\n"
        "model: {candidates: [ridge]}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(retrain_module, "smoke_test", lambda *_: "сломано")
    result = runner.invoke(app, ["retrain", "--config", str(cfg)])
    assert result.exit_code == EXIT_ROLLED_BACK
