"""Автоматическое переобучение: триггер → данные → обучение → проверки → реестр → выкладка.

Схема процесса и описание этапов — в ``Model_retrain_BPMN.md``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from soft_sensor.data import DataValidationError, load_dataset
from soft_sensor.model import Metrics
from soft_sensor.predict import SoftSensor
from soft_sensor.registry import ModelRegistry
from soft_sensor.train import fit_model

if TYPE_CHECKING:
    import pandas as pd

    from soft_sensor.config import GateConfig, MonitorConfig, TrainConfig

logger = logging.getLogger(__name__)


class Trigger(StrEnum):
    """Что запустило переобучение."""

    DRIFT = "drift"
    """Мониторинг: качество текущей модели на свежих анализах упало."""
    SCHEDULE = "schedule"
    """Плановое переобучение по расписанию."""
    MANUAL = "manual"
    """Событие на установке: ремонт, смена сырья, перекалибровка датчиков."""
    INITIAL = "initial"
    """Первая версия: рабочей модели ещё нет."""


class Decision(StrEnum):
    """Чем закончился процесс."""

    PROMOTED = "promoted"
    """Все проверки пройдены, новая версия в работе."""
    REJECTED = "rejected"
    """Проверки качества не пройдены, в работе осталась прежняя версия."""
    ROLLED_BACK = "rolled_back"
    """Версия выложена, но не прошла пробный запуск — возвращена прежняя."""


class InsufficientDataError(DataValidationError):
    """Новых данных слишком мало для переобучения."""


# ---------------------------------------------------------------- мониторинг


@dataclass(frozen=True)
class DriftReport:
    """Оценка текущей модели на последних размеченных точках."""

    n_points: int
    rmse: float
    bias: float
    reference_rmse: float
    """RMSE модели на тесте при её обучении (из metadata.json)."""
    reasons: list[str]

    @property
    def triggered(self) -> bool:
        """Нужно ли переобучение."""
        return bool(self.reasons)


def check_drift(sensor: SoftSensor, frame: pd.DataFrame, cfg: MonitorConfig) -> DriftReport:
    """Сравнивает прогнозы рабочей модели с лабораторией на последних ``cfg.window`` точках.

    Args:
        sensor: рабочая модель.
        frame: свежие теги вместе с колонкой результатов анализа ``y``.
        cfg: пороги срабатывания.
    """
    tail = frame.tail(cfg.window + sensor.history_required)
    predictions = sensor.predict(tail.drop(columns="y"))
    metrics = Metrics.compute(tail.loc[predictions.index, "y"], predictions.to_numpy())
    reference = sensor.metadata.test_metrics.rmse

    reasons = []
    if metrics.rmse > reference * cfg.max_rmse_ratio:
        reasons.append(
            f"RMSE {metrics.rmse:.4f} > {cfg.max_rmse_ratio} x эталонный {reference:.4f}"
        )
    if abs(metrics.bias) > cfg.max_abs_bias:
        reasons.append(f"|смещение| {abs(metrics.bias):.4f} > {cfg.max_abs_bias}")
    return DriftReport(len(predictions), metrics.rmse, metrics.bias, reference, reasons)


# ---------------------------------------------------------------- проверки


@dataclass(frozen=True)
class GateResult:
    """Результат одной проверки качества."""

    name: str
    passed: bool
    detail: str


def evaluate_gates(
    challenger: Metrics, champion: Metrics | None, gates: GateConfig
) -> list[GateResult]:
    """Проверки качества новой версии.

    Абсолютные — модель вообще пригодна; относительная — новая версия
    не хуже текущей на **одних и тех же** свежих данных.
    """
    results = [
        GateResult(
            "rmse",
            challenger.rmse <= gates.max_rmse,
            f"RMSE {challenger.rmse:.4f} ≤ {gates.max_rmse}",
        ),
        GateResult("r2", challenger.r2 >= gates.min_r2, f"R² {challenger.r2:.3f} ≥ {gates.min_r2}"),
        GateResult(
            "bias",
            abs(challenger.bias) <= gates.max_abs_bias,
            f"|смещение| {abs(challenger.bias):.4f} ≤ {gates.max_abs_bias}",
        ),
    ]
    if champion is None:
        results.append(GateResult("vs_champion", True, "рабочей версии нет — сравнение пропущено"))
    else:
        limit = champion.rmse * (1.0 + gates.max_degradation)
        results.append(
            GateResult(
                "vs_champion",
                challenger.rmse <= limit,
                f"RMSE {challenger.rmse:.4f} ≤ RMSE текущей {champion.rmse:.4f}"
                f" × {1 + gates.max_degradation:.2f}",
            )
        )
    return results


def smoke_test(sensor: SoftSensor, frame: pd.DataFrame, gates: GateConfig) -> str | None:
    """Пробный запуск выложенной модели. Возвращает текст ошибки или ``None``."""
    try:
        predictions = sensor.predict(frame.drop(columns="y", errors="ignore").tail(50))
    except Exception as exc:
        return f"модель не запускается: {exc}"
    low, high = gates.smoke_output_range
    values = predictions.to_numpy()
    if not np.isfinite(values).all():
        return "в прогнозах есть NaN/inf"
    if values.min() < low or values.max() > high:
        return f"прогнозы вне диапазона [{low}, {high}]"
    return None


# ---------------------------------------------------------------- процесс


@dataclass
class RetrainReport:
    """Итог одного запуска переобучения — сохраняется в реестре рядом с версией."""

    trigger: str
    decision: str
    version: str
    production_version: str | None
    previous_version: str | None
    challenger: dict[str, float]
    champion: dict[str, float] | None
    gates: list[dict[str, Any]]
    n_train: int
    data_path: str
    message: str
    started_at: str
    finished_at: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        """Сериализация для JSON."""
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_retrain(cfg: TrainConfig, trigger: Trigger) -> RetrainReport:
    """Полный цикл переобучения.

    1. Получение и проверка свежих данных (``cfg.data.path``).
    2. Обучение новой версии (challenger) и её оценка на отложенном хвосте.
    3. Оценка текущей версии (champion) на тех же данных.
    4. Проверки качества → решение о публикации.
    5. Регистрация версии в реестре (всегда — для аудита).
    6. Выкладка и пробный запуск; при сбое — откат на прежнюю версию.

    Raises:
        DataValidationError: данные не прошли проверку; рабочая модель не тронута.
    """
    started = _now()
    rcfg = cfg.retrain
    registry = ModelRegistry(rcfg.registry_dir)
    previous = registry.production_version()

    # 1. Данные
    frame = load_dataset(cfg.data.path, cfg.data)
    if len(frame) < rcfg.gates.min_train_samples:
        msg = f"данных {len(frame)} строк, нужно минимум {rcfg.gates.min_train_samples}"
        raise InsufficientDataError(msg)

    # 2. Новая версия
    fitted = fit_model(frame, cfg)
    challenger = fitted.metadata.test_metrics

    # 3. Текущая версия на тех же данных
    champion: Metrics | None = None
    if previous is not None:
        current = SoftSensor.load(registry.path(previous))
        test = fitted.test_frame
        predictions = current.predict(test.drop(columns=cfg.data.target_column))
        champion = Metrics.compute(
            test.loc[predictions.index, cfg.data.target_column], predictions.to_numpy()
        )
        logger.info("Текущая %s на свежих данных: RMSE=%.4f", previous, champion.rmse)

    # 4. Проверки
    gates = evaluate_gates(challenger, champion, rcfg.gates)
    failed = [g for g in gates if not g.passed]

    # 5. Регистрация
    version = registry.register(fitted.pipeline, fitted.metadata)

    # 6. Выкладка
    if failed:
        decision = Decision.REJECTED
        message = "не пройдены проверки: " + "; ".join(f"{g.name} [{g.detail}]" for g in failed)
    else:
        registry.promote(version, cfg.artifacts_dir, reason=str(trigger))
        error = smoke_test(SoftSensor.load(cfg.artifacts_dir), frame, rcfg.gates)
        if error is None:
            decision = Decision.PROMOTED
            message = f"версия {version} выложена"
        else:
            decision = Decision.ROLLED_BACK
            if previous is not None:
                registry.rollback(cfg.artifacts_dir, reason=f"smoke test {version}: {error}")
                message = f"пробный запуск не пройден ({error}); возвращена {previous}"
            else:
                registry.retire(cfg.artifacts_dir, reason=f"smoke test {version}: {error}")
                message = f"пробный запуск не пройден ({error}); рабочей модели нет"

    report = RetrainReport(
        trigger=str(trigger),
        decision=str(decision),
        version=version,
        production_version=registry.production_version(),
        previous_version=previous,
        challenger=asdict(challenger),
        champion=asdict(champion) if champion else None,
        gates=[asdict(g) for g in gates],
        n_train=fitted.n_train,
        data_path=str(cfg.data.path),
        message=message,
        started_at=started,
        finished_at=_now(),
    )
    registry.write_report(version, report.to_dict())
    log = logger.info if decision is Decision.PROMOTED else logger.warning
    log("Переобучение (%s): %s — %s", trigger, decision, message)
    return report
