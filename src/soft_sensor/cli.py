"""Командная строка: ``soft-sensor generate-data | train | predict``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from soft_sensor import __version__
from soft_sensor.config import TrainConfig, load_config
from soft_sensor.data import DataValidationError, load_dataset
from soft_sensor.logging_setup import setup_logging
from soft_sensor.predict import SoftSensor
from soft_sensor.registry import ModelRegistry, RegistryError
from soft_sensor.retrain import Decision, Trigger, check_drift, run_retrain
from soft_sensor.synthetic import generate_debutanizer
from soft_sensor.train import run_training

# Коды завершения — по ним планировщик / CI решает, что делать дальше.
EXIT_DATA_ERROR = 1
EXIT_REJECTED = 2
EXIT_DRIFT = 3
EXIT_ROLLED_BACK = 4

ConfigOption = Annotated[
    Path | None, typer.Option("--config", exists=True, dir_okay=False, help="YAML-конфиг")
]

app = typer.Typer(
    help="Виртуальный анализатор бутана в кубе дебутанизатора.",
    no_args_is_help=True,
    add_completion=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING, ERROR")] = "INFO",
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Версия пакета")
    ] = False,
) -> None:
    """Общие опции для всех команд."""
    setup_logging(log_level)


@app.command("generate-data")
def generate_data(
    out: Annotated[Path, typer.Option(help="Куда сохранить CSV")] = Path(
        "data/raw/debutanizer.csv"
    ),
    n_samples: Annotated[int, typer.Option(min=10, help="Длина ряда")] = 2394,
    seed: Annotated[int, typer.Option(help="Зерно генератора")] = 42,
    regime_shift: Annotated[
        float, typer.Option(help="Сдвиг режима процесса (демо дрейфа), 0 — номинал")
    ] = 0.0,
) -> None:
    """Сгенерировать синтетический датасет дебутанизатора."""
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_debutanizer(n_samples=n_samples, seed=seed, regime_shift=regime_shift)
    frame.to_csv(out, index=False)
    typer.echo(f"Сохранено {n_samples} строк в {out}")


@app.command()
def train(config: ConfigOption = None) -> None:
    """Обучить модель и сохранить артефакт (ручной путь без проверок и реестра)."""
    cfg = load_config(config) if config else TrainConfig()
    result = run_training(cfg)
    m = result.metadata.test_metrics
    typer.echo(
        f"Модель: {result.metadata.model_name} | "
        f"RMSE={m.rmse:.4f} MAE={m.mae:.4f} R2={m.r2:.3f} | {result.model_dir}"
    )


@app.command()
def predict(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="CSV с тегами")],
    model_dir: Annotated[Path, typer.Option(help="Каталог артефакта")] = Path("models"),
    out: Annotated[
        Path | None, typer.Option(help="CSV для прогнозов (по умолчанию stdout)")
    ] = None,
) -> None:
    """Рассчитать прогноз по CSV с технологическими тегами."""
    sensor = SoftSensor.load(model_dir)
    predictions = sensor.predict(pd.read_csv(input_csv))
    if out is None:
        typer.echo(predictions.to_csv(index_label="row"), nl=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out, index_label="row")
        typer.echo(f"Сохранено {len(predictions)} прогнозов в {out}")


@app.command()
def monitor(
    labeled_csv: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="CSV с тегами и анализами y")
    ],
    config: ConfigOption = None,
) -> None:
    """Проверить рабочую модель на свежих анализах. Код 3 — нужно переобучение."""
    cfg = load_config(config) if config else TrainConfig()
    sensor = SoftSensor.load(cfg.artifacts_dir)
    frame = load_dataset(labeled_csv, cfg.data)
    report = check_drift(sensor, frame, cfg.retrain.monitor)
    typer.echo(
        f"Точек: {report.n_points} | RMSE={report.rmse:.4f} "
        f"(эталон {report.reference_rmse:.4f}) | смещение={report.bias:+.4f}"
    )
    if report.triggered:
        typer.echo("ДРЕЙФ: " + "; ".join(report.reasons))
        raise typer.Exit(EXIT_DRIFT)
    typer.echo("Качество в норме")


@app.command()
def retrain(
    trigger: Annotated[Trigger, typer.Option(help="Что запустило переобучение")] = Trigger.MANUAL,
    config: ConfigOption = None,
) -> None:
    """Переобучить, проверить и при успехе выложить новую версию модели."""
    cfg = load_config(config) if config else TrainConfig()
    try:
        report = run_retrain(cfg, trigger)
    except (DataValidationError, FileNotFoundError) as exc:
        typer.echo(f"ОШИБКА ДАННЫХ: {exc}. Рабочая модель не изменена.", err=True)
        raise typer.Exit(EXIT_DATA_ERROR) from exc

    for gate in report.gates:
        typer.echo(f"  [{'OK' if gate['passed'] else 'FAIL'}] {gate['detail']}")
    typer.echo(f"{report.decision.upper()}: {report.message}")
    typer.echo(f"В работе: {report.production_version or 'нет модели'}")
    if report.decision == Decision.REJECTED:
        raise typer.Exit(EXIT_REJECTED)
    if report.decision == Decision.ROLLED_BACK:
        raise typer.Exit(EXIT_ROLLED_BACK)


@app.command()
def rollback(config: ConfigOption = None) -> None:
    """Вернуть в работу предыдущую выложенную версию."""
    cfg = load_config(config) if config else TrainConfig()
    try:
        version = ModelRegistry(cfg.retrain.registry_dir).rollback(
            cfg.artifacts_dir, reason="manual rollback"
        )
    except RegistryError as exc:
        typer.echo(f"ОШИБКА: {exc}", err=True)
        raise typer.Exit(EXIT_DATA_ERROR) from exc
    typer.echo(f"В работе версия {version}")


if __name__ == "__main__":  # pragma: no cover
    app()
