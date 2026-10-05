"""Командная строка: ``soft-sensor generate-data | train | predict``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from soft_sensor import __version__
from soft_sensor.config import TrainConfig, load_config
from soft_sensor.logging_setup import setup_logging
from soft_sensor.predict import SoftSensor
from soft_sensor.synthetic import generate_debutanizer
from soft_sensor.train import run_training

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
) -> None:
    """Сгенерировать синтетический датасет дебутанизатора."""
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_debutanizer(n_samples=n_samples, seed=seed).to_csv(out, index=False)
    typer.echo(f"Сохранено {n_samples} строк в {out}")


@app.command()
def train(
    config: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="YAML-конфиг")
    ] = None,
) -> None:
    """Обучить модель и сохранить артефакт."""
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


if __name__ == "__main__":  # pragma: no cover
    app()
