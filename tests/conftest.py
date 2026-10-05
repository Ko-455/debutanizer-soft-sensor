from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from soft_sensor.config import DataConfig, HgbParams, ModelConfig, SplitConfig, TrainConfig
from soft_sensor.synthetic import generate_debutanizer

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return generate_debutanizer(n_samples=600, seed=7)


@pytest.fixture
def csv_path(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def train_cfg(tmp_path: Path, csv_path: Path) -> TrainConfig:
    """Быстрый конфиг: мало итераций бустинга, 3 фолда."""
    return TrainConfig(
        data=DataConfig(path=csv_path),
        split=SplitConfig(test_fraction=0.2, cv_splits=3),
        model=ModelConfig(hgb=HgbParams(max_iter=50)),
        artifacts_dir=tmp_path / "models",
    )
