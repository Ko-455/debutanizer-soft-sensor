from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from soft_sensor.config import INPUT_COLUMNS, DataConfig
from soft_sensor.data import DataValidationError, load_dataset, time_split, validate_frame
from soft_sensor.synthetic import generate_debutanizer

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def test_synthetic_is_deterministic() -> None:
    a = generate_debutanizer(100, seed=1)
    b = generate_debutanizer(100, seed=1)
    assert a.equals(b)
    assert list(a.columns) == [*INPUT_COLUMNS, "y"]
    assert a.to_numpy().min() >= 0.0
    assert a.to_numpy().max() <= 1.0


def test_synthetic_too_short() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        generate_debutanizer(3)


def test_missing_column(frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="u3"):
        validate_frame(frame.drop(columns="u3"), DataConfig(), require_target=True)


def test_target_optional_for_inference(frame: pd.DataFrame) -> None:
    out = validate_frame(frame.drop(columns="y"), DataConfig(), require_target=False)
    assert "y" not in out.columns


def test_non_numeric_column(frame: pd.DataFrame) -> None:
    bad = frame.assign(u1="сломанный датчик")
    with pytest.raises(DataValidationError, match="нечисловые"):
        validate_frame(bad, DataConfig(), require_target=True)


def test_too_many_missing(frame: pd.DataFrame) -> None:
    bad = frame.copy()
    bad.loc[: len(bad) // 10, "u2"] = np.nan
    with pytest.raises(DataValidationError, match="пропусков"):
        validate_frame(bad, DataConfig(), require_target=True)


def test_sparse_gaps_are_filled(frame: pd.DataFrame) -> None:
    gappy = frame.copy()
    gappy.loc[10, "u4"] = np.nan
    out = validate_frame(gappy, DataConfig(), require_target=True)
    assert not out.isna().any().any()
    assert out.loc[10, "u4"] == frame.loc[9, "u4"]


def test_load_dataset(csv_path: Path) -> None:
    assert len(load_dataset(csv_path, DataConfig())) == 600


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv", DataConfig())


def test_time_split_keeps_order(frame: pd.DataFrame) -> None:
    split = time_split(frame, 0.25)
    assert len(split.train) == 450
    assert split.train.index.max() < split.test.index.min()


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_time_split_bad_fraction(frame: pd.DataFrame, fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        time_split(frame, fraction)


def test_time_split_too_small(frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        time_split(frame.head(1), 0.4)
