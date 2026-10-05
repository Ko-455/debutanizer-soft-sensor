from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from soft_sensor.config import INPUT_COLUMNS, FeatureConfig
from soft_sensor.features import BOTTOM_TEMPERATURE_MEAN, build_features, feature_names

if TYPE_CHECKING:
    import pandas as pd


def test_shape_and_names(frame: pd.DataFrame) -> None:
    cfg = FeatureConfig(lags=(1, 3))
    x = build_features(frame, INPUT_COLUMNS, cfg)
    assert len(x) == len(frame) - 3
    assert list(x.columns) == feature_names(INPUT_COLUMNS, cfg)
    assert len(x.columns) == 7 + 7 * 2 + 1
    assert not x.isna().any().any()


def test_lag_values_come_from_the_past(frame: pd.DataFrame) -> None:
    x = build_features(frame, INPUT_COLUMNS, FeatureConfig(lags=(2,)))
    t = 100
    assert x.loc[t, "u5_lag2"] == frame.loc[t - 2, "u5"]


def test_no_future_leakage(frame: pd.DataFrame) -> None:
    """Изменение будущего не должно менять признаки прошлого."""
    cfg = FeatureConfig()
    base = build_features(frame, INPUT_COLUMNS, cfg)
    tampered = frame.copy()
    tampered.loc[300:, list(INPUT_COLUMNS)] = 0.0
    changed = build_features(tampered, INPUT_COLUMNS, cfg)
    np.testing.assert_array_equal(base.loc[:299].to_numpy(), changed.loc[:299].to_numpy())


def test_bottom_mean_optional(frame: pd.DataFrame) -> None:
    x = build_features(frame, INPUT_COLUMNS, FeatureConfig(add_bottom_temperature_mean=False))
    assert BOTTOM_TEMPERATURE_MEAN not in x.columns


def test_index_aligns_with_target(frame: pd.DataFrame) -> None:
    x = build_features(frame, INPUT_COLUMNS, FeatureConfig())
    assert x.index[0] == 6
    assert frame.loc[x.index, "y"].shape[0] == len(x)
