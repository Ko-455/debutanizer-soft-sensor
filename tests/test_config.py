from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from soft_sensor.config import FeatureConfig, ModelConfig, TrainConfig, load_config

if TYPE_CHECKING:
    from pathlib import Path


def test_defaults_are_valid() -> None:
    cfg = TrainConfig()
    assert cfg.features.max_lag == 6
    assert cfg.model.candidates == ("ridge", "hgb")


def test_lags_sorted() -> None:
    assert FeatureConfig(lags=(3, 1, 2)).lags == (1, 2, 3)


@pytest.mark.parametrize("lags", [(1, 1), (0,), (-2,)])
def test_bad_lags_rejected(lags: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        FeatureConfig(lags=lags)


def test_empty_candidates_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(candidates=())


def test_load_config_rejects_typos(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("split:\n  tset_fraction: 0.3\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_partial(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("split:\n  cv_splits: 3\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.split.cv_splits == 3
    assert cfg.split.test_fraction == 0.2


def test_load_empty_config(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == TrainConfig()


def test_repo_config_is_valid() -> None:
    from pathlib import Path

    repo_cfg = Path(__file__).parents[1] / "configs" / "train.yaml"
    assert load_config(repo_cfg).model.random_state == 42
