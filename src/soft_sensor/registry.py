"""Реестр версий модели на файловой системе.

Структура::

    models/registry/
    ├── v0001/  model.joblib, metadata.json, report.json
    ├── v0002/  ...
    └── production.json   ← какая версия сейчас в работе + история выкладок

В реестр попадает **каждая** обученная версия, в том числе отклонённая
проверками, — чтобы по отчёту было видно, почему её не выпустили.
Рабочая модель копируется в ``artifacts_dir`` (по умолчанию ``models/``),
откуда её загружает сервис. На предприятии роль реестра выполняет
MLflow Model Registry или сетевое хранилище; интерфейс тот же.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from soft_sensor.model import METADATA_FILE, MODEL_FILE, save_artifact

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from soft_sensor.model import ModelMetadata

logger = logging.getLogger(__name__)

PRODUCTION_FILE = "production.json"
REPORT_FILE = "report.json"
_VERSION_RE = re.compile(r"^v(\d{4,})$")


class RegistryError(RuntimeError):
    """Операция с реестром невозможна (нет версии, некуда откатываться)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ModelRegistry:
    """Версии модели в каталоге ``root`` и указатель на рабочую версию."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # ------------------------------------------------------------------ чтение
    def versions(self) -> list[str]:
        """Все зарегистрированные версии по возрастанию."""
        if not self.root.is_dir():
            return []
        found = [p.name for p in self.root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
        return sorted(found, key=lambda v: int(v[1:]))

    def path(self, version: str) -> Path:
        """Каталог версии; ошибка, если её нет."""
        path = self.root / version
        if not (path / MODEL_FILE).is_file():
            msg = f"версии {version} нет в реестре {self.root}"
            raise RegistryError(msg)
        return path

    def _state(self) -> dict[str, Any]:
        file = self.root / PRODUCTION_FILE
        if not file.is_file():
            return {"current": None, "history": []}
        state: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
        return state

    def production_version(self) -> str | None:
        """Версия, которая сейчас в работе, или ``None``, если выкладок ещё не было."""
        current: str | None = self._state()["current"]
        return current

    def history(self) -> list[dict[str, str]]:
        """История выкладок: версия, время, причина."""
        history: list[dict[str, str]] = self._state()["history"]
        return history

    def read_report(self, version: str) -> dict[str, Any]:
        """Отчёт о переобучении, сохранённый вместе с версией."""
        report: dict[str, Any] = json.loads(
            (self.path(version) / REPORT_FILE).read_text(encoding="utf-8")
        )
        return report

    # ------------------------------------------------------------------ запись
    def register(self, pipeline: Pipeline, metadata: ModelMetadata) -> str:
        """Сохраняет новую версию (не выкладывая её) и возвращает её номер."""
        existing = self.versions()
        number = int(existing[-1][1:]) + 1 if existing else 1
        version = f"v{number:04d}"
        save_artifact(pipeline, metadata, self.root / version)
        logger.info("Зарегистрирована версия %s", version)
        return version

    def write_report(self, version: str, report: dict[str, Any]) -> None:
        """Сохраняет отчёт о переобучении рядом с версией."""
        (self.path(version) / REPORT_FILE).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def promote(self, version: str, deploy_dir: Path, reason: str) -> None:
        """Делает версию рабочей: копирует артефакт в ``deploy_dir`` и пишет историю.

        Файлы сначала копируются во временный каталог рядом с целевым и
        только потом подменяют рабочие, чтобы сервис не прочитал
        наполовину скопированную модель.
        """
        source = self.path(version)
        deploy_dir.mkdir(parents=True, exist_ok=True)
        staging = deploy_dir / ".staging"
        staging.mkdir(exist_ok=True)
        for name in (MODEL_FILE, METADATA_FILE):
            shutil.copy2(source / name, staging / name)
        for name in (MODEL_FILE, METADATA_FILE):
            (staging / name).replace(deploy_dir / name)
        staging.rmdir()

        self._set_current(version, reason)
        logger.info("Версия %s выложена в %s (%s)", version, deploy_dir, reason)

    def retire(self, deploy_dir: Path, reason: str) -> None:
        """Снимает модель с работы, когда откатываться некуда: сервис выдаёт «нет оценки»."""
        for name in (MODEL_FILE, METADATA_FILE):
            (deploy_dir / name).unlink(missing_ok=True)
        self._set_current(None, reason)
        logger.warning("Рабочая модель снята: %s", reason)

    def _set_current(self, version: str | None, reason: str) -> None:
        state = self._state()
        state["current"] = version
        if version is not None:
            state["history"].append({"version": version, "promoted_at": _now(), "reason": reason})
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / PRODUCTION_FILE).write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def rollback(self, deploy_dir: Path, reason: str = "rollback") -> str:
        """Возвращает в работу предыдущую выложенную версию и возвращает её номер."""
        history = self.history()
        current = self.production_version()
        previous = [h["version"] for h in history if h["version"] != current]
        if current is None or not previous:
            msg = "откатываться некуда: другой выложенной версии нет"
            raise RegistryError(msg)
        target = previous[-1]
        self.promote(target, deploy_dir, reason)
        return target
