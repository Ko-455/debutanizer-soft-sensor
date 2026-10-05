"""Виртуальный анализатор (soft sensor) содержания бутана в кубе дебутанизатора."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("soft-sensor")
except PackageNotFoundError:  # пакет запущен из исходников без установки
    __version__ = "0.0.0"

__all__ = ["__version__"]
