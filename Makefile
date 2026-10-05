.DEFAULT_GOAL := help
RUN := poetry run

.PHONY: help setup lint format typecheck test check data train predict clean

help:  ## Список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Создать .venv по poetry.lock и поставить git-хуки
	poetry install --with dev
	$(RUN) pre-commit install --install-hooks

lint:  ## ruff: линтер и проверка форматирования
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## ruff: автоисправление и форматирование
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck:  ## mypy --strict
	$(RUN) mypy

test:  ## pytest с покрытием
	$(RUN) pytest

check:  ## Все хуки pre-commit по всем файлам (то же, что в CI)
	$(RUN) pre-commit run --all-files
	$(RUN) pytest

data:  ## Сгенерировать синтетический датасет
	$(RUN) soft-sensor generate-data

train: data  ## Обучить модель
	$(RUN) soft-sensor train --config configs/train.yaml

predict:  ## Прогноз по data/raw/debutanizer.csv
	$(RUN) soft-sensor predict data/raw/debutanizer.csv --out data/predictions.csv

clean:  ## Удалить кэши и артефакты (не трогает .venv)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist
	find . -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
