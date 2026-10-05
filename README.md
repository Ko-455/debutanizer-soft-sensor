# soft-sensor: виртуальный анализатор бутана в кубе дебутанизатора

Содержание C4 (бутана) в кубовом продукте дебутанизатора — ключевой показатель
качества, но лаборатория определяет его хроматографом редко и с задержкой.
Виртуальный анализатор (soft sensor) восстанавливает его непрерывно по тегам
АСУ ТП, которые и так пишутся каждую минуту.

| Тег | Что это |
|-----|---------|
| `u1` | температура верха колонны |
| `u2` | давление верха |
| `u3` | расход флегмы |
| `u4` | расход на следующую установку |
| `u5` | температура 6-й тарелки |
| `u6`, `u7` | температура низа (два датчика) |
| `y` | содержание бутана в кубе (цель) |

Структура данных повторяет эталонный датасет Fortuna et al. (2007,
*Soft Sensors for Monitoring and Control of Industrial Processes*). Пока нет
реальной выгрузки, `soft-sensor generate-data` создаёт синтетический ряд с
похожей динамикой; реальный CSV с теми же колонками кладётся в
`data/raw/debutanizer.csv` без изменений в коде.

## Быстрый старт

Нужны Python 3.11–3.13, [Poetry](https://python-poetry.org/docs/#installation) ≥ 2.0 и `make`.

```bash
git clone <repo> && cd debutanizer-soft-sensor
make setup        # .venv по poetry.lock + git-хуки pre-commit
make train        # синтетические данные → обучение → models/
make predict      # прогноз → data/predictions.csv
```

Без `make`:

```bash
poetry install --with dev
poetry run pre-commit install --install-hooks
poetry run soft-sensor generate-data
poetry run soft-sensor train --config configs/train.yaml
poetry run soft-sensor predict data/raw/debutanizer.csv --out data/predictions.csv
```

## Виртуальное окружение и Git

Окружение «интегрировано» в репозиторий через описание, а не через файлы:

| В Git | Зачем |
|-------|-------|
| `pyproject.toml` | зависимости и настройки ruff / mypy / pytest |
| `poetry.lock` | точные версии всех пакетов, включая транзитивные |
| `poetry.toml` | `in-project = true` — окружение создаётся в `./.venv` |
| `.python-version` | версия интерпретатора для pyenv / uv / IDE |
| `.vscode/settings.json` | IDE сама подхватывает `.venv/bin/python` |

Сама папка `.venv/` в `.gitignore`. Её не коммитят: это сотни мегабайт
бинарников, привязанных к ОС и пути на диске, — на чужой машине они не
запустятся. `poetry install` воссоздаёт её из `poetry.lock` бит-в-бит,
а хук `poetry check --lock` не даёт закоммитить `pyproject.toml`,
рассинхронизированный с lock-файлом.

Добавить зависимость: `poetry add <pkg>` (или `poetry add --group dev <pkg>`) —
обновятся оба файла, их и коммитим вместе.

## Качество кода

| Инструмент | Что проверяет | Когда |
|------------|---------------|-------|
| ruff check | pyflakes, pycodestyle, isort, bugbear, pandas-vet, bandit, docstrings | commit |
| ruff format | форматирование (аналог black) | commit |
| mypy --strict | типы, плагин pydantic | commit |
| nbstripout | выводы ноутбуков не попадают в Git | commit |
| pre-commit-hooks | пробелы, YAML/TOML, большие файлы, приватные ключи, конфликты | commit |
| poetry check --lock | согласованность pyproject и lock | commit |
| pytest + coverage ≥ 85 % | тесты | push, CI |

`make check` прогоняет всё то же, что CI (`.github/workflows/ci.yml`).

## Структура

```
├── configs/train.yaml        # гиперпараметры и пути (валидируются pydantic)
├── data/raw/                 # данные — вне Git
├── models/                   # артефакты — вне Git
├── notebooks/01_prototype.ipynb  # исходный исследовательский код («до»)
├── src/soft_sensor/
│   ├── config.py             # схема конфига
│   ├── synthetic.py          # генератор данных
│   ├── data.py               # загрузка, валидация, разбиение по времени
│   ├── features.py           # лаговые признаки — одна функция для train и predict
│   ├── model.py              # пайплайны, выбор по CV, метрики, артефакт
│   ├── train.py              # сценарий обучения
│   ├── predict.py            # SoftSensor — применение модели
│   └── cli.py                # soft-sensor generate-data | train | predict
└── tests/                    # pytest, в т.ч. тест на утечку будущего
```

Что именно и почему изменилось по сравнению с ноутбуком — в
[docs/REFACTORING.md](docs/REFACTORING.md).

## Артефакт модели

`models/model.joblib` — sklearn-`Pipeline` (скейлер + регрессор),
`models/metadata.json` — имя модели, список признаков, лаги, метрики на
тесте, RMSE на кросс-валидации, версия пакета, время обучения. При
прогнозе признаки строятся по параметрам из метаданных, и состав
признаков сверяется с тем, на котором училась модель.

Загружайте `model.joblib` только из доверенного источника: joblib/pickle
исполняет код при загрузке.
