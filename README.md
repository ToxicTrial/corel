# CutoutNet Training

Минимальный репозиторий для обучения собственной сети удаления фона CutoutNet.

Это **не архив старых версий CutoutLab**. В репозитории оставлены только новая нейросеть, обучение, оценка качества и инструменты подготовки датасета.

## Что намеренно НЕ хранится в Git

- P3M-10K / AM-2K / AIM-500 и любые другие датасеты;
- синтетические изображения;
- экспортированные RGBA cutouts;
- checkpoints (`*.pt`, `*.pth`, `*.ckpt`);
- кэш моделей;
- benchmark/output/logs;
- виртуальное окружение `.venv`;
- zip-архивы старых версий.

Все эти пути закрыты в `.gitignore`.

## Быстрый старт на новом Windows ПК

1. Установить Python 3.12 и Git.
2. Клонировать репозиторий.
3. Запустить `install_windows.bat`.
4. Установить CUDA-сборку PyTorch с https://pytorch.org/get-started/locally/ для своей видеокарты.
5. Проверить `python check_gpu.py`. Должно быть `CUDA available: True`.
6. Скачать датасет отдельно и положить его в `data/raw/...`.
7. Запустить `dataset_wizard_windows.bat` и выполнить import + QA.
8. Для smoke-test поставить в `config/train.yaml` `epochs: 2`.
9. Запустить `train_windows.bat`.
10. После проверки вернуть нужное число эпох и продолжить обучение.

## Основные файлы

- `cutoutnet/model.py` — CutoutNet: foreground, boundary, detail, residual-background, uncertainty, objectness и fusion heads.
- `cutoutnet/trainer.py` — обучение и checkpoints.
- `cutoutnet/dataset.py` — чтение image/mask pairs и manifests.
- `tools/dataset_manager.py` — импорт, manifests и QA.
- `tools/build_synthetic_dataset.py` — синтетические multi-object сцены.
- `config/train.yaml` — параметры обучения.

## Данные

Репозиторий хранит только пустую структуру каталогов. Сам датасет переносится отдельно или скачивается заново.

После импорта Dataset Wizard создаёт локально:

```text
data/cutoutnet/registry.jsonl
data/cutoutnet/manifests/train.jsonl
data/cutoutnet/manifests/val.jsonl
data/cutoutnet/manifests/test.jsonl
```

Они также не коммитятся, потому что содержат локальные пути и зависят от конкретного ПК.

## Checkpoints

Во время обучения создаются:

```text
checkpoints/cutoutnet/best.pt
checkpoints/cutoutnet/last.pt
checkpoints/cutoutnet/history.json
```

Весовые файлы специально исключены из Git. Для переноса `best.pt` лучше использовать отдельное облачное хранилище или GitHub Release, если размер позволяет.

## GitHub / перенос на обучающий ПК

Репозиторий специально не содержит датасеты, checkpoints, outputs, кэш моделей и архивы. После `git clone` на другом ПК нужно заново положить исходный датасет в `data/raw/` и запустить Dataset Wizard.

Для проверки, что в Git случайно не попали крупные файлы, перед push можно выполнить:

```bash
git status
git ls-files
```

Вес `best.pt` также не коммитится. Его нужно переносить отдельно либо прикладывать к GitHub Release после обучения.

Cloud portability fixes in v0.2.4:
- manifests keep project-relative paths through data/raw symlinks
- dataset loader remaps stale Windows and previous Colab absolute paths
