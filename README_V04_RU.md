# CutoutNet v0.4 — полная модульная архитектура для наклеек

**Расширение поверх исходного репозитория `ToxicTrial/corel` и v0.3, без удаления старых моделей и чекпоинтов.** Архитектура полностью написана, но её *качество зависит от обучения*. Новый код НЕ включает обученные InstanceNet, ColorCleanNet или ROI Refiner. На неподготовленных весах результата PhotoRoom не будет.

## Состав

- **CutoutNet (ConvNeXt-Tiny + декодер + 6 голов + Fusion)** — существующая основная сегментация и старая совместимая модель.
- **Validator** — карта риска на исходном разрешении, выбор ROI.
- **ROIRefiner** — локальная сверточная сеть (v0.3), обучается отдельно на синтетических масках или на **реальных ошибках основной сети**.
- **InstanceNet** — отдельная CNN с центрами объектов и векторами смещения к центрам, позволяет разделять соприкасающиеся экземпляры после обучения на integer-ID масках; резервный режим выделяет только отдельные связные компоненты.
- **ColorCleanNet** — отдельная CNN для коррекции RGB именно в узкой полосе по контуру, не меняя бинарную альфу.
- **StickerLoss** — расширение для дообучения базовой сети: BCE/Dice и головы из v0.2 + дополнительная взвешенная ошибка границ и soft-clDice.
- **FullPipeline** — единый инференс с поддержкой выборочного экспорта объектов, PNG RGBA с alpha 0/255 и SVG линии реза.
- **calibrate_v04.py** — выбор порога сегментации по *валидационным* изображениям.

## 0. Установка

В корне `corel` (обновлённая ветка v0.4):

```powershell
pip install -r requirements.txt
python check_gpu.py
```

PyTorch под CUDA устанавливается согласно базовому README. Все модели работают локально, но обучение на CPU крайне медленное. Скрипты используют только уже доступные изображения и чекпоинты.

## 1. Обработка прямо сейчас — старая обученная CutoutNet

```powershell
python cutout_v04.py "C:\images\sample.jpg" --checkpoint "checkpoints\cutoutnet\best.pt" --output outputs\v04 --diagnostics --export-instances --cut-svg
```

Если веса находятся на Google Drive, нужно скачать `best.pt` или `last_full.pt` в локальную папку либо смонтировать Drive в Colab. Они не находятся в GitHub (их исключает `.gitignore`).

Без `--instances` раздельные наклейки создаются по *связным компонентам*: соприкасающиеся экземпляры останутся одним компонентом. Исходная маска сохраняет все области, выбранные основной сетью.

## 2. Дообучение основной сети с дополнительными геометрическими потерями

В `config/train_sticker.yaml` укажите `train.init_checkpoint` на существующий `best.pt`. **Новый каталог `checkpoints/sticker` не перезаписывает старый `checkpoints/cutoutnet`.**

```powershell
python train_sticker.py --config config/train_sticker.yaml
```

Если модель ConvNeXt-Tiny уже обучалась и требуется аккуратная донастройка, не начинайте с нуля. Этап использует ранее обученные параметры, а оптимизатор создаётся новый. Внутренняя логика `trainer.py` расширена; старый запуск `train_colab.py` не меняется.

## 3. Обучение ROIRefiner на ошибках основной сети

Первый запуск можно сделать на синтетических повреждениях из v0.3:

```powershell
python -m cutoutnet.refiner_train --train-root data/cutoutnet/manifests/train.jsonl --val-root data/cutoutnet/manifests/val.jsonl --output checkpoints/refiner --crop 512 --epochs 20
```

Следующий этап — сформировать ошибки вашей **реально обученной** основной сети на отдельных train и val:

```powershell
python -m cutoutnet.real_refiner cache --dataset data/cutoutnet/manifests/train.jsonl --checkpoint checkpoints/sticker/best.pt --output data/cutoutnet/coarse/train
python -m cutoutnet.real_refiner cache --dataset data/cutoutnet/manifests/val.jsonl --checkpoint checkpoints/sticker/best.pt --output data/cutoutnet/coarse/val
python -m cutoutnet.real_refiner train --train-manifest data/cutoutnet/coarse/train/manifest.jsonl --val-manifest data/cutoutnet/coarse/val/manifest.jsonl --init checkpoints/refiner/best.pt --output checkpoints/refiner_real --epochs 20 --crop 512
```

Если `--init` нет, ROIRefiner обучается с нуля. Сравните валидационную ошибку `validation` с `no_refiner` и обязательно посмотрите настоящие изображения. Если сеть ухудшает маски, **не включайте её**.

## 4. Обучение InstanceNet

Нужны *экземплярные* маски, а не обычные бинарные: например, фон=0, первый кот=1, второй кот=2. Это 16-битные PNG, строго той же ширины/высоты, что и фотографии.

```text
data/instances/train/images/001.jpg
                         /002.jpg
                    /instances/001.png
                              /002.png
data/instances/val/images/...
                  /instances/...
```

```powershell
python -m cutoutnet.instance_train --train data/instances/train --val data/instances/val --output checkpoints/instance --epochs 40
```

Это отдельная небольшая CNN. Предсказание центров объектов и векторов смещения позволяет разделять соприкасающиеся объекты **после хорошего обучения**, но эта способность не гарантируется без верных ID-масок. Модель не выбирает семантически «главный» объект автоматически: выбор делается через ID, чтобы не угадывать намерения пользователя.

## 5. Обучение ColorCleanNet

Нужны чистые вырезанные PNG с альфа-каналом и сохранённым RGB объектов. Скрипт имитирует загрязнение границ случайными фонами.

```text
data/color/train/a.png
                 /b.png
          /val/c.png
              /d.png
```

```powershell
python -m cutoutnet.color_train --train data/color/train --val data/color/val --output checkpoints/color --epochs 30
```

Только пиксели в пределах нескольких пикселей **внутри** бинарной маски получают корректировку цвета. Полупрозрачности никогда не добавляются.

## 6. Проверка всей системы

```powershell
python calibrate_v04.py --data data/cutoutnet/manifests/val.jsonl --checkpoint checkpoints/sticker/best.pt --refiner checkpoints/refiner_real/best.pt --output checkpoints/sticker/calibration.json
python cutout_v04.py "C:\images" --checkpoint checkpoints/sticker/best.pt --refiner checkpoints/refiner_real/best.pt --instances checkpoints/instance/best.pt --color checkpoints/color/best.pt --calibration checkpoints/sticker/calibration.json --export-instances --diagnostics --cut-svg --output outputs/v04
```

Можно временно убрать `--refiner`, `--instances`, `--color` до появления *обученных* весов. Без этих аргументов случайные вспомогательные сети не запускаются.

Выбор нужных объектов:

```powershell
python cutout_v04.py "C:\images\sample.jpg" --checkpoint checkpoints/sticker/best.pt --instances checkpoints/instance/best.pt --keep-ids 1,3 --export-instances
```

Номера ID просмотрите в `*_diagnostics.json` и индивидуальных PNG. Метрика качества есть в `evaluate_v03.py`, её следует запускать на **test**, который не участвовал ни в обучении, ни в калибровке.

## Ограничения — критично

1. В репозитории исходного проекта нет файлов обученных весов. Их перенос и обучение на GPU нельзя подменить программным кодом.
2. InstanceNet учится только при наличии масок с разными ID для соприкасающихся объектов. Если таких данных нет, он не сможет надёжно различать соседние предметы.
3. Цветовая CNN помогает с ореолами, но без чистых RGBA-эталонов не гарантирует восстановления реального цвета.
4. Uncertainty Head в v0.2 обучен на широкой полосе около GT границы, что является эвристической разметкой, а не настоящей калиброванной вероятностью ошибки. Validator оценивает риск, но не предоставляет доказанную вероятность ошибки.
5. SVG-контур подходит как геометрическая заготовка, но перед печатью нужно проверить реальные DPI, отступ и требования плоттера. Режим `unified` использует выпуклую оболочку.
6. Ни одна проверка кода не заменяет оценку качества обученной модели на реальных фотографиях. Цель уровня PhotoRoom требует серьёзного сравнения и дообучения на сложных изображениях.

Локальные smoke-тесты `tests/test_v04.py`: архитектура instance, градиенты, синтетическое разделение объектов, сохранение внутренних отверстий и тонких компонент, наборы данных и изменение RGB только на границе. Повторная CI-проверка полного базового репозитория выполняется отдельно.
