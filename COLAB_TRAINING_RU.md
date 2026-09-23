# CutoutNet — Colab training

Рекомендуемый первый production run на Tesla T4 16 GB:

- 640x640
- batch_size=2
- gradient accumulation=2 (effective batch 4)
- AMP on
- 30 epochs
- ConvNeXt-Tiny pretrained encoder
- encoder frozen only at epoch 1
- validation each epoch
- full resume checkpoint every epoch

В Colab измените `train.output_dir` в `config/train_colab.yaml` на каталог Google Drive, например:

`/content/drive/MyDrive/CutoutNet/checkpoints/t4_640`

`resume: auto` автоматически продолжит обучение с `last_full.pt`, включая optimizer, scheduler и GradScaler.

Запуск:

`python train_colab.py`

После 30 эпох можно сделать отдельный fine-tune 768x768 на 5-10 эпох с batch=1. Для него лучше начать с `best.pt` как pretrained model, а не продолжать scheduler старого 640-run вслепую.
