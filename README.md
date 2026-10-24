# Code107M — генерация Python-кода

Encoder–decoder Transformer с **106 977 792 параметрами**, обученный с нуля
на `SlavaYus/tinysms_qa_12m`. Предобученные веса и дополнительные обучающие
данные не используются.

## Архитектура

- 8 слоёв энкодера, 4 слоя декодера; размерность 768, 12 голов внимания.
- SwiGLU 2048, RoPE, RMSNorm, Q/K RMS-нормализация.
- Общая матрица embeddings и выходной проекции; собственный BPE-токенизатор.
- Контекст 256 токенов; BF16 при обучении, KV-cache при генерации.

## Окружение

Python 3.11+, PyTorch с CUDA и зависимости из `requirements.txt`.
Для GPU используйте совместимую CUDA-сборку PyTorch. В исходном запуске
использовались H100 и PyTorch `2.9.1+cu128`; перед установкой зависимостей
учтите закреплённую в `requirements.txt` версию PyTorch.

## Обучение

```bash
python prepare.py --output data/processed --max-source 256 --max-target 256
python preflight.py -cn reproduce
python train.py -cn reproduce
```

Данные и токенизатор готовятся только по обучающей части датасета. Validation
выделяется по хешу вопроса; длинные пары удаляются целиком, без обрезки программ.
Конфиг `reproduce` фиксирует 77 474 шага исходного запуска. Фактические настройки
и логи находятся в `artifacts/`.

## Веса модели

Части архива весов (~399 МБ) не включены в эту компактную копию.
Для восстановления модели выполните из корня проекта:

```bash
for part in 000 001 002 003 004 005 006 007; do
  curl -fL --retry 3 \
    "https://raw.githubusercontent.com/Kera-0/-/94a7bb8ea94f33c0de64efdab8796ee409a300ed/delivery/hw2_submission.zip.$part" \
    -o "delivery/hw2_submission.zip.$part"
done
python scripts/assemble_checkpoint.py --extract
```

Скрипт проверяет контрольные суммы. После распаковки доступны
`model/best.pt` и `model/tokenizer.json`.

## Инференс

```bash
python inference.py --checkpoint model/best.pt --tokenizer model/tokenizer.json \
  --questions pass1_questions.jsonl --output predictions.jsonl \
  --max-new-tokens 64 --batch-size 128 --dtype fp16
```

Сохраняется исходный порядок вопросов. Параметры генерации соответствуют
исходному репозиторию. Для запуска в Kaggle/Colab предусмотрен
`notebooks/inference_t4.ipynb`.

## Проверки

```bash
python -m pytest -q
```

## Происхождение

[Исходный репозиторий](https://github.com/Kera-0/-), commit
`94a7bb8ea94f33c0de64efdab8796ee409a300ed`.
Код и остальные скачанные файлы сохранены без изменений. Сокращён только README;
оригинал сохранён в `README_UPSTREAM.md`, контрольные суммы — в `DOWNLOAD_MANIFEST.json`.
