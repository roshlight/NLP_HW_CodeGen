# HSE NLP HW2 — генерация кода

Encoder–decoder Transformer: **106 977 792 параметра**, контекст **256 токенов**.
Модель и BPE-токенизатор обучаются с нуля только на `SlavaYus/tinysms_qa_12m`
(revision `b2724689ff079e9c22efd548fe134bfb29036dbb`). Предобученные веса,
данные и токенизаторы прошлых экспериментов не используются.

Обучение завершено: **77,474 шагов**, 3.54 ч на H100. Готовый файл для бота: [submission.jsonl](submission.jsonl). Отчёт: [report/report.pdf](report/report.pdf). Генерация 1919 программ на H100 заняла 4.55 с с учётом загрузки; фактический замер на T4 пока отсутствует.

## Генерация для бота

Нужны веса этого запуска и его `tokenizer.json`. Вопросы взяты из официального
`test` датасета по ссылке из задания: **1919 строк, только поле `question`**.
Они сохранены в [pass1_questions.jsonl](pass1_questions.jsonl) без изменения
порядка. Источник и SHA-256 записаны в [artifacts/question_source.json](artifacts/question_source.json).
Отдельный файл от Telegram-бота для сравнения не получен; используется порядок
официального теста задания. Веса находятся в полном архиве в `delivery/`; контрольные суммы включены.

Полный архив хранится частями до 48 MiB в `delivery/` из-за ограничения
GitHub на размер одного файла. Сборка и проверка:

```bash
python scripts/assemble_checkpoint.py --extract
```

```bash
pip install -r requirements.txt
python scripts/prepare_questions.py
python inference.py --checkpoint model/best.pt --tokenizer model/tokenizer.json \
  --questions pass1_questions.jsonl --output submission.jsonl \
  --max-new-tokens 64 --batch-size 128 --dtype fp16
python scripts/validate_submission.py --questions pass1_questions.jsonl \
  --submission submission.jsonl --details submission.details.jsonl
```

Отправлять боту нужно только `submission.jsonl`: 1919 объектов вида
`{"code": "..."}`, по одному на строку, в исходном порядке вопросов.
Используется жадная генерация одной программы на задачу, максимум 64 новых токена.
Предсказания сохраняются без исправления кода или подстановки ответов.

Ноутбук [notebooks/inference_t4.ipynb](notebooks/inference_t4.ipynb) предназначен
для Kaggle/Colab с T4. Он проверяет реальную модель GPU и измеряет время загрузки
модели, токенизации, генерации и записи файла. **Замер на T4 пока не выполнен**;
скорость на H100 не подтверждает ограничение 180 секунд.

## Обучение

Нужен Python 3.11+ и CUDA PyTorch (на H100 используется `2.9.1+cu128`).
Поставьте подходящий CUDA wheel PyTorch, затем остальные зависимости из
`requirements.txt`. В файле закреплена локальная версия CPU PyTorch;
на GPU не следует заменять уже установленную рабочую CUDA-сборку.

```bash
python prepare.py --output data/processed --max-source 256 --max-target 256
python preflight.py -cn reproduce
python train.py -cn reproduce
python -m pytest -q
```

Подготовка скачивает только обучающие шарды официального набора. Валидация
выделяется по хешу вопроса; её примеры исключены из обучения токенизатора.
Примеры длиннее 256 токенов удаляются целиком, без обрезки программ.
Конфигурация `reproduce` фиксирует фактически выбранные 77 474 шага и отключает
повторную калибровку расписания по скорости другой машины. Исходная конфигурация
реального запуска сохранена в артефактах и в архиве модели.

Архитектура: 8 слоёв энкодера, 4 декодера, размерность 768, 12 голов,
SwiGLU 2048, RoPE в self-attention, RMSNorm, общие embedding/output веса.
Q/K RMS-нормализация без обучаемых параметров ограничивает рост attention logits.
Обучение: BF16, AdamW, LR 1e-4, warmup 1000, cosine decay, effective batch 256.

История текущей работы и участие агента описаны в
[report/experiments.md](report/experiments.md). Диагностические метрики не являются
официальным pass@1; его считает бот после отправки файла пользователем.

PDF собирается из финальных логов: установите `requirements-report.txt` и
запустите `python scripts/build_report.py --font /path/to/cyrillic-font.ttf`.
Скрипт откажется выпускать итоговый PDF до завершения обучения.

`operations/export_watch.py` ожидает завершения H100-запуска и генерирует
`submission.jsonl` на переданном официальном тесте. Проверка связывает каждую
программу с исходной строкой вопроса; происхождение вопросов записывается отдельно.
