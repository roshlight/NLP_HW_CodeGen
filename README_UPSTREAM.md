# HSE NLP HW2 — генерация кода

Encoder–decoder Transformer: **106 977 792 параметра**, контекст **256 токенов**.
Модель и BPE-токенизатор обучаются с нуля только на `SlavaYus/tinysms_qa_12m`
(revision `b2724689ff079e9c22efd548fe134bfb29036dbb`). Предобученные веса,
данные и токенизаторы прошлых экспериментов не используются.

Обучение завершено: **77 474 шага**, 3.54 ч на H100. **Текущий файл для бота:
[submission_256.jsonl](submission_256.jsonl)** — лимит генерации увеличен с 64 до
256 токенов на том же чекпойнте. Генерация всех 1919 программ заняла **8.85 с
на H100**, включая загрузку модели. Фактический замер на T4 пока отсутствует.

Первая посылка [submission.jsonl](submission.jsonl), лимит 64, получила
**47.71% public pass@1 (458/960)** по сообщению бота. Результат нового файла
пока неизвестен. В полном выводе синтаксических ошибок **1 вместо 194**,
программ без `return` среди разобранных — **5 вместо 596**.
Это проверки структуры кода, не оценка математической правильности.

## Генерация для бота

Нужны веса этого запуска и его `tokenizer.json`. Вопросы взяты из официального
`test` датасета по ссылке из задания: **1919 строк, только поле `question`**.
Они сохранены в [pass1_questions.jsonl](pass1_questions.jsonl) без изменения
порядка. Источник и SHA-256 записаны в [artifacts/question_source.json](artifacts/question_source.json).
Отдельный файл от Telegram-бота для сравнения не получен; используется порядок
официального теста задания. Веса находятся в полном архиве в `delivery/`; контрольные суммы включены.

Исходный полный архив с весами хранится частями до 48 MiB в `delivery/`.
Он сохраняет первую версию с лимитом 64. Обновлённые исходники, ноутбук,
отчёт и файл ответов находятся в [delivery/update_256.zip](delivery/update_256.zip);
обновление использует те же веса. Для новой версии распакуйте базовый архив
в отдельную папку, чтобы сохранить актуальные файлы репозитория:

```bash
python scripts/assemble_checkpoint.py --output /tmp/hw2_submission.zip
python -m zipfile -e /tmp/hw2_submission.zip weights_bundle
```

```bash
pip install -r requirements.txt
python scripts/prepare_questions.py
python inference.py --checkpoint weights_bundle/model/best.pt --tokenizer weights_bundle/model/tokenizer.json \
  --questions pass1_questions.jsonl --output submission_256.jsonl \
  --max-new-tokens 256 --batch-size 128 --dtype fp16
python scripts/validate_submission.py --questions pass1_questions.jsonl \
  --submission submission_256.jsonl --details submission_256.details.jsonl
```

Отправлять боту нужно только **`submission_256.jsonl`**: 1919 объектов вида
`{"code": "..."}`, по одному на строку, в исходном порядке вопросов.
Используется жадная генерация одной программы на задачу, максимум 256 новых токенов.
Предсказания сохраняются без исправления кода или подстановки ответов.

Ноутбук [notebooks/inference_t4_256.ipynb](notebooks/inference_t4_256.ipynb) предназначен
для Kaggle/Colab с T4. Он проверяет реальную модель GPU и измеряет время загрузки
модели, токенизации, генерации и записи файла. **Замер на T4 пока не выполнен**;
скорость на H100 не подтверждает ограничение 180 секунд.
Исходный [ноутбук с лимитом 64](notebooks/inference_t4.ipynb) оставлен для
воспроизведения первой посылки. Сравнение версий:
[artifacts/revision_256/comparison.json](artifacts/revision_256/comparison.json).

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
