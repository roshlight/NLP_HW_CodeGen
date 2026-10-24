"""Build a portable notebook without credentials or machine-specific paths."""

import base64
import argparse
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(max_new_tokens=64, target=None):
    if max_new_tokens not in (64, 256):
        raise ValueError("Choose the baseline 64 or revised 256 token limit")
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "src").rglob("*")):
            if p.suffix in (".py", ".yaml"):
                z.write(p, p.relative_to(ROOT))
        for name in ["inference.py", "scripts/assemble_checkpoint.py", "scripts/validate_submission.py", "scripts/prepare_questions.py"]:
            z.write(ROOT / name, name)
    encoded = base64.b64encode(packed.getvalue()).decode()
    cells = [
        cell("markdown", """# Генерация для бота: Transformer 107M / контекст 256

Выберите **NVIDIA T4** в Colab или Kaggle. Ноутбук скачивает официальный
`test` из датасета по ссылке в задании: 1919 вопросов без эталонных ответов.
Файл `pass1_questions.jsonl` создаётся в исходном порядке строк этого теста.
Одна программа на вопрос, жадная генерация, максимум **64 новых токена**.
Последняя ячейка измеряет загрузку модели, токенизацию, генерацию и запись.
Установка зависимостей и скачивание файлов выполняются заранее.
Выполнение на H100 не подтверждает ограничение 180 секунд на T4.
"""),
        cell("code", "%pip -q install hydra-core==1.3.2 tokenizers==0.23.3 pyarrow==23.0.1\n"),
        cell("code", f"""from pathlib import Path
import base64, io, json, sys, zipfile
work = Path.cwd()
root = work / 'code107m_inference'
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as archive:
    archive.extractall(root)
sys.path.insert(0, str(root))
questions = work / 'pass1_questions.jsonl'
from scripts.prepare_questions import prepare_questions
if not questions.is_file():
    print(prepare_questions(questions))
assert len(questions.read_text(encoding='utf-8').splitlines()) == 1919
"""),
        cell("markdown", """## Веса текущего запуска

Можно заранее распаковать `hw2_submission.zip` в рабочий каталог.
Следующая ячейка при отсутствии весов скачает части архива из репозитория,
проверит SHA-256 каждой части и всего архива и распакует его.
Иные чекпойнты не загружаются.
"""),
        cell("code", """import concurrent.futures
import urllib.request
from scripts.assemble_checkpoint import assemble, extract
model_dir = work / 'model'
if not (model_dir / 'best.pt').is_file():
    base = 'https://raw.githubusercontent.com/Kera-0/-/main/delivery/'
    parts_dir = work / 'delivery'
    parts_dir.mkdir(exist_ok=True)
    def download(name):
        assert Path(name).name == name
        target = parts_dir / name
        if not target.is_file():
            with urllib.request.urlopen(base + name, timeout=180) as response:
                temporary = target.with_suffix(target.suffix + '.part')
                with temporary.open('wb') as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                temporary.replace(target)
        return target
    manifest_path = download('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, [p['name'] for p in manifest['parts']]))
    archive = assemble(manifest_path, work / 'hw2_submission.zip')
    extract(archive, work)
assert (model_dir / 'tokenizer.json').is_file()
"""),
        cell("code", """from argparse import Namespace
from inference import run
report = run(Namespace(
    checkpoint=str(model_dir / 'best.pt'),
    tokenizer=str(model_dir / 'tokenizer.json'),
    questions=str(questions), output=str(work / 'submission.jsonl'),
    batch_size=128, max_new_tokens=64, device='cuda', dtype='fp16',
    expected_count=1919, require_t4=True, expected_parameters=106977792,
))
from scripts.validate_submission import validate
validation = validate(questions, work / 'submission.jsonl', work / 'submission.details.jsonl')
assert validation['order_verified']
print(validation)
assert report['under_180_seconds'], f"Превышен лимит T4: {report['total_seconds']:.2f} с"
print('Готово: отправьте submission.jsonl боту для вычисления pass@1')
"""),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):
        c["id"] = f"inference-{i}"
        if max_new_tokens == 256:
            source = "".join(c["source"])
            source = source.replace("**64 новых токена**", "**256 новых токенов**")
            source = source.replace("max_new_tokens=64", "max_new_tokens=256")
            source = source.replace("submission.jsonl", "submission_256.jsonl")
            source = source.replace("submission.details.jsonl", "submission_256.details.jsonl")
            source = source.replace("model_dir = work / 'model'\n", "model_dir = work / 'model'\nif not (model_dir / 'best.pt').is_file():\n    model_dir = work / 'weights_bundle' / 'model'\n")
            source = source.replace("extract(archive, work)", "extract(archive, model_dir.parent)")
            c["source"] = source.splitlines(True)
    target = Path(target) if target else ROOT / ("notebooks/inference_t4_256.ipynb" if max_new_tokens == 256 else "notebooks/inference_t4.ipynb")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, choices=[64, 256], default=64)
    parser.add_argument("--output")
    args = parser.parse_args()
    build(args.max_new_tokens, args.output)
