"""Verify the downloaded run, then build its report and complete submission archive."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def finalize(delivery, font, archive):
    sys.path.insert(0, str(ROOT))
    from scripts.validate_submission import validate
    from scripts.build_inference_notebook import build as build_notebook
    from scripts.build_report import build as build_report
    from scripts.package_submission import package

    delivery = Path(delivery)
    artifacts = ROOT / "artifacts"
    summary = json.loads((artifacts / "training_summary.json").read_text())
    export = json.loads((artifacts / "export_status.json").read_text())
    if not summary.get("completed") or export.get("phase") != "completed":
        raise ValueError("Training and final generation must both be completed")
    timing = json.loads((artifacts / "submission.timing.json").read_text())
    if timing["max_new_tokens"] != 64 or timing["parameters"] != 106977792 or timing["questions"] != 1919:
        raise ValueError("Submission does not match the requested generation settings")
    questions = ROOT / "pass1_questions.jsonl"
    source = json.loads((artifacts / "question_source.json").read_text())
    if hashlib.sha256(questions.read_bytes()).hexdigest() != source["questions_sha256"]:
        raise ValueError("Official test questions changed")
    validation = validate(questions, ROOT / "submission.jsonl", artifacts / "submission.details.jsonl")
    if validation["submission_sha256"] != timing["submission_sha256"] or validation["questions_sha256"] != timing["questions_sha256"]:
        raise ValueError("Inference timing does not belong to the supplied questions and answers")
    validation["question_source"] = source
    (artifacts / "submission_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    import torch
    from src.model.transformer import CodeTransformer
    state = torch.load(delivery / "model/best.pt", map_location="cpu", weights_only=True)
    metrics = [json.loads(line) for line in (artifacts / "metrics.jsonl").read_text().splitlines()]
    best_validation = min((row for row in metrics if "val/nll" in row), key=lambda row: row["val/nll"])
    if state["step"] != best_validation["step"]:
        raise ValueError("Downloaded checkpoint is not the recorded best validation checkpoint")
    config = state["config"]["model"].copy()
    config.pop("_target_")
    with torch.device("meta"):
        model = CodeTransformer(**config)
    model.load_state_dict(state["model"], assign=True)
    if sum(p.numel() for p in model.parameters()) != 106977792:
        raise ValueError("Incorrect model size")
    if model.max_length != 256 or not config.get("qk_norm"):
        raise ValueError("Wrong final architecture")
    for name, tensor in state["model"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    tokenizer_hash = hashlib.sha256((delivery / "model/tokenizer.json").read_bytes()).hexdigest()
    if tokenizer_hash != state["metadata"]["tokenizer_sha256"]:
        raise ValueError("Wrong tokenizer for the checkpoint")
    checkpoint_info = {"step": state["step"], "parameters": 106977792,
                       "max_length": model.max_length, "qk_norm": True,
                       "all_weights_finite": True, "tokenizer_sha256": tokenizer_hash}
    (artifacts / "checkpoint_verification.json").write_text(json.dumps(checkpoint_info, indent=2) + "\n")
    del model, state
    provenance = json.loads((artifacts / "source_provenance.json").read_text())
    provenance["training_complete"] = True
    (artifacts / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    readme = ROOT / "README.md"
    text = readme.read_text()
    text = text.replace("Обучение текущего запуска ещё выполняется. Файл ответов, финальные веса и PDF\nбудут добавлены после его завершения.",
        f"Обучение завершено: **{summary['steps']:,} шагов**, {summary['training_hours']:.2f} ч на H100. "
        f"Готовый файл для бота: [submission.jsonl](submission.jsonl). "
        f"Отчёт: [report/report.pdf](report/report.pdf). "
        f"Генерация 1919 программ на H100 заняла {timing['total_seconds']:.2f} с с учётом загрузки; "
        "фактический замер на T4 пока отсутствует.")
    text = text.replace("Веса будут опубликованы с контрольными суммами.", "Веса находятся в полном архиве в `delivery/`; контрольные суммы включены.")
    text = text.replace("Полный архив будет храниться частями", "Полный архив хранится частями")
    readme.write_text(text)
    build_notebook()
    build_report(font)
    package(delivery / "model", questions, archive)
    print(json.dumps({"submission": str(ROOT / "submission.jsonl"), "archive": str(archive),
                      "validation": validation, "checkpoint": checkpoint_info}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--font", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    finalize(args.delivery, args.font, Path(args.archive))
