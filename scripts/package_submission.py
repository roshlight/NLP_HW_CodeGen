"""Package the verified completed run and split it below GitHub's file limit."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(model, questions, output):
    model, questions, output = Path(model), Path(questions), Path(output)
    required = [ROOT / "submission.jsonl", ROOT / "report/report.pdf",
                ROOT / "artifacts/submission_validation.json", ROOT / "notebooks/inference_t4.ipynb"]
    required += [model / name for name in ["best.pt", "tokenizer.json", "config.yaml", "data_metadata.json"]]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError("Final submission, PDF, validation and model files must all exist")
    validation = json.loads((ROOT / "artifacts/submission_validation.json").read_text())
    if not validation["order_verified"] or validation["rows"] != 1919:
        raise ValueError("Exact question order must be verified before packaging")
    if hashlib.sha256(questions.read_bytes()).hexdigest() != validation["questions_sha256"]:
        raise ValueError("Question file changed after validation")
    if hashlib.sha256((ROOT / "submission.jsonl").read_bytes()).hexdigest() != validation["submission_sha256"]:
        raise ValueError("Submission changed after validation")
    files = [ROOT / name for name in ["README.md", "requirements.txt", "requirements-report.txt", "pyproject.toml",
             "inference.py", "train.py", "prepare.py", "preflight.py", "submission.jsonl"]]
    for name in ["src", "scripts", "operations", "tests", "notebooks", "report", "artifacts"]:
        files += [p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pt"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for p in sorted(set(files)):
            archive.write(p, p.relative_to(ROOT))
        archive.write(questions, "pass1_questions.jsonl")
        for name in ["best.pt", "tokenizer.json", "config.yaml", "data_metadata.json"]:
            archive.write(model / name, "model/" + name)
    delivery = ROOT / "delivery"
    delivery.mkdir(exist_ok=True)
    parts, total = [], hashlib.sha256()
    with output.open("rb") as stream:
        for index in range(1000):
            chunk = stream.read(48 * 1024 * 1024)
            if not chunk:
                break
            total.update(chunk)
            name = f"hw2_submission.zip.{index:03d}"
            (delivery / name).write_bytes(chunk)
            parts.append({"name": name, "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    manifest = {"archive": "hw2_submission.zip", "size": output.stat().st_size,
                "sha256": total.hexdigest(), "parts": parts}
    (delivery / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    package(args.model, args.questions, args.output)
