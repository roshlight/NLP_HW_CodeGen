"""Export the completed run on explicitly selected, provenance-recorded questions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def watch(project, run, exports, delivery, timeout_hours=12):
    project, run, exports, delivery = map(Path, (project, run, exports, delivery))
    delivery.mkdir(parents=True, exist_ok=True)
    status_path = delivery / "export_status.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(project / "dependencies"), str(project)])
    deadline = time.monotonic() + timeout_hours * 3600

    def status(phase, **extra):
        record = {"phase": phase, "max_new_tokens": 64, "updated_unix": time.time(), **extra}
        status_path.write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)

    def generate(questions, name):
        output = delivery / name
        subprocess.run([
            sys.executable, "-u", str(project / "inference.py"),
            "--checkpoint", str(run / "best.pt"), "--tokenizer", str(run / "tokenizer.json"),
            "--questions", str(questions), "--output", str(output),
            "--batch-size", "128", "--max-new-tokens", "64", "--dtype", "fp16",
        ], cwd=project, env=env, check=True, timeout=900)
        return output

    status("waiting_for_training")
    while time.monotonic() < deadline:
        record = json.loads((project / "worker_status.json").read_text())
        if record["phase"] == "failed":
            status("training_failed")
            return
        if record["phase"] == "completed":
            break
        time.sleep(30)
    else:
        raise TimeoutError("Training/export wait timed out")
    original_questions = delivery / "pass1_questions.jsonl"
    # This is a separate preview. It is never named submission.jsonl.
    if not original_questions.is_file():
        hf_questions = exports / "questions_used.jsonl"
        if hf_questions.is_file():
            status("generating_hf_preview_original_bot_order_unverified")
            generate(hf_questions, "preview_hf64.jsonl")
        status("waiting_for_original_pass1_questions")
    while not original_questions.is_file() and time.monotonic() < deadline:
        time.sleep(30)
    if not original_questions.is_file():
        status("original_pass1_questions_missing")
        return
    records = [json.loads(s) for s in original_questions.read_text().splitlines()]
    if len(records) != 1919 or any(not isinstance(r.get("question"), str) for r in records):
        raise ValueError("Original pass1_questions.jsonl must have 1919 question records")
    status("generating_original_bot_order")
    output = generate(original_questions, "submission.jsonl")
    details = [json.loads(s) for s in output.with_suffix(".details.jsonl").read_text().splitlines()]
    answers = [json.loads(s) for s in output.read_text().splitlines()]
    if len(details) != 1919 or len(answers) != 1919:
        raise ValueError("Incomplete submission")
    for i, (q, answer, detail) in enumerate(zip(records, answers, details)):
        if set(answer) != {"code"} or not isinstance(answer["code"], str):
            raise ValueError("Invalid submission schema")
        if detail["id"] != i or detail["question"] != q["question"] or detail["code"] != answer["code"]:
            raise ValueError("Answer/question order mismatch")
    durable = project / "delivery_256_fresh" / "bot"
    durable.mkdir(parents=True, exist_ok=True)
    for p in delivery.glob("submission.*"):
        shutil.copyfile(p, durable / p.name)
    shutil.copyfile(original_questions, durable / original_questions.name)
    question_source = delivery / "question_source.json"
    provenance = json.loads(question_source.read_text()) if question_source.is_file() else {}
    status("completed", rows=1919, original_question_order_verified=True,
           question_source=provenance,
           separate_bot_file_compared=provenance.get("separate_bot_file_compared", False),
           questions_sha256=hashlib.sha256(original_questions.read_bytes()).hexdigest(),
           submission_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    shutil.copyfile(status_path, durable / status_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--exports", required=True)
    parser.add_argument("--delivery", required=True)
    args = parser.parse_args()
    watch(args.project, args.run, args.exports, args.delivery)
