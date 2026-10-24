"""Batched greedy decoding with static self KV + precomputed cross KV cache."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from hydra.utils import instantiate
import pyarrow.parquet as pq
import torch

from src.text_encoder.tokenizer import load_tokenizer


def read_questions(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pq.read_table(path, columns=["question"]).column("question").to_pylist()
    if path.suffix == ".jsonl":
        return [
            json.loads(line)["question"] for line in path.read_text().splitlines() if line.strip()
        ]
    if path.suffix == ".json":
        records = json.loads(path.read_text())
        return [r if isinstance(r, str) else r["question"] for r in records]
    raise ValueError("Expected parquet, JSON list, or JSONL containing question")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def run(args):
    load_start = time.perf_counter()
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    if getattr(args, "require_t4", False) and "T4" not in gpu:
        raise RuntimeError(f"This benchmark requires a real NVIDIA T4, found {gpu}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tok = load_tokenizer(args.tokenizer)
    if (
        hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
        != state["metadata"]["tokenizer_sha256"]
    ):
        raise ValueError("Tokenizer does not match the checkpoint")
    model = instantiate(state["config"]["model"])
    model.load_state_dict(state["model"])
    parameters = sum(p.numel() for p in model.parameters())
    expected_parameters = getattr(args, "expected_parameters", None)
    if expected_parameters is not None and parameters != expected_parameters:
        raise ValueError(f"Expected the supplied {expected_parameters}-parameter model, got {parameters}")
    model.eval().to(device)
    if args.dtype == "fp16":
        model.half()
    elif args.dtype == "bf16":
        model.bfloat16()
    questions = read_questions(args.questions)
    if not questions:
        raise ValueError("Empty questions file")
    expected = getattr(args, "expected_count", 1919)
    if expected is not None and len(questions) != expected:
        raise ValueError(f"Expected {expected} questions in original order, got {len(questions)}")
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("Every question must be a nonempty string")
    sync(device)
    load_seconds = time.perf_counter() - load_start
    started = time.perf_counter()  # tokenization, sort, transfer, generation, decode AND writing
    encoded = [e.ids + [2] for e in tok.encode_batch(questions)]
    if max(map(len, encoded)) > model.max_length:
        raise ValueError("Test prompt exceeds model context; refusing silent truncation")
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    predictions, terminated = [None] * len(order), [False] * len(order)
    for start in range(0, len(order), args.batch_size):
        ids = order[start : start + args.batch_size]
        length = (max(len(encoded[i]) for i in ids) + 7) // 8 * 8
        source = torch.zeros((len(ids), length), dtype=torch.long)
        for j, i in enumerate(ids):
            source[j, : len(encoded[i])] = torch.tensor(encoded[i])
        output = model.generate(
            source.to(device), args.max_new_tokens, valid_vocab_size=tok.get_vocab_size()
        ).cpu()
        texts = tok.decode_batch(output.tolist())
        for j, i in enumerate(ids):
            predictions[i] = texts[j]
            terminated[i] = bool(output[j].eq(2).any())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps({"code": prediction}, ensure_ascii=False) + "\n")
    with output_path.with_suffix(".details.jsonl").open("w", encoding="utf-8") as f:
        for i, (q, p, ended) in enumerate(zip(questions, predictions, terminated)):
            f.write(
                json.dumps({"id": i, "question": q, "code": p, "eos": ended}, ensure_ascii=False)
                + "\n"
            )
    with output_path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "code"])
        writer.writeheader()
        writer.writerows({"id": i, "code": p} for i, p in enumerate(predictions))
    sync(device)
    elapsed = time.perf_counter() - started
    checked = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(checked) == len(questions) and all(set(r) == {"code"} and isinstance(r["code"], str) for r in checked)
    total = time.perf_counter() - load_start
    report = {
        "questions": len(questions),
        "parameters": parameters,
        "generation_pipeline_seconds": elapsed,
        "load_seconds": load_seconds,
        "total_seconds": total,
        "eos_rate": sum(terminated) / len(terminated),
        "dtype": args.dtype,
        "device": gpu,
        "actual_t4": "T4" in gpu,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "under_180_seconds": total <= 180,
        "submission_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "questions_sha256": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".timing.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="saved/first_train/best.pt")
    p.add_argument("--tokenizer", default="saved/first_train/tokenizer.json")
    p.add_argument("--questions", required=True)
    p.add_argument("--output", default="outputs/submission_107m.jsonl")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--expected-count", type=int, default=1919)
    p.add_argument("--require-t4", action="store_true")
    p.add_argument("--expected-parameters", type=int, default=106977792)
    run(p.parse_args())
