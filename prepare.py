"""Download only the assigned dataset, train BPE, write mmap token arrays atomically."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from src.text_encoder.tokenizer import train_tokenizer, validation_question

DATASET = "SlavaYus/tinysms_qa_12m"
REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"


def rows(path, batch_size=4096):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def prepare(args):
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; use a new output path")
    work = dest.with_name(dest.name + ".building")
    if work.exists():
        raise FileExistsError(f"Incomplete preparation at {work}; inspect/remove it explicitly")
    work.mkdir(parents=True)
    if args.local_parquet:
        paths = sorted(Path(args.local_parquet).glob("*.parquet"))
        origin = {"local_parquet": str(Path(args.local_parquet).resolve())}
    else:
        root = Path(
            snapshot_download(
                DATASET,
                repo_type="dataset",
                revision=REVISION,
                allow_patterns=["data/train/*.parquet"],
                token=False,
            )
        )
        paths = sorted((root / "data/train").glob("*.parquet"))
        origin = {"dataset": DATASET, "revision": REVISION}
    if not paths:
        raise ValueError("No train parquet files")

    def texts():
        rng = np.random.default_rng(args.seed)
        for path in paths:
            # Uniform per-shard subset. At most one raw shard is resident at a time.
            table = pq.read_table(path, columns=["question", "code"])
            ids = rng.choice(
                len(table), min(args.tokenizer_rows_per_shard, len(table)), replace=False
            )
            for row in table.take(ids).to_pylist():
                if not validation_question(row["question"], args.val_modulus):
                    yield row["question"]
                    yield row["code"]

    tok = train_tokenizer(texts(), work / "tokenizer.json", args.vocab_size)
    count, lengths = Counter(), {"source": [], "target": []}
    offsets = {"train": 0, "val": 0}
    files = {
        s: (open(work / f"{s}.tokens.bin", "wb"), open(work / f"{s}.index.raw", "wb"))
        for s in ("train", "val")
    }
    val_file = open(work / "val.jsonl", "w")
    try:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                records = batch.to_pylist()
                questions = tok.encode_batch([r["question"] for r in records])
                codes = tok.encode_batch([r["code"] for r in records])
                for row, q, c in zip(records, questions, codes):
                    count["seen"] += 1
                    source, target = q.ids + [2], c.ids + [2]
                    # Bounded length sample for quantiles, no raw text stored in logs.
                    if count["seen"] % 100 == 1:
                        lengths["source"].append(len(source))
                        lengths["target"].append(len(target))
                    split = (
                        "val" if validation_question(row["question"], args.val_modulus) else "train"
                    )
                    if len(source) > args.max_source or len(target) > args.max_target:
                        count[f"{split}_overlength"] += 1
                        continue  # Never cut a Python function in half.
                    if not row["question"].strip() or not row["code"].strip():
                        count["empty"] += 1
                        continue
                    token_file, idx_file = files[split]
                    token_file.write(np.asarray(source + target, dtype=np.uint16).tobytes())
                    idx_file.write(np.asarray([offsets[split], len(source), len(target)], dtype=np.uint64).tobytes())
                    offsets[split] += len(source) + len(target)
                    count[split] += 1
                    count[f"{split}_target_tokens"] += len(target)
                    if split == "val":
                        val_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"shard": path.name, **count}), flush=True)
    finally:
        val_file.close()
        for fs in files.values():
            for f in fs:
                f.close()
    for split in files:
        if count[split] == 0:
            raise ValueError(f"Empty {split}; check source and val_modulus")
        raw = np.memmap(work / f"{split}.index.raw", mode="r", dtype=np.uint64).reshape(-1, 3)
        out = np.lib.format.open_memmap(
            work / f"{split}.index.npy", mode="w+", dtype=np.uint64, shape=raw.shape
        )
        out[:] = raw
        out.flush()
        del out, raw
        (work / f"{split}.index.raw").unlink()
    digest = hashlib.sha256((work / "tokenizer.json").read_bytes()).hexdigest()
    metadata = {
        **origin,
        "counts": dict(count),
        "tokenizer_sha256": digest,
        "vocab_size": tok.get_vocab_size(),
        "model_vocab_size": args.vocab_size,
        "max_source": args.max_source,
        "max_target": args.max_target,
        "seed": args.seed,
        "val_modulus": args.val_modulus,
        "tokenizer_rows_per_shard": args.tokenizer_rows_per_shard,
        "length_quantiles": {
            k: dict(
                zip(
                    ["p50", "p90", "p95", "p99", "max"],
                    np.quantile(v, [0.5, 0.9, 0.95, 0.99, 1]).tolist(),
                )
            )
            for k, v in lengths.items()
            if v
        },
    }
    (work / "metadata.json").write_text(json.dumps(metadata, indent=2))
    work.rename(dest)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed")
    parser.add_argument(
        "--local-parquet", default=None, help="Offline copy of assigned train shards"
    )
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--max-source", type=int, default=384)
    parser.add_argument("--max-target", type=int, default=384)
    parser.add_argument("--tokenizer-rows-per-shard", type=int, default=4096)
    parser.add_argument("--val-modulus", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 260 <= args.vocab_size <= 65536 or args.val_modulus < 2:
        parser.error("vocab-size must be in [260, 65536]; val-modulus >= 2")
    prepare(args)
