"""Fetch only the official public test split linked from the assignment."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import pyarrow.parquet as pq

DATASET = "SlavaYus/tinysms_qa_12m"
REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"
PARQUET_SHA256 = "8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a"
QUESTIONS_SHA256 = "a3a128e5acf694f9b9a828dcc582bacb79367eecf11af2f6041c0c5f788e7003"


def prepare_questions(output, cache=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache) if cache else output.with_suffix(".parquet")
    if not cache.is_file():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/test/test.parquet"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != PARQUET_SHA256:
            raise ValueError("Official test parquet checksum mismatch")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    if hashlib.sha256(cache.read_bytes()).hexdigest() != PARQUET_SHA256:
        raise ValueError("Cached official test parquet checksum mismatch")
    table = pq.read_table(cache)
    if table.column_names != ["question"] or len(table) != 1919:
        raise ValueError("Expected the official questions-only test with 1919 rows")
    questions = table.column("question").to_pylist()
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("Empty or invalid question")
    data = "".join(json.dumps({"question": q}, ensure_ascii=False) + "\n" for q in questions).encode("utf-8")
    if hashlib.sha256(data).hexdigest() != QUESTIONS_SHA256:
        raise ValueError("Official question order or serialization changed")
    output.write_bytes(data)
    return {
        "dataset": DATASET, "revision": REVISION, "file": "data/test/test.parquet",
        "rows": len(questions), "original_dataset_row_order_preserved": True,
        "source": "official_test_split_linked_from_assignment",
        "parquet_sha256": PARQUET_SHA256, "questions_sha256": QUESTIONS_SHA256,
        "separate_bot_file_compared": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pass1_questions.jsonl")
    parser.add_argument("--cache")
    parser.add_argument("--metadata", default="artifacts/question_source.json")
    args = parser.parse_args()
    metadata = prepare_questions(args.output, args.cache)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
