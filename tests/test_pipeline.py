from argparse import Namespace
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from prepare import prepare
from inference import run
from src.trainer.trainer import Trainer


def test_prepare_train_resume_and_inference(tmp_path):
    records = json.loads(
        (Path(__file__).parent / "fixtures/assigned_train_sample.json").read_text()
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(pa.Table.from_pylist(records), raw / "train.parquet")
    processed = tmp_path / "processed"
    prepare(
        Namespace(
            output=str(processed),
            local_parquet=str(raw),
            vocab_size=512,
            max_source=256,
            max_target=256,
            tokenizer_rows_per_shard=32,
            val_modulus=5,
            seed=42,
        )
    )
    with initialize_config_dir(version_base=None, config_dir=str(Path("src/configs").resolve())):
        cfg = compose(
            config_name="smoke",
            overrides=[
                f"data_dir={processed}",
                f"run_dir={tmp_path}/continuous",
                "model.vocab_size=512",
                "model.dropout=0.1",
                "trainer.max_steps=4",
                "trainer.tensorboard=false",
            ],
        )
    continuous = Trainer(cfg)
    continuous.train()
    expected = {k: v.clone() for k, v in continuous.model.state_dict().items()}
    cfg.run_dir = str(tmp_path / "resumed")
    interrupted = Trainer(cfg)
    original_step = interrupted.optimizer.step
    calls = 0

    def stop_after_two(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == 2:
            interrupted.request_stop()
        return result

    interrupted.optimizer.step = stop_after_two
    interrupted.train()
    cfg.resume = str(tmp_path / "resumed/last.pt")
    resumed = Trainer(cfg)
    resumed.train()
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([r["question"] for r in records[:3]]))
    out = tmp_path / "answers.jsonl"
    run(
        Namespace(
            checkpoint=str(tmp_path / "resumed/best.pt"),
            tokenizer=str(tmp_path / "resumed/tokenizer.json"),
            questions=str(questions),
            output=str(out),
            batch_size=2,
            max_new_tokens=16,
            device="cpu",
            dtype="fp32",
            expected_count=3,
        )
    )
    actual = [json.loads(s) for s in out.read_text().splitlines()]
    assert all(set(row) == {"code"} and isinstance(row["code"], str) for row in actual)
    details = [json.loads(s) for s in out.with_suffix(".details.jsonl").read_text().splitlines()]
    assert [row["code"] for row in actual] == [row["code"] for row in details]
    actual = details
    assert [x["id"] for x in actual] == [0, 1, 2]
    assert [x["question"] for x in actual] == [r["question"] for r in records[:3]]
