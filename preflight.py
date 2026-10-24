"""Allocate full architecture and stress-test the configured microbatch before the real run."""

import contextlib
import json
from pathlib import Path
import time

import hydra
from hydra.utils import instantiate
import torch

from src.loss.token_loss import token_loss


@hydra.main(version_base=None, config_path="src/configs", config_name="first_train")
def main(cfg):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run this preflight on the training GPU")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    metadata = json.loads((Path(cfg.data_dir) / "metadata.json").read_text())
    model = instantiate(cfg.model).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, fused=device.type == "cuda"
    )
    b = cfg.hardware.micro_batch
    source = torch.randint(4, cfg.model.vocab_size, (b, metadata["max_source"]), device=device)
    target = torch.randint(4, cfg.model.vocab_size, (b, metadata["max_target"]), device=device)
    decoder = torch.cat((torch.ones_like(target[:, :1]), target[:, :-1]), dim=1)
    seconds = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if cfg.hardware.precision == "bf16"
            else contextlib.nullcontext()
        )
        with amp:
            h = model(source, decoder)
            total, n = token_loss(model, h, target, cfg.trainer.loss_chunk_size)
            loss = total / n
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        print({"step": step, "loss": float(loss.detach()), "grad_norm": float(grad)}, flush=True)
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "micro_batch": b,
        "accumulation": cfg.hardware.accumulation,
        "source_length": source.size(1),
        "target_length": target.size(1),
        "max_length_micro_step_seconds": sum(seconds[1:]) / 3,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
        "note": "Synthetic allocation check only; weights discarded, not an experiment.",
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
