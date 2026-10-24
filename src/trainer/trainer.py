import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import signal
import time

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.collate import collate
from src.datasets.token_dataset import TokenDataset, BucketSampler
from src.logger.local import LocalWriter
from src.loss.token_loss import token_loss
from src.metrics.code import generation_metrics
from src.metrics.arithmetic import arithmetic_metrics
from src.text_encoder.tokenizer import load_tokenizer
from src.utils.checkpoint import atomic_save, random_state, restore_random
from src.utils.provenance import save_sources


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required. Use -cn smoke for CPU checks.")
        if cfg.hardware.precision == "bf16" and (
            self.device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("bf16 training requires A100/H100 or another bf16 CUDA GPU")
        if cfg.hardware.precision not in ("bf16", "fp32"):
            raise ValueError("Training supports bf16 or fp32; inference supports fp16 on T4")
        torch.set_float32_matmul_precision("high")
        self.root, self.run = Path(cfg.data_dir), Path(cfg.run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        if (self.run / "last.pt").exists() and not cfg.resume:
            raise FileExistsError("Existing run: resume explicitly or choose a new run_dir")
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self.tokenizer = load_tokenizer(self.root / "tokenizer.json")
        if (
            hashlib.sha256((self.root / "tokenizer.json").read_bytes()).hexdigest()
            != self.metadata["tokenizer_sha256"]
        ):
            raise ValueError("Tokenizer checksum mismatch")
        if self.metadata["model_vocab_size"] != cfg.model.vocab_size:
            raise ValueError("Configured vocabulary differs from prepared data")
        if max(self.metadata["max_source"], self.metadata["max_target"]) > cfg.model.max_length:
            raise ValueError("Data exceeds configured context")
        self.model = instantiate(cfg.model).to(self.device)
        decay, no_decay = [], []
        for p in self.model.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.optimizer.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            fused=self.device.type == "cuda",
        )
        self.train_data = TokenDataset(self.root, "train")
        self.sampler = BucketSampler(self.train_data, cfg.hardware.micro_batch, cfg.seed)
        if len(self.sampler) < 1:
            raise ValueError("Dataset smaller than micro_batch")
        self.val_data = TokenDataset(self.root, "val")
        # Fixed random subset instead of first shard only.
        rng = np.random.default_rng(cfg.seed)
        self.val_ids = rng.permutation(len(self.val_data))[: cfg.trainer.val_examples].tolist()
        self.val_loader = DataLoader(
            Subset(self.val_data, self.val_ids),
            batch_size=cfg.hardware.micro_batch,
            collate_fn=collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        self.step, self.epoch, self.batch_cursor = 0, 0, 0
        self.best, self.elapsed, self.horizon = float("inf"), 0.0, cfg.trainer.max_steps
        self.calibrated, self.stop_requested = False, False
        if cfg.resume:
            # Training checkpoints contain RNG Python objects. Load only your own files.
            state = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            current_config = OmegaConf.to_container(cfg, resolve=True)
            for key in ("model", "optimizer", "seed", "hardware", "trainer"):
                if state["config"][key] != current_config[key]:
                    raise ValueError(f"Resume config mismatch: {key}. Use the original config.")
            if state["metadata"] != self.metadata:
                raise ValueError("Resume dataset differs")
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.step, self.epoch, self.batch_cursor = (
                state["step"],
                state["epoch"],
                state["batch_cursor"],
            )
            self.best, self.elapsed = state["best"], state["elapsed"]
            self.horizon, self.calibrated = state["horizon"], state["calibrated"]
            restore_random(state["rng"])
        save_sources(self.run)
        self.writer = LocalWriter(self.run, cfg.trainer.tensorboard)
        OmegaConf.save(cfg, self.run / "config.yaml")
        shutil.copy2(self.root / "tokenizer.json", self.run / "tokenizer.json")
        (self.run / "data_metadata.json").write_text(json.dumps(self.metadata, indent=2))
        (self.run / "environment.json").write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if self.device.type == "cuda" else "cpu",
                    "parameters": sum(p.numel() for p in self.model.parameters()),
                },
                indent=2,
            )
        )
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def request_stop(self, *_):
        self.stop_requested = True
        print("Stop requested; saving after the current optimizer step.", flush=True)

    def amp(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.cfg.hardware.precision == "bf16"
            else contextlib.nullcontext()
        )

    def move(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def new_iterator(self):
        self.sampler.epoch, self.sampler.start_batch = self.epoch, self.batch_cursor
        # Dedicated generator keeps worker initialization from perturbing model dropout RNG.
        gen = torch.Generator().manual_seed(self.cfg.seed + self.epoch)
        return iter(
            DataLoader(
                self.train_data,
                batch_sampler=self.sampler,
                collate_fn=collate,
                num_workers=self.cfg.hardware.workers,
                pin_memory=self.device.type == "cuda",
                generator=gen,
            )
        )

    def lr(self):
        warmup = min(self.cfg.trainer.warmup_steps, max(1, self.horizon // 10))
        if self.step < warmup:
            factor = (self.step + 1) / warmup
        else:
            progress = min(1.0, (self.step - warmup) / max(1, self.horizon - warmup - 1))
            floor = self.cfg.optimizer.min_lr_ratio
            factor = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        return self.cfg.optimizer.lr * factor

    def save(self, name="last.pt", inference_only=False):
        state = {
            "model": self.model.state_dict(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "metadata": self.metadata,
            "step": self.step,
        }
        if not inference_only:
            state.update(
                optimizer=self.optimizer.state_dict(),
                epoch=self.epoch,
                batch_cursor=self.batch_cursor,
                best=self.best,
                elapsed=self.elapsed + time.monotonic() - self.started,
                horizon=self.horizon,
                calibrated=self.calibrated,
                rng=random_state(),
            )
        atomic_save(state, self.run / name)

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        loss, count = 0.0, 0
        for batch in self.val_loader:
            batch = self.move(batch)
            with self.amp():
                hidden = self.model(batch["source"], batch["decoder_input"])
                total, n = token_loss(
                    self.model, hidden, batch["labels"], self.cfg.trainer.loss_chunk_size
                )
            loss += total.item()
            count += n.item()
        result = {"val/nll": loss / count, "val/perplexity": math.exp(min(20, loss / count))}
        n_gen = min(self.cfg.trainer.generation_examples, len(self.val_ids))
        preds, refs, ended = [], [], 0
        for start in range(0, n_gen, self.cfg.hardware.micro_batch):
            ids = self.val_ids[start : min(start + self.cfg.hardware.micro_batch, n_gen)]
            rows = [self.val_data[i] for i in ids]
            batch = self.move(collate(rows))
            with self.amp():
                tokens = self.model.generate(
                    batch["source"],
                    self.cfg.trainer.generation_max_tokens,
                    valid_vocab_size=self.tokenizer.get_vocab_size(),
                )
            ended += tokens.eq(2).any(-1).sum().item()
            preds.extend(self.tokenizer.decode_batch(tokens.cpu().tolist()))
            refs.extend(self.tokenizer.decode(list(map(int, r[1]))) for r in rows)
        if n_gen:
            result.update({f"val/{k}": v for k, v in generation_metrics(preds, refs).items()})
            result.update({f"val/{k}": v for k, v in arithmetic_metrics(preds, refs).items()})
            result["val/eos_rate"] = ended / n_gen
            with open(self.run / f"generations_{self.step:06d}.jsonl", "w") as f:
                for p, r in zip(preds, refs):
                    f.write(json.dumps({"prediction": p, "reference": r}) + "\n")
        self.writer.log(self.step, **result)
        if result["val/nll"] < self.best:
            self.best = result["val/nll"]
            self.save("best.pt", inference_only=True)
        self.model.train()

    def train(self):
        cfg = self.cfg
        self.started = time.monotonic()
        iterator = self.new_iterator()
        self.model.train()
        last_log = self.started
        log_loss, log_tokens, log_examples = 0.0, 0, 0
        try:
            while self.step < self.horizon and not self.stop_requested:
                if self.elapsed + time.monotonic() - self.started >= cfg.trainer.max_hours * 3600:
                    break
                batches = []
                for _ in range(cfg.hardware.accumulation):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        self.epoch += 1
                        self.batch_cursor = 0
                        iterator = self.new_iterator()
                        batch = next(iterator)
                    self.batch_cursor += 1
                    batches.append(batch)
                # One denominator across all microbatches = correct token-weighted accumulation.
                denominator = sum(b["labels"].ne(-100).sum().item() for b in batches)
                rate = self.lr()
                for group in self.optimizer.param_groups:
                    group["lr"] = rate
                self.optimizer.zero_grad(set_to_none=True)
                for batch in batches:
                    batch = self.move(batch)
                    with self.amp():
                        hidden = self.model(batch["source"], batch["decoder_input"])
                        total, _ = token_loss(
                            self.model, hidden, batch["labels"], cfg.trainer.loss_chunk_size
                        )
                        loss = total / denominator
                    loss.backward()
                    log_loss += total.detach().item()
                    log_examples += batch["source"].size(0)
                grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.trainer.clip_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                self.step += 1
                log_tokens += denominator
                elapsed = self.elapsed + time.monotonic() - self.started
                if (
                    cfg.trainer.calibrate_budget
                    and not self.calibrated
                    and self.step >= cfg.trainer.calibration_steps
                ):
                    # Reserve 20% for validation/checkpoints and slower long batches.
                    seconds_per_step = elapsed / self.step
                    feasible = int(0.80 * cfg.trainer.max_hours * 3600 / seconds_per_step)
                    self.horizon = min(cfg.trainer.max_steps, max(self.step + 1, feasible))
                    self.calibrated = True
                    self.writer.log(
                        self.step,
                        planned_steps=self.horizon,
                        estimated_step_seconds=seconds_per_step,
                    )
                if self.step % cfg.trainer.log_every == 0:
                    seconds = time.monotonic() - last_log
                    self.writer.log(
                        self.step,
                        **{
                            "train/nll": log_loss / log_tokens,
                            "train/lr": rate,
                            "train/grad_norm": float(grad),
                            "train/target_tokens_per_second": log_tokens / seconds,
                            "train/examples_per_second": log_examples / seconds,
                            "train/elapsed_hours": elapsed / 3600,
                            "train/peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9
                            if self.device.type == "cuda"
                            else 0,
                        },
                    )
                    last_log, log_loss, log_tokens, log_examples = time.monotonic(), 0.0, 0, 0
                if self.step % cfg.trainer.eval_every == 0:
                    self.validate()
                if self.step % cfg.trainer.save_every == 0:
                    self.save()
            self.validate()
            self.save()
            self.save("final.pt", inference_only=True)
        finally:
            self.writer.close()
