import os
from pathlib import Path
import random

import numpy as np
import torch


def atomic_save(state, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with open(temp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def random_state():
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
