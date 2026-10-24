import json
import time


class LocalWriter:
    def __init__(self, directory, tensorboard=True):
        self.file = open(directory / "metrics.jsonl", "a", buffering=1)
        self.tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(directory / "tensorboard"))

    def log(self, step, **metrics):
        record = {"step": step, "time": time.time(), **metrics}
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        if self.tb:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb.add_scalar(key, value, step)

    def close(self):
        self.file.close()
        if self.tb:
            self.tb.close()
