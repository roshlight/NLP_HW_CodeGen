from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Sampler


class TokenDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.index = np.load(root / f"{split}.index.npy", mmap_mode="r")
        self.tokens = np.memmap(root / f"{split}.tokens.bin", dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        start, src_len, tgt_len = map(int, self.index[i])
        return (
            self.tokens[start : start + src_len].astype(np.int64),
            self.tokens[start + src_len : start + src_len + tgt_len].astype(np.int64),
        )


class BucketSampler(Sampler):
    """Deterministic global shuffle + local length buckets, resumable by batch offset."""

    def __init__(self, dataset, batch_size, seed=42, bucket_size=8192, drop_last=True):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.bucket_size = max(batch_size, (bucket_size // batch_size) * batch_size)
        self.drop_last = drop_last
        self.epoch, self.start_batch = 0, 0

    def __len__(self):
        n = len(self.dataset)
        return (
            n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.dataset))
        lengths = self.dataset.index[:, 1] + self.dataset.index[:, 2]
        batches = []
        for start in range(0, len(order), self.bucket_size):
            ids = order[start : start + self.bucket_size]
            ids = ids[np.argsort(lengths[ids], kind="stable")]
            for k in range(0, len(ids), self.batch_size):
                batch = ids[k : k + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch.tolist())
        rng.shuffle(batches)
        yield from batches[self.start_batch :]
