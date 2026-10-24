import numpy as np
import torch

from src.datasets.collate import collate
from src.datasets.token_dataset import BucketSampler
from src.text_encoder.tokenizer import train_tokenizer, validation_question


def test_loss_shift_and_padding():
    b = collate([([4, 2], [5, 6, 2]), ([7, 2], [8, 2])])
    assert b["decoder_input"][0, :3].tolist() == [1, 5, 6]
    assert b["labels"][1, :4].tolist() == [8, 2, -100, -100]
    assert b["source"].shape[1] % 8 == 0


def test_group_split():
    assert validation_question("a  question\n") == validation_question("a question")


def test_bpe_preserves_code_and_unicode(tmp_path):
    text = "def solve():\n    x = 12.75\n    # café\n    return x / 3\n"
    tok = train_tokenizer([text] * 10, tmp_path / "tok.json", 512)
    assert tok.decode(tok.encode(text).ids) == text
    assert tok.decode(tok.encode("未知 🧮").ids) == "未知 🧮"


def test_sampler_resume_and_coverage():
    class Dummy:
        index = np.array([[0, i % 10, i % 7] for i in range(101)])

        def __len__(self):
            return len(self.index)

    sampler = BucketSampler(Dummy(), 8, bucket_size=30, drop_last=False)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for b in batches for i in b) == list(range(101))
    sampler.start_batch = 3
    assert list(sampler) == batches[3:]
    sampler.start_batch, sampler.epoch = 0, 1
    assert list(sampler) != batches


def test_token_weighted_accumulation():
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 5)
    x, y = torch.randn(5, 3), torch.tensor([1, 2, 3, 4, 1])
    torch.nn.functional.cross_entropy(model(x), y).backward()
    expected = model.weight.grad.clone()
    model.zero_grad()
    for ids in (slice(0, 2), slice(2, 5)):
        (torch.nn.functional.cross_entropy(model(x[ids]), y[ids], reduction="sum") / 5).backward()
    torch.testing.assert_close(expected, model.weight.grad)
