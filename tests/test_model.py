import torch
from torch.nn import functional as F

from src.model.transformer import CodeTransformer
from src.loss.token_loss import token_loss
from src.datasets.collate import collate


def tiny(**kw):
    return CodeTransformer(
        vocab_size=32,
        dim=32,
        heads=4,
        hidden=64,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0,
        max_length=32,
        **kw,
    )


def test_cached_matches_full_teacher_forcing():
    torch.manual_seed(1)
    m = tiny().eval()
    src = torch.tensor([[5, 6, 2, 0], [7, 8, 9, 2]])
    tgt = torch.tensor([[1, 12, 13, 14], [1, 15, 16, 17]])
    with torch.no_grad():
        full = m.logits(m(src, tgt))
        memory, mask = m.encode(src)
        cache = m.init_cache(memory, 4)
        actual = torch.stack(
            [m.decode_step(tgt[:, i : i + 1], memory, mask, cache, i) for i in range(4)], dim=1
        )
    torch.testing.assert_close(actual, full, atol=2e-6, rtol=2e-5)


def test_future_tokens_do_not_leak():
    m = tiny().eval()
    src = torch.tensor([[4, 2]])
    a, b = torch.tensor([[1, 5, 6, 7]]), torch.tensor([[1, 5, 20, 21]])
    torch.testing.assert_close(m(src, a)[:, :2], m(src, b)[:, :2])


def test_source_padding_invariance():
    m = tiny().eval()
    a, b = torch.tensor([[4, 5, 2]]), torch.tensor([[4, 5, 2, 0, 0]])
    target = torch.tensor([[1, 6, 7]])
    torch.testing.assert_close(m(a, target), m(b, target), atol=2e-6, rtol=2e-5)


def test_chunked_loss_gradient_matches_full():
    torch.manual_seed(12)
    m = tiny().train()
    batch = collate([([4, 5, 2], [7, 8, 2]), ([9, 2], [11, 2])])
    h = m(batch["source"], batch["decoder_input"])
    total, count = token_loss(m, h, batch["labels"], chunk_size=3)
    (total / count).backward()
    grads = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    logits = m.logits(m(batch["source"], batch["decoder_input"]))
    expected = F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=-100)
    expected.backward()
    torch.testing.assert_close(total / count, expected)
    for actual, p in zip(grads, m.parameters()):
        torch.testing.assert_close(actual, p.grad, atol=2e-6, rtol=2e-4)


def test_parameter_budget():
    with torch.device("meta"):
        m = CodeTransformer()
    count = sum(p.numel() for p in m.parameters())
    assert 100_000_000 <= count <= 120_000_000


def test_activation_checkpointing_gradient():
    torch.manual_seed(13)
    a, b = tiny(), tiny(activation_checkpointing=True)
    b.load_state_dict(a.state_dict())
    for m in (a, b):
        m(torch.tensor([[4, 5, 2]]), torch.tensor([[1, 7, 8]])).square().mean().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_tiny_batch_can_overfit():
    torch.manual_seed(4)
    m = tiny().train()
    opt = torch.optim.AdamW(m.parameters(), lr=0.01)
    source, target = torch.tensor([[4, 5, 2], [6, 7, 2]]), torch.tensor([[8, 9, 2], [10, 11, 2]])
    dec = torch.cat((torch.ones_like(target[:, :1]), target[:, :-1]), 1)
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(m.logits(m(source, dec)).flatten(0, 1), target.flatten())
        loss.backward()
        opt.step()
    assert loss.item() < 0.02
    m.eval()
    assert torch.equal(m.generate(source, 6), target)


def test_bfloat16_cache_projection_dtype():
    m = tiny().eval()
    src = torch.tensor([[5, 6, 2, 0], [7, 8, 9, 2]])
    tgt = torch.tensor([[1, 12, 13], [1, 15, 16]])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        full = m.logits(m(src, tgt))
        memory, mask = m.encode(src)
        cache = m.init_cache(memory, 3)
        assert cache[0][0][0].dtype == torch.bfloat16
        actual = torch.stack(
            [m.decode_step(tgt[:, i : i + 1], memory, mask, cache, i) for i in range(3)], dim=1
        )
    torch.testing.assert_close(actual, full, atol=0.005, rtol=0.03)


def test_qk_normalization_resists_projection_magnitude_growth():
    import copy
    from src.model.transformer import Attention
    torch.manual_seed(29)
    attention = Attention(32, 4, 0, 32).eval()
    enlarged = copy.deepcopy(attention)
    with torch.no_grad():
        enlarged.q.weight.mul_(10000)
        enlarged.k.weight.mul_(10000)
    x = torch.randn(2, 7, 32)
    expected = attention(x)
    actual = enlarged(x)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    actual.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in enlarged.parameters())
