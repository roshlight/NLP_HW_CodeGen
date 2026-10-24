import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def token_loss(model, hidden, labels, chunk_size=1024):
    """Summed CE and valid-token count. Recompute logits to bound peak memory."""
    hidden, labels = hidden.reshape(-1, hidden.size(-1)), labels.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)

    def chunk_loss(h, y):
        return F.cross_entropy(model.logits(h).float(), y, ignore_index=-100, reduction="sum")

    for h, y in zip(hidden.split(chunk_size), labels.split(chunk_size)):
        if model.training and torch.is_grad_enabled():
            total = total + checkpoint(chunk_loss, h, y, use_reentrant=False)
        else:
            total = total + chunk_loss(h, y)
    return total, labels.ne(-100).sum()
