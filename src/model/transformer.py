"""From-scratch asymmetric encoder-decoder; no pretrained components."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class Rotary(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        angles = torch.outer(torch.arange(max_length).float(), inv)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, offset=0):
        c = self.cos[offset : offset + x.size(-2)].to(x.dtype)
        s = self.sin[offset : offset + x.size(-2)].to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, max_length, rotary=True, qk_norm=True):
        super().__init__()
        assert dim % heads == 0 and (dim // heads) % 2 == 0
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.qk_norm = qk_norm
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = Rotary(self.head_dim, max_length) if rotary else None

    def split(self, x):
        return x.view(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)

    def kv(self, x):
        return self.split(self.k(x)), self.split(self.v(x))

    def forward(self, x, context=None, mask=None, causal=False, cache=None, offset=0):
        q = self.split(self.q(x))
        if context is not None and cache is not None:
            k, v = cache  # precomputed encoder keys/values
        else:
            k, v = self.kv(x if context is None else context)
        if self.qk_norm:
            # Bound attention logits despite growth of Q/K projection magnitudes.
            # RMS normalization has no learned parameters; the 107M count is unchanged.
            q = F.rms_norm(q, (self.head_dim,), eps=1e-6).to(q.dtype)
            k = F.rms_norm(k, (self.head_dim,), eps=1e-6).to(k.dtype)
        if self.rope is not None:
            q, k = self.rope(q, offset), self.rope(k, offset)
        if context is None and cache is not None:
            end = offset + x.size(1)
            cache[0][:, :, offset:end] = k
            cache[1][:, :, offset:end] = v
            k, v = cache[0][:, :, :end], cache[1][:, :, :end]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).contiguous().flatten(2))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, dropout, max_length, qk_norm=True, decoder=False):
        super().__init__()
        self.self_attn = Attention(dim, heads, dropout, max_length, qk_norm=qk_norm)
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.ff = FeedForward(dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.decoder = decoder
        if decoder:
            self.cross = Attention(dim, heads, dropout, max_length, rotary=False, qk_norm=qk_norm)
            self.nc = RMSNorm(dim)

    def forward(self, x, memory=None, src_mask=None, cache=None, offset=0):
        x = x + self.drop(
            self.self_attn(
                self.n1(x),
                mask=None if self.decoder else src_mask,
                causal=self.decoder and cache is None,
                cache=cache[0] if cache else None,
                offset=offset,
            )
        )
        if self.decoder:
            x = x + self.drop(
                self.cross(
                    self.nc(x),
                    context=memory,
                    mask=src_mask,
                    cache=cache[1] if cache else None,
                )
            )
        return x + self.drop(self.ff(self.n2(x)))


class CodeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=16384,
        dim=768,
        heads=12,
        hidden=2048,
        encoder_layers=8,
        decoder_layers=4,
        dropout=0.1,
        max_length=512,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        activation_checkpointing=False,
        qk_norm=True,
    ):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.max_length, self.dim = max_length, dim
        self.activation_checkpointing = activation_checkpointing
        self.embedding = nn.Embedding(vocab_size, dim)
        args = (dim, heads, hidden, dropout, max_length, qk_norm)
        self.encoder = nn.ModuleList([Block(*args) for _ in range(encoder_layers)])
        self.decoder = nn.ModuleList([Block(*args, decoder=True) for _ in range(decoder_layers)])
        self.enc_norm, self.dec_norm = RMSNorm(dim), RMSNorm(dim)
        self.apply(self._init)
        for blocks in (self.encoder, self.decoder):
            for layer in blocks:
                nn.init.normal_(layer.self_attn.out.weight, std=0.02 / math.sqrt(2 * len(blocks)))
                nn.init.normal_(layer.ff.down.weight, std=0.02 / math.sqrt(2 * len(blocks)))
                if layer.decoder:
                    nn.init.normal_(layer.cross.out.weight, std=0.02 / math.sqrt(3 * len(blocks)))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def encode(self, source):
        if source.size(1) > self.max_length:
            raise ValueError("Source exceeds model context; refusing silent truncation")
        mask = source.ne(self.pad_id)[:, None, None, :]
        x = self.embedding(source)
        for block in self.encoder:
            if self.training and self.activation_checkpointing:
                x = checkpoint(block, x, None, mask, use_reentrant=False)
            else:
                x = block(x, src_mask=mask)
        return self.enc_norm(x), mask

    def forward(self, source, decoder_input):
        if decoder_input.size(1) > self.max_length:
            raise ValueError("Target exceeds model context")
        memory, mask = self.encode(source)
        x = self.embedding(decoder_input)
        for block in self.decoder:
            if self.training and self.activation_checkpointing:
                x = checkpoint(block, x, memory, mask, use_reentrant=False)
            else:
                x = block(x, memory, mask)
        return self.dec_norm(x)

    def logits(self, hidden):
        return F.linear(hidden, self.embedding.weight)

    def init_cache(self, memory, max_new_tokens):
        result = []
        for block in self.decoder:
            cross = block.cross.kv(memory)
            shape = (
                memory.size(0),
                block.self_attn.heads,
                max_new_tokens,
                block.self_attn.head_dim,
            )
            # Use projection dtype (autocast), not embedding dtype.
            kv = (
                torch.empty(shape, device=memory.device, dtype=cross[0].dtype),
                torch.empty(shape, device=memory.device, dtype=cross[0].dtype),
            )
            result.append((kv, cross))
        return result

    def decode_step(self, token, memory, mask, cache, offset):
        x = self.embedding(token)
        for block, layer_cache in zip(self.decoder, cache):
            x = block(x, memory, mask, layer_cache, offset)
        return self.logits(self.dec_norm(x))[:, -1]

    @torch.inference_mode()
    def generate(self, source, max_new_tokens=256, valid_vocab_size=None):
        if self.training:
            raise RuntimeError("Call model.eval() before generation")
        if not 1 <= max_new_tokens <= self.max_length:
            raise ValueError("Invalid generation length")
        memory, mask = self.encode(source)
        cache = self.init_cache(memory, max_new_tokens)
        token = torch.full((source.size(0), 1), self.bos_id, device=source.device)
        result = torch.full((source.size(0), max_new_tokens), self.pad_id, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for step in range(max_new_tokens):
            logits = self.decode_step(token, memory, mask, cache, step)
            logits[:, [self.pad_id, self.bos_id, 3]] = -torch.inf
            if valid_vocab_size is not None:
                logits[:, valid_vocab_size:] = -torch.inf
            nxt = logits.argmax(-1)
            nxt = torch.where(finished, self.pad_id, nxt)
            result[:, step] = nxt
            finished |= nxt.eq(self.eos_id)
            token = nxt[:, None]
            if finished.all():
                break
        return result[:, : step + 1]
