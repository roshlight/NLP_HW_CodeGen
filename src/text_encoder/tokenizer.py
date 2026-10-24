import hashlib
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def validation_question(question, modulus=1000):
    """Group exact/whitespace-equivalent questions, never split their programs."""
    normalized = " ".join(question.split())
    digest = hashlib.sha256(normalized.encode()).digest()
    return int.from_bytes(digest[:8], "little") % modulus == 0


def train_tokenizer(texts, path, vocab_size=16384):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # No normalization: indentation, newlines and case are semantically relevant.
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))
    return tok


def load_tokenizer(path):
    tok = Tokenizer.from_file(str(path))
    assert [tok.token_to_id(s) for s in SPECIAL] == list(range(4))
    return tok
