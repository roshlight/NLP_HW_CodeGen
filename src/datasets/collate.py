import torch


def collate(rows):
    def padded(seqs, fill):
        length = (max(map(len, seqs)) + 7) // 8 * 8
        out = torch.full((len(seqs), length), fill, dtype=torch.long)
        for i, seq in enumerate(seqs):
            out[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
        return out

    return {
        "source": padded([r[0] for r in rows], 0),
        "decoder_input": padded([[1] + list(r[1][:-1]) for r in rows], 0),
        "labels": padded([r[1] for r in rows], -100),
    }
