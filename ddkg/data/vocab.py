from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class Vocabulary:
    def __init__(self, words: List[str] | None = None):
        self.itos = SPECIALS + list(words or [])
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists: Iterable[List[str]], min_freq: int = 5) -> "Vocabulary":
        c: Counter = Counter()
        for toks in token_lists:
            c.update(toks)
        words = [w for w, n in sorted(c.items(), key=lambda x: (-x[1], x[0])) if n >= min_freq and w not in SPECIALS]
        return cls(words)

    def encode(self, tokens: List[str], max_len: int) -> List[int]:
        ids = [SOS] + [self.stoi.get(t, UNK) for t in tokens][: max_len - 2] + [EOS]
        return ids + [PAD] * (max_len - len(ids))

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i in (PAD, SOS):
                continue
            out.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return " ".join(out)

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path) -> "Vocabulary":
        with open(path) as f:
            itos = json.load(f)
        v = cls(itos[len(SPECIALS):])
        assert v.itos[: len(SPECIALS)] == SPECIALS
        return v
