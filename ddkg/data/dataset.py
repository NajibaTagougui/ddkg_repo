from __future__ import annotations
import random
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """One item = (image features, one caption). Eval mode yields one item per image (caption 0 unused)."""
    def __init__(self, records: List[dict], store, vocab, max_len: int, train: bool, all_captions: bool = False):
        self.r, self.store, self.vocab, self.max_len, self.train = records, store, vocab, max_len, train
        if train and all_captions:
            self.items = [(i, j) for i, r in enumerate(records) for j in range(len(r["tokens"]))]
        elif train:
            self.items = [(i, -1) for i in range(len(records))]
        else:
            self.items = [(i, 0) for i in range(len(records))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, j = self.items[k]
        rec = self.r[i]
        if j < 0:
            j = random.randrange(len(rec["tokens"]))
        row = self.store.row[rec["id"]]
        return {
            "feats": torch.from_numpy(np.asarray(self.store.feats[row], dtype=np.float32)),
            "boxes": torch.from_numpy(np.asarray(self.store.boxes[row], dtype=np.float32)),
            "nvalid": torch.tensor(int(self.store.nvalid[row]), dtype=torch.long),
            "tokens": torch.tensor(self.vocab.encode(rec["tokens"][j], self.max_len), dtype=torch.long),
            "idx": torch.tensor(i, dtype=torch.long),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
