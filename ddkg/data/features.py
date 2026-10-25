"""Pre-extracted detector features stored as memory-mapped .npy files (no extra dependencies).
Layout in <root>:  <split>_feats.npy [n,N,D] fp16 | <split>_boxes.npy [n,N,4] fp16 (normalised xyxy)
                   <split>_nvalid.npy [n] uint8   | <split>_ids.json (row -> image id)"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import List
import numpy as np


def _p(root, split, name):
    return os.path.join(root, f"{split}_{name}")


def create_store(root: str, split: str, ids: List[int], n_obj: int, feat_dim: int):
    Path(root).mkdir(parents=True, exist_ok=True)
    n = len(ids)
    ids_path = _p(root, split, "ids.json")
    resume = os.path.exists(ids_path) and os.path.exists(_p(root, split, "feats.npy"))
    mode = "r+" if resume else "w+"
    mm = np.lib.format.open_memmap
    feats = mm(_p(root, split, "feats.npy"), mode=mode, dtype=np.float16, shape=(n, n_obj, feat_dim))
    boxes = mm(_p(root, split, "boxes.npy"), mode=mode, dtype=np.float16, shape=(n, n_obj, 4))
    nvalid = mm(_p(root, split, "nvalid.npy"), mode=mode, dtype=np.uint8, shape=(n,))
    with open(ids_path, "w") as f:
        json.dump([int(i) for i in ids], f)
    return feats, boxes, nvalid


class FeatureStore:
    def __init__(self, root: str, split: str):
        self.feats = np.load(_p(root, split, "feats.npy"), mmap_mode="r")
        self.boxes = np.load(_p(root, split, "boxes.npy"), mmap_mode="r")
        self.nvalid = np.load(_p(root, split, "nvalid.npy"), mmap_mode="r")
        with open(_p(root, split, "ids.json")) as f:
            ids = json.load(f)
        self.row = {int(i): r for r, i in enumerate(ids)}
        missing = int((np.asarray(self.nvalid) == 0).sum())
        if missing:
            raise RuntimeError(f"{missing} rows in {split} store have no features; re-run extract_features.py")
