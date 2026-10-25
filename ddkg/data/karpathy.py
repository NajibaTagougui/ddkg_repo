"""Karpathy-split loader for COCO and Flickr30k (dataset_coco.json / dataset_flickr30k.json).
Download: https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"""
from __future__ import annotations
import json, os
from typing import Dict, List


def load_karpathy(json_path: str, dataset: str, image_root: str) -> Dict[str, List[dict]]:
    with open(json_path) as f:
        d = json.load(f)
    out: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for im in d["images"]:
        split = "train" if im["split"] == "restval" else im["split"]
        if split not in out:
            continue
        if dataset == "coco":
            path = os.path.join(image_root, im["filepath"], im["filename"])
            iid = int(im["cocoid"])
        elif dataset == "flickr30k":
            path = os.path.join(image_root, im["filename"])
            iid = int(im["imgid"])
        else:
            raise ValueError(f"unknown dataset {dataset!r}")
        out[split].append({
            "id": iid,
            "path": path,
            "tokens": [s["tokens"] for s in im["sentences"]],
            "raw": [s["raw"] for s in im["sentences"]],
        })
    return out
