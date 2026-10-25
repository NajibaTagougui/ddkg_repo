#!/usr/bin/env python3
"""Build and save the caption vocabulary from the Karpathy train split; print dataset statistics."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddkg.config import load_config
from ddkg.data.karpathy import load_karpathy
from ddkg.data.vocab import Vocabulary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    recs = load_karpathy(cfg.data.karpathy_json, cfg.data.dataset, cfg.data.image_root)
    for s, r in recs.items():
        print(f"{s}: {len(r)} images, {sum(len(x['tokens']) for x in r)} captions")
    v = Vocabulary.build((t for r in recs["train"] for t in r["tokens"]), cfg.data.min_word_freq)
    v.save(cfg.data.vocab_path)
    print(f"vocab: {len(v)} words (min_freq={cfg.data.min_word_freq}) -> {cfg.data.vocab_path}")
    missing = [r["path"] for r in recs["train"][:50] if not Path(r["path"]).exists()]
    if missing:
        print("WARNING: image files not found, e.g.", missing[:3])


if __name__ == "__main__":
    main()
