#!/usr/bin/env python3
"""Convert ConceptNet Numberbatch (English) to the .npy + vocab.json used by the KG retriever.
Download numberbatch-en-19.08.txt.gz from https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/
Concepts are single lowercase words. Words that occur in the caption vocabulary come first, then the file
is filled in file order up to --max-concepts (the file is NOT frequency-sorted; this is a documented choice)."""
import argparse, gzip, json, re
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vocab-json", default=None, help="caption vocab.json whose words get priority")
    ap.add_argument("--max-concepts", type=int, default=60000)
    a = ap.parse_args()
    priority = set(json.load(open(a.vocab_json))) if a.vocab_json else set()
    opener = gzip.open if a.input.endswith(".gz") else open
    prio, rest = {}, {}
    with opener(a.input, "rt", encoding="utf-8") as f:
        first = f.readline().split()
        header = len(first) == 2 and all(x.isdigit() for x in first)
        lines = f if header else [" ".join(first)] + list(f)
        for line in lines:
            parts = line.rstrip().split(" ")
            w = parts[0]
            if w.startswith("/c/"):
                if not w.startswith("/c/en/"):
                    continue
                w = w[len("/c/en/"):].split("/")[0]
            if not re.fullmatch(r"[a-z]+", w) or w in prio or w in rest:
                continue
            vec = np.asarray(parts[1:], dtype=np.float32)
            (prio if w in priority else rest)[w] = vec
    words = list(prio)
    for w in rest:
        if len(words) >= a.max_concepts:
            break
        words.append(w)
    table = {**prio, **rest}
    emb = np.stack([table[w] for w in words]).astype(np.float16)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "conceptnet_numberbatch.npy", emb)
    json.dump(words, open(out / "conceptnet_vocab.json", "w"))
    print(f"saved {emb.shape} ({len(prio)} caption-vocab words + {len(words) - len(prio)} others) -> {out}")


if __name__ == "__main__":
    main()
