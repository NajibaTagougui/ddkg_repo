#!/usr/bin/env python3
"""python scripts/train.py --config configs/base.yaml --set run.seed=1 --set model.caption_gating=false"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddkg.config import load_config
from ddkg.engine import fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. training.epochs=10")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    fit(load_config(a.config, a.set), resume=not a.no_resume)


if __name__ == "__main__":
    main()
