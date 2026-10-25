#!/usr/bin/env python3
"""Re-evaluate a finished run: python scripts/evaluate.py --run-dir results/coco/full/seed0 --split test"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddkg.engine import evaluate_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    evaluate_run(a.run_dir, a.split, a.set)


if __name__ == "__main__":
    main()
