#!/usr/bin/env python3
"""python scripts/smoke_test.py [--device cuda]   Runs ddkg/selftest.py for every ablation variant. No data needed."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ddkg.selftest import run_selftest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    print(f"torch {torch.__version__} | device {a.device}")
    run_selftest(a.device)
    if a.device == "cuda":
        from ddkg.selftest import make_cfg, make_batch, V
        from ddkg.models import DDKG
        cfg = make_cfg()
        model = DDKG(cfg, V).cuda()
        batch = make_batch(cfg, torch.device("cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(batch)
        loss.backward()
        assert torch.isfinite(loss)
        print("  OK  bf16 autocast forward/backward")
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
