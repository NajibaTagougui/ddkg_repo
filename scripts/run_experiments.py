#!/usr/bin/env python3
"""Run a suite of experiments x seeds, one subprocess per run (frees the GPU between runs, skips finished runs).
  python scripts/run_experiments.py --suite ablation --seeds 0 1 2 --dataset coco
  python scripts/run_experiments.py --suite quick --seeds 0 --set training.epochs=5 --dry-run"""
import argparse, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def fmt(v):
    return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="quick")
    ap.add_argument("--experiments", nargs="*", default=None, help="override the suite's experiment list")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--base-config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--exp-config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--set", action="append", default=[], help="global overrides applied to every run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ec = yaml.safe_load(open(a.exp_config))
    names = a.experiments or ec["suites"][a.suite]
    failed = []
    for name in names:
        if name not in ec["experiments"]:
            sys.exit(f"unknown experiment {name!r}; known: {sorted(ec['experiments'])}")
        for seed in a.seeds:
            out = Path(a.results_dir) / a.dataset / name / f"seed{seed}"
            if (out / "test_metrics.json").exists():
                print(f"[skip] {name} seed{seed} (finished)")
                continue
            cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--config", a.base_config]
            ov = [f"data.dataset={a.dataset}", f"run.seed={seed}", f"run.name={name}", f"run.out_dir={out}"]
            ov += [f"{k}={fmt(v)}" for k, v in ec["experiments"][name].items()] + a.set
            for o in ov:
                cmd += ["--set", o]
            print("[run]", " ".join(cmd))
            if a.dry_run:
                continue
            if subprocess.run(cmd).returncode != 0:
                failed.append(f"{name}/seed{seed}")
    if failed:
        sys.exit("FAILED runs: " + ", ".join(failed))
    print("all requested runs finished; now: python scripts/aggregate_results.py --dataset", a.dataset)


if __name__ == "__main__":
    main()
