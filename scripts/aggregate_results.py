#!/usr/bin/env python3
"""Aggregate results/<dataset>/<exp>/seed*/ into mean +- std tables, paired significance tests on per-image CIDEr,
and Markdown/LaTeX tables ready to paste into the paper. Reads only files written by real runs."""
import argparse, json
from pathlib import Path
import numpy as np

METRICS = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"]


def load_runs(root: Path):
    runs = {}
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seeds = []
        for sd in sorted(exp_dir.glob("seed*")):
            mf = sd / "test_metrics.json"
            if mf.exists():
                m = json.load(open(mf))
                per = json.load(open(sd / "test_per_image_cider.json")) if (sd / "test_per_image_cider.json").exists() else {}
                summ = json.load(open(sd / "run_summary.json")) if (sd / "run_summary.json").exists() else {}
                seeds.append({"metrics": m, "per": per, "summary": summ, "seed": sd.name})
        if seeds:
            runs[exp_dir.name] = seeds
    return runs


def mean_std(seeds, key):
    v = [s["metrics"][key] for s in seeds if key in s["metrics"]]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else None


def mean_per_image(seeds):
    ids = set.intersection(*[set(s["per"]) for s in seeds]) if all(s["per"] for s in seeds) else set()
    ids = sorted(ids)
    return ids, np.mean([[s["per"][i] for i in ids] for s in seeds], axis=0) if ids else None


def paired_tests(a, b, n_boot=10000, seed=0):
    """a, b: per-image scores (same images). Returns mean diff (a-b), 95% CI, bootstrap p (two-sided), Wilcoxon p."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    try:
        from scipy.stats import wilcoxon
        p_w = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p_w = float("nan")
    return float(d.mean()), float(lo), float(hi), p_boot, p_w


def fmt(ms):
    return "-" if ms is None else (f"{ms[0]:.2f} ± {ms[1]:.2f}" if ms[2] > 1 else f"{ms[0]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--reference", default="full", help="experiment the others are compared against")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    root = Path(a.results_dir) / a.dataset
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no finished runs under {root}")
    ref = runs.get(a.reference)
    ref_ids, ref_pi = mean_per_image(ref) if ref else ([], None)
    md = [f"## {a.dataset}: test metrics, mean ± std over seeds (x100)", "",
          "| Experiment | seeds | official | " + " | ".join(METRICS) + " | Δ CIDEr vs " + a.reference + " [95% CI] | p (boot) | p (Wilcoxon) |",
          "|---|---|---|" + "---|" * len(METRICS) + "---|---|---|"]
    summary = {}
    for name, seeds in runs.items():
        official = all(s["metrics"].get("official", False) for s in seeds)
        cells = [fmt(mean_std(seeds, m)) for m in METRICS]
        delta = pb = pw = "-"
        entry = {"seeds": len(seeds), "official": official, "metrics": {m: mean_std(seeds, m) for m in METRICS}}
        if ref is not None and name != a.reference:
            ids, pi = mean_per_image(seeds)
            common = sorted(set(ids) & set(ref_ids))
            if common:
                ia = {i: k for k, i in enumerate(ids)}
                ib = {i: k for k, i in enumerate(ref_ids)}
                x = np.array([pi[ia[i]] for i in common])
                y = np.array([ref_pi[ib[i]] for i in common])
                d, lo, hi, p1, p2 = paired_tests(x, y, a.n_boot)
                delta, pb, pw = f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]", f"{p1:.4f}", f"{p2:.4g}"
                entry["delta_cider_x100"] = {"diff": 100 * d, "ci": [100 * lo, 100 * hi], "p_boot": p1, "p_wilcoxon": p2}
        md.append(f"| {name} | {len(seeds)} | {'yes' if official else 'NO'} | " + " | ".join(cells) + f" | {delta} | {pb} | {pw} |")
        summary[name] = entry
    md += ["", "`official = NO` means PTBTokenizer/METEOR/SPICE did not all run (Java missing?); such numbers are NOT comparable with the literature.",
           "Per-image tests are paired over test images, using CIDEr averaged over seeds; with <3 seeds treat p-values as indicative only."]
    out_md = Path(a.results_dir) / f"{a.dataset}_tables.md"
    out_md.write_text("\n".join(md) + "\n")
    (Path(a.results_dir) / f"{a.dataset}_summary.json").write_text(json.dumps(summary, indent=2))
    tex = ["\\begin{tabular}{l" + "c" * len(METRICS) + "}", "\\toprule", "Model & " + " & ".join(METRICS) + " \\\\", "\\midrule"]
    for name, seeds in runs.items():
        tex.append(name.replace("_", "\\_") + " & " + " & ".join(fmt(mean_std(seeds, m)).replace("±", "$\\pm$") for m in METRICS) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (Path(a.results_dir) / f"{a.dataset}_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_md}, {a.dataset}_summary.json, {a.dataset}_table.tex")


if __name__ == "__main__":
    main()
