"""Training / evaluation engine."""
from __future__ import annotations
import logging, math, time
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .config import Cfg, load_config
from .utils import set_seed, setup_logging, save_json, load_json, count_params
from .data.karpathy import load_karpathy
from .data.vocab import Vocabulary
from .data.features import FeatureStore
from .data.dataset import CaptionDataset, collate
from .models import DDKG
from .evaluation.coco_eval import cider_scores, full_eval

log = logging.getLogger(__name__)


def get_device(cfg: Cfg) -> torch.device:
    if cfg.hardware.device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA requested but not available; falling back to CPU")
    return torch.device("cpu")


def to_device(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def amp_ctx(cfg: Cfg, dev: torch.device):
    if dev.type == "cuda" and cfg.hardware.amp:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_data(cfg: Cfg, splits):
    recs = load_karpathy(cfg.data.karpathy_json, cfg.data.dataset, cfg.data.image_root)
    vp = Path(cfg.data.vocab_path)
    if vp.exists():
        vocab = Vocabulary.load(vp)
    else:
        vocab = Vocabulary.build((t for r in recs["train"] for t in r["tokens"]), cfg.data.min_word_freq)
        vocab.save(vp)
        log.info("built vocab (%d words) -> %s", len(vocab), vp)
    stores = {s: FeatureStore(cfg.data.feature_root, s) for s in splits}
    return recs, vocab, stores


@torch.no_grad()
def predict(model, records, store, vocab, cfg: Cfg, dev, beam: int, limit=None):
    recs = records[:limit] if limit else records
    ds = CaptionDataset(recs, store, vocab, cfg.data.max_caption_len, train=False)
    dl = DataLoader(ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collate)
    model.eval()
    preds = {}
    for batch in dl:
        idx = batch["idx"].tolist()
        b = to_device(batch, dev)
        with amp_ctx(cfg, dev):
            out = model.generate(b, beam=beam, max_len=cfg.eval.max_len, length_penalty=cfg.eval.length_penalty)
        for i, ids in zip(idx, out):
            preds[recs[i]["id"]] = vocab.decode(ids)
    return preds


def final_test(cfg: Cfg, model, recs, vocab, store, dev, out: Path, split: str = "test"):
    preds = predict(model, recs[split], store, vocab, cfg, dev, beam=cfg.eval.test_beam)
    if cfg.eval.official:
        metrics, per = full_eval(preds, {r["id"]: r["raw"] for r in recs[split]})
    else:
        c, per = cider_scores(preds, {r["id"]: [" ".join(t) for t in r["tokens"]] for r in recs[split]})
        metrics = {"CIDEr": 100 * c, "official": False, "notes": ["cfg.eval.official=false"]}
    save_json(metrics, out / f"{split}_metrics.json")
    save_json({str(k): v for k, v in preds.items()}, out / f"{split}_predictions.json")
    save_json({str(k): v for k, v in per.items()}, out / f"{split}_per_image_cider.json")
    log.info("%s metrics: %s", split, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in metrics.items()})
    return metrics


def _param_groups(model, wd):
    dec, nodec = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (nodec if p.ndim <= 1 else dec).append(p)
    return [{"params": dec, "weight_decay": wd}, {"params": nodec, "weight_decay": 0.0}]


def fit(cfg: Cfg, resume: bool = True):
    out = Path(cfg.run.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out / "train.log"))
    save_json(cfg.to_dict(), out / "config.json")
    set_seed(cfg.run.seed)
    dev = get_device(cfg)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    recs, vocab, stores = build_data(cfg, ["train", "val", "test"])
    tr = CaptionDataset(recs["train"], stores["train"], vocab, cfg.data.max_caption_len, True, cfg.training.all_captions)
    dl = DataLoader(tr, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                    collate_fn=collate, drop_last=True, pin_memory=dev.type == "cuda",
                    persistent_workers=cfg.data.num_workers > 0)
    raw = DDKG(cfg, len(vocab)).to(dev)
    net = torch.compile(raw) if (cfg.hardware.compile and dev.type == "cuda") else raw
    tot, trn = count_params(raw)
    log.info("params: total %.2fM | trainable %.2fM | train items %d | batches/epoch %d", tot / 1e6, trn / 1e6, len(tr), len(dl))

    tcfg = cfg.training
    opt = torch.optim.AdamW(_param_groups(raw, tcfg.weight_decay), lr=tcfg.lr, betas=(0.9, 0.98), eps=1e-8)
    steps_per_epoch = max(1, len(dl) // tcfg.grad_accum)
    total_steps = steps_per_epoch * tcfg.epochs
    floor = tcfg.min_lr / tcfg.lr

    def lr_lambda(s):
        if s < tcfg.warmup_steps:
            return (s + 1) / tcfg.warmup_steps
        p = min(1.0, (s - tcfg.warmup_steps) / max(1, total_steps - tcfg.warmup_steps))
        return max(floor, 0.5 * (1 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_ep, step, best, best_ep, bad, hist = 0, 0, -1.0, -1, 0, []
    last = out / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, step, best, best_ep, bad, hist = ck["epoch"] + 1, ck["step"], ck["best"], ck["best_ep"], ck["bad"], ck["hist"]
        log.info("resumed from epoch %d", start_ep)

    t_train = time.time()
    for ep in range(start_ep, tcfg.epochs):
        net.train()
        t0 = time.time()
        sums, nb = {}, 0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(dl):
            b = to_device(batch, dev)
            with amp_ctx(cfg, dev):
                loss, aux = net(b)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep} iter {it}: {loss.item()}")
            (loss / tcfg.grad_accum).backward()
            if (it + 1) % tcfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(raw.parameters(), tcfg.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            sums["loss"] = sums.get("loss", 0) + loss.detach()
            for k, v in aux.items():
                sums[k] = sums.get(k, 0) + v
            nb += 1
            if (it + 1) % (cfg.hardware.log_every * tcfg.grad_accum) == 0:
                log.info("ep %d it %d/%d loss %.4f lr %.2e", ep, it + 1, len(dl), float(sums["loss"]) / nb, sched.get_last_lr()[0])
        avg = {k: float(v) / max(nb, 1) for k, v in sums.items()}
        preds = predict(raw, recs["val"], stores["val"], vocab, cfg, dev, cfg.eval.val_beam, cfg.eval.val_images)
        vc, _ = cider_scores(preds, {r["id"]: [" ".join(t) for t in r["tokens"]] for r in recs["val"][: cfg.eval.val_images]})
        sample = next(iter(preds.items()))
        log.info("epoch %d | %s | val CIDEr %.4f | %.0fs | e.g. %r", ep, {k: round(v, 4) for k, v in avg.items()}, vc, time.time() - t0, sample[1])
        hist.append({"epoch": ep, "val_cider": vc, **avg})
        if vc > best:
            best, best_ep, bad = vc, ep, 0
            torch.save({"model": raw.state_dict(), "epoch": ep, "val_cider": vc}, out / "best.pt")
        else:
            bad += 1
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "step": step, "best": best, "best_ep": best_ep, "bad": bad, "hist": hist}, last)
        save_json(hist, out / "history.json")
        if bad >= tcfg.patience:
            log.info("early stopping (no val improvement for %d epochs)", bad)
            break

    raw.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=False)["model"])
    metrics = final_test(cfg, raw, recs, vocab, stores["test"], dev, out)
    summary = {"params_total": tot, "params_trainable": trn, "best_val_cider": best, "best_epoch": best_ep,
               "epochs_run": len(hist), "train_seconds": time.time() - t_train,
               "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else None}
    save_json(summary, out / "run_summary.json")
    return metrics


def evaluate_run(run_dir: str, split: str = "test", overrides=()):
    out = Path(run_dir)
    d = load_json(out / "config.json")
    cfg = Cfg(d)
    for o in overrides:
        from .config import parse_override
        k, v = parse_override(o)
        sub = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            sub = getattr(sub, p)
        setattr(sub, parts[-1], v)
    setup_logging(None)
    dev = get_device(cfg)
    recs, vocab, stores = build_data(cfg, [split])
    model = DDKG(cfg, len(vocab)).to(dev)
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=False)["model"])
    return final_test(cfg, model, recs, vocab, stores[split], dev, out, split)
