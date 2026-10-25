"""Caption metrics. Final numbers use the official coco-caption code (pycocoevalcap + Java)."""
from __future__ import annotations
import logging, re
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)


def simple_tokenize(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


def cider_scores(preds: Dict[int, str], refs: Dict[int, List[str]]) -> Tuple[float, Dict[int, float]]:
    """CIDEr-D only (pure Python in pycocoevalcap, no Java). Fast; used for validation/model selection."""
    from pycocoevalcap.cider.cider import Cider
    ids = list(preds)
    gts = {i: [simple_tokenize(r) for r in refs[i]] for i in ids}
    res = {i: [simple_tokenize(preds[i])] for i in ids}
    score, arr = Cider().compute_score(gts, res)
    return float(score), {i: float(a) for i, a in zip(ids, arr)}


def full_eval(preds: Dict[int, str], raw_refs: Dict[int, List[str]]):
    """BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE (x100). Returns (metrics, per_image_cider).
    metrics['official'] is True only if PTBTokenizer and all five scorers ran."""
    ids = list(preds)
    gts = {i: [{"caption": c} for c in raw_refs[i]] for i in ids}
    res = {i: [{"caption": preds[i]}] for i in ids}
    metrics: Dict = {"official": True, "notes": []}
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tok = PTBTokenizer()
        gts_t, res_t = tok.tokenize(gts), tok.tokenize(res)
    except Exception as e:  # Java missing etc.
        metrics["official"] = False
        metrics["notes"].append(f"PTBTokenizer unavailable ({e}); simple tokenizer used")
        gts_t = {i: [simple_tokenize(c["caption"]) for c in gts[i]] for i in ids}
        res_t = {i: [simple_tokenize(res[i][0]["caption"])] for i in ids}

    per_cider: Dict[int, float] = {}
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    scorers = [("bleu", Bleu(4)), ("rouge", Rouge()), ("cider", Cider())]
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        scorers.append(("meteor", Meteor()))
    except Exception as e:
        metrics["official"] = False; metrics["notes"].append(f"METEOR unavailable ({e})")
    try:
        from pycocoevalcap.spice.spice import Spice
        scorers.append(("spice", Spice()))
    except Exception as e:
        metrics["official"] = False; metrics["notes"].append(f"SPICE unavailable ({e})")

    for name, sc in scorers:
        try:
            score, scores = sc.compute_score(gts_t, res_t)
        except Exception as e:
            metrics["official"] = False; metrics["notes"].append(f"{name} failed ({e})")
            continue
        if name == "bleu":
            for n, s in enumerate(score, 1):
                metrics[f"BLEU-{n}"] = 100 * float(s)
        elif name == "rouge":
            metrics["ROUGE-L"] = 100 * float(score)
        elif name == "cider":
            metrics["CIDEr"] = 100 * float(score)
            per_cider = {i: float(a) for i, a in zip(list(gts_t), scores)}
        elif name == "meteor":
            metrics["METEOR"] = 100 * float(score)
        elif name == "spice":
            metrics["SPICE"] = 100 * float(score)
    return metrics, per_cider
