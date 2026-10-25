"""Self-tests for the model (run on CPU or GPU with tiny dimensions; no data, detector or ConceptNet needed).
Checks, for every ablation variant: finite loss, backward pass, gradient reaches the modules that were
untrained in the first draft (edge head, edge pruner, retrieval projection, gate, per-token gate),
beam search output, padding invariance (padded nodes must not change valid outputs) and decoder causality."""
from __future__ import annotations
from pathlib import Path
import torch
from .config import load_config
from .models import DDKG

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
TINY = ["knowledge_graph.require_real=false", "knowledge_graph.top_k=5", "features.num_objects=10",
        "features.feat_dim=32", "model.hidden_dim=64", "model.ffn_dim=128", "model.num_layers=2",
        "model.decoder_layers=2", "model.gat_heads=2", "model.attn_heads=4", "model.gate_dim=16",
        "model.relation_hidden=32", "data.max_caption_len=12", "model.dropout=0.0", "hardware.amp=false"]
VARIANTS = {
    "full": [],
    "image_only_gate": ["model.caption_gating=false"],
    "static_gate": ["knowledge_graph.static_gate=true", "model.caption_gating=false"],
    "no_kg": ["model.use_kg=false"],
    "no_cross_attn": ["model.use_cross_attn=false"],
    "no_edge_pruning": ["model.edge_pruning=false"],
    "dense_kg_attention": ["model.kg_attention=dense"],
    "baseline_transformer": ["model.mode=baseline"],
}
V = 50


def make_cfg(extra=()):
    return load_config(str(BASE), TINY + list(extra))


def make_batch(cfg, dev, B=3):
    N, D, T = cfg.features.num_objects, cfg.features.feat_dim, cfg.data.max_caption_len
    g = torch.Generator().manual_seed(1)
    x1, y1 = torch.rand(B, N, generator=g) * 0.5, torch.rand(B, N, generator=g) * 0.5
    boxes = torch.stack([x1, y1, x1 + 0.05 + torch.rand(B, N, generator=g) * 0.4, y1 + 0.05 + torch.rand(B, N, generator=g) * 0.4], -1)
    tokens = torch.randint(4, V, (B, T), generator=g)
    tokens[:, 0] = 1
    for b, end in enumerate([T - 1, 8, 5][:B]):
        tokens[b, end] = 2
        tokens[b, end + 1:] = 0
    batch = {"feats": torch.randn(B, N, D, generator=g), "boxes": boxes,
             "nvalid": torch.tensor([N, 6, 1][:B]), "tokens": tokens}
    return {k: v.to(dev) for k, v in batch.items()}


def _grad_alive(model, name):
    p = dict(model.named_parameters()).get(name)
    return None if p is None else (p.grad is not None and float(p.grad.abs().sum()) > 0)


def check_variant(name, extra, dev):
    cfg = make_cfg(extra)
    torch.manual_seed(0)
    model = DDKG(cfg, V).to(dev)
    batch = make_batch(cfg, dev)
    model.train()
    loss, aux = model(batch)
    assert torch.isfinite(loss), f"[{name}] loss not finite"
    loss.backward()
    must_have = {
        "sg.edge_head.weight": not model.baseline,
        "enc.pruners.0.out.weight": (not model.baseline) and cfg.model.edge_pruning,
        "kg_ret.q.0.weight": model.use_kg and not cfg.knowledge_graph.static_gate,
        "kg_ret.gate.0.weight": model.use_kg and not cfg.knowledge_graph.static_gate,
        "decoder.layers.0.ca.gq.weight": model.use_kg and cfg.model.caption_gating,
    }
    for pname, expected in must_have.items():
        alive = _grad_alive(model, pname)
        if expected:
            assert alive, f"[{name}] no gradient reaches {pname}"
    model.eval()
    with torch.no_grad():
        for beam in (1, 3):
            out = model.generate(batch, beam=beam, max_len=8)
            assert len(out) == batch["feats"].shape[0] and all(isinstance(t, int) for s in out for t in s)
        # padding invariance: perturbing padded nodes must not change valid-node encodings or logits
        e1 = model.encode(batch)
        l1, _ = model.decoder(batch["tokens"][:, :-1], e1["vis"], e1["kg"], e1["vis_bias"], e1["kg_bias"])
        pert = {k: v.clone() for k, v in batch.items()}
        pert["feats"][1, 6:] = torch.randn_like(pert["feats"][1, 6:]) * 5
        pert["boxes"][1, 6:] = torch.rand_like(pert["boxes"][1, 6:]).sort(-1).values
        pert["feats"][2, 1:] = torch.randn_like(pert["feats"][2, 1:]) * 5
        e2 = model.encode(pert)
        l2, _ = model.decoder(pert["tokens"][:, :-1], e2["vis"], e2["kg"], e2["vis_bias"], e2["kg_bias"])
        assert torch.allclose(e1["vis"][1, :6], e2["vis"][1, :6], atol=1e-4), f"[{name}] padded nodes leak into vis (item 1)"
        assert torch.allclose(e1["vis"][2, :1], e2["vis"][2, :1], atol=1e-4), f"[{name}] padded nodes leak into vis (item 2)"
        assert torch.allclose(l1[1], l2[1], atol=1e-3), f"[{name}] padded nodes change logits"
        # causality: changing future tokens must not change earlier logits
        t2 = batch["tokens"][:, :-1].clone()
        t2[:, 6:] = (t2[:, 6:] + 7) % V
        l3, _ = model.decoder(t2, e1["vis"], e1["kg"], e1["vis_bias"], e1["kg_bias"])
        assert torch.allclose(l1[:, :6], l3[:, :6], atol=1e-4), f"[{name}] decoder is not causal"
    total = sum(p.numel() for p in model.parameters())
    return {"variant": name, "loss": float(loss), "params": total, "aux": {k: round(float(v), 4) for k, v in aux.items()}}


def run_selftest(device: str = "cpu"):
    dev = torch.device(device)
    results = []
    for name, extra in VARIANTS.items():
        results.append(check_variant(name, extra, dev))
        print(f"  OK  {name:22s} loss={results[-1]['loss']:.3f} params={results[-1]['params']:,} aux={results[-1]['aux']}")
    return results
