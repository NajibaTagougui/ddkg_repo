"""Image-conditioned ConceptNet retrieval with a learned gate.
Selection of the top-K concepts is non-differentiable, so the similarity of every SELECTED concept is
recomputed with gradients and fed to the gate MLP; this is how the retrieval projection is trained."""
from __future__ import annotations
import json, logging, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def load_kg_table(embed_path: str, vocab_path: str, require_real: bool, fallback_size: int = 2000, dim: int = 300):
    if os.path.exists(embed_path) and os.path.exists(vocab_path):
        emb = np.load(embed_path).astype(np.float32)
        with open(vocab_path) as f:
            vocab = json.load(f)
        assert emb.shape[0] == len(vocab), "embedding rows and vocab length differ"
        return torch.from_numpy(emb), vocab
    if require_real:
        raise FileNotFoundError(
            f"ConceptNet files not found ({embed_path}, {vocab_path}). Run scripts/prepare_conceptnet.py, "
            "or set knowledge_graph.require_real=false ONLY for smoke tests (random embeddings are meaningless).")
    log.warning("Using RANDOM concept embeddings (smoke test only).")
    g = torch.Generator().manual_seed(0)
    return torch.randn(fallback_size, dim, generator=g), [f"c{i}" for i in range(fallback_size)]


class KGRetriever(nn.Module):
    def __init__(self, table: torch.Tensor, hidden_dim: int, top_k: int, threshold: float, static_gate: bool):
        super().__init__()
        self.K, self.thr, self.static = top_k, threshold, static_gate
        D = table.shape[1]
        self.D = D
        self.register_buffer("kg_emb", table.half())
        self.register_buffer("kg_norm", F.normalize(table.float(), dim=-1).half())
        self.q = nn.Sequential(nn.Linear(hidden_dim, D), nn.LayerNorm(D))
        self.ctx = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(D + hidden_dim + 1, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, nodes: torch.Tensor, node_mask: torch.Tensor):
        """nodes [B,N,H]; node_mask [B,N]. Returns emb [B,N,K,D], prior [B,N,K] in [0,1], idx [B,N,K]."""
        B, N, H = nodes.shape
        q = F.normalize(self.q(nodes).float(), dim=-1)                          # [B,N,D]
        tbl_n = self.kg_norm if q.is_cuda else self.kg_norm.float()
        with torch.no_grad():
            sims = torch.matmul(q.detach().reshape(B * N, self.D).to(tbl_n.dtype), tbl_n.t())
            idx = sims.topk(self.K, dim=-1).indices.view(B, N, self.K)
        emb = self.kg_emb[idx].float()                                          # [B,N,K,D]
        sim = (q[:, :, None, :] * self.kg_norm[idx].float()).sum(-1)            # [B,N,K] (differentiable)
        if self.static:
            prior = torch.ones(B, N, self.K, device=nodes.device)
        else:
            m = node_mask[..., None].float()
            u = torch.tanh(self.ctx((nodes * m).sum(1) / m.sum(1).clamp_min(1.0)))   # [B,H] image summary
            gin = torch.cat([emb, u[:, None, None, :].expand(B, N, self.K, H).to(emb.dtype), sim.unsqueeze(-1)], dim=-1)
            g = torch.sigmoid(self.gate(gin).squeeze(-1))
            prior = g * (g > self.thr).float()
        return emb, prior, idx
