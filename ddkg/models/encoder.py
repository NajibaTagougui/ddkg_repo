"""Dual Graph Encoder: per layer = visual GAT (soft, pruned edges) -> grouped concept GAT -> bidirectional
cross-attention -> FFN. Also a plain region-Transformer encoder used as the baseline."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MHA, FFN, GATLayer, key_bias


class SoftEdgePruner(nn.Module):
    """Scores edges from node states; soft weight = base_weight * score (differentiable), and the lowest
    `ratio` fraction of the surviving edges is hard-removed (cumulative across layers)."""
    def __init__(self, node_d: int, pair_d: int, ratio: float, hidden: int = 128):
        super().__init__()
        self.ratio = ratio
        self.si, self.sj, self.sp = nn.Linear(node_d, hidden), nn.Linear(node_d, hidden), nn.Linear(pair_d, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, h, pair_h, base_w, keep_prev, node_mask):
        B, N, _ = base_w.shape
        sc = torch.sigmoid(self.out(F.gelu(self.si(h)[:, :, None, :] + self.sj(h)[:, None, :, :] + self.sp(pair_h))).squeeze(-1).float())
        soft = base_w.float() * sc
        off = keep_prev                                               # alive off-diagonal edges [B,N,N] bool
        if self.ratio > 0:
            flat = soft.detach().masked_fill(~off, -1.0).reshape(B, -1)
            cnt = off.reshape(B, -1).sum(1)
            k = ((1.0 - self.ratio) * cnt.float()).floor().long().clamp_min(1)
            thr = flat.sort(dim=1, descending=True).values.gather(1, (k - 1).unsqueeze(1))
            keep = (flat >= thr).reshape(B, N, N) & off
        else:
            keep = off
        w = torch.where(keep, soft, torch.zeros_like(soft))
        w = w + torch.eye(N, device=h.device)[None] * node_mask[:, :, None].float()
        sparsity = (soft * off.float()).sum() / off.float().sum().clamp_min(1.0)
        return w, keep, sparsity


class BiCross(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.v2k, self.k2v = MHA(d, heads, dropout), MHA(d, heads, dropout)
        self.nv, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, vis, kg, vis_bias, kg_bias):
        vis = self.nv(vis + self.drop(self.v2k(vis, kg, kg, bias=kg_bias)))
        kg = self.nk(kg + self.drop(self.k2v(kg, vis, vis, bias=vis_bias)))
        return vis, kg


class DualGraphEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        H, L = m.hidden_dim, m.num_layers
        self.m = m
        self.vis_gat = nn.ModuleList(GATLayer(H, H, m.gat_heads, m.dropout) for _ in range(L))
        self.pruners = nn.ModuleList(SoftEdgePruner(H, m.relation_hidden, m.pruning_ratio) for _ in range(L)) if m.edge_pruning else None
        self.ffn_v = nn.ModuleList(FFN(H, m.ffn_dim, m.dropout) for _ in range(L))
        if m.use_kg:
            self.kg_gat = nn.ModuleList(GATLayer(H, H, m.gat_heads, m.dropout) for _ in range(L))
            self.ffn_k = nn.ModuleList(FFN(H, m.ffn_dim, m.dropout) for _ in range(L))
            self.cross = nn.ModuleList(BiCross(H, m.attn_heads, m.dropout) for _ in range(L)) if m.use_cross_attn else None

    def forward(self, vis, node_mask, base_w, pair_h, kg=None, kg_bias=None):
        """vis [B,N,H]; kg [B,N,K,H] or None; kg_bias [B,N*K] (log prior, NEG on padded objects).
        Returns vis [B,N,H], kg_flat [B,N*K,H] | None, sparsity (scalar tensor)."""
        B, N, H = vis.shape
        vis_bias = key_bias(node_mask)
        eye = torch.eye(N, dtype=torch.bool, device=vis.device)[None]
        keep = node_mask[:, :, None] & node_mask[:, None, :] & ~eye
        w = base_w
        spars = vis.new_zeros(()).float()
        K = kg.shape[2] if kg is not None else 0
        for l in range(len(self.vis_gat)):
            if self.pruners is not None:
                w, keep, s = self.pruners[l](vis, pair_h, base_w, keep, node_mask)
                spars = spars + s / len(self.vis_gat)
            vis = self.vis_gat[l](vis, edge_bias=torch.log(w.clamp_min(1e-6)), key_valid=node_mask)
            if kg is not None:
                if self.m.kg_attention == "grouped":
                    kg = self.kg_gat[l](kg.reshape(B * N, K, H)).reshape(B, N, K, H)
                else:
                    kvalid = node_mask[:, :, None].expand(B, N, K).reshape(B, N * K)
                    kg = self.kg_gat[l](kg.reshape(B, N * K, H), key_valid=kvalid).reshape(B, N, K, H)
                if self.cross is not None:
                    vis, kf = self.cross[l](vis, kg.reshape(B, N * K, H), vis_bias, kg_bias)
                    kg = kf.reshape(B, N, K, H)
                kg = kg + self.ffn_k[l](kg)
            vis = vis + self.ffn_v[l](vis)
        return vis, (kg.reshape(B, N * K, H) if kg is not None else None), spars


class RegionEncoder(nn.Module):
    """Baseline: Transformer encoder over region features (no graphs, no KG)."""
    def __init__(self, m):
        super().__init__()
        H = m.hidden_dim
        self.attn = nn.ModuleList(MHA(H, m.attn_heads, m.dropout) for _ in range(m.num_layers))
        self.n1 = nn.ModuleList(nn.LayerNorm(H) for _ in range(m.num_layers))
        self.ffn = nn.ModuleList(FFN(H, m.ffn_dim, m.dropout) for _ in range(m.num_layers))
        self.drop = nn.Dropout(m.dropout)

    def forward(self, x, node_mask):
        b = key_bias(node_mask)
        for a, n, f in zip(self.attn, self.n1, self.ffn):
            xl = n(x)
            x = x + self.drop(a(xl, xl, xl, bias=b))
            x = x + f(x)
        return x
