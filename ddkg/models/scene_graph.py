"""Scene-graph encoder on cached detector features.
Edges are SOFT weights in (0,1) predicted from node pairs, so the edge head is trained end-to-end by the
caption loss. Relation types are a latent pair representation (no relation labels are used)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def pair_spatial(b: torch.Tensor) -> torch.Tensor:
    """b [B,N,4] normalised xyxy -> pairwise spatial features [B,N,N,5]."""
    x1, y1, x2, y2 = b.unbind(-1)
    w = (x2 - x1).clamp_min(1e-3)
    h = (y2 - y1).clamp_min(1e-3)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx = ((cx[:, :, None] - cx[:, None, :]) / w[:, :, None]).clamp(-10, 10)
    dy = ((cy[:, :, None] - cy[:, None, :]) / h[:, :, None]).clamp(-10, 10)
    dw = torch.log(w[:, :, None] / w[:, None, :]).clamp(-5, 5)
    dh = torch.log(h[:, :, None] / h[:, None, :]).clamp(-5, 5)
    ix1 = torch.maximum(x1[:, :, None], x1[:, None, :])
    iy1 = torch.maximum(y1[:, :, None], y1[:, None, :])
    ix2 = torch.minimum(x2[:, :, None], x2[:, None, :])
    iy2 = torch.minimum(y2[:, :, None], y2[:, None, :])
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    area = w * h
    iou = inter / (area[:, :, None] + area[:, None, :] - inter + 1e-6)
    return torch.stack([dx, dy, dw, dh, iou], dim=-1)


class SceneGraphEncoder(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, rel_hidden: int, dropout: float, with_pairs: bool = True):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        if not with_pairs:          # baseline: node projection only, so parameter counts are comparable
            return
        self.p_s = nn.Linear(hidden_dim, rel_hidden)
        self.p_o = nn.Linear(hidden_dim, rel_hidden)
        self.p_sp = nn.Linear(5, rel_hidden)
        self.pair_mlp = nn.Sequential(nn.GELU(), nn.Linear(rel_hidden, rel_hidden), nn.GELU())
        self.edge_head = nn.Linear(rel_hidden, 1)

    def embed(self, feats):
        return self.node(feats)

    def forward(self, feats, boxes, node_mask):
        """feats [B,N,Df]; boxes [B,N,4]; node_mask [B,N] bool.
        Returns dict(nodes [B,N,H], edge_w [B,N,N] (diag=1 on valid nodes, 0 on padding), pair_h [B,N,N,R])."""
        B, N, _ = feats.shape
        nodes = self.node(feats)
        phi = pair_spatial(boxes.float())
        pair = self.p_s(nodes)[:, :, None, :] + self.p_o(nodes)[:, None, :, :] + self.p_sp(phi.to(nodes.dtype))
        pair_h = self.pair_mlp(pair)
        w = torch.sigmoid(self.edge_head(pair_h).squeeze(-1).float())
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        eye = torch.eye(N, dtype=torch.bool, device=feats.device)[None]
        w = torch.where(eye, torch.ones_like(w), w) * valid.float()
        return {"nodes": nodes, "edge_w": w, "pair_h": pair_h}
