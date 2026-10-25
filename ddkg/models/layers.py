"""Shared building blocks. Attention uses F.scaled_dot_product_attention with ONE additive float mask
(never nn.MultiheadAttention's mixed bool/float masks), so padding and gating biases are explicit."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4  # finite "minus infinity" that is safe in bf16/fp16


def key_bias(valid: torch.Tensor) -> torch.Tensor:
    """valid [B,S] bool -> additive bias [B,S] (0 for valid keys, NEG for padding)."""
    return torch.zeros(valid.shape, device=valid.device, dtype=torch.float32).masked_fill(~valid, NEG)


class MHA(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.p = heads, d // heads, dropout
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q, k, v, bias=None, causal: bool = False):
        """q [B,T,D]; k,v [B,S,D]; bias None | [B,S] | [B,T,S] additive float."""
        B, T, _ = q.shape
        S = k.shape[1]
        Q = self.q(q).view(B, T, self.h, self.dh).transpose(1, 2)
        K = self.k(k).view(B, S, self.h, self.dh).transpose(1, 2)
        V = self.v(v).view(B, S, self.h, self.dh).transpose(1, 2)
        mask = None
        if bias is not None:
            if bias.dim() == 2:
                bias = bias[:, None, None, :]
            elif bias.dim() == 3:
                bias = bias[:, None, :, :]
            mask = bias.to(Q.dtype)
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.p if self.training else 0.0,
            is_causal=(causal and mask is None))
        return self.o(out.transpose(1, 2).reshape(B, T, -1))


class FFN(nn.Module):
    """Pre-norm feed-forward; returns the residual DELTA."""
    def __init__(self, d: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class GATLayer(nn.Module):
    """Dense multi-head graph attention with the efficient decomposition
    a^T[Wh_i || Wh_j] = a_l.Wh_i + a_r.Wh_j (no [B,N,N,H,2D] tensor).
    edge_bias [B,N,N] is added to the attention logits (log edge weights); key_valid [B,N] masks padded keys."""
    def __init__(self, in_d: int, out_d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_d % heads == 0
        self.h, self.dh = heads, out_d // heads
        self.W = nn.Linear(in_d, out_d, bias=False)
        self.al = nn.Parameter(torch.empty(heads, self.dh))
        self.ar = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.al)
        nn.init.xavier_uniform_(self.ar)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_d)
        self.res = nn.Linear(in_d, out_d, bias=False) if in_d != out_d else nn.Identity()

    def forward(self, x, edge_bias=None, key_valid=None):
        B, N, _ = x.shape
        Wx = self.W(x).view(B, N, self.h, self.dh)
        el = (Wx * self.al).sum(-1)                      # [B,N,H]
        er = (Wx * self.ar).sum(-1)                      # [B,N,H]
        e = F.leaky_relu(el[:, :, None, :] + er[:, None, :, :], 0.2)   # [B,N(i),N(j),H]
        if edge_bias is not None:
            e = e + edge_bias.to(e.dtype).unsqueeze(-1)
        if key_valid is not None:
            e = e.masked_fill(~key_valid[:, None, :, None], NEG)
        a = self.drop(torch.softmax(e.float(), dim=2).to(Wx.dtype))
        out = torch.einsum("bijh,bjhd->bihd", a, Wx).reshape(B, N, -1)
        return self.norm(out + self.res(x))
