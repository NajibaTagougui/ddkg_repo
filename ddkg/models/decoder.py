"""Transformer decoder with caption-conditioned concept gating and batched beam search."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MHA, FFN

PAD, SOS, EOS = 0, 1, 2


class SinPE(nn.Module):
    def __init__(self, d: int, max_len: int, dropout: float):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class GatedCrossAttn(nn.Module):
    """Cross-attention over [visual || concept] nodes. If caption_gating, every query token t gates every
    concept m:  gate_tm = sigmoid((Wq x_t).(Wk c_m)/sqrt(g) + b);  the logit bias is log gate_tm + kg_bias_m
    (kg_bias = log image-level prior, NEG on padded objects). Visual keys only get their padding bias."""
    def __init__(self, d, heads, dropout, gate_dim, caption_gating):
        super().__init__()
        self.cg, self.gd = caption_gating, gate_dim
        self.vp = nn.Linear(d, d, bias=False)
        self.kp = nn.Linear(d, d, bias=False)
        self.attn = MHA(d, heads, dropout)
        if caption_gating:
            self.gq, self.gk = nn.Linear(d, gate_dim), nn.Linear(d, gate_dim)
            self.gb = nn.Parameter(torch.full((1,), 2.0))

    def forward(self, q, vis, kg, vis_bias, kg_bias):
        B, T, _ = q.shape
        vb = vis_bias[:, None, :].expand(B, T, vis_bias.shape[1])
        if kg is None:
            return self.attn(q, self.vp(vis), self.vp(vis), bias=vb), None
        kv = torch.cat([self.vp(vis), self.kp(kg)], dim=1)
        kb = kg_bias[:, None, :].expand(B, T, kg_bias.shape[1])
        gate = None
        if self.cg:
            s = torch.einsum("btg,bmg->btm", self.gq(q), self.gk(kg)).float() / math.sqrt(self.gd) + self.gb.float()
            kb = kb + F.logsigmoid(s)
            gate = torch.sigmoid(s).detach()
        return self.attn(q, kv, kv, bias=torch.cat([vb, kb], dim=-1)), gate


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ffn_dim, dropout, gate_dim, caption_gating):
        super().__init__()
        self.sa, self.n1 = MHA(d, heads, dropout), nn.LayerNorm(d)
        self.ca, self.n2 = GatedCrossAttn(d, heads, dropout, gate_dim, caption_gating), nn.LayerNorm(d)
        self.ffn = FFN(d, ffn_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, vis, kg, vis_bias, kg_bias):
        xl = self.n1(x)
        x = x + self.drop(self.sa(xl, xl, xl, causal=True))
        ca, gate = self.ca(self.n2(x), vis, kg, vis_bias, kg_bias)
        x = x + self.drop(ca)
        return x + self.ffn(x), gate


class Decoder(nn.Module):
    def __init__(self, m, vocab_size: int, max_len: int):
        super().__init__()
        d = m.hidden_dim
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()
        self.pe = SinPE(d, max_len + 2, m.dropout)
        cg = bool(m.caption_gating and m.use_kg and m.mode == "ddkg")
        self.layers = nn.ModuleList(DecoderLayer(d, m.attn_heads, m.ffn_dim, m.dropout, m.gate_dim, cg) for _ in range(m.decoder_layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size, bias=False)

    def forward(self, tokens, vis, kg, vis_bias, kg_bias):
        x = self.pe(self.emb(tokens))
        gates = []
        for layer in self.layers:
            x, g = layer(x, vis, kg, vis_bias, kg_bias)
            if g is not None:
                gates.append(g)
        return self.out(self.norm(x)), gates

    @torch.no_grad()
    def beam_search(self, vis, kg, vis_bias, kg_bias, beam=3, max_len=20, length_penalty=0.7):
        """Batched beam search. Returns list (len B) of token-id lists without <sos>/<eos>/<pad>."""
        B, dev = vis.size(0), vis.device
        V = self.out.out_features
        rep = lambda t: None if t is None else t.repeat_interleave(beam, dim=0)
        vis, kg, vb, kb = rep(vis), rep(kg), rep(vis_bias), rep(kg_bias)
        seq = torch.full((B * beam, 1), SOS, dtype=torch.long, device=dev)
        scores = torch.zeros(B, beam, device=dev)
        scores[:, 1:] = -1e9                                      # only beam 0 is live at step 0
        finished = torch.zeros(B * beam, dtype=torch.bool, device=dev)
        pad_row = torch.full((V,), float("-inf"), device=dev)
        pad_row[PAD] = 0.0
        for _ in range(max_len):
            logits, _ = self.forward(seq, vis, kg, vb, kb)
            lp = F.log_softmax(logits[:, -1].float(), dim=-1)
            lp = torch.where(finished[:, None], pad_row[None, :], lp)
            cand = (scores.reshape(B * beam, 1) + lp).reshape(B, beam * V)
            top, idx = cand.topk(beam, dim=-1)
            src = idx // V
            tok = idx % V
            sel = (torch.arange(B, device=dev)[:, None] * beam + src).reshape(-1)
            seq = torch.cat([seq[sel], tok.reshape(-1, 1)], dim=1)
            finished = finished[sel] | (tok.reshape(-1) == EOS)
            scores = top
            if bool(finished.all()):
                break
        lens = ((seq[:, 1:] != PAD).sum(1)).clamp_min(1).float()
        norm = scores.reshape(-1) / lens.pow(length_penalty)
        best = norm.reshape(B, beam).argmax(dim=1)
        out = []
        for b in range(B):
            ids = seq[b * beam + int(best[b]), 1:].tolist()
            res = []
            for t in ids:
                if t in (EOS, PAD):
                    break
                res.append(t)
            out.append(res)
        return out
