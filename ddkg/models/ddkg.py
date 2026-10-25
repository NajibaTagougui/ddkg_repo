from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import key_bias, NEG
from .scene_graph import SceneGraphEncoder
from .kg import KGRetriever, load_kg_table
from .encoder import DualGraphEncoder, RegionEncoder
from .decoder import Decoder


class DDKG(nn.Module):
    """batch: feats [B,N,Df], boxes [B,N,4], nvalid [B], tokens [B,T]."""
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        m, kgc = cfg.model, cfg.knowledge_graph
        self.m, self.cfg = m, cfg
        self.baseline = m.mode == "baseline"
        self.use_kg = (not self.baseline) and m.use_kg
        H = m.hidden_dim
        self.sg = SceneGraphEncoder(cfg.features.feat_dim, H, m.relation_hidden, m.dropout, with_pairs=not self.baseline)
        if self.baseline:
            self.region = RegionEncoder(m)
        else:
            self.enc = DualGraphEncoder(m)
            if self.use_kg:
                table, _ = load_kg_table(kgc.embed_path, kgc.vocab_path, kgc.require_real)
                self.kg_ret = KGRetriever(table, H, kgc.top_k, kgc.gate_threshold, kgc.static_gate)
                self.kg_ada = nn.Sequential(nn.Linear(table.shape[1], H), nn.LayerNorm(H))
                self.gate_proj = nn.Linear(1, H)
        self.decoder = Decoder(m, vocab_size, cfg.data.max_caption_len)
        lw = cfg.training.loss_weights
        self.lw_sp, self.lw_gate, self.lw_cons = lw.sparsity, lw.gate_l1, lw.consistency
        self.ls = cfg.training.label_smoothing

    @staticmethod
    def node_mask(batch):
        N = batch["feats"].shape[1]
        return torch.arange(N, device=batch["feats"].device)[None, :] < batch["nvalid"][:, None]

    def encode(self, batch):
        mask = self.node_mask(batch)
        vis_bias = key_bias(mask)
        out = {"mask": mask, "vis_bias": vis_bias, "kg": None, "kg_bias": None, "sparsity": None, "prior": None}
        if self.baseline:
            out["vis"] = self.region(self.sg.embed(batch["feats"]), mask)
            return out
        sg = self.sg(batch["feats"], batch["boxes"], mask)
        nodes = sg["nodes"]
        kg_h = kg_bias = None
        if self.use_kg:
            emb, prior, _ = self.kg_ret(nodes, mask)                         # [B,N,K,D], [B,N,K]
            prior = prior * mask[:, :, None].float()                        # padded objects carry no concepts
            B, N, K, _ = emb.shape
            pn = prior / prior.flatten(1).amax(dim=1).clamp_min(1e-6)[:, None, None]
            kg_h = self.kg_ada(emb) + self.gate_proj(prior.unsqueeze(-1))     # [B,N,K,H]
            kb = torch.log(pn.clamp_min(1e-6)).masked_fill(~mask[:, :, None], NEG)
            kg_bias = kb.reshape(B, N * K)
            out["prior"] = pn
        vis, kg, sparsity = self.enc(nodes, mask, sg["edge_w"], sg["pair_h"], kg_h, kg_bias)
        out.update(vis=vis, kg=kg, kg_bias=kg_bias, sparsity=sparsity)
        return out

    def forward(self, batch):
        """Returns (total_loss, dict of detached scalar tensors)."""
        e = self.encode(batch)
        tok = batch["tokens"]
        logits, gates = self.decoder(tok[:, :-1], e["vis"], e["kg"], e["vis_bias"], e["kg_bias"])
        cap = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tok[:, 1:].reshape(-1),
                              ignore_index=0, label_smoothing=self.ls)
        total = cap
        aux = {"caption": cap.detach()}
        if e["sparsity"] is not None:
            total = total + self.lw_sp * e["sparsity"]
            aux["sparsity"] = e["sparsity"].detach()
        if e["prior"] is not None:
            m = e["mask"][:, :, None].float()
            pn = e["prior"]
            if self.lw_gate > 0:
                gl1 = (pn * m).sum() / (m.sum() * pn.shape[2]).clamp_min(1.0)
                total = total + self.lw_gate * gl1
            if self.lw_cons > 0 and e["kg"] is not None:
                B, N, K = pn.shape
                w = pn * m
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
                kg_obj = (e["kg"].reshape(B, N, K, -1) * w.unsqueeze(-1)).sum(2)
                cos = F.cosine_similarity(e["vis"].float(), kg_obj.float(), dim=-1)
                cons = ((1 - cos) * e["mask"].float()).sum() / e["mask"].float().sum().clamp_min(1.0)
                total = total + self.lw_cons * cons
                aux["consistency"] = cons.detach()
            aux["concepts_kept"] = ((pn > 0).float() * m).sum() / (m.sum() * pn.shape[2]).clamp_min(1.0)
        if gates:
            aux["cap_gate_mean"] = torch.stack([g.mean() for g in gates]).mean()
        return total, aux

    @torch.no_grad()
    def generate(self, batch, beam=3, max_len=20, length_penalty=0.7):
        e = self.encode(batch)
        return self.decoder.beam_search(e["vis"], e["kg"], e["vis_bias"], e["kg_bias"], beam, max_len, length_penalty)
