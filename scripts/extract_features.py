#!/usr/bin/env python3
"""Extract frozen Faster R-CNN (torchvision ResNet-50-FPN v2) region features ONCE for every split.
Writes memory-mapped files under data.feature_root (resumable: re-run to finish an interrupted split).
Per image: up to features.num_objects boxes (class-agnostic NMS 0.5, score>--score-thresh, at least
--min-objects kept), 1024-d box-head features, normalised boxes. Images with no detection get a whole-image box.
Disk (fp16): COCO ~9 GB, Flickr30k ~2.3 GB. Training then never touches images or the detector."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import nms
from tqdm import tqdm

from ddkg.config import load_config
from ddkg.data.karpathy import load_karpathy
from ddkg.data.features import create_store


class Imgs(Dataset):
    def __init__(self, paths):
        self.p = paths
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return TF.to_tensor(Image.open(self.p[i]).convert("RGB")), i


def collate(b):
    return [x[0] for x in b], [x[1] for x in b]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-size", type=int, default=600)
    ap.add_argument("--max-size", type=int, default=1000)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--min-objects", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    N, D = cfg.features.num_objects, cfg.features.feat_dim
    recs = load_karpathy(cfg.data.karpathy_json, cfg.data.dataset, cfg.data.image_root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                     box_score_thresh=0.05, box_detections_per_img=100).eval().to(dev)
    det.transform.min_size, det.transform.max_size = (args.min_size,), args.max_size

    for split in args.splits:
        rs = recs[split]
        feats, boxes, nvalid = create_store(cfg.data.feature_root, split, [r["id"] for r in rs], N, D)
        todo = [i for i in range(len(rs)) if int(nvalid[i]) == 0]
        print(f"[{split}] {len(rs)} images, {len(todo)} to extract")
        dl = DataLoader(Imgs([rs[i]["path"] for i in todo]), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
        for imgs, js in tqdm(dl, desc=split):
            images, _ = det.transform([im.to(dev) for im in imgs])
            fm = det.backbone(images.tensors)
            props, _ = det.rpn(images, fm)
            dets, _ = det.roi_heads(fm, props, images.image_sizes)
            sel = []
            for d, (h, w) in zip(dets, images.image_sizes):
                b, s = d["boxes"], d["scores"]
                if b.numel() > 0:
                    keep = nms(b, s, 0.5)
                    b, s = b[keep], s[keep]
                    k = int((s > args.score_thresh).sum())
                    k = min(N, max(k, min(args.min_objects, b.shape[0])))
                    b = b[:k]
                if b.shape[0] == 0:
                    b = torch.tensor([[0.0, 0.0, float(w), float(h)]], device=dev)
                sel.append(b)
            bf = det.roi_heads.box_head(det.roi_heads.box_roi_pool(fm, sel, images.image_sizes))   # [sumN,1024]
            off = 0
            for b, j, (h, w) in zip(sel, js, images.image_sizes):
                n = b.shape[0]
                row = todo[j]
                feats[row, :n] = bf[off:off + n].half().cpu().numpy()
                feats[row, n:] = 0
                nb = (b / torch.tensor([w, h, w, h], device=dev, dtype=b.dtype)).clamp(0, 1)
                boxes[row, :n] = nb.half().cpu().numpy()
                boxes[row, n:] = 0
                nvalid[row] = n
                off += n
        feats.flush(); boxes.flush(); nvalid.flush()
    print("done")


if __name__ == "__main__":
    main()
