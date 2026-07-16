#!/usr/bin/env python
"""Train the sign class-verifier: fine-tune ConvNeXt-Tiny on the REAL single-sign crops
(21 subset classes). Reports accuracy on a held-out REAL val split (sanity: must be high)
and saves weights, so verify_signgen.py can measure the valid-label rate of GENERATED signs.

Usage:
  conda activate longtail-synth
  python scripts/detection/train_verifier.py --device 0 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.verifier import build_convnext, build_crop_index, imagenet_transform, load_crop  # noqa: E402


def stratified_split(index, val_frac, seed):
    """Per-class 80/20 split; classes with <2 crops go entirely to train (can't eval)."""
    by_cls = defaultdict(list)
    for rec in index:
        by_cls[rec[1]].append(rec)
    rng = random.Random(seed)
    train, val, val_starved = [], [], []
    for cid, recs in by_cls.items():
        rng.shuffle(recs)
        k = int(round(len(recs) * val_frac))
        if len(recs) < 2:
            train += recs
            val_starved.append(cid)
        else:
            val += recs[:max(1, k)]
            train += recs[max(1, k):]
    return train, val, val_starved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--out", default="data/tt100k/verifier")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    tiles, prepared = Path(args.tiles), Path(args.prepared)
    sub = json.loads((prepared / "subset.json").read_text())
    class_ids = sorted(c["id"] for c in sub["classes"])         # model index -> class_id
    id2name = {c["id"]: c["name"] for c in sub["classes"]}
    cid2idx = {c: i for i, c in enumerate(class_ids)}

    index = build_crop_index(tiles / "train" / "labels")
    train_recs, val_recs, starved = stratified_split(index, args.val_frac, args.seed)
    print(f"[verif] {len(index)} crops, {len(class_ids)} classes | train={len(train_recs)} "
          f"val={len(val_recs)} | sem-val (starved): {[id2name[c] for c in starved]}")

    class CropDS(Dataset):
        def __init__(self, recs, train):
            self.recs, self.tf = recs, imagenet_transform(train=train)

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            stem, cid, box = self.recs[i]
            crop = load_crop(tiles / "train" / "images", stem, box)
            return self.tf(Image.fromarray(crop)), cid2idx[cid]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_convnext(len(class_ids), pretrained=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss()
    tl = DataLoader(CropDS(train_recs, True), batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    vl = DataLoader(CropDS(val_recs, False), batch_size=args.batch, num_workers=args.num_workers) if val_recs else None

    for ep in range(args.epochs):
        net.train()
        tot = correct = 0
        run = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = net(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
        print(f"  ép {ep + 1}/{args.epochs}  loss={run / tot:.4f}  train_acc={correct / tot:.4f}")

    # ---- eval on REAL val (sanity) + confusion ----
    val_acc, conf_mat, per_cls = None, None, {}
    if vl:
        net.eval()
        C = len(class_ids)
        conf_mat = np.zeros((C, C), int)
        with torch.no_grad():
            for x, y in vl:
                p = net(x.to(dev)).argmax(1).cpu().numpy()
                for yt, yp in zip(y.numpy(), p):
                    conf_mat[yt, yp] += 1
        val_acc = float(np.trace(conf_mat) / conf_mat.sum())
        for i, cid in enumerate(class_ids):
            n = conf_mat[i].sum()
            per_cls[id2name[cid]] = round(float(conf_mat[i, i] / n), 4) if n else None
        print(f"[verif] val REAL top-1 acc = {val_acc:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "class_ids": class_ids},
               out / "convnext_signcls.pt")
    report = {"n_crops": len(index), "n_train": len(train_recs), "n_val": len(val_recs),
              "classes": [id2name[c] for c in class_ids], "val_starved": [id2name[c] for c in starved],
              "val_real_top1_acc": val_acc, "per_class_val_acc": per_cls,
              "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
              "caveat": ("treinado em crops REAIS (pequenos/borrados); ao medir amostras GERADAS "
                         "(limpas), parte da rejeição pode ser DOMAIN-GAP e não classe errada. "
                         "top1_acc (argmax entre classes) é robusto a isso; mean_conf é sensível. "
                         "Ver REQ2 (sim-to-real).")}
    (out / "verifier_report.json").write_text(json.dumps(report, indent=2))
    print(f"-> {out / 'convnext_signcls.pt'}  +  verifier_report.json")


if __name__ == "__main__":
    main()
