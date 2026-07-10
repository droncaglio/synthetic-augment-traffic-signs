#!/usr/bin/env python3
"""Análise de raridade/diversidade de FUNDOS (imagens nosign do TT100K).

Embeddings de cena (CLIP ViT-B/32 via transformers) + KMeans → mede diversidade
de fundos e identifica tipos de cena raros (clusters pequenos). Serve à ideia de
copy-paste: escolher fundos (inclusive raros) para colar placas reais/sintéticas.

Saídas:
  - reports/tt100k_backgrounds.md            (tamanhos de cluster, raridade, guia)
  - reports/tt100k_background_clusters.csv   (imagem -> cluster)
  - analysis/bg_cluster_sizes.png            (distribuição de tamanho de cluster)
  - analysis/bg_tsne.png                     (mapa 2D t-SNE colorido por cluster)
  - analysis/bg_montages/cluster_XX.png      (mosaico de amostras por cluster)

Uso:
  python scripts/analyze_backgrounds.py [--dir data/tt100k/nosign_1] [--k 20] [--sample 6000]
"""
from __future__ import annotations
import argparse, math, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def list_images(d: Path, sample: int, seed: int = 42):
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in d.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    if sample and len(files) > sample:
        rnd = random.Random(seed)
        files = rnd.sample(files, sample)
    return files


def embed(files, batch=64, device=None):
    """Features de cena via ConvNeXt-Tiny (ImageNet, torchvision) — robusto e rápido."""
    import torch, numpy as np
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] carregando ConvNeXt-Tiny (ImageNet) em {device}")
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    net = convnext_tiny(weights=weights).to(device).eval()
    feat_net = torch.nn.Sequential(net.features, net.avgpool, torch.nn.Flatten())  # -> 768d
    tf = T.Compose([
        T.Resize(232), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats, ok = [], []
    with torch.no_grad():
        for i in range(0, len(files), batch):
            chunk = files[i:i + batch]
            tens = []
            for p in chunk:
                try:
                    tens.append(tf(Image.open(p).convert("RGB")))
                    ok.append(p)
                except Exception:
                    pass
            if not tens:
                continue
            x = torch.stack(tens).to(device)
            f = feat_net(x)
            f = torch.nn.functional.normalize(f, dim=-1)
            feats.append(f.cpu().numpy())
            print(f"\r[embed] {min(i+batch,len(files))}/{len(files)}", end="", flush=True)
    print()
    return np.concatenate(feats, 0), ok


def montage(paths, out, cols=3, thumb=256):
    from PIL import Image
    paths = paths[:cols * cols]
    if not paths:
        return
    rows = math.ceil(len(paths) / cols)
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), (20, 20, 20))
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb))
            canvas.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
        except Exception:
            pass
    canvas.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO / "data" / "tt100k" / "nosign_1")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sample", type=int, default=6000)
    args = ap.parse_args()

    import numpy as np
    if not args.dir.exists():
        raise SystemExit(f"[erro] dir de fundos não existe: {args.dir}")
    files = list_images(args.dir, args.sample)
    print(f"[info] {len(files)} imagens amostradas de {args.dir}")

    X, paths = embed(files)
    print(f"[info] embeddings: {X.shape}")

    from sklearn.cluster import KMeans
    k = min(args.k, len(paths))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels = km.labels_

    # distância ao centróide → "outlier score" (fundos atípicos)
    d = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)

    from collections import Counter
    sizes = Counter(labels.tolist())

    adir = REPO / "analysis"; (adir / "bg_montages").mkdir(parents=True, exist_ok=True)
    (REPO / "reports").mkdir(exist_ok=True)

    # CSV imagem -> cluster
    with open(REPO / "reports" / "tt100k_background_clusters.csv", "w") as f:
        f.write("image,cluster,dist_to_centroid\n")
        for p, l, dd in zip(paths, labels, d):
            f.write(f"{p.name},{l},{dd:.4f}\n")

    # montagens por cluster (amostras mais próximas do centróide = representativas)
    order = np.argsort(d)
    per_cluster = {c: [] for c in range(k)}
    for idx in order:
        c = labels[idx]
        if len(per_cluster[c]) < 9:
            per_cluster[c].append(paths[idx])
    for c in range(k):
        montage(per_cluster[c], adir / "bg_montages" / f"cluster_{c:02d}.png")

    # plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # tamanhos de cluster
        cs = sorted(sizes.items(), key=lambda kv: -kv[1])
        plt.figure(figsize=(9, 4))
        plt.bar([str(c) for c, _ in cs], [n for _, n in cs])
        plt.xlabel("cluster"); plt.ylabel("nº imagens de fundo")
        plt.title(f"TT100K nosign — diversidade de fundos (KMeans k={k}, n={len(paths)})")
        plt.tight_layout(); plt.savefig(adir / "bg_cluster_sizes.png", dpi=130); plt.close()
        # t-SNE 2D
        from sklearn.manifold import TSNE
        n_ts = min(2500, len(paths))
        sel = np.random.RandomState(42).choice(len(paths), n_ts, replace=False)
        emb2 = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(X[sel])
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=labels[sel], cmap="tab20", s=6)
        plt.title("TT100K nosign — mapa t-SNE de fundos (cor=cluster)")
        plt.tight_layout(); plt.savefig(adir / "bg_tsne.png", dpi=130); plt.close()
        print("[ok] plots de fundo salvos")
    except Exception as e:
        print(f"[warn] plots falharam: {e}")

    # relatório
    md = REPO / "reports" / "tt100k_backgrounds.md"
    cs = sorted(sizes.items(), key=lambda kv: kv[1])  # menor -> maior (raros primeiro)
    with open(md, "w") as f:
        w = f.write
        w("# TT100K — análise de fundos (nosign)\n\n")
        w(f"Fonte: `{args.dir.relative_to(REPO) if args.dir.is_relative_to(REPO) else args.dir}` · "
          f"amostra **{len(paths)}** imagens · embedding **ConvNeXt-Tiny (ImageNet, 768d)** · **KMeans k={k}**.\n\n")
        w("Raridade de fundo = clusters pequenos (tipos de cena sub-representados) e imagens com "
          "alta distância ao centróide (fundos atípicos). Úteis para copy-paste: colar placas — "
          "sobretudo classes raras — em fundos diversos e incomuns aumenta a variabilidade de contexto.\n\n")
        w("## Tamanho dos clusters (raros primeiro)\n\n| cluster | imagens | % | montagem |\n|--:|--:|--:|---|\n")
        for c, n in cs:
            w(f"| {c} | {n} | {100*n/len(paths):.1f}% | `analysis/bg_montages/cluster_{c:02d}.png` |\n")
        w("\n## Fundos mais atípicos (maior distância ao centróide)\n\n| imagem | cluster | dist |\n|---|--:|--:|\n")
        far = np.argsort(-d)[:15]
        for idx in far:
            w(f"| {paths[idx].name} | {labels[idx]} | {d[idx]:.3f} |\n")
        w("\n_Veja `analysis/bg_tsne.png`, `analysis/bg_cluster_sizes.png` e as montagens por cluster._\n")
    print(f"[ok] {md}")
    print(f"[resumo] {len(paths)} fundos | k={k} | menor cluster={cs[0][1]} | maior={cs[-1][1]}")


if __name__ == "__main__":
    main()
