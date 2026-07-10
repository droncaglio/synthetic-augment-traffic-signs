#!/usr/bin/env python3
"""Análise exploratória do TT100K (classes, raridade, tamanhos, densidade).

Lê o(s) annotations json do TT100K (formato {"types": [...], "imgs": {id: {path, objects:[{category,bbox}]}}})
e produz:
  - reports/tt100k_analysis.md   (relatório legível)
  - reports/tt100k_class_counts.csv
  - analysis/*.png               (distribuição de classes, tamanhos, densidade)

Uso:
  python scripts/analyze_tt100k.py [--ann CAMINHO/annotations.json] [--root data/tt100k]
Se --ann não for dado, procura recursivamente por um json com chave "imgs" sob --root.
"""
from __future__ import annotations
import argparse, json, sys, math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Bins de tamanho estilo COCO (área em px²): small < 32², medium 32²–96², large > 96²
SMALL, LARGE = 32 * 32, 96 * 96

# Famílias semânticas TT100K por prefixo do código de categoria
FAMILY = {
    "p": "prohibitory (proibição/regulamentação)",
    "i": "indication/mandatory (indicação, azul)",
    "w": "warning (advertência, triangular)",
}


def find_ann(root: Path) -> Path | None:
    cands = sorted(root.rglob("*.json"), key=lambda p: (len(p.parts), p.name))
    for p in cands:
        try:
            with open(p) as f:
                head = f.read(4096)
            if '"imgs"' in head or '"objects"' in head:
                return p
        except Exception:
            continue
    return cands[0] if cands else None


def bbox_wh(obj) -> tuple[float, float]:
    b = obj.get("bbox") or {}
    if {"xmin", "ymin", "xmax", "ymax"} <= set(b):
        return b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]
    if {"x", "y", "w", "h"} <= set(b):
        return b["w"], b["h"]
    return 0.0, 0.0


def prefix(cat: str) -> str:
    for i, ch in enumerate(cat):
        if ch.isdigit():
            return cat[:i] or cat
    return cat


def family(cat: str) -> str:
    return FAMILY.get(cat[:1], "other/misc")


def split_of(path: str) -> str:
    p = path.replace("\\", "/").lower()
    for s in ("train", "test", "val", "other"):
        if f"/{s}/" in f"/{p}" or p.startswith(s + "/"):
            return s
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=REPO / "data" / "tt100k")
    args = ap.parse_args()

    ann_path = args.ann or find_ann(args.root)
    if not ann_path or not ann_path.exists():
        sys.exit(f"[erro] annotations json não encontrado (root={args.root}). Passe --ann.")
    print(f"[info] lendo {ann_path}")
    with open(ann_path) as f:
        data = json.load(f)

    imgs = data.get("imgs", data)  # tolera formato achatado
    types = data.get("types")

    inst = Counter()             # instâncias por categoria
    img_per_cat = defaultdict(set)
    per_split_imgs = Counter()
    per_split_inst = Counter()
    signs_per_img = []
    empty_imgs = Counter()       # imagens sem placa (fundo puro) por split
    areas = []                   # (cat, area_px, w, h)

    for img_id, im in imgs.items():
        objs = im.get("objects", []) or []
        sp = split_of(im.get("path", ""))
        per_split_imgs[sp] += 1
        n = 0
        for o in objs:
            cat = o.get("category") or o.get("label") or "?"
            w, h = bbox_wh(o)
            if w <= 0 or h <= 0:
                continue
            inst[cat] += 1
            img_per_cat[cat].add(img_id)
            per_split_inst[sp] += 1
            areas.append((cat, w * h, w, h))
            n += 1
        signs_per_img.append(n)
        if n == 0:
            empty_imgs[sp] += 1

    total_inst = sum(inst.values())
    total_imgs = len(imgs)
    ncat = len(inst)

    # famílias e prefixos
    fam_inst = Counter()
    pref_inst = Counter()
    for cat, c in inst.items():
        fam_inst[family(cat)] += c
        pref_inst[prefix(cat)] += c

    # thresholds long-tail
    ge100 = sorted([c for c in inst if inst[c] >= 100])
    ge50 = [c for c in inst if inst[c] >= 50]
    lt10 = [c for c in inst if inst[c] < 10]
    lt5 = [c for c in inst if inst[c] < 5]

    # tamanhos
    def size_bin(a):
        return "small" if a < SMALL else ("large" if a > LARGE else "medium")
    size_counter = Counter(size_bin(a) for _, a, _, _ in areas)

    # ---- CSV ----
    (REPO / "reports").mkdir(exist_ok=True)
    csv_path = REPO / "reports" / "tt100k_class_counts.csv"
    rows = sorted(inst.items(), key=lambda kv: -kv[1])
    with open(csv_path, "w") as f:
        f.write("category,family,prefix,instances,images,pct_instances,cum_pct\n")
        cum = 0
        for cat, c in rows:
            cum += c
            f.write(f"{cat},{family(cat)},{prefix(cat)},{c},{len(img_per_cat[cat])},"
                    f"{100*c/total_inst:.3f},{100*cum/total_inst:.3f}\n")
    print(f"[ok] {csv_path}")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        adir = REPO / "analysis"; adir.mkdir(exist_ok=True)

        # 1. Distribuição de instâncias por classe (rank x count, log-y)
        counts = [c for _, c in rows]
        plt.figure(figsize=(11, 4))
        plt.bar(range(len(counts)), counts, width=1.0)
        plt.yscale("log"); plt.xlabel("classe (rank por frequência)")
        plt.ylabel("nº instâncias (log)")
        plt.axhline(100, color="r", ls="--", lw=1, label="limiar 100 inst.")
        plt.title(f"TT100K — distribuição long-tail de classes (n={ncat})")
        plt.legend(); plt.tight_layout()
        plt.savefig(adir / "class_distribution.png", dpi=130); plt.close()

        # 2. Histograma de tamanho de bbox (lado = sqrt(area))
        sides = [math.sqrt(a) for _, a, _, _ in areas]
        plt.figure(figsize=(8, 4))
        plt.hist([s for s in sides if s <= 300], bins=60)
        plt.axvline(32, color="r", ls="--", lw=1, label="32 px (small)")
        plt.axvline(96, color="g", ls="--", lw=1, label="96 px (large)")
        plt.xlabel("lado da placa = √área (px)"); plt.ylabel("nº instâncias")
        plt.title("TT100K — distribuição de tamanho das placas"); plt.legend()
        plt.tight_layout(); plt.savefig(adir / "bbox_size.png", dpi=130); plt.close()

        # 3. Placas por imagem (densidade)
        plt.figure(figsize=(8, 4))
        mx = max(signs_per_img) if signs_per_img else 0
        plt.hist(signs_per_img, bins=range(0, min(mx, 30) + 2))
        plt.xlabel("placas por imagem"); plt.ylabel("nº imagens")
        plt.title("TT100K — densidade de placas por imagem")
        plt.tight_layout(); plt.savefig(adir / "signs_per_image.png", dpi=130); plt.close()
        print(f"[ok] plots em {adir}")
    except Exception as e:
        print(f"[warn] plots falharam: {e}")

    # ---- Relatório MD ----
    md = REPO / "reports" / "tt100k_analysis.md"
    with_signs = sum(1 for n in signs_per_img if n > 0)
    top20 = rows[:20]
    bottom20 = rows[-20:]
    with open(md, "w") as f:
        w = f.write
        w("# TT100K — análise exploratória\n\n")
        w(f"Fonte de anotação: `{ann_path.relative_to(REPO) if ann_path.is_relative_to(REPO) else ann_path}`\n\n")
        w("## Visão geral\n\n")
        w(f"- Imagens totais: **{total_imgs}**\n")
        w(f"- Imagens com ≥1 placa: **{with_signs}** | sem placa (fundo): **{total_imgs-with_signs}**\n")
        w(f"- Instâncias de placas: **{total_inst}**\n")
        w(f"- Categorias observadas (com ≥1 instância): **{ncat}**"
          + (f" | declaradas em `types`: {len(types)}\n" if types else "\n"))
        w(f"- Placas por imagem (só imgs com placa): média **{total_inst/max(with_signs,1):.2f}**, "
          f"máx **{max(signs_per_img) if signs_per_img else 0}**\n\n")

        w("## Splits\n\n| split | imagens | imgs sem placa | instâncias |\n|---|--:|--:|--:|\n")
        for sp in sorted(per_split_imgs):
            w(f"| {sp} | {per_split_imgs[sp]} | {empty_imgs[sp]} | {per_split_inst[sp]} |\n")
        w("\n")

        w("## Cauda longa\n\n")
        w(f"- Classes com **≥100** instâncias: **{len(ge100)}** (subset \"treinável\" clássico)\n")
        w(f"- Classes com ≥50: {len(ge50)} | com <10: **{len(lt10)}** | com <5: **{len(lt5)}**\n")
        top1 = rows[0][1] if rows else 0
        med = sorted(inst.values())[ncat // 2] if ncat else 0
        w(f"- Razão head/tail: classe mais comum = **{top1}** inst. | mediana = **{med}** inst. "
          f"| razão ≈ **{top1/max(med,1):.0f}×**\n")
        share = sum(c for _, c in rows[:10])
        w(f"- Top-10 classes concentram **{100*share/total_inst:.1f}%** de todas as instâncias\n\n")

        w("## Famílias semânticas (por prefixo do código)\n\n| família | instâncias | % |\n|---|--:|--:|\n")
        for fam, c in fam_inst.most_common():
            w(f"| {fam} | {c} | {100*c/total_inst:.1f}% |\n")
        w("\n")

        w("## Tamanho das placas (bins COCO, área px²)\n\n| bin | definição | instâncias | % |\n|---|---|--:|--:|\n")
        for b, dfn in [("small", "< 32²=1024"), ("medium", "32²–96²"), ("large", "> 96²=9216")]:
            c = size_counter.get(b, 0)
            w(f"| {b} | {dfn} | {c} | {100*c/max(len(areas),1):.1f}% |\n")
        w("\n")

        w("## Top-20 classes (mais frequentes)\n\n| categoria | família | instâncias | imagens |\n|---|---|--:|--:|\n")
        for cat, c in top20:
            w(f"| `{cat}` | {family(cat)} | {c} | {len(img_per_cat[cat])} |\n")
        w("\n## Bottom-20 classes (mais raras)\n\n| categoria | família | instâncias | imagens |\n|---|---|--:|--:|\n")
        for cat, c in bottom20:
            w(f"| `{cat}` | {family(cat)} | {c} | {len(img_per_cat[cat])} |\n")
        w("\n_Ver `reports/tt100k_class_counts.csv` (contagem completa) e `analysis/*.png`._\n")
    print(f"[ok] {md}")
    print(f"\n[resumo] {total_imgs} imgs | {total_inst} inst | {ncat} classes | "
          f"≥100={len(ge100)} | small={100*size_counter.get('small',0)/max(len(areas),1):.0f}%")


if __name__ == "__main__":
    main()
