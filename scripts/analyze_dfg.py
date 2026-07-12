#!/usr/bin/env python3
"""Análise exploratória do DFG traffic-sign dataset (Tabernik & Skočaj, 2020).

Lê as anotações COCO do DFG (`train.json` + `test.json`, formato COCO padrão com
`images`/`annotations`/`categories`) e produz o mesmo relatório do `analyze_tt100k.py`,
acrescido de uma seção de **viabilidade de subset** (espelho do critério do P3:
`min_instances=80` + `>=10` instâncias no split de teste) — a resposta direta a
"o DFG tem tudo que precisamos para compor um subset head/mid/tail equivalente ao do TT100K?".

Anotações `ignore=true` são excluídas (o DFG marca como difícil bbox < 30 px).

Uso:
  python scripts/analyze_dfg.py [--root data/dfg]
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMALL, LARGE = 32 * 32, 96 * 96  # bins COCO (área px²)


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data" / "dfg")
    args = ap.parse_args()

    tr_p, te_p = args.root / "train.json", args.root / "test.json"
    if not tr_p.exists() or not te_p.exists():
        sys.exit(f"[erro] esperava train.json e test.json em {args.root} (baixe via go.vicos.si/dfgannot)")
    tr, te = load_coco(tr_p), load_coco(te_p)
    id2name = {c["id"]: c["name"] for c in tr["categories"]}
    ncat_declared = len(tr["categories"])

    # contagens por split (exclui ignore), imagens por categoria, tamanhos
    inst = Counter()                 # instâncias globais por categoria (nome)
    inst_split = {"train": Counter(), "test": Counter()}
    img_per_cat = defaultdict(set)
    per_split_imgs = {"train": len(tr["images"]), "test": len(te["images"])}
    per_split_inst = Counter()
    empty_imgs = Counter()
    signs_per_img = []
    areas = []                       # (name, area_px)
    res = Counter()                  # resolução das imagens (w x h)

    for split, d in (("train", tr), ("test", te)):
        anns_by_img = defaultdict(list)
        for a in d["annotations"]:
            if a.get("ignore", False):
                continue
            anns_by_img[a["image_id"]].append(a)
        for im in d["images"]:
            res[(im["width"], im["height"])] += 1
            objs = anns_by_img.get(im["id"], [])
            signs_per_img.append(len(objs))
            if not objs:
                empty_imgs[split] += 1
            for a in objs:
                name = id2name[a["category_id"]]
                inst[name] += 1
                inst_split[split][name] += 1
                per_split_inst[split] += 1
                img_per_cat[name].add((split, a["image_id"]))
                w, h = a["bbox"][2], a["bbox"][3]
                areas.append((name, w * h))

    total_inst = sum(inst.values())
    total_imgs = sum(per_split_imgs.values())
    ncat = len(inst)
    rows = sorted(inst.items(), key=lambda kv: (-kv[1], kv[0]))

    ge100 = [c for c in inst if inst[c] >= 100]
    ge80 = [c for c in inst if inst[c] >= 80]
    ge50 = [c for c in inst if inst[c] >= 50]
    lt10 = [c for c in inst if inst[c] < 10]

    def size_bin(a):
        return "small" if a < SMALL else ("large" if a > LARGE else "medium")
    size_counter = Counter(size_bin(a) for _, a in areas)

    # --- viabilidade de subset (espelho do P3) ---
    MIN_INST, MIN_TEST = 80, 10
    elig = [c for c in inst if inst[c] >= MIN_INST]
    elig_test = [c for c in elig if inst_split["test"][c] >= MIN_TEST]

    # ---- CSV ----
    (REPO / "reports").mkdir(exist_ok=True)
    csv_path = REPO / "reports" / "dfg_class_counts.csv"
    with open(csv_path, "w") as f:
        f.write("category,instances,instances_train,instances_test,images,pct_instances,cum_pct\n")
        cum = 0
        for cat, c in rows:
            cum += c
            f.write(f"{cat},{c},{inst_split['train'][cat]},{inst_split['test'][cat]},"
                    f"{len(img_per_cat[cat])},{100*c/total_inst:.3f},{100*cum/total_inst:.3f}\n")
    print(f"[ok] {csv_path}")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        adir = REPO / "analysis"; adir.mkdir(exist_ok=True)
        counts = [c for _, c in rows]
        plt.figure(figsize=(11, 4))
        plt.bar(range(len(counts)), counts, width=1.0)
        plt.yscale("log"); plt.xlabel("classe (rank por frequência)")
        plt.ylabel("nº instâncias (log)")
        plt.axhline(100, color="r", ls="--", lw=1, label="limiar 100 inst.")
        plt.axhline(20, color="orange", ls=":", lw=1, label="piso DFG 20 inst.")
        plt.title(f"DFG — distribuição long-tail de classes (n={ncat})")
        plt.legend(); plt.tight_layout()
        plt.savefig(adir / "dfg_class_distribution.png", dpi=130); plt.close()

        sides = [math.sqrt(a) for _, a in areas]
        plt.figure(figsize=(8, 4))
        plt.hist([s for s in sides if s <= 400], bins=60)
        plt.axvline(32, color="r", ls="--", lw=1, label="32 px (small)")
        plt.axvline(96, color="g", ls="--", lw=1, label="96 px (large)")
        plt.xlabel("lado da placa = √área (px)"); plt.ylabel("nº instâncias")
        plt.title("DFG — distribuição de tamanho das placas"); plt.legend()
        plt.tight_layout(); plt.savefig(adir / "dfg_bbox_size.png", dpi=130); plt.close()
        print(f"[ok] plots em {adir}")
    except Exception as e:
        print(f"[warn] plots falharam: {e}")

    # ---- Relatório MD ----
    md = REPO / "reports" / "dfg_analysis.md"
    with_signs = sum(1 for n in signs_per_img if n > 0)
    top20, bottom20 = rows[:20], rows[-20:]
    med = sorted(inst.values())[ncat // 2]
    top1 = rows[0][1]
    with open(md, "w") as f:
        w = f.write
        w("# DFG traffic-sign dataset — análise exploratória\n\n")
        w("Fonte: `data/dfg/{train,test}.json` (COCO; Tabernik & Skočaj 2020). "
          "Anotações `ignore=true` (bbox < 30 px, marcadas difíceis) excluídas.\n\n")
        w("## Visão geral\n\n")
        w(f"- Imagens totais: **{total_imgs}** (train {per_split_imgs['train']} / test {per_split_imgs['test']})\n")
        w(f"- Imagens com ≥1 placa: **{with_signs}** | sem placa: **{total_imgs-with_signs}**\n")
        w(f"- Instâncias de placas: **{total_inst}**\n")
        w(f"- Categorias observadas: **{ncat}** / {ncat_declared} declaradas\n")
        w(f"- Placas por imagem (só imgs com placa): média **{total_inst/max(with_signs,1):.2f}**, "
          f"máx **{max(signs_per_img)}**\n")
        w(f"- Resolução: {', '.join(f'{ww}×{hh} ({n})' for (ww,hh),n in res.most_common(3))}\n\n")

        w("## Splits\n\n| split | imagens | imgs sem placa | instâncias |\n|---|--:|--:|--:|\n")
        for sp in ("train", "test"):
            w(f"| {sp} | {per_split_imgs[sp]} | {empty_imgs[sp]} | {per_split_inst[sp]} |\n")
        w("\n")

        w("## Cauda longa\n\n")
        w(f"- Classes com **≥100** instâncias: **{len(ge100)}** | ≥80: {len(ge80)} | ≥50: {len(ge50)} "
          f"| <10: **{len(lt10)}**\n")
        w(f"- Razão head/tail: classe mais comum = **{top1}** inst. | mediana = **{med}** inst. "
          f"| razão ≈ **{top1/max(med,1):.0f}×**\n")
        w(f"- Piso curado: DFG garante ≥20 instâncias/classe (por construção) → cauda **truncada**, "
          f"não natural como no TT100K.\n")
        share = sum(c for _, c in rows[:10])
        w(f"- Top-10 classes concentram **{100*share/total_inst:.1f}%** das instâncias\n\n")

        w("## Tamanho das placas (bins COCO, área px²)\n\n| bin | definição | instâncias | % |\n|---|---|--:|--:|\n")
        for b, dfn in [("small", "< 32²=1024"), ("medium", "32²–96²"), ("large", "> 96²=9216")]:
            c = size_counter.get(b, 0)
            w(f"| {b} | {dfn} | {c} | {100*c/max(len(areas),1):.1f}% |\n")
        sides = sorted(math.sqrt(a) for _, a in areas)
        w(f"\nLado da bbox (√área): mediana **{sides[len(sides)//2]:.0f} px** "
          f"(p10 {sides[len(sides)//10]:.0f}, p90 {sides[9*len(sides)//10]:.0f}). "
          f"Imagens 1920×1080 → placas **grandes** relativas ao TT100K (2048², placas minúsculas).\n\n")

        w("## Viabilidade de subset (espelho do critério P3)\n\n")
        w(f"Critério: `min_instances={MIN_INST}` (piso) **e** `≥{MIN_TEST}` instâncias no split de teste "
          "(restrição dura para estratificar head/mid/tail e ter suporte de avaliação na cauda).\n\n")
        w(f"- Classes elegíveis (≥{MIN_INST} inst. globais): **{len(elig)}**\n")
        w(f"- Dessas, com ≥{MIN_TEST} no teste: **{len(elig_test)}** "
          f"→ {'✅ sobra folga' if len(elig_test) >= 20 else '⚠️ apertado'} para um subset de ~20 classes\n\n")

        w("## Top-20 classes (mais frequentes)\n\n| categoria | instâncias | train | test | imagens |\n|---|--:|--:|--:|--:|\n")
        for cat, c in top20:
            w(f"| `{cat}` | {c} | {inst_split['train'][cat]} | {inst_split['test'][cat]} | {len(img_per_cat[cat])} |\n")
        w("\n## Bottom-20 classes (mais raras)\n\n| categoria | instâncias | train | test | imagens |\n|---|--:|--:|--:|--:|\n")
        for cat, c in bottom20:
            w(f"| `{cat}` | {c} | {inst_split['train'][cat]} | {inst_split['test'][cat]} | {len(img_per_cat[cat])} |\n")
        w("\n_Ver `reports/dfg_class_counts.csv` (contagem completa) e `analysis/dfg_*.png`._\n")
    print(f"[ok] {md}")
    print(f"\n[resumo] {total_imgs} imgs | {total_inst} inst | {ncat} classes | "
          f"≥100={len(ge100)} | small={100*size_counter.get('small',0)/max(len(areas),1):.0f}% | "
          f"elegíveis(≥80,≥10test)={len(elig_test)}")


if __name__ == "__main__":
    main()
