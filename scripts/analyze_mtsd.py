#!/usr/bin/env python3
"""Análise exploratória do MTSD (Mapillary Traffic Sign Dataset, Ertler et al. 2020).

Lê as anotações COCO do conjunto *fully-annotated* (`train_coco.json` + `val_coco.json`)
e produz o mesmo relatório do `analyze_dfg.py`, com a seção de **viabilidade de subset**
(critério do P3: `min_instances=80` + `>=10` no split de avaliação — aqui o `val`, pois o
`test` do MTSD tem rótulos retidos).

Origem das anotações: mirror Kaggle `zeuss2k3/mapillary-traffic-sign-dataset`, pasta
"annotations for mtsd in coco format" (extraída via range, sem baixar as imagens de 35 GB).

Ressalvas do mirror (declarar no paper, não usar como fonte canônica):
  - Categorias vêm anonimizadas (`category_N`) — perde-se o nome semântico da taxonomia oficial.
  - Cobre só o conjunto fully-annotated (~30k imgs); o partially-annotated (~48k) não entra.
  - Licença do mirror ("MIT") é incorreta; a oficial é CC-BY-SA + termos acadêmicos Mapillary.

Uso:
  python scripts/analyze_mtsd.py [--root data/mtsd]
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMALL, LARGE = 32 * 32, 96 * 96


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data" / "mtsd")
    args = ap.parse_args()

    tr_p, va_p = args.root / "train_coco.json", args.root / "val_coco.json"
    if not tr_p.exists() or not va_p.exists():
        sys.exit(f"[erro] esperava train_coco.json e val_coco.json em {args.root}")
    tr, va = load_coco(tr_p), load_coco(va_p)
    id2name = {c["id"]: c["name"] for c in tr["categories"]}

    inst = Counter()
    inst_split = {"train": Counter(), "val": Counter()}
    img_per_cat = defaultdict(set)
    per_split_imgs = {"train": len(tr["images"]), "val": len(va["images"])}
    per_split_inst = Counter()
    empty_imgs = Counter()
    signs_per_img = []
    areas = []
    widths = []
    res = Counter()

    for split, d in (("train", tr), ("val", va)):
        anns_by_img = defaultdict(list)
        for a in d["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for im in d["images"]:
            res[(im["width"], im["height"])] += 1
            widths.append(im["width"])
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
                areas.append(w * h)

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
    size_counter = Counter(size_bin(a) for a in areas)

    MIN_INST, MIN_EVAL = 80, 10
    elig = [c for c in inst if inst[c] >= MIN_INST]
    elig_eval = [c for c in elig if inst_split["val"][c] >= MIN_EVAL]

    # ---- CSV ----
    (REPO / "reports").mkdir(exist_ok=True)
    csv_path = REPO / "reports" / "mtsd_class_counts.csv"
    with open(csv_path, "w") as f:
        f.write("category,instances,instances_train,instances_val,images,pct_instances,cum_pct\n")
        cum = 0
        for cat, c in rows:
            cum += c
            f.write(f"{cat},{c},{inst_split['train'][cat]},{inst_split['val'][cat]},"
                    f"{len(img_per_cat[cat])},{100*c/total_inst:.3f},{100*cum/total_inst:.3f}\n")
    print(f"[ok] {csv_path}")

    # ---- Relatório MD ----
    md = REPO / "reports" / "mtsd_analysis.md"
    with_signs = sum(1 for n in signs_per_img if n > 0)
    med = sorted(inst.values())[ncat // 2]
    top1 = rows[0][1]
    sides = sorted(math.sqrt(a) for a in areas)
    ws = sorted(widths)
    with open(md, "w") as f:
        w = f.write
        w("# MTSD (Mapillary Traffic Sign Dataset) — análise exploratória\n\n")
        w("Fonte: `data/mtsd/{train,val}_coco.json` (mirror Kaggle `zeuss2k3`, conjunto "
          "*fully-annotated*, formato COCO). Ertler et al. 2020. **Categorias anonimizadas** "
          "(`category_N`) neste mirror; licença canônica = CC-BY-SA + termos Mapillary.\n\n")
        w("## Visão geral\n\n")
        w(f"- Imagens (fully-annotated): **{total_imgs}** (train {per_split_imgs['train']} / val {per_split_imgs['val']})\n")
        w(f"- Imagens com ≥1 placa: **{with_signs}** | sem placa: **{total_imgs-with_signs}**\n")
        w(f"- Instâncias de placas: **{total_inst}**\n")
        w(f"- Categorias observadas: **{ncat}**\n")
        w(f"- Placas por imagem (só imgs com placa): média **{total_inst/max(with_signs,1):.2f}**, "
          f"máx **{max(signs_per_img)}**\n")
        w(f"- Resolução: **{len(res)} distintas**, largura mediana **{ws[len(ws)//2]}** px "
          f"(p10 {ws[len(ws)//10]}, p90 {ws[9*len(ws)//10]}, min {ws[0]}, max {ws[-1]}) — "
          f"**variável**, não panorama fixo 2048².\n\n")

        w("## Splits\n\n| split | imagens | imgs sem placa | instâncias |\n|---|--:|--:|--:|\n")
        for sp in ("train", "val"):
            w(f"| {sp} | {per_split_imgs[sp]} | {empty_imgs[sp]} | {per_split_inst[sp]} |\n")
        w("\n(o split `test` oficial do MTSD tem rótulos retidos → usa-se `val` como avaliação)\n\n")

        w("## Cauda longa\n\n")
        w(f"- Classes com **≥100** instâncias: **{len(ge100)}** | ≥80: {len(ge80)} | ≥50: {len(ge50)} "
          f"| <10: **{len(lt10)}**\n")
        w(f"- Razão head/tail: classe mais comum = **{top1}** inst. | mediana = **{med}** inst. "
          f"| razão ≈ **{top1/max(med,1):.0f}×** (cauda natural, sem piso curado)\n")
        share = sum(c for _, c in rows[:10])
        w(f"- Top-10 classes concentram **{100*share/total_inst:.1f}%** (sem categoria 'other' "
          f"dominante engolindo o dataset)\n\n")

        w("## Tamanho das placas (bins COCO, área px²)\n\n| bin | definição | instâncias | % |\n|---|---|--:|--:|\n")
        for b, dfn in [("small", "< 32²=1024"), ("medium", "32²–96²"), ("large", "> 96²=9216")]:
            c = size_counter.get(b, 0)
            w(f"| {b} | {dfn} | {c} | {100*c/max(len(areas),1):.1f}% |\n")
        w(f"\nLado da bbox (√área): mediana **{sides[len(sides)//2]:.0f} px** "
          f"(p10 {sides[len(sides)//10]:.0f}, p90 {sides[9*len(sides)//10]:.0f}) — "
          f"**regime small-object**, próximo do TT100K.\n\n")

        w("## Viabilidade de subset (espelho do critério P3)\n\n")
        w(f"Critério: `min_instances={MIN_INST}` **e** `≥{MIN_EVAL}` no split de avaliação (`val`).\n\n")
        w(f"- Classes elegíveis (≥{MIN_INST} inst.): **{len(elig)}**\n")
        w(f"- Dessas, com ≥{MIN_EVAL} no val: **{len(elig_eval)}** "
          f"→ {'✅ folga enorme' if len(elig_eval) >= 20 else '⚠️ apertado'} para um subset "
          f"de ~20 classes head/mid/tail (poderia ir bem além).\n\n")

        w("## Top-20 classes (mais frequentes)\n\n| categoria | instâncias | train | val | imagens |\n|---|--:|--:|--:|--:|\n")
        for cat, c in rows[:20]:
            w(f"| `{cat}` | {c} | {inst_split['train'][cat]} | {inst_split['val'][cat]} | {len(img_per_cat[cat])} |\n")
        w("\n_Ver `reports/mtsd_class_counts.csv` (contagem completa). Categorias anonimizadas no mirror; "
          "para o paper, mapear aos nomes da taxonomia oficial Mapillary._\n")
    print(f"[ok] {md}")
    print(f"\n[resumo] {total_imgs} imgs | {total_inst} inst | {ncat} classes | "
          f"≥100={len(ge100)} | small={100*size_counter.get('small',0)/max(len(areas),1):.0f}% | "
          f"elegíveis(≥80,≥10val)={len(elig_eval)}")


if __name__ == "__main__":
    main()
