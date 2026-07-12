#!/usr/bin/env python3
"""Seleção determinística de subset head/mid/tail para datasets COCO (DFG, MTSD).

Aplica o MESMO critério do TT100K (`src/detection/subset.py`): `min_instances` de piso +
partição em 3 faixas + pick uniformemente espaçado, sem RNG e sem espiar AP. Como esses
datasets já têm split fixo, o suporte de avaliação (`>=min_eval` instâncias no split de
avaliação) é imposto AQUI (pré-filtro de elegibilidade), dispensando o reparo do `splits.py`.

Presets:
  dfg  : train.json / test.json        (eval = test)
  mtsd : train_coco.json / val_coco.json (eval = val)

Uso:
  python scripts/select_subset_coco.py --dataset mtsd [--n-classes 20] [--min-instances 80] [--min-eval 10]
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from detection.subset import select_subset, save_subset  # noqa: E402

PRESETS = {
    "dfg":  {"train": "train.json",      "eval": "test.json", "eval_name": "test"},
    "mtsd": {"train": "train_coco.json", "eval": "val_coco.json", "eval_name": "val"},
}


def coco_counts(path: Path):
    """-> (Counter inst por nome de categoria, Counter imgs por nome)."""
    d = json.loads(path.read_text())
    id2name = {c["id"]: c["name"] for c in d["categories"]}
    inst = Counter()
    imgs = defaultdict(set)
    for a in d["annotations"]:
        if a.get("ignore", False):
            continue
        name = id2name[a["category_id"]]
        inst[name] += 1
        imgs[name].add(a["image_id"])
    return inst, {k: len(v) for k, v in imgs.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=PRESETS, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--n-classes", type=int, default=20)
    ap.add_argument("--min-instances", type=int, default=80)
    ap.add_argument("--min-eval", type=int, default=10)
    args = ap.parse_args()

    pre = PRESETS[args.dataset]
    root = args.root or REPO / "data" / args.dataset
    tr_inst, tr_imgs = coco_counts(root / pre["train"])
    ev_inst, _ = coco_counts(root / pre["eval"])

    total = Counter(tr_inst)
    for k, v in ev_inst.items():
        total[k] += v

    # elegibilidade: piso de instâncias globais E suporte de avaliação
    eligible = {c: n for c, n in total.items()
                if n >= args.min_instances and ev_inst.get(c, 0) >= args.min_eval}
    dropped_eval = [c for c in total if total[c] >= args.min_instances and ev_inst.get(c, 0) < args.min_eval]

    # catalog no formato esperado por select_subset (ordenado por -inst, nome)
    ordered = dict(sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0])))
    catalog = {"categories": {c: {"instances": n} for c, n in ordered.items()}}

    subset = select_subset(catalog, args.n_classes, args.min_instances)
    # anexa contagens train/eval por classe escolhida
    for c in subset["classes"]:
        c["instances_train"] = tr_inst.get(c["name"], 0)
        c["instances_eval"] = ev_inst.get(c["name"], 0)
        c["images_train"] = tr_imgs.get(c["name"], 0)
    subset["dataset"] = args.dataset
    subset["eval_split"] = pre["eval_name"]
    subset["min_eval"] = args.min_eval
    subset["n_eligible"] = len(eligible)

    out = root / "subset.json"
    save_subset(subset, out)

    print(f"[{args.dataset}] elegíveis (≥{args.min_instances} inst, ≥{args.min_eval} eval): {len(eligible)}"
          f"  | descartadas por eval<{args.min_eval}: {len(dropped_eval)}")
    print(f"selecionadas {subset['n_classes']} classes:")
    print(f"  {'tier':4s} {'classe':16s} {'total':>6s} {'train':>6s} {args.dataset[:4]+'.eval':>6s} {'imgs_tr':>7s}")
    for c in subset["classes"]:
        print(f"  {c['tier']:4s} {c['name']:16s} {c['instances']:6d} {c['instances_train']:6d} "
              f"{c['instances_eval']:6d} {c['images_train']:7d}")
    # heurística de 'catch-all' (ex.: other-sign): classe muito acima da 2ª
    tops = sorted(total.values(), reverse=True)[:2]
    if len(tops) == 2 and tops[0] > 3 * tops[1]:
        print(f"[aviso] classe mais frequente ({tops[0]}) é >3× a 2ª ({tops[1]}) — possível bucket 'other', checar.")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
