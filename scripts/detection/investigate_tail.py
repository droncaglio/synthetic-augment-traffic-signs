#!/usr/bin/env python
"""Investigate whether the null tail result hides a real effect: per-class ΔAP
heterogeneity, correlation with allocation/real-count, allocation-stratified contrasts,
and the minimum detectable effect (power) at n=7.

WHY: GATE A showed AP-tail all-size real_duplicate/photometric_full − da_only is ns.
Before accepting the negative thesis, test the obvious rescue: maybe augmentation helps
the classes it actually augmented (high synthetic allocation), and the 42-class macro
dilutes it with untouched/starved classes. Also quantify "ns ≠ null" (MDE).

NON-DESTRUCTIVE: reads dets_<split>.json + allocation.json, writes only
reports/det/tail_heterogeneity_v2.md. Uses the SAME per_class_ap matching as recompute_metrics.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "detection"))

from scipy import stats as ss  # noqa: E402

from detection.report import gts_by_pid, load_runs           # noqa: E402
from detection.budget import budget_tag                        # noqa: E402
from detection.stats import paired_ci                          # noqa: E402
from detection.ap_by_size import Detection, GroundTruth, PANORAMA_SIZE_DEFAULT  # noqa: E402
from recompute_metrics import per_class_ap, paired_p           # noqa: E402

KEY_CONTRASTS = [("real_duplicate", "da_only"), ("photometric_full", "da_only"),
                 ("diffusion_bg", "da_only"), ("signgen_controlnet", "da_only")]


def mde(sd: float, n: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable paired mean-delta at given power (two-sided t approx via normal)."""
    z_a = ss.norm.ppf(1 - alpha / 2)
    z_b = ss.norm.ppf(power)
    return (z_a + z_b) * sd / (n ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="experiments_full201/tt100k")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--arms", nargs="+", default=[
        "zero_aug", "da_only", "real_duplicate", "copy_paste",
        "photometric_full", "diffusion_bg", "signgen_controlnet"])
    ap.add_argument("--out", default="reports/det/tail_heterogeneity_v2.md")
    args = ap.parse_args()

    prepared = Path(args.prepared)
    subset = json.loads((prepared / "subset.json").read_text())
    names = subset["names"]
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    tail_names = subset["by_tier"]["tail"]
    tail_ids = [subset_ids[n] for n in tail_names]
    id2name = {subset_ids[n]: n for n in tail_names}
    alloc_j = json.loads((prepared / "allocation.json").read_text())
    alloc = {int(k): v for k, v in alloc_j["alloc"].items()}
    real = {int(k): v for k, v in alloc_j["train_counts"].items()}

    records = {r["id"]: r for r in
               (json.loads(l) for l in (prepared / "panoramas.jsonl").read_text().splitlines()
                if l.strip())}
    split_ids = json.loads((prepared / "splits.json").read_text())[args.eval_split]
    bm = budget_tag(args.K)
    image_area = float(PANORAMA_SIZE_DEFAULT) ** 2

    gts_pid = gts_by_pid(records, split_ids, subset_ids)
    id_of = {pid: i for i, pid in enumerate(gts_pid)}
    gt_list = [GroundTruth(id_of[pid], o["class_id"], *o["box"])
               for pid, lst in gts_pid.items() for o in lst]

    # per (arm, seed): {class_id -> all-size AP} for tail classes
    ap_cs: dict = {}  # arm -> {seed -> {cid -> ap_all}}
    for arm in args.arms:
        ap_cs[arm] = {}
        for s, (_hl, dets_pid) in load_runs(args.project, arm, args.seeds, bm,
                                            eval_split=args.eval_split).items():
            det_list = [Detection(id_of[pid], d["class_id"], d["conf"], *d["box"])
                        for pid, dl in dets_pid.items() if pid in id_of for d in dl]
            pc = per_class_ap(det_list, gt_list, len(names), image_area)
            ap_cs[arm][s] = {c: pc[c]["all"] for c in tail_ids}

    def seed_macro(arm, s, ids):
        vals = [ap_cs[arm][s][c] for c in ids if ap_cs[arm][s][c] == ap_cs[arm][s][c]]
        return st.mean(vals) if vals else float("nan")

    L = ["# Investigação da cauda — heterogeneidade, alocação, poder (all-size)\n",
         f"- split **{args.eval_split}**, K={args.K}, 42 classes de cauda, 7 seeds",
         f"- alocação na cauda: total {sum(alloc[c] for c in tail_ids)} synth; "
         f"starved (0) = {sum(1 for c in tail_ids if alloc[c]==0)} classes\n"]

    # ---- 1. Per-class mean ΔAP (all-size), key contrasts ----
    for treat, base in KEY_CONTRASTS[:2]:
        common = sorted(set(ap_cs[treat]) & set(ap_cs[base]))
        per_cls_delta = {}
        for c in tail_ids:
            ds = [ap_cs[treat][s][c] - ap_cs[base][s][c] for s in common]
            per_cls_delta[c] = st.mean(ds)
        up = sum(1 for c in tail_ids if per_cls_delta[c] > 1e-6)
        dn = sum(1 for c in tail_ids if per_cls_delta[c] < -1e-6)
        fl = 42 - up - dn
        # Spearman ΔAP vs alloc, vs real count
        cs = [c for c in tail_ids]
        rho_a, p_a = ss.spearmanr([alloc[c] for c in cs], [per_cls_delta[c] for c in cs])
        rho_r, p_r = ss.spearmanr([real[c] for c in cs], [per_cls_delta[c] for c in cs])
        L += [f"\n## {treat} − {base} (all-size) — por classe",
              f"- classes que SOBEM: **{up}**, descem: {dn}, ~0: {fl} (de 42)",
              f"- Spearman ρ(ΔAP, synth alocado) = **{rho_a:+.2f}** (p={p_a:.3f}) "
              f"→ {'aumento ajuda onde foi aplicado' if rho_a>0.3 and p_a<0.1 else 'sem relação clara com quanto se aumentou'}",
              f"- Spearman ρ(ΔAP, nº real train) = {rho_r:+.2f} (p={p_r:.3f})",
              "- top 6 ganhos / top 6 perdas (ΔAP médio):"]
        srt = sorted(tail_ids, key=lambda c: -per_cls_delta[c])
        gain = ", ".join(f"{id2name[c]} {per_cls_delta[c]:+.3f}(a{alloc[c]})" for c in srt[:6])
        loss = ", ".join(f"{id2name[c]} {per_cls_delta[c]:+.3f}(a{alloc[c]})" for c in srt[-6:])
        L += [f"  - ↑ {gain}", f"  - ↓ {loss}"]

    # ---- 2. Allocation-stratified contrast (does the effect concentrate in high-synth?) ----
    L += ["\n## Contraste estratificado por alocação (all-size, paired-t)\n",
          "Divide a cauda em terços por synth alocado; recomputa ΔAP-tail em cada estrato.\n",
          "| contraste | estrato | n_cls | ΔAP | IC95 | p |", "|---|---|---|---|---|---|"]
    srt_by_alloc = sorted(tail_ids, key=lambda c: alloc[c])
    t = len(srt_by_alloc) // 3
    strata = {"baixo-synth": srt_by_alloc[:t], "médio": srt_by_alloc[t:2*t],
              "alto-synth": srt_by_alloc[2*t:]}
    for treat, base in KEY_CONTRASTS[:2]:
        common = sorted(set(ap_cs[treat]) & set(ap_cs[base]))
        for sname, ids in strata.items():
            deltas = [seed_macro(treat, s, ids) - seed_macro(base, s, ids) for s in common]
            ci = paired_ci(deltas)
            L.append(f"| {treat}−{base} | {sname} | {len(ids)} | {ci['mean']:+.4f} | "
                     f"[{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}] | {paired_p(deltas):.3f} |")

    # ---- 3. Power / MDE for the all-size headline contrasts ----
    L += ["\n## Poder — efeito mínimo detectável (MDE) a n=7, 80% poder\n",
          "| contraste | ΔAP obs | dp(Δ) | MDE(80%) | detectável? |", "|---|---|---|---|---|"]
    for treat, base in KEY_CONTRASTS:
        common = sorted(set(ap_cs[treat]) & set(ap_cs[base]))
        deltas = [seed_macro(treat, s, tail_ids) - seed_macro(base, s, tail_ids) for s in common]
        m, sd = st.mean(deltas), (st.stdev(deltas) if len(deltas) > 1 else 0.0)
        d = mde(sd, len(deltas))
        L.append(f"| {treat}−{base} | {m:+.4f} | {sd:.4f} | {d:.4f} | "
                 f"{'sim, se real' if abs(m) >= d else 'NÃO — poder insuf. p/ este efeito'} |")
    L += ["\n> Leitura: se |ΔAP obs| < MDE, o 'ns' é **falta de poder**, não prova de efeito nulo. "
          "O honesto é reportar 'não detectamos ganho ≥ MDE', não 'não há ganho'.\n"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
