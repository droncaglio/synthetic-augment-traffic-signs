#!/usr/bin/env python
"""Convergência por run: épocas rodadas, steps igualados e trajetória de loss.

Para cada run do grid lê `results.csv` (loss de treino por época; + mAP de val por
época SE o run foi treinado com --val) e o `ap_report*.json` (meta: epochs/steps).
Responde "o orçamento de steps convergiu?" sem precisar retreinar:
  - se há coluna de mAP de val (probe val-on): reporta BEST EPOCH e mAP@best vs @last;
  - senão (runs oficiais val-off): reporta a queda relativa da loss no último 20%
    das épocas (proxy de platô) — alarme se ainda em descida acentuada.

Saídas: reports/det/epoch_report.md (+ analysis/train_loss_curves.png).
Uso (na workstation): python scripts/detection/epoch_report.py
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from detection.budget import budget_tag           # noqa: E402
from detection.run_naming import experiment_name  # noqa: E402

ARMS = ["zero_aug", "da_only", "real_duplicate", "bg_photometric", "copy_paste", "diffusion_bg"]


def read_csv(p: Path):
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k.strip(): [float(r[k]) for r in rows if r.get(k, "").strip() not in ("", "nan")]
            for k in (rows[0].keys() if rows else [])}
    return cols


def pick(cols, *cands):
    for c in cands:
        for k in cols:
            if k.strip() == c:
                return cols[k]
    # fallback: substring
    for c in cands:
        for k in cols:
            if c in k:
                return cols[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    args = ap.parse_args()
    project = Path(args.project).resolve()
    bm = budget_tag(args.K)

    rows = []
    curves = {}  # arm -> list of (seed, epochs, train_loss_series, val_map_series|None)
    for arm in args.arms:
        curves[arm] = []
        for s in args.seeds:
            d = project / experiment_name(arm, s, budget_tag=bm)
            rcsv = d / "results.csv"
            if not rcsv.exists():
                continue
            cols = read_csv(rcsv)
            box = pick(cols, "train/box_loss")
            cls = pick(cols, "train/cls_loss")
            dfl = pick(cols, "train/dfl_loss")
            # loss total de treino por época (soma das componentes disponíveis)
            comps = [x for x in (box, cls, dfl) if x]
            n = min(len(x) for x in comps) if comps else 0
            tot = [sum(c[i] for c in comps) for i in range(n)] if comps else []
            vmap = pick(cols, "metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50(B)")
            # meta epochs/steps
            meta = {}
            for fn in ("ap_report_test.json", "ap_report.json"):
                fp = d / fn
                if fp.exists():
                    meta = json.loads(fp.read_text()).get("meta", {})
                    break
            ep = meta.get("epochs", n)
            row = {"arm": arm, "seed": s, "epochs": ep,
                   "steps": meta.get("steps") or meta.get("realized_steps"),
                   "dev": meta.get("deviation")}
            has_val = bool(vmap) and len(set(round(v, 4) for v in vmap)) > 1
            if has_val:
                best_i = max(range(len(vmap)), key=lambda i: vmap[i])
                row["best_epoch"] = best_i + 1
                row["mAP_best"] = round(vmap[best_i], 4)
                row["mAP_last"] = round(vmap[-1], 4)
                row["gap_best_last"] = round(vmap[best_i] - vmap[-1], 4)
            elif tot:
                # proxy de platô: queda relativa da loss no último 20% das épocas
                k = max(1, n // 5)
                drop = (tot[-k] - tot[-1]) / max(abs(tot[-k]), 1e-9)
                row["loss_first"] = round(tot[0], 3)
                row["loss_last"] = round(tot[-1], 3)
                row["tail_drop_%"] = round(100 * drop, 2)  # ~0 => achatou; alto => ainda caindo
            rows.append(row)
            curves[arm].append((s, ep, tot, vmap if has_val else None))

    # plot curvas de loss (e mAP val se houver) por braço
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        adir = REPO / "analysis"; adir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
        for ax, arm in zip(axes.ravel(), ARMS):
            any_val = False
            for (s, ep, tot, vmap) in curves.get(arm, []):
                if tot:
                    ax.plot(range(1, len(tot) + 1), tot, lw=0.8, alpha=0.7, label=f"s{s} loss")
                if vmap:
                    any_val = True
                    ax2 = ax.twinx()
                    ax2.plot(range(1, len(vmap) + 1), vmap, lw=1.2, color="green")
                    ax2.set_ylabel("val mAP50-95", color="green", fontsize=8)
            ax.set_title(arm, fontsize=10); ax.set_xlabel("época"); ax.set_ylabel("train loss")
        fig.suptitle("Convergência por braço — train loss (val mAP em verde onde houver probe val-on)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(adir / "train_loss_curves.png", dpi=130)
        print(f"[ok] {adir/'train_loss_curves.png'}")
    except Exception as e:
        print(f"[warn] plot: {e}")

    # markdown
    out = REPO / "reports" / "det" / "epoch_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    val_mode = any("best_epoch" in r for r in rows)
    with open(out, "w") as f:
        f.write("# Convergência por run (épocas / steps / loss)\n\n")
        if val_mode:
            f.write("Runs com probe val-on (best epoch por mAP de val):\n\n")
            f.write("| arm | seed | epochs | best_epoch | mAP_best | mAP_last | gap |\n|---|--:|--:|--:|--:|--:|--:|\n")
            for r in rows:
                if "best_epoch" in r:
                    f.write(f"| {r['arm']} | {r['seed']} | {r['epochs']} | {r['best_epoch']} "
                            f"| {r['mAP_best']} | {r['mAP_last']} | {r['gap_best_last']} |\n")
            f.write("\n")
        f.write("Runs val-off (proxy de platô = queda da loss no último 20%; ~0 = achatou):\n\n")
        f.write("| arm | seed | epochs | steps | dev | loss_first | loss_last | tail_drop_% |\n"
                "|---|--:|--:|--:|--:|--:|--:|--:|\n")
        for r in rows:
            if "tail_drop_%" in r:
                f.write(f"| {r['arm']} | {r['seed']} | {r['epochs']} | {r.get('steps','?')} "
                        f"| {r.get('dev','?')} | {r['loss_first']} | {r['loss_last']} | {r['tail_drop_%']} |\n")
    print(f"[ok] {out}")
    # resumo no stdout
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    print("\n[resumo] épocas por braço (equalized steps -> variam):")
    for arm in ARMS:
        eps = sorted(set(r["epochs"] for r in by_arm.get(arm, [])))
        drops = [r["tail_drop_%"] for r in by_arm.get(arm, []) if "tail_drop_%" in r]
        d = f"tail_drop%~{max(drops):.1f}(máx)" if drops else ""
        print(f"  {arm:16} epochs={eps} {d}")


if __name__ == "__main__":
    main()
