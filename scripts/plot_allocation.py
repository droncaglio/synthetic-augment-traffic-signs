#!/usr/bin/env python3
"""Gráfico antes/depois da alocação por frequência (water-filling K=0.5).

Barras empilhadas por classe: base = instâncias reais no treino; topo (outra cor) =
tiles sintéticos alocados. Mostra o preenchimento da cauda até um piso comum.
Lê data/tt100k/prepared/{allocation.json,subset.json}.

Uso: python scripts/plot_allocation.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PREP = REPO / "data" / "tt100k" / "prepared"


def main():
    alloc = json.load(open(PREP / "allocation.json"))
    subset = json.load(open(PREP / "subset.json"))
    id2 = {c["id"]: c for c in subset["classes"]}

    ids = sorted(int(k) for k in alloc["train_counts"])
    real = np.array([alloc["train_counts"][str(i)] for i in ids])
    synth = np.array([alloc["alloc"][str(i)] for i in ids])
    names = [id2[i]["name"] for i in ids]
    tiers = [id2[i].get("tier", "?") for i in ids]
    total = real + synth
    floor = int(total.max() if synth.sum() else 0)
    # piso real = maior total entre classes que receberam sintético
    filled = total[synth > 0]
    floor = int(filled.max()) if filled.size else int(total.max())

    tier_color = {"head": "#7f7f7f", "mid": "#7f7f7f", "tail": "#7f7f7f"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(ids))
    C_REAL, C_SYN = "#2c6fbb", "#e8833a"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1]})

    for ax, zoom in ((ax1, False), (ax2, True)):
        ax.bar(x, real, color=C_REAL, label="reais (treino)")
        ax.bar(x, synth, bottom=real, color=C_SYN, label="sintéticos alocados")
        if synth.sum():
            ax.axhline(floor, ls="--", lw=1, color="#c0392b",
                       label=f"piso ≈ {floor} inst.")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}" for n in names], rotation=60, ha="right", fontsize=8)
        # marca o tier sob o rótulo
        for xi, t in zip(x, tiers):
            ax.text(xi, -0.06, t[:1].upper(), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color="#555")
        ax.set_ylabel("instâncias no treino")
        if zoom:
            ax.set_ylim(0, floor * 1.35)
            ax.set_title(f"Zoom no piso (0–{int(floor*1.35)})")
            # anota as barras estouradas
            for xi, r in zip(x, real):
                if r > floor * 1.35:
                    ax.annotate(f"{r}", (xi, floor * 1.30), ha="center", va="top",
                                fontsize=7, color=C_REAL, rotation=90)
        else:
            ax.set_title("Escala completa")
        ax.legend(loc="upper right", fontsize=9)

    B = alloc["B"]
    fig.suptitle(f"TT100K subset ({len(ids)} classes) — alocação por frequência "
                 f"(water-filling, K={alloc['K']}, B={B} tiles)\n"
                 f"cabeça já acima do piso → 0 sintético; cauda preenchida até ~{floor}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = REPO / "analysis" / "allocation_before_after.png"
    fig.savefig(out, dpi=140)
    print(f"[ok] {out}")
    # resumo textual
    print(f"B={B} | piso≈{floor} | classes c/ sintético: {(synth>0).sum()}/{len(ids)} | "
          f"cabeça sem alocação: {[names[i] for i in range(len(ids)) if synth[i]==0]}")


if __name__ == "__main__":
    main()
