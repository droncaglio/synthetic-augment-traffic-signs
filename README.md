# synthetic-augment-traffic-signs

Código do **Paper 3 (WVC)** — aumento de dados para **detecção de placas de trânsito**
sob cauda longa. Base de dataset: **TT100K (Tsinghua-Tencent 100K)**.

> Repositório **público**: nunca versionar dados, pesos ou LaTeX. Ver `.gitignore`.
> Texto do paper vive em `mestrado-dissertacao/papers/wvc/` (privado).

## Ambiente

Reaproveita o env conda do pipeline de detecção legado:

```bash
conda activate longtail-synth   # numpy, pandas, matplotlib, ultralytics 8.4.x
```

## Dataset (TT100K)

Não versionado. Baixar sob `data/raw/` a partir da fonte oficial da Tsinghua
(https://cg.cs.tsinghua.edu.cn/traffic-sign/):

- `tt100k_2021.zip` (~18 GB) — imagens com placas + `annotations*.json` (versão 2021, cauda mais preenchida).
- `nosign_1.zip` (~18 GB) — amostra de imagens **de fundo sem placa** (para copy-paste / estudo de fundos).

Extrair para `data/tt100k/`.

## Análise exploratória

```bash
conda activate longtail-synth
python scripts/analyze_tt100k.py            # auto-detecta annotations sob data/tt100k
```

Gera:
- `reports/tt100k_analysis.md` — visão geral, splits, cauda longa, famílias, tamanhos, densidade.
- `reports/tt100k_class_counts.csv` — contagem por classe (instâncias, imagens, % acumulado).
- `analysis/*.png` — distribuição de classes (log), tamanho de bbox, placas por imagem.

## Objetivo de pesquisa

Estudar classes e distribuição do TT100K para **compor um subset** adequado (não só o
45-classes padrão) e avaliar técnicas de aumento sob cauda longa + objetos pequenos:
novos fundos ao redor de placas reais, placas em fundos existentes, copy-paste, sintéticos.

Ver nota do vault: `mestrado-dissertacao/vault/datasets/tt100k-traffic-signs.md`.
