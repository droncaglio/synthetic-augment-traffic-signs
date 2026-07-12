# DFG traffic-sign dataset — análise exploratória

Fonte: `data/dfg/{train,test}.json` (COCO; Tabernik & Skočaj 2020). Anotações `ignore=true` (bbox < 30 px, marcadas difíceis) excluídas.

## Visão geral

- Imagens totais: **6957** (train 5254 / test 1703)
- Imagens com ≥1 placa: **6896** | sem placa: **61**
- Instâncias de placas: **13253**
- Categorias observadas: **200** / 200 declaradas
- Placas por imagem (só imgs com placa): média **1.92**, máx **10**
- Resolução: 1920×1080 (6794), 720×576 (162), 864×691 (1)

## Splits

| split | imagens | imgs sem placa | instâncias |
|---|--:|--:|--:|
| train | 5254 | 59 | 9815 |
| test | 1703 | 2 | 3438 |

## Cauda longa

- Classes com **≥100** instâncias: **24** | ≥80: 37 | ≥50: 108 | <10: **0**
- Razão head/tail: classe mais comum = **673** inst. | mediana = **51** inst. | razão ≈ **13×**
- Piso curado: DFG garante ≥20 instâncias/classe (por construção) → cauda **truncada**, não natural como no TT100K.
- Top-10 classes concentram **24.2%** das instâncias

## Tamanho das placas (bins COCO, área px²)

| bin | definição | instâncias | % |
|---|---|--:|--:|
| small | < 32²=1024 | 200 | 1.5% |
| medium | 32²–96² | 4547 | 34.3% |
| large | > 96²=9216 | 8506 | 64.2% |

Lado da bbox (√área): mediana **123 px** (p10 45, p90 223). Imagens 1920×1080 → placas **grandes** relativas ao TT100K (2048², placas minúsculas).

## Viabilidade de subset (espelho do critério P3)

Critério: `min_instances=80` (piso) **e** `≥10` instâncias no split de teste (restrição dura para estratificar head/mid/tail e ter suporte de avaliação na cauda).

- Classes elegíveis (≥80 inst. globais): **37**
- Dessas, com ≥10 no teste: **37** → ✅ sobra folga para um subset de ~20 classes

## Top-20 classes (mais frequentes)

| categoria | instâncias | train | test | imagens |
|---|--:|--:|--:|--:|
| `IV-5` | 673 | 492 | 181 | 628 |
| `II-1` | 446 | 344 | 102 | 424 |
| `X-1.1` | 382 | 306 | 76 | 360 |
| `IV-2` | 344 | 238 | 106 | 343 |
| `II-2` | 297 | 230 | 67 | 288 |
| `III-3` | 289 | 211 | 78 | 286 |
| `III-6` | 269 | 207 | 62 | 242 |
| `III-35` | 172 | 129 | 43 | 165 |
| `II-30-40` | 171 | 133 | 38 | 171 |
| `II-34` | 167 | 132 | 35 | 164 |
| `II-47` | 161 | 124 | 37 | 150 |
| `III-105` | 145 | 105 | 40 | 145 |
| `III-86-1` | 143 | 104 | 39 | 112 |
| `IV-1` | 140 | 98 | 42 | 137 |
| `III-14` | 137 | 108 | 29 | 137 |
| `III-54` | 133 | 100 | 33 | 133 |
| `X-1.2` | 132 | 104 | 28 | 123 |
| `VII-4.3-1` | 117 | 87 | 30 | 69 |
| `I-15` | 108 | 83 | 25 | 108 |
| `II-40` | 108 | 87 | 21 | 108 |

## Bottom-20 classes (mais raras)

| categoria | instâncias | train | test | imagens |
|---|--:|--:|--:|--:|
| `VI-3-2` | 22 | 16 | 6 | 20 |
| `VII-4.1-1` | 22 | 13 | 9 | 20 |
| `I-29` | 21 | 16 | 5 | 21 |
| `II-42.1` | 21 | 13 | 8 | 21 |
| `II-46` | 21 | 12 | 9 | 21 |
| `II-6` | 21 | 12 | 9 | 21 |
| `III-105.1` | 21 | 11 | 10 | 21 |
| `III-112` | 21 | 16 | 5 | 17 |
| `III-42` | 21 | 11 | 10 | 21 |
| `III-46` | 21 | 11 | 10 | 21 |
| `IV-12` | 21 | 16 | 5 | 21 |
| `IV-13.1-2` | 21 | 14 | 7 | 18 |
| `IV-13.1-3` | 21 | 12 | 9 | 21 |
| `VI-2.1` | 21 | 12 | 9 | 20 |
| `VII-4` | 21 | 12 | 9 | 14 |
| `II-46.1` | 20 | 14 | 6 | 20 |
| `II-46.2` | 20 | 12 | 8 | 20 |
| `III-29-40` | 20 | 14 | 6 | 20 |
| `III-39` | 20 | 13 | 7 | 20 |
| `III-59` | 20 | 15 | 5 | 20 |

_Ver `reports/dfg_class_counts.csv` (contagem completa) e `analysis/dfg_*.png`._
