# TT100K — análise exploratória

Fonte de anotação: `data/tt100k/tt100k_2021/annotations_all.json`

## Visão geral

- Imagens totais: **10592**
- Imagens com ≥1 placa: **10592** | sem placa (fundo): **0**
- Instâncias de placas: **27346**
- Categorias observadas (com ≥1 instância): **201** | declaradas em `types`: 232
- Placas por imagem (só imgs com placa): média **2.58**, máx **17**

## Splits

| split | imagens | imgs sem placa | instâncias |
|---|--:|--:|--:|
| other | 1542 | 0 | 2336 |
| test | 3016 | 0 | 8261 |
| train | 6034 | 0 | 16749 |

## Cauda longa

- Classes com **≥100** instâncias: **45** (subset "treinável" clássico)
- Classes com ≥50: 68 | com <10: **84** | com <5: **57**
- Razão head/tail: classe mais comum = **3176** inst. | mediana = **16** inst. | razão ≈ **198×**
- Top-10 classes concentram **54.0%** de todas as instâncias

## Famílias semânticas (por prefixo do código)

| família | instâncias | % |
|---|--:|--:|
| prohibitory (proibição/regulamentação) | 20103 | 73.5% |
| indication/mandatory (indicação, azul) | 5437 | 19.9% |
| warning (advertência, triangular) | 1806 | 6.6% |

## Tamanho das placas (bins COCO, área px²)

| bin | definição | instâncias | % |
|---|---|--:|--:|
| small | < 32²=1024 | 11914 | 43.6% |
| medium | 32²–96² | 13423 | 49.1% |
| large | > 96²=9216 | 2009 | 7.3% |

## Top-20 classes (mais frequentes)

| categoria | família | instâncias | imagens |
|---|---|--:|--:|
| `pn` | prohibitory (proibição/regulamentação) | 3176 | 3016 |
| `pne` | prohibitory (proibição/regulamentação) | 2384 | 2142 |
| `i5` | indication/mandatory (indicação, azul) | 1734 | 1599 |
| `p11` | prohibitory (proibição/regulamentação) | 1582 | 1560 |
| `pl40` | prohibitory (proibição/regulamentação) | 1413 | 1376 |
| `pl50` | prohibitory (proibição/regulamentação) | 1073 | 1062 |
| `pl80` | prohibitory (proibição/regulamentação) | 904 | 827 |
| `p26` | prohibitory (proibição/regulamentação) | 840 | 786 |
| `pl60` | prohibitory (proibição/regulamentação) | 835 | 803 |
| `i4` | indication/mandatory (indicação, azul) | 814 | 774 |
| `pl100` | prohibitory (proibição/regulamentação) | 673 | 394 |
| `pl30` | prohibitory (proibição/regulamentação) | 640 | 637 |
| `pl5` | prohibitory (proibição/regulamentação) | 537 | 386 |
| `il60` | indication/mandatory (indicação, azul) | 489 | 361 |
| `i2` | indication/mandatory (indicação, azul) | 472 | 463 |
| `i2r` | indication/mandatory (indicação, azul) | 429 | 422 |
| `p5` | prohibitory (proibição/regulamentação) | 421 | 408 |
| `w57` | warning (advertência, triangular) | 420 | 393 |
| `p13` | prohibitory (proibição/regulamentação) | 379 | 244 |
| `p10` | prohibitory (proibição/regulamentação) | 374 | 358 |

## Bottom-20 classes (mais raras)

| categoria | família | instâncias | imagens |
|---|---|--:|--:|
| `pnlc` | prohibitory (proibição/regulamentação) | 2 | 2 |
| `w28` | warning (advertência, triangular) | 1 | 1 |
| `w48` | warning (advertência, triangular) | 1 | 1 |
| `w5` | warning (advertência, triangular) | 1 | 1 |
| `pr5` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `w49` | warning (advertência, triangular) | 1 | 1 |
| `pa18` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `pa6` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `w44` | warning (advertência, triangular) | 1 | 1 |
| `w1` | warning (advertência, triangular) | 1 | 1 |
| `w62` | warning (advertência, triangular) | 1 | 1 |
| `pw4.2` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `pw2.5` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `pr10` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `pa8` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `p7` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `w14` | warning (advertência, triangular) | 1 | 1 |
| `pclr` | prohibitory (proibição/regulamentação) | 1 | 1 |
| `w56` | warning (advertência, triangular) | 1 | 1 |
| `ph3.8` | prohibitory (proibição/regulamentação) | 1 | 1 |

_Ver `reports/tt100k_class_counts.csv` (contagem completa) e `analysis/*.png`._
