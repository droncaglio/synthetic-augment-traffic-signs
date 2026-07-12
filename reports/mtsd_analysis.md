# MTSD (Mapillary Traffic Sign Dataset) — análise exploratória

Fonte: `data/mtsd/{train,val}_coco.json` (mirror Kaggle `zeuss2k3`, conjunto *fully-annotated*, formato COCO). Ertler et al. 2020. **Categorias anonimizadas** (`category_N`) neste mirror; licença canônica = CC-BY-SA + termos Mapillary.

## Visão geral

- Imagens (fully-annotated): **29769** (train 25956 / val 3813)
- Imagens com ≥1 placa: **29769** | sem placa: **0**
- Instâncias de placas: **68815**
- Categorias observadas: **400**
- Placas por imagem (só imgs com placa): média **2.31**, máx **33**
- Resolução: **352 distintas**, largura mediana **3840** px (p10 1920, p90 4160, min 320, max 13312) — **variável**, não panorama fixo 2048².

## Splits

| split | imagens | imgs sem placa | instâncias |
|---|--:|--:|--:|
| train | 25956 | 0 | 60078 |
| val | 3813 | 0 | 8737 |

(o split `test` oficial do MTSD tem rótulos retidos → usa-se `val` como avaliação)

## Cauda longa

- Classes com **≥100** instâncias: **173** | ≥80: 211 | ≥50: 287 | <10: **6**
- Razão head/tail: classe mais comum = **2713** inst. | mediana = **86** inst. | razão ≈ **32×** (cauda natural, sem piso curado)
- Top-10 classes concentram **24.4%** (sem categoria 'other' dominante engolindo o dataset)

## Tamanho das placas (bins COCO, área px²)

| bin | definição | instâncias | % |
|---|---|--:|--:|
| small | < 32²=1024 | 19372 | 28.2% |
| medium | 32²–96² | 35506 | 51.6% |
| large | > 96²=9216 | 13937 | 20.3% |

Lado da bbox (√área): mediana **48 px** (p10 21, p90 144) — **regime small-object**, próximo do TT100K.

## Viabilidade de subset (espelho do critério P3)

Critério: `min_instances=80` **e** `≥10` no split de avaliação (`val`).

- Classes elegíveis (≥80 inst.): **211**
- Dessas, com ≥10 no val: **209** → ✅ folga enorme para um subset de ~20 classes head/mid/tail (poderia ir bem além).

## Top-20 classes (mais frequentes)

| categoria | instâncias | train | val | imagens |
|---|--:|--:|--:|--:|
| `category_78` | 2713 | 2376 | 337 | 2205 |
| `category_106` | 2128 | 1864 | 264 | 1357 |
| `category_240` | 1957 | 1709 | 248 | 1590 |
| `category_164` | 1763 | 1537 | 226 | 704 |
| `category_349` | 1723 | 1497 | 226 | 714 |
| `category_11` | 1553 | 1358 | 195 | 1187 |
| `category_154` | 1347 | 1179 | 168 | 1154 |
| `category_267` | 1209 | 1055 | 154 | 1050 |
| `category_172` | 1206 | 1055 | 151 | 1046 |
| `category_104` | 1182 | 1023 | 159 | 1066 |
| `category_98` | 1157 | 1003 | 154 | 1016 |
| `category_372` | 1107 | 969 | 138 | 867 |
| `category_280` | 998 | 876 | 122 | 902 |
| `category_366` | 760 | 667 | 93 | 682 |
| `category_222` | 698 | 610 | 88 | 633 |
| `category_92` | 667 | 584 | 83 | 565 |
| `category_51` | 584 | 511 | 73 | 488 |
| `category_345` | 572 | 499 | 73 | 506 |
| `category_195` | 571 | 500 | 71 | 438 |
| `category_37` | 564 | 495 | 69 | 532 |

_Ver `reports/mtsd_class_counts.csv` (contagem completa). Categorias anonimizadas no mirror; para o paper, mapear aos nomes da taxonomia oficial Mapillary._
