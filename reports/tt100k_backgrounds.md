# TT100K — análise de fundos (nosign)

Fonte: `data/tt100k/nosign_1` · amostra **6000** imagens · embedding **ConvNeXt-Tiny (ImageNet, 768d)** · **KMeans k=20**.

Raridade de fundo = clusters pequenos (tipos de cena sub-representados) e imagens com alta distância ao centróide (fundos atípicos). Úteis para copy-paste: colar placas — sobretudo classes raras — em fundos diversos e incomuns aumenta a variabilidade de contexto.

## Tamanho dos clusters (raros primeiro)

| cluster | imagens | % | montagem |
|--:|--:|--:|---|
| 17 | 85 | 1.4% | `analysis/bg_montages/cluster_17.png` |
| 12 | 92 | 1.5% | `analysis/bg_montages/cluster_12.png` |
| 18 | 159 | 2.6% | `analysis/bg_montages/cluster_18.png` |
| 10 | 216 | 3.6% | `analysis/bg_montages/cluster_10.png` |
| 1 | 220 | 3.7% | `analysis/bg_montages/cluster_01.png` |
| 0 | 260 | 4.3% | `analysis/bg_montages/cluster_00.png` |
| 9 | 261 | 4.3% | `analysis/bg_montages/cluster_09.png` |
| 11 | 290 | 4.8% | `analysis/bg_montages/cluster_11.png` |
| 6 | 291 | 4.8% | `analysis/bg_montages/cluster_06.png` |
| 5 | 295 | 4.9% | `analysis/bg_montages/cluster_05.png` |
| 16 | 297 | 5.0% | `analysis/bg_montages/cluster_16.png` |
| 4 | 299 | 5.0% | `analysis/bg_montages/cluster_04.png` |
| 15 | 304 | 5.1% | `analysis/bg_montages/cluster_15.png` |
| 8 | 312 | 5.2% | `analysis/bg_montages/cluster_08.png` |
| 14 | 366 | 6.1% | `analysis/bg_montages/cluster_14.png` |
| 7 | 371 | 6.2% | `analysis/bg_montages/cluster_07.png` |
| 2 | 393 | 6.5% | `analysis/bg_montages/cluster_02.png` |
| 3 | 418 | 7.0% | `analysis/bg_montages/cluster_03.png` |
| 19 | 437 | 7.3% | `analysis/bg_montages/cluster_19.png` |
| 13 | 634 | 10.6% | `analysis/bg_montages/cluster_13.png` |

## Fundos mais atípicos (maior distância ao centróide)

| imagem | cluster | dist |
|---|--:|--:|
| 13790.jpg | 5 | 1.007 |
| 8383.jpg | 5 | 0.983 |
| 4749.jpg | 15 | 0.983 |
| 11748.jpg | 5 | 0.978 |
| 5994.jpg | 15 | 0.976 |
| 1919.jpg | 5 | 0.976 |
| 19354.jpg | 5 | 0.972 |
| 5812.jpg | 4 | 0.961 |
| 19151.jpg | 2 | 0.959 |
| 16738.jpg | 5 | 0.958 |
| 6650.jpg | 7 | 0.956 |
| 78.jpg | 17 | 0.953 |
| 32.jpg | 15 | 0.949 |
| 17742.jpg | 7 | 0.948 |
| 14118.jpg | 5 | 0.947 |

_Veja `analysis/bg_tsne.png`, `analysis/bg_cluster_sizes.png` e as montagens por cluster._
