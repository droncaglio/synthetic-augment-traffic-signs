# Detection grid report — eval=test, K=0.5

## Per-arm AP (mean ± std over seeds)

| arm | n | AP-tail | AP@small(macro) |
|---|---|---|---|
| zero_aug | 7 | 0.6376±0.0707 | 0.6875±0.0288 |
| da_only | 7 | 0.8597±0.0508 | 0.8652±0.0214 |
| real_duplicate | 7 | 0.8662±0.0505 | 0.8722±0.0214 |
| bg_photometric | 7 | 0.9023±0.044 | 0.8893±0.0153 |
| bg_photometric_mask | 7 | 0.8587±0.0495 | 0.8714±0.0212 |
| photometric_full | 7 | 0.9059±0.0365 | 0.8779±0.017 |
| copy_paste | 7 | 0.82±0.0255 | 0.8574±0.0116 |
| copy_paste_mask | 7 | 0.8465±0.0366 | 0.87±0.0123 |
| diffusion_bg | 7 | 0.8821±0.0535 | 0.8796±0.0231 |

## Primary contrasts — paired-seed ΔAP (t CI 95%)

sig = IC exclui 0. seeds+ = em quantas seeds o tratamento > baseline.

| treatment vs baseline | n | ΔAP-tail [CI] | sig | ΔAP@small [CI] | sig | tail seeds+ |
|---|---|---|---|---|---|---|
| da_only vs zero_aug | 7 | 0.2221 [0.1617, 0.2826] | ✓ | 0.1777 [0.1587, 0.1968] | ✓ | 7/7 |
| real_duplicate vs da_only | 7 | 0.0065 [-0.0312, 0.0442] | – | 0.007 [-0.0076, 0.0216] | – | 5/7 |
| bg_photometric vs real_duplicate | 7 | 0.036 [-0.0272, 0.0993] | – | 0.0171 [-0.0127, 0.047] | – | 6/7 |
| copy_paste vs real_duplicate | 7 | -0.0462 [-0.0976, 0.0052] | – | -0.0148 [-0.0363, 0.0067] | – | 1/7 |
| diffusion_bg vs real_duplicate | 7 | 0.0159 [-0.042, 0.0738] | – | 0.0074 [-0.0129, 0.0278] | – | 3/7 |
| diffusion_bg vs copy_paste | 7 | 0.0621 [0.0158, 0.1083] | ✓ | 0.0223 [0.0007, 0.0438] | ✓ | 5/7 |
| diffusion_bg vs bg_photometric | 7 | -0.0202 [-0.086, 0.0457] | – | -0.0097 [-0.0362, 0.0168] | – | 2/7 |
| photometric_full vs bg_photometric | 7 | 0.0037 [-0.0477, 0.055] | – | -0.0114 [-0.0393, 0.0165] | – | 5/7 |
| copy_paste_mask vs copy_paste | 7 | 0.0264 [-0.0065, 0.0593] | – | 0.0126 [0.0061, 0.0191] | ✓ | 5/7 |
| bg_photometric_mask vs bg_photometric | 7 | -0.0436 [-0.0927, 0.0056] | – | -0.0179 [-0.0406, 0.0048] | – | 2/7 |

> ΔAP × custo (fronteira de Pareto): custo de geração por braço a preencher (GPU-h por amostra aceita) — auditoria separada.
