# synthetic-augment-traffic-signs

Código do **Paper 3 (WVC)** — aumento de dados para **detecção de placas de trânsito**
sob cauda longa. Base de dataset: **TT100K (Tsinghua-Tencent 100K)**.

> Repositório **público**: nunca versionar dados, pesos ou LaTeX. Ver `.gitignore`.
> Texto do paper vive em `mestrado-dissertacao/papers/wvc/` (privado).

## Ambiente

Env conda dedicado (deps de difusão são pesadas — não compartilha com os outros repos):

```bash
conda env create -f env/environment.yml
conda activate augment-traffic-signs
```

Testes unitários (sem GPU): `pytest tests/unit -q`.

## Reprodução (um comando)

`reproduce.py` orquestra o experimento de ponta a ponta:

```bash
python reproduce.py                 # check → download → prepare → train → report
```

Steps (rodáveis isolados com `--step <nome>`):

| step | o que faz |
|---|---|
| `check` | valida env/GPU/deps **e** os extras de difusão (`bitsandbytes`, `HF_TOKEN` no `.env`) — falha rápido antes do passe longo |
| `download` | verifica o TT100K 2021; auto-baixa o zip se faltar |
| `prepare` | espinha de dados idempotente: `prepare_tt100k → select_subset → make_splits` (split por panorama + pHash near-dup + `assert_no_leak`) `→ tile_panoramas{train,val,test} → build_allocation` (K=0.5) |
| `generate` | materializa os tiles sintéticos dos braços (difusão com `--resume` + scanner = zero_aug seed 0). Opcional: `train`/`all` já auto-preparam cada braço sob demanda |
| `train` | `batch_run_det.py` — grid 6 braços × 7 seeds, steps de otimização igualados, status resumível |
| `report` | `det_report.py` — AP por braço + contrastes primários com bootstrap CI pareado por seed |

Modos úteis:

```bash
python reproduce.py --smoke                 # valida a espinha (1 seed, 2 épocas, sem difusão)
python reproduce.py --dry-run               # prevê tudo sem executar
python reproduce.py --step generate --arm diffusion_bg   # só o passe de difusão (~17h)
python reproduce.py --step report --eval-split test      # contrastes finais no test
```

**Escada de custo de novidade de contexto** (braços): `zero_aug` · `da_only` ·
`real_duplicate` · `bg_photometric` · `copy_paste` · `diffusion_bg`. A difusão
regenera o **fundo** (FluxFill máscara invertida), preservando a placa real
recomposta com feather; uma varredura anti-alucinação (scanner) rejeita/regenera
tiles com placa inventada. Detector `yolo11n`, avaliação por **reconstrução no
panorama 2048 + NMS global**. Saídas: `experiments/tt100k/<arm>_<seed>/` e
`reports/det/report.md`. Notificações de progresso via Telegram (`.env`, ver `.env.example`).

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
