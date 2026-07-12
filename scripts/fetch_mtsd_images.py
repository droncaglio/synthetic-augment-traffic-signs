#!/usr/bin/env python3
"""Download SELETIVO das imagens do MTSD — só os .jpg das classes do subset.

Em vez de baixar o zip de 35 GB do mirror Kaggle, extrai por HTTP range apenas as
imagens que contêm ≥1 instância de alguma classe do `subset.json` (~5.9k imgs, ~5 GB).
Self-contained: resolve sozinho a URL assinada (pública) do dataset. Resumível: pula
o que já existe no destino.

STANDALONE por ora — NÃO está plugado no `reproduce.py`. Quando for integrar o MTSD ao
pipeline, este script é o passo "download" (equivalente ao download do TT100K); o "prepare"
consumirá `data/mtsd/{train,val}_coco.json` + as imagens que este script deixa em `--out`.

Uso (rodar na workstation, onde há disco + GPU):
  python scripts/fetch_mtsd_images.py --out data/mtsd/images          # tudo do subset (~5 GB)
  python scripts/fetch_mtsd_images.py --out data/mtsd/images --limit 5  # smoke test
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
API = "https://www.kaggle.com/api/v1/datasets/download/{slug}"


def resolve_signed_url(slug: str) -> str:
    """Segue o 302 do endpoint público do Kaggle → URL assinada do GCS (válida ~3 dias)."""
    import requests
    r = requests.get(API.format(slug=slug), allow_redirects=False, timeout=60)
    loc = r.headers.get("Location")
    if r.status_code not in (301, 302) or not loc:
        sys.exit(f"[erro] esperava redirect do Kaggle, veio {r.status_code}. Dataset privado/removido?")
    return loc


def needed_keys(root: Path, subset: dict) -> dict[str, str]:
    """-> {KEY (file_name sem .jpg): split} para imagens com ≥1 classe do subset."""
    names = set(subset["names"])
    out: dict[str, str] = {}
    for split, fn in (("train", "train_coco.json"), ("val", "val_coco.json")):
        d = json.loads((root / fn).read_text())
        ok_ids = {c["id"] for c in d["categories"] if c["name"] in names}
        imgid_file = {im["id"]: im["file_name"] for im in d["images"]}
        keep = {a["image_id"] for a in d["annotations"] if a["category_id"] in ok_ids}
        for iid in keep:
            out[imgid_file[iid][:-4]] = split  # tira '.jpg'
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=REPO / "data" / "mtsd")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "mtsd" / "images")
    ap.add_argument("--slug", default="zeuss2k3/mapillary-traffic-sign-dataset")
    ap.add_argument("--limit", type=int, default=0, help="baixar só N imagens (smoke test)")
    args = ap.parse_args()

    from remotezip import RemoteZip

    subset = json.loads((args.root / "subset.json").read_text())
    keys = needed_keys(args.root, subset)
    print(f"[info] subset com {subset['n_classes']} classes → {len(keys)} imagens-alvo "
          f"(train {sum(v=='train' for v in keys.values())} / val {sum(v=='val' for v in keys.values())})")

    print("[info] resolvendo URL assinada do dataset...")
    url = resolve_signed_url(args.slug)

    got = skipped = missing = 0
    nbytes = 0
    t0 = time.time()
    with RemoteZip(url) as z:
        key2member = {n.rsplit("/", 1)[-1][:-4]: n
                      for n in z.namelist() if n.lower().endswith(".jpg")}
        todo = list(keys.items())
        if args.limit:
            todo = todo[:args.limit]
        for i, (key, split) in enumerate(todo, 1):
            dst = args.out / split / f"{key}.jpg"
            if dst.exists() and dst.stat().st_size > 0:
                skipped += 1
                continue
            member = key2member.get(key)
            if not member:
                missing += 1
                print(f"[warn] sem membro no zip para {key}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(dst, "wb") as f:
                data = src.read()
                f.write(data)
            got += 1
            nbytes += len(data)
            if got % 100 == 0 or args.limit:
                rate = nbytes / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {i}/{len(todo)}  baixadas={got} puladas={skipped}  "
                      f"{nbytes/1e6:.0f} MB  ({rate:.1f} MB/s)")

    print(f"\n[ok] baixadas={got} | já existiam={skipped} | sem membro={missing} | "
          f"{nbytes/1e9:.2f} GB em {time.time()-t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
