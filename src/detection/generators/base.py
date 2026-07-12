"""ArmGenerator: turn the shared source manifest into an arm's synthetic train tiles.

Reviewer-relevant design:
- The in-place arms (real_duplicate, bg_photometric, diffusion_bg) consume the
  IDENTICAL source manifest and keep the source tile's FULL labels — so the core
  "context ladder" is perfectly paired (same source instances, same in-tile positions;
  only the background treatment differs). copy_paste consumes the same sources but
  relocates the sign into a background tile (its orthogonal reference role).
- The arm's train set = the real train tiles + these synthetic tiles (run_det lists
  BOTH dirs in dataset.yaml; no file copies). Equalized optimizer steps keep the
  training budget fair across arms.
- Anti-leak: sources/backgrounds come only from the TRAIN tiles.
"""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from detection.generators.manifests import per_class_counts


def feather_alpha(th: int, tw: int) -> np.ndarray:
    """Alpha (th,tw) = 1 in the center, ramping to ~0 over a small border (soft paste).

    Shared by every arm that composites a real sign crop onto a new background
    (copy_paste, diffusion_bg) so the blend — and thus the edge artifact — is
    IDENTICAL across arms and can't confound the comparison.
    """
    f = max(1, min(tw, th) // 10)
    ay = np.ones(th, np.float32)
    ax = np.ones(tw, np.float32)
    for i in range(f):
        v = (i + 1) / (f + 1)
        ay[i] = ay[-1 - i] = min(ay[i], v)
        ax[i] = ax[-1 - i] = min(ax[i], v)
    return np.minimum(ay[:, None], ax[None, :])


class ArmGenerator(ABC):
    """Base class. Subclasses implement make_tile() for one source instance."""

    name: str = "base"

    def __init__(self, tiles_dir: str | Path, seed: int = 0):
        self.tiles_dir = Path(tiles_dir)
        self.train_img = self.tiles_dir / "train" / "images"
        self.train_lbl = self.tiles_dir / "train" / "labels"
        self.seed = seed

    # -- helpers -----------------------------------------------------------
    def load_tile(self, stem: str):
        """Return (np_image HxWx3 uint8, label_lines, ignore_boxes) for a train tile."""
        arr = np.asarray(Image.open(self.train_img / f"{stem}.jpg").convert("RGB"))
        lbl = self.train_lbl / f"{stem}.txt"
        labels = [ln for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
        ig = self.train_lbl / f"{stem}.ignore.json"
        ignores = json.loads(ig.read_text()) if ig.exists() else []
        return arr, labels, ignores

    # -- arm-specific ------------------------------------------------------
    @abstractmethod
    def make_tile(self, source: dict, rng: random.Random):
        """Produce ONE synthetic tile from a source instance.

        Returns (np_image HxWx3 uint8, label_lines: list[str]) or None to skip
        (e.g. diffusion rejected by the hallucination scan).
        """

    # -- orchestration -----------------------------------------------------
    def generate(self, sources: list[dict], out_dir: str | Path) -> dict:
        """Write synthetic tiles for `sources` into out_dir/{images,labels} + manifest."""
        out_dir = Path(out_dir)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
        rng = random.Random(self.seed)
        n_written, realized_labels = 0, []
        for i, src in enumerate(sources):
            res = self.make_tile(src, rng)
            if res is None:
                continue
            img, labels = res
            name = f"syn_{self.name}_{i:06d}"
            Image.fromarray(img).save(out_dir / "images" / f"{name}.jpg", quality=95)
            (out_dir / "labels" / f"{name}.txt").write_text("\n".join(labels))
            n_written += 1
            for ln in labels:
                realized_labels.append({"class_id": int(ln.split()[0])})
        manifest = {
            "arm": self.name, "seed": self.seed,
            "n_sources": len(sources), "n_tiles_written": n_written,
            "allocated_per_class": per_class_counts(sources),
            "realized_per_class": per_class_counts(realized_labels),  # incl. co-occurring signs
        }
        if getattr(self, "_scan_stats", None) is not None:
            manifest["scan_stats"] = self._scan_stats  # diffusion anti-hallucination audit
        (out_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
