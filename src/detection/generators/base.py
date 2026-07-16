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

    def _restore_scan_stats(self, manifest_path: Path) -> None:
        """On resume, carry forward the previous run's scan_stats accumulators so the
        anti-hallucination audit stays complete across restarts (counters, not scanner)."""
        stats = getattr(self, "_scan_stats", None)
        if stats is None or not manifest_path.exists():
            return
        prev = json.loads(manifest_path.read_text()).get("scan_stats", {})
        for k in ("tiles", "attempts", "regenerated", "rejected", "scan_fired", "scan_detections"):
            stats[k] += int(prev.get(k, 0))

    # -- arm-specific ------------------------------------------------------
    @abstractmethod
    def make_tile(self, source: dict, rng: random.Random):
        """Produce ONE synthetic tile from a source instance.

        Returns (np_image HxWx3 uint8, label_lines: list[str]) or None to skip
        (e.g. diffusion rejected by the hallucination scan).
        """

    # -- orchestration -----------------------------------------------------
    def generate(self, sources: list[dict], out_dir: str | Path, resume: bool = False) -> dict:
        """Write synthetic tiles for `sources` into out_dir/{images,labels} + manifest.

        resume=True: skip sources whose tile is already on disk (crash-safe for the long
        diffusion pass). Existing tiles are re-counted from disk so the manifest stays
        correct, and prior scan_stats are carried forward from the previous manifest.
        """
        out_dir = Path(out_dir)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
        if resume:
            self._restore_scan_stats(out_dir / "generation_manifest.json")
        rng = random.Random(self.seed)
        # periodic Telegram progress for the LONG GPU passes (diffusion/signgen). Silent
        # _NullNotifier if no creds; cheap arms finish before the interval and never ping.
        import time
        from detection.notifications.telegram import TelegramNotifier
        notifier = TelegramNotifier.from_env()
        t0 = _last_ping = time.time()
        ping_every, n_src = 20 * 60, max(1, len(sources))
        n_written, realized_labels = 0, []
        for i, src in enumerate(sources):
            if time.time() - _last_ping >= ping_every:
                _last_ping = time.time()
                rej = self._scan_stats.get("rejected") if getattr(self, "_scan_stats", None) else None
                notifier.send_message(
                    f"⏳ GEN <code>{self.name}</code>: {i}/{n_src} ({100 * i // n_src}%) · "
                    f"{n_written} escritas" + (f" · {rej} rejeit." if rej is not None else "")
                    + f" · {int((time.time() - t0) // 60)}min")
            name = f"syn_{self.name}_{i:06d}"
            img_path, lbl_path = out_dir / "images" / f"{name}.jpg", out_dir / "labels" / f"{name}.txt"
            if resume and img_path.exists():   # already generated -> count from disk, skip
                for ln in (lbl_path.read_text().splitlines() if lbl_path.exists() else []):
                    if ln.strip():
                        realized_labels.append({"class_id": int(ln.split()[0])})
                n_written += 1
                continue
            res = self.make_tile(src, rng)
            if res is None:
                continue
            img, labels = res
            Image.fromarray(img).save(img_path, quality=95)
            lbl_path.write_text("\n".join(labels))
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
