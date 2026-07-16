"""Sim-to-real degradation for the signgen arm (REQ2).

The template+ControlNet generator produces big, sharp, close-up signs; the real TT100K tail
is tiny and blurry (max-dim median 32-71px, sharpness lapvar ~1500-2600). Pasting a sharp sign
shrunk down does NOT reproduce the real small-object statistics — the same pathology that made
the img2img MVP useless. This matches the synthetic sign to the real distribution of (a) SIZE
(sampled from the class's real bboxes — same pool copy_paste uses, so it's paired) and (b)
SHARPNESS/NOISE (downscale + light blur + noise + JPEG round-trip, calibrated to real lapvar).

Pure (cv2/numpy) — unit-tested.
"""
from __future__ import annotations

import random

import cv2
import numpy as np


def lapvar(crop: np.ndarray) -> float:
    """Sharpness = variance of the Laplacian (higher = sharper). The calibration target."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_real_bbox(class_id: int, index: dict, rng: random.Random) -> list:
    """Draw a real [cx,cy,bw,bh] of `class_id` from the index (index_instances_by_class) —
    the target SIZE from the real distribution, the same pool copy_paste samples (paired)."""
    pool = index.get(class_id, [])
    if not pool:
        raise ValueError(f"classe {class_id} sem instâncias reais no index")
    _tile, box = pool[rng.randrange(len(pool))]
    return list(box)


def degrade_to_real(sign_rgb: np.ndarray, target_px: int, rng: random.Random, *,
                    blur=(0.35, 0.85), noise=(1.5, 4.0), jpeg_q=(45, 80)) -> np.ndarray:
    """Shrink a clean sign to target_px and add small-object degradation to match real crops.

    downscale (INTER_AREA) -> gaussian blur -> gaussian noise -> JPEG round-trip. Defaults
    CALIBRATED (qa_sim2real.py, 4 tail classes) so the output lapvar lands ~in the real range
    (~1400-2800), NEVER as sharp as a naive downscale (~7000). Off-class casualties of the
    degradation (e.g. small triangles losing their glyph) are caught by the verifier filter.
    """
    t = max(4, int(target_px))
    small = cv2.resize(sign_rgb, (t, t), interpolation=cv2.INTER_AREA)
    sigma = rng.uniform(*blur)
    if sigma > 0:
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    nstd = rng.uniform(*noise)
    if nstd > 0:
        nrng = np.random.default_rng(rng.randrange(2 ** 31))
        small = np.clip(small.astype(np.float32) + nrng.normal(0, nstd, small.shape), 0, 255
                        ).astype(np.uint8)
    q = int(rng.uniform(*jpeg_q))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        small = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return small
