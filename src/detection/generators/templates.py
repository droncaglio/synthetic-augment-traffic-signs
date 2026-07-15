"""Official-icon templates for sign-generation (method #5-bis, template+ControlNet).

The TT100K download ships `marks/` — clean official sign ICONS (RGBA, ~280px). We use them
as the CLASS ANCHOR for ControlNet: the Canny edges of the template pin the sign's shape and
glyph, so the diffuser renders a photorealistic in-the-wild sign whose CLASS is guaranteed by
construction (exact label). Pose diversity comes from warping the template BEFORE the Canny.

- 10/21 subset classes have a direct PNG; the parametric families (pl*/il*/ph*/pm*) have one
  exemplar each (pl40/il50/ph3.5/pm10) -> we render the class number onto the base ring.

Pure helpers (no GPU) — unit-tested. cv2/numpy/PIL are already deps.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# parametric family -> base exemplar in marks/ ; and the unit suffix drawn in the icon
FAMILY_BASE = {"pl": "pl40", "il": "il50", "ph": "ph3.5", "pm": "pm10"}
FAMILY_SUFFIX = {"ph": "m", "pm": "t"}
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _to_rgba(im: Image.Image, size: int) -> np.ndarray:
    return np.asarray(im.convert("RGBA").resize((size, size), Image.LANCZOS))


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()  # fallback (POC still runs, less pretty)


def _fit_font(text: str, max_w: float, max_h: float) -> ImageFont.FreeTypeFont:
    """Largest bold font whose `text` fits within (max_w, max_h)."""
    size = int(max_h)
    while size > 6:
        f = _load_font(size)
        l, t, r, b = f.getbbox(text)
        if (r - l) <= max_w and (b - t) <= max_h:
            return f
        size -= 2
    return _load_font(8)


def _sign_geometry(alpha: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, R) of the opaque sign region (bbox of alpha>10)."""
    ys, xs = np.where(alpha > 10)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    R = min(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
    return cx, cy, R


def render_parametric(base_png: str | Path, text: str, size: int = 280,
                      disc_frac: float = 0.66) -> np.ndarray:
    """Render `text` onto a base ring icon (erase its number, draw the new one centered)."""
    im = Image.open(base_png).convert("RGBA").resize((size, size), Image.LANCZOS)
    cx, cy, R = _sign_geometry(np.asarray(im)[..., 3])
    d = ImageDraw.Draw(im)
    rw = disc_frac * R
    d.ellipse([cx - rw, cy - rw, cx + rw, cy + rw], fill=(255, 255, 255, 255))  # erase old number
    font = _fit_font(text, max_w=1.75 * rw, max_h=1.5 * rw)
    d.text((cx, cy), text, fill=(0, 0, 0, 255), font=font, anchor="mm")
    return np.asarray(im)


def load_template(class_name: str, marks_dir: str | Path, size: int = 280) -> np.ndarray:
    """RGBA (size,size,4) official icon for a subset class. Direct PNG if present, else a
    parametric render (pl*/il*/ph*/pm* number on the family base ring)."""
    direct = Path(marks_dir) / f"{class_name}.png"
    if direct.exists():
        return _to_rgba(Image.open(direct), size)
    m = re.match(r"[a-z]+", class_name or "")
    if m is None:
        raise ValueError(f"class_name inválido '{class_name}': esperado prefixo de família (ex. 'pl70')")
    fam = m.group(0)
    base = FAMILY_BASE.get(fam)
    if base is None:
        raise ValueError(f"'{class_name}': sem PNG direto e sem base paramétrica p/ família '{fam}'")
    base_path = Path(marks_dir) / f"{base}.png"
    if not base_path.exists():
        raise FileNotFoundError(f"base paramétrica '{base}.png' ausente em {marks_dir} "
                                f"(necessária p/ a classe '{class_name}')")
    number = class_name[len(fam):] + FAMILY_SUFFIX.get(fam, "")
    return render_parametric(base_path, number, size)


def pose_warp(rgba: np.ndarray, rng, max_rot: float = 15.0, max_persp: float = 0.18
              ) -> tuple[np.ndarray, np.ndarray]:
    """Random rotation + perspective on the RGBA template -> a new POSE. Returns (warped, H).

    This is what gives the diffuser pose diversity the img2img POC lacked: the Canny of the
    warped template carries the new pose, so the generated sign inherits it.
    """
    h, w = rgba.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = max_persp * min(h, w)
    dst = src + np.float32([[rng.uniform(0, j), rng.uniform(0, j)],
                            [rng.uniform(-j, 0), rng.uniform(0, j)],
                            [rng.uniform(-j, 0), rng.uniform(-j, 0)],
                            [rng.uniform(0, j), rng.uniform(-j, 0)]])
    ang = math.radians(rng.uniform(-max_rot, max_rot))
    c, s, cx, cy = math.cos(ang), math.sin(ang), w / 2.0, h / 2.0
    rot = np.array([[c, -s], [s, c]], np.float32)
    dst = (dst - [cx, cy]) @ rot.T + [cx, cy]
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(rgba, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderValue=(0, 0, 0, 0))
    return warped, H


def template_canny(rgba: np.ndarray, lo: int = 80, hi: int = 180) -> np.ndarray:
    """Canny edges (HxW uint8 {0,255}) of the template composited on WHITE — the ControlNet
    control image. White bg keeps the sign's outer outline as an edge (red ring vs white)."""
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    comp = np.where((alpha > 10)[..., None], rgb, 255).astype(np.uint8)
    gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, lo, hi)
