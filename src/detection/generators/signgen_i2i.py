"""Sign-Gen (img2img) — POC generator: take a REAL sign crop and re-synthesize its
APPEARANCE with low/medium-strength SD1.5 img2img. Low strength preserves the glyph;
the diffusion adds pose/lighting/wear variation. This is the 'right button' probe — the
context ladder (background arms) was flat, so we test novelty of the SIGN itself.

*** GPU + MODEL (SD1.5 ~4GB). Heavy imports (torch/diffusers) are LAZY so this module
imports on a CPU box and the pure helpers (crop / letterbox / contact-sheet) stay
unit-testable. This is a POC probe, not a full arm yet. ***
"""
from __future__ import annotations

import random

import cv2
import numpy as np

from detection.generators.bg_photometric import _yolo_to_px

# safe, class-agnostic prompt: we DON'T let the model invent the glyph (low strength keeps
# the real number); the words only steer texture/lighting/realism.
DEFAULT_PROMPT = "a close-up photo of a traffic sign, realistic, detailed, sharp focus, daylight"
DEFAULT_NEG = "blurry, low quality, distorted, deformed, extra text, watermark, jpeg artifacts"


# -- pure helpers (no GPU, unit-tested) ---------------------------------------
def crop_bbox(tile_rgb: np.ndarray, bbox_norm) -> np.ndarray:
    """Cut the pixel region of a normalized [cx,cy,w,h] bbox out of a tile (RGB uint8)."""
    h, w = tile_rgb.shape[:2]
    x1, y1, x2, y2 = _yolo_to_px([float(v) for v in bbox_norm], w, h)
    return tile_rgb[y1:y2, x1:x2].copy()


def to_square(crop: np.ndarray, size: int = 512, pad: int = 114) -> tuple[np.ndarray, dict]:
    """Letterbox `crop` into a size×size square (aspect preserved, gray pad) for the diffuser.

    Returns (square, meta) — meta records the inner placement so `resize_back` undoes it
    exactly. Letterbox (not stretch) keeps the sign's true aspect through the diffusion.
    """
    ch, cw = crop.shape[:2]
    scale = size / max(ch, cw)
    nh, nw = max(1, round(ch * scale)), max(1, round(cw * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (nw, nh), interpolation=interp)
    sq = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    sq[top:top + nh, left:left + nw] = resized
    return sq, {"top": top, "left": left, "nh": nh, "nw": nw, "orig_hw": (ch, cw)}


def resize_back(square: np.ndarray, meta: dict) -> np.ndarray:
    """Undo `to_square`: cut the inner (letterboxed) region and resize to the original crop."""
    top, left, nh, nw = meta["top"], meta["left"], meta["nh"], meta["nw"]
    inner = square[top:top + nh, left:left + nw]
    ch, cw = meta["orig_hw"]
    interp = cv2.INTER_AREA if nh > ch else cv2.INTER_CUBIC
    return cv2.resize(inner, (cw, ch), interpolation=interp)


def _fit(img: np.ndarray, box: int, bg: int = 30) -> np.ndarray:
    """Center `img` inside a box×box canvas, aspect kept (for contact-sheet cells)."""
    h, w = img.shape[:2]
    s = min(box / h, box / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.full((box, box, 3), bg, np.uint8)
    t, l = (box - nh) // 2, (box - nw) // 2
    canvas[t:t + nh, l:l + nw] = r
    return canvas


def contact_sheet(rows: list[list], col_labels: list[str], *, cell: int = 160,
                  pad: int = 6, header: int = 26, note: int = 18) -> np.ndarray:
    """Assemble a labeled grid (RGB uint8). rows[r][c] = (img_rgb, note_str|None) or None.

    col_labels headers each column; the per-cell note is drawn under the image.
    """
    ncol, nrow = len(col_labels), len(rows)
    W = pad + ncol * (cell + pad)
    H = header + nrow * (cell + note + pad) + pad
    sheet = np.full((H, W, 3), 20, np.uint8)
    for c, lab in enumerate(col_labels):
        x = pad + c * (cell + pad)
        cv2.putText(sheet, lab, (x + 4, header - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for r, row in enumerate(rows):
        y = header + r * (cell + note + pad)
        for c, item in enumerate(row):
            if item is None:
                continue
            img, txt = item
            x = pad + c * (cell + pad)
            sheet[y:y + cell, x:x + cell] = _fit(img, cell)
            if txt:
                cv2.putText(sheet, txt, (x + 4, y + cell + note - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (170, 230, 170), 1, cv2.LINE_AA)
    return sheet


# -- GPU probe (lazy) ---------------------------------------------------------
class SignGenI2I:
    """SD1.5 img2img wrapper: crop -> square -> img2img(strength sweep) -> crop back."""

    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", *,
                 steps: int = 40, guidance: float = 7.5,
                 prompt: str = DEFAULT_PROMPT, neg_prompt: str = DEFAULT_NEG,
                 size: int = 512):
        self.model_id = model_id
        self.steps, self.guidance = steps, guidance
        self.prompt, self.neg_prompt = prompt, neg_prompt
        self.size = size
        self._pipe = None

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32
        # safety_checker=None: the NSFW filter can blank valid sign images (false positive);
        # the domain (traffic signs from real crops) makes genuine NSFW output negligible.
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype, safety_checker=None)
        pipe = pipe.to("cuda" if cuda else "cpu")
        pipe.enable_attention_slicing()          # fit the 8GB 2070S
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def generate_variants(self, crop_rgb: np.ndarray, strengths, rng: random.Random) -> dict:
        """{strength: variant_crop (orig size)} for one real crop, over a strength sweep."""
        import torch
        from PIL import Image
        pipe = self._load_pipe()
        sq, meta = to_square(crop_rgb, self.size)
        init = Image.fromarray(sq)
        out: dict[float, np.ndarray] = {}
        for s in strengths:
            # CPU generator for reproducibility: diffusers moves the noise to the pipe device,
            # and CUDA RNG is less deterministic across runs (matters for paper figures).
            g = torch.Generator(device="cpu").manual_seed(rng.randrange(2 ** 31))
            res = pipe(prompt=self.prompt, negative_prompt=self.neg_prompt, image=init,
                       strength=float(s), num_inference_steps=self.steps,
                       guidance_scale=self.guidance, generator=g).images[0]
            out[s] = resize_back(np.array(res.convert("RGB")), meta)
        return out
