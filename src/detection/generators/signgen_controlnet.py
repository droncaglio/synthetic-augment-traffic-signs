"""Sign-Gen (template + ControlNet) — POC generator: render a photorealistic in-the-wild sign
from the OFFICIAL TEMPLATE (marks/), with the CLASS PINNED by ControlNet-Canny. Pose diversity
comes from warping the template before the Canny. This is the lever the flat context ladder
pointed at: genuinely NEW sign appearance/pose, with the class guaranteed BY CONSTRUCTION
(exact label) — the thing img2img-of-the-crop (method #4) could not produce.

*** GPU + MODEL (SD1.5 ~4GB + ControlNet-canny ~1.4GB). Heavy imports (torch/diffusers) are
LAZY so this module imports on CPU; the geometry helpers live in `templates.py` (unit-tested).
POC probe, not a full arm yet. ***
"""
from __future__ import annotations

import random

import cv2
import numpy as np

from detection.generators.templates import pose_warp, template_canny

DEFAULT_PROMPT = ("a photo of a traffic sign mounted on a pole, city street, realistic, "
                  "natural daylight, sharp focus")
DEFAULT_NEG = ("cartoon, drawing, illustration, painting, blurry, low quality, distorted, "
               "deformed, extra text, watermark")


class SignGenControlNet:
    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 controlnet_id: str = "lllyasviel/sd-controlnet-canny", *,
                 steps: int = 30, guidance: float = 7.5, cn_scale: float = 1.0,
                 prompt: str = DEFAULT_PROMPT, neg_prompt: str = DEFAULT_NEG, size: int = 512):
        self.model_id, self.controlnet_id = model_id, controlnet_id
        self.steps, self.guidance, self.cn_scale = steps, guidance, cn_scale
        self.prompt, self.neg_prompt, self.size = prompt, neg_prompt, size
        self._pipe = None

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32
        cn = ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=dtype)
        # safety_checker=None: NSFW filter can blank valid sign renders (false positive).
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id, controlnet=cn, torch_dtype=dtype, safety_checker=None)
        pipe = pipe.to("cuda" if cuda else "cpu")
        pipe.enable_attention_slicing()        # fit the 8GB 2070S
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def make_control(self, template_rgba: np.ndarray, rng: random.Random):
        """Warp the template (new pose) -> Canny 3-ch control image (size×size). Returns
        (control_rgb, warped_rgba) — warped kept for the side-by-side display."""
        warped, _ = pose_warp(template_rgba, rng)
        edges = template_canny(warped)                       # HxW uint8 {0,255}
        ctrl = np.stack([edges] * 3, axis=-1)                # ControlNet wants 3-ch
        ctrl = cv2.resize(ctrl, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return ctrl, warped

    def generate(self, template_rgba: np.ndarray, n: int, rng: random.Random) -> list[dict]:
        """n variants, each a new pose+render: [{control, warped, image}]."""
        import torch
        from PIL import Image
        pipe = self._load_pipe()
        tpl = cv2.resize(template_rgba, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)
        out = []
        for _ in range(n):
            ctrl, warped = self.make_control(tpl, rng)
            g = torch.Generator(device="cpu").manual_seed(rng.randrange(2 ** 31))
            img = pipe(prompt=self.prompt, negative_prompt=self.neg_prompt,
                       image=Image.fromarray(ctrl),
                       num_inference_steps=self.steps, guidance_scale=self.guidance,
                       controlnet_conditioning_scale=float(self.cn_scale),
                       height=self.size, width=self.size, generator=g).images[0]
            out.append({"control": ctrl, "warped": warped,
                        "image": np.array(img.convert("RGB"))})
        return out
