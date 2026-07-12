"""Diffusion-Bg arm: real sign fixed, regenerate the BACKGROUND with FluxFill inpainting
(inverted mask). The expensive rung of the context ladder — synthetic coherent context.

*** GPU + MODEL heavy — validate/tune on the workstation, NOT unit-tested here. ***
Requires diffusers/peft/bitsandbytes + FLUX.1-Fill-dev (~heavy download). All heavy
imports are lazy so this module imports fine on a CPU box (the pipeline loads on the
first make_tile call and is cached).

Design (matches p3-plano-experimentos + code-review guardrails):
  1. Mask is INVERTED: white over the background, BLACK over every subset sign in the
     tile -> FluxFill regenerates only the background, sign pixels untouched.
  2. After generation, the ORIGINAL sign crops are alpha-composited back over their
     bboxes (guards against edge bleed) -> label stays exactly valid.
  3. Hallucination scan: a light detector is run over the generated tile OUTSIDE the
     sign bboxes; if it fires (conf >= scan_conf) the tile invented a sign -> reject and
     regenerate (up to max_regen), else fall back (skip -> caller can use bg_photometric).
  4. LoRA (optional) trained on TRAIN-only backgrounds keeps the context in-domain.
Anti-leak: source tiles + any LoRA backgrounds come only from the TRAIN split.
"""
from __future__ import annotations

import random

import numpy as np

from detection.generators.base import ArmGenerator, feather_alpha
from detection.generators.bg_photometric import _yolo_to_px


class DiffusionBg(ArmGenerator):
    name = "diffusion_bg"

    def __init__(self, tiles_dir, seed: int = 0, *,
                 model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
                 lora_dir: str | None = None,
                 prompt: str = "an empty urban street scene, road, buildings, sky, realistic photo",
                 steps: int = 30, guidance: float = 30.0,
                 scan_weights: str | None = None, scan_conf: float = 0.25,
                 max_regen: int = 3, imgsz: int = 640):
        super().__init__(tiles_dir, seed)
        self.model_id, self.lora_dir, self.prompt = model_id, lora_dir, prompt
        self.steps, self.guidance = steps, guidance
        self.scan_weights, self.scan_conf, self.max_regen = scan_weights, scan_conf, max_regen
        self.imgsz = imgsz
        self._pipe = None
        self._scanner = None
        # scan telemetry -> manifest (evidence: scan is alive + rejection rate).
        # scan_detections>0 proves the scanner fires at all (it sees the real signs);
        # scan_fired = tiles where it detected a sign OUTSIDE the bbox (a hallucination).
        self._scan_stats = {"tiles": 0, "attempts": 0, "regenerated": 0, "rejected": 0,
                            "scan_fired": 0, "scan_detections": 0,
                            "scanner": scan_weights, "scan_conf": scan_conf}

    # -- lazy heavy resources (GPU) ---------------------------------------
    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import FluxFillPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig
        nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_id, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16)
        if self.lora_dir:
            from peft import PeftModel
            transformer = PeftModel.from_pretrained(transformer, str(self.lora_dir))
        self._pipe = FluxFillPipeline.from_pretrained(
            self.model_id, transformer=transformer, torch_dtype=torch.bfloat16)
        self._pipe.enable_model_cpu_offload()
        return self._pipe

    def _load_scanner(self):
        if self._scanner is None and self.scan_weights:
            from ultralytics import YOLO
            self._scanner = YOLO(str(self.scan_weights))
        return self._scanner

    # -- helpers ----------------------------------------------------------
    def _sign_boxes_px(self, labels, w, h):
        return [_yolo_to_px([float(v) for v in ln.split()[1:5]], w, h) for ln in labels]

    def _inverted_mask(self, labels, w, h):
        """PIL 'L' mask: 255 (regenerate) everywhere except sign bboxes = 0 (keep)."""
        from PIL import Image
        m = np.full((h, w), 255, np.uint8)
        for x1, y1, x2, y2 in self._sign_boxes_px(labels, w, h):
            m[y1:y2, x1:x2] = 0
        return Image.fromarray(m)

    def _hallucinated(self, out_img, sign_boxes) -> bool:
        """True if the scanner detects a sign OUTSIDE the preserved sign bboxes."""
        scanner = self._load_scanner()
        if scanner is None:
            return False  # no scanner configured -> can't check (warn upstream)
        h, w = out_img.shape[:2]
        r = scanner.predict(out_img, conf=self.scan_conf, verbose=False, save=False)
        b = r[0].boxes
        n = 0 if b is None else len(b)
        self._scan_stats["scan_detections"] += n  # >0 total => scanner is alive
        if n == 0:
            return False
        for cx, cy, bw, bh in b.xywhn.cpu().numpy():
            px, py = cx * w, cy * h
            if not any(x1 <= px <= x2 and y1 <= py <= y2 for x1, y1, x2, y2 in sign_boxes):
                self._scan_stats["scan_fired"] += 1
                return True  # a detection whose center is not on a real sign -> hallucination
        return False

    # -- arm --------------------------------------------------------------
    def make_tile(self, source: dict, rng: random.Random):
        import torch
        from PIL import Image
        img, labels, _ignores = self.load_tile(source["source_tile"])
        if not labels:
            return None
        h, w = img.shape[:2]
        pipe = self._load_pipe()
        mask = self._inverted_mask(labels, w, h)
        bg = Image.fromarray(img)
        sign_boxes = self._sign_boxes_px(labels, w, h)

        self._scan_stats["tiles"] += 1
        for attempt in range(self.max_regen):
            self._scan_stats["attempts"] += 1
            gen = torch.Generator("cpu").manual_seed(rng.randrange(2 ** 31))
            out = pipe(prompt=self.prompt, image=bg, mask_image=mask,
                       height=self.imgsz, width=self.imgsz,
                       num_inference_steps=self.steps, guidance_scale=self.guidance,
                       generator=gen).images[0]
            out = np.array(out.convert("RGB"))  # copy -> writable for the sign composite
            # composite the real sign crops back with a feathered edge (same blend as
            # copy_paste) -> no hard rectangular seam, sign core stays intact, label valid
            for (x1, y1, x2, y2) in sign_boxes:
                th, tw = y2 - y1, x2 - x1
                if th <= 0 or tw <= 0:
                    continue
                a = feather_alpha(th, tw)[..., None]
                region = out[y1:y2, x1:x2].astype(np.float32)
                crop = img[y1:y2, x1:x2].astype(np.float32)
                out[y1:y2, x1:x2] = (a * crop + (1 - a) * region).astype(np.uint8)
            if not self._hallucinated(out, sign_boxes):
                if attempt > 0:
                    self._scan_stats["regenerated"] += 1  # succeeded but needed retries
                return out, labels
        self._scan_stats["rejected"] += 1
        return None  # rejected max_regen times -> skip (caller may fall back)
