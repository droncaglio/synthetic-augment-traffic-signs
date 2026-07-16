"""Sign-Gen arm: paste a SYNTHETIC sign (template+ControlNet) into a real background tile —
the flagship arm the flat context ladder pointed at ("generate NEW sign appearance"). It is a
CopyPaste subclass that only swaps the pasted crop: instead of relocating a REAL sign it
GENERATES one of the same class, at the SAME placement copy_paste uses (shared sources + seed +
deterministic assign_placements) → paired 1:1 with copy_paste, so the only variable is
real-relocated vs synthetic. Per instance: generate (color-anchor) → degrade to real size
(REQ2) → verify/reject with the class-verifier (REQ1) → paste (reuses CopyPaste).

*** GPU + MODELS (ControlNet + ConvNeXt verifier). Heavy work is lazy inside SignGenControlNet /
SignClassifier, so this module imports on CPU. ***
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from detection.generators.copy_paste import CopyPaste
from detection.generators.degrade import degrade_to_real
from detection.generators.signgen_controlnet import SignGenControlNet
from detection.generators.templates import load_template
from detection.verifier import SignClassifier


class SignGenArm(CopyPaste):
    name = "signgen_controlnet"

    def __init__(self, tiles_dir, seed: int = 0, *, verifier_weights, marks_dir,
                 max_regen: int = 4, strength: float = 0.6, conf_thr: float = 0.5,
                 steps: int = 30):
        super().__init__(tiles_dir, seed)
        self.marks_dir = str(marks_dir)
        self.max_regen, self.conf_thr = max_regen, conf_thr
        self.gen = SignGenControlNet(color_anchor=True, strength=strength, steps=steps)
        self.clf = SignClassifier(weights_path=verifier_weights)
        sub = json.loads((self.tiles_dir.parent / "prepared" / "subset.json").read_text())
        self._id2name = {c["id"]: c["name"] for c in sub["classes"]}
        # classes without any official template (nem PNG direto nem base paramétrica, ex. pcl)
        # can't be generated -> skip their sources gracefully (don't crash the whole pass).
        self._no_template = set()
        for nm in self._id2name.values():
            try:
                load_template(nm, self.marks_dir)
            except Exception:
                self._no_template.add(nm)
        if self._no_template:
            print(f"[signgen] classes SEM template (puladas, sem sintético): "
                  f"{sorted(self._no_template)}")
        # trust-construction for classes the verifier can't judge (too few real crops -> its
        # class head is degenerate, e.g. il100 with 1 crop). Read from the verifier report.
        self.skip_verify: set[str] = set()
        rep = Path(verifier_weights).parent / "verifier_report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            self.skip_verify = set(r.get("val_starved", []))
            self.skip_verify |= {k for k, v in (r.get("per_class_val_acc") or {}).items()
                                 if v is not None and v < 0.5}
        self._scan_stats = {"tiles": 0, "attempts": 0, "regenerated": 0, "rejected": 0,
                            "no_template": 0, "skip_verify": sorted(self.skip_verify),
                            "conf_thr": conf_thr}

    def _sign_crop(self, source: dict, tw: int, th: int, rng: random.Random):
        name = self._id2name.get(source["class_id"])
        if name is None:   # fail fast: a silent skip would break the copy_paste pairing invisibly
            raise KeyError(f"signgen_controlnet: class_id {source.get('class_id')} não está no "
                           f"subset.json ({sorted(self._id2name)}) — subset desatualizado?")
        if name in self._no_template:   # classe sem template (ex. pcl) -> sem sintético (skip)
            self._scan_stats["no_template"] += 1
            return None
        tpl = load_template(name, self.marks_dir)
        target = max(tw, th)
        self._scan_stats["tiles"] += 1
        for attempt in range(self.max_regen):
            self._scan_stats["attempts"] += 1
            v = self.gen.generate(tpl, 1, rng)[0]
            ys, xs = np.where(v["warped"][..., 3] > 10)
            if len(xs) == 0:
                continue
            crop = v["image"][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            degr = degrade_to_real(crop, target, rng)
            if name in self.skip_verify:                       # trust construction (ultra-rare)
                return cv2.resize(degr, (tw, th), interpolation=cv2.INTER_AREA)
            pid, conf, _ = self.clf.predict(degr)
            if pid == source["class_id"] and conf >= self.conf_thr:
                if attempt > 0:
                    self._scan_stats["regenerated"] += 1
                return cv2.resize(degr, (tw, th), interpolation=cv2.INTER_AREA)
        self._scan_stats["rejected"] += 1                      # all attempts off-class -> skip
        return None
