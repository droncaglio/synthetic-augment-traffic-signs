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
from detection.generators.masks import shape_alpha, sign_shape
from detection.generators.signgen_controlnet import SignGenControlNet
from detection.generators.templates import load_template
from detection.verifier import SignClassifier, load_crop


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
        # trust-construction for classes the verifier can't judge (too few real crops -> its
        # class head is degenerate, e.g. il100 with 1 crop). Read from the verifier report.
        self.skip_verify: set[str] = set()
        rep = Path(verifier_weights).parent / "verifier_report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            self.skip_verify = set(r.get("val_starved", []))
            self.skip_verify |= {k for k, v in (r.get("per_class_val_acc") or {}).items()
                                 if v is not None and v < 0.5}
        self._scan_stats = {"tiles": 0, "attempts": 0, "passed": 0, "regenerated": 0,
                            "fallback_best": 0, "no_template": 0,
                            "skip_verify": sorted(self.skip_verify), "conf_thr": conf_thr}
        # Classes without an official icon (nem PNG direto nem base paramétrica, ex. pcl: glifo
        # complexo) get a FALLBACK template from a real crop (RGB + silhueta) so EVERY class is
        # generatable -> signgen realizes the FULL allocation, identical count to copy_paste.
        self._fallback_tpl: dict[str, np.ndarray] = {}
        self._no_template: set[str] = set()
        self._build_templates()

    def _build_templates(self) -> None:
        from detection.generators.manifests import index_instances_by_class
        idx = None
        name2id = {v: k for k, v in self._id2name.items()}
        for nm in self._id2name.values():
            try:
                load_template(nm, self.marks_dir)
                continue                                          # official template exists
            except Exception:
                pass
            if idx is None:
                idx = index_instances_by_class(self.train_lbl, single_sign_only=True)
            pool = idx.get(name2id[nm], [])
            if not pool:
                self._no_template.add(nm)                         # truly nothing -> skip (rare)
                continue
            stem, box = max(pool, key=lambda tb: tb[1][2] * tb[1][3])   # largest real crop
            crop = load_crop(self.train_img, stem, box)
            h, w = crop.shape[:2]
            alpha = (shape_alpha(h, w, sign_shape(nm)) * 255).astype(np.uint8)  # class silhouette
            rgba = np.dstack([crop, alpha])
            self._fallback_tpl[nm] = cv2.resize(rgba, (280, 280), interpolation=cv2.INTER_AREA)
        if self._fallback_tpl:
            print(f"[signgen] template-fallback (de crop real) p/ classes sem ícone: "
                  f"{sorted(self._fallback_tpl)}")
        if self._no_template:
            print(f"[signgen] SEM template nem crop real (puladas): {sorted(self._no_template)}")

    def _template(self, name: str):
        return self._fallback_tpl.get(name) if name in self._fallback_tpl \
            else load_template(name, self.marks_dir)

    def _sign_crop(self, source: dict, tw: int, th: int, rng: random.Random):
        name = self._id2name.get(source["class_id"])
        if name is None:   # fail fast: a silent skip would break the copy_paste pairing invisibly
            raise KeyError(f"signgen_controlnet: class_id {source.get('class_id')} não está no "
                           f"subset.json ({sorted(self._id2name)}) — subset desatualizado?")
        if name in self._no_template:      # no icon AND no real crop (rare) -> can't generate
            self._scan_stats["no_template"] += 1
            return None
        tpl = self._template(name)
        cid, target = source["class_id"], max(tw, th)
        self._scan_stats["tiles"] += 1
        # BEST-OF-N: never skip -> every placement is filled, so signgen realizes the SAME count
        # as copy_paste (fair comparison). The verifier SELECTS the best of N candidates and its
        # telemetry (passed vs fallback_best) reports fidelity — it does NOT reduce the count.
        best_degr, best_p = None, -1.0
        for attempt in range(self.max_regen):
            self._scan_stats["attempts"] += 1
            v = self.gen.generate(tpl, 1, rng)[0]
            ys, xs = np.where(v["warped"][..., 3] > 10)
            if len(xs) == 0:
                continue
            crop = v["image"][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            degr = degrade_to_real(crop, target, rng)
            if name in self.skip_verify:                       # trust construction (ultra-rare)
                self._scan_stats["passed"] += 1
                return cv2.resize(degr, (tw, th), interpolation=cv2.INTER_AREA)
            pid, conf, probs = self.clf.predict(degr)
            if pid == cid and conf >= self.conf_thr:
                self._scan_stats["passed"] += 1
                if attempt > 0:
                    self._scan_stats["regenerated"] += 1
                return cv2.resize(degr, (tw, th), interpolation=cv2.INTER_AREA)
            j = self.clf.class_ids.index(cid) if cid in (self.clf.class_ids or []) else -1
            p_int = float(probs[j]) if j >= 0 else (conf if pid == cid else 0.0)
            if p_int > best_p:
                best_p, best_degr = p_int, degr
        if best_degr is None:              # every attempt had an empty warped (shouldn't happen)
            self._scan_stats["no_template"] += 1
            return None
        self._scan_stats["fallback_best"] += 1                 # none passed thr -> keep best-of-N
        return cv2.resize(best_degr, (tw, th), interpolation=cv2.INTER_AREA)
