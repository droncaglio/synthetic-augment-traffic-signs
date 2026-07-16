"""Sign class-verifier: a classifier trained on the REAL single-sign crops (21 subset
classes) that, run on GENERATED samples, gives the per-class VALID-LABEL RATE — the number
the review demanded ("classe verificada a posteriori", not "garantida por construção"). It
also doubles as the rejection filter of the future signgen arm (generate -> classify ->
reject if off-class), mirroring diffusion_bg's hallucination scan.

*** torch/torchvision are LAZY (imported inside methods) so this module imports on CPU and
the valid_rate logic is unit-testable with a fake net. Reuses the ConvNeXt-Tiny pattern from
scripts/analyze_backgrounds.py. ***
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from detection.generators.manifests import index_instances_by_class
from detection.generators.signgen_i2i import crop_bbox

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_crop_index(labels_dir: str | Path) -> list[tuple[str, int, list]]:
    """(tile_stem, class_id, [cx,cy,w,h]) for every single-sign train instance (pure, no images).

    Reuses index_instances_by_class(single_sign_only=True) — the same real-instance pool the
    generators draw sources from, so the verifier trains on exactly the real signs."""
    idx = index_instances_by_class(labels_dir, single_sign_only=True)
    return [(stem, cid, box) for cid, items in idx.items() for (stem, box) in items]


def build_convnext(n_classes: int, pretrained: bool = True):
    """ConvNeXt-Tiny with a fresh n_classes head (torchvision). Shared by train + inference."""
    import torch.nn as nn
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    net = convnext_tiny(weights=w)
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, n_classes)  # 768 -> n_classes
    return net


def imagenet_transform(train: bool = False):
    """ImageNet eval transform; train adds light aug (NO flip — signs are directional)."""
    import torchvision.transforms as T
    steps = [T.Resize(232), T.CenterCrop(224)]
    if train:
        steps += [T.RandomRotation(10), T.ColorJitter(0.2, 0.2, 0.2)]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class SignClassifier:
    """Inference wrapper: predict the class of a sign crop + measure valid-rate of a batch."""

    def __init__(self, class_ids: list[int] | None = None, weights_path: str | Path | None = None,
                 device: str | None = None):
        self.class_ids = list(class_ids) if class_ids else None  # model index -> real class_id
        self.weights_path = str(weights_path) if weights_path else None
        self.device = device
        self._net = None
        self._tf = None

    def _load_net(self):
        if self._net is not None:
            return self._net
        import torch
        if self.weights_path:
            ckpt = torch.load(self.weights_path, map_location="cpu")
            self.class_ids = list(ckpt["class_ids"])
            net = build_convnext(len(self.class_ids), pretrained=False)
            net.load_state_dict(ckpt["state_dict"])
        else:
            assert self.class_ids, "class_ids required when no weights_path"
            net = build_convnext(len(self.class_ids), pretrained=True)
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tf = imagenet_transform(train=False)
        self._net = net.to(self._device).eval()
        return self._net

    def predict(self, crop_rgb: np.ndarray) -> tuple[int, float, np.ndarray]:
        """(predicted class_id, confidence, full prob vector) for one RGB crop."""
        import torch
        from PIL import Image
        net = self._load_net()
        x = self._tf(Image.fromarray(crop_rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(net(x), dim=1)[0]
        i = int(probs.argmax())
        return self.class_ids[i], float(probs[i]), probs.cpu().numpy()

    def valid_rate(self, crops: list[np.ndarray], intended_class_id: int,
                   conf_thr: float = 0.0) -> dict:
        """Per-class QA over `crops` all of the SAME intended class.

        top1_acc = frac(argmax == intended); accept_rate = top1 AND conf>=thr (the arm's keep
        rule); mean_conf over the intended-class probability."""
        if not crops:
            return {"n": 0, "top1_acc": 0.0, "accept_rate": 0.0, "mean_conf": 0.0}
        idx_map = {c: i for i, c in enumerate(self.class_ids)} if self.class_ids else {}
        j = idx_map.get(intended_class_id, -1)     # index of the intended class (once)
        top1 = accept = 0
        confs = []
        for c in crops:
            cid, conf, probs = self.predict(c)
            hit = cid == intended_class_id
            top1 += hit
            accept += hit and conf >= conf_thr
            # confidence assigned to the INTENDED class (not the argmax) — honest signal
            confs.append(float(probs[j]) if j >= 0 else conf)
        n = len(crops)
        return {"n": n, "top1_acc": top1 / n, "accept_rate": accept / n,
                "mean_conf": float(np.mean(confs))}


def load_crop(images_dir: str | Path, stem: str, bbox: list) -> np.ndarray:
    """Load a train tile and cut the sign crop (for the training dataset / QA)."""
    from PIL import Image
    tile = np.asarray(Image.open(Path(images_dir) / f"{stem}.jpg").convert("RGB"))
    return crop_bbox(tile, bbox)
