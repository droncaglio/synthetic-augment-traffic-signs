"""Real-duplicate arm: the zero-point of content (pure oversampling).

Duplicates the source tile unchanged (same sign, same background, same position).
The model just sees the selected instances more times — no new content/context.
This is the ENIAC 'oversample_real' control: if a content arm doesn't beat this,
the synthetic content added nothing beyond repetition.
"""
from __future__ import annotations

import random

from detection.generators.base import ArmGenerator


class RealDuplicate(ArmGenerator):
    name = "real_duplicate"

    def make_tile(self, source: dict, rng: random.Random):
        img, labels, _ignores = self.load_tile(source["source_tile"])
        return img, labels
