"""Budget multiplier helpers (K) for the detection allocation.

The deterministic tag ``bm{int(round(K*100)):03d}`` namespaces artifacts by
budget (allocation manifests, synth dirs, experiment dirs).

Conventions:
  K=0.25 -> "bm025"
  K=0.50 -> "bm050"   (fixed default for Paper 3, see p3-plano-experimentos)
  K=1.00 -> "bm100"

K is the fraction of the subset-class training instances turned into synthetic
budget: ``B = round(K * N_train_subset_instances)``. Adapted verbatim from the
Paper 1 (synthetic-allocation-derma) budget helper — the tag semantics are
dataset-agnostic.
"""
from __future__ import annotations

import re

# Fixed default for Paper 3 (K=0.5). Kept as fallback when a batch omits K.
BUDGET_MULTIPLIER_DEFAULT: float = 0.5

# Valid range bounded by 3 digits (bm999 = K=9.99); lower bound > 0.
_K_MIN: float = 0.01
_K_MAX: float = 9.99

_TAG_RX = re.compile(r"^bm(?P<digits>\d{3})$")


def budget_tag(K: float) -> str:
    """K (float) -> tag ``bm###`` (3 zero-padded digits).

    Examples:
        >>> budget_tag(0.5)
        'bm050'
        >>> budget_tag(0.25)
        'bm025'
    """
    if not isinstance(K, (int, float)):
        raise ValueError(f"budget_tag: K must be numeric, got {type(K).__name__}")
    if K < _K_MIN or K > _K_MAX:
        raise ValueError(f"budget_tag: K={K} outside range [{_K_MIN}, {_K_MAX}]")
    return f"bm{int(round(K * 100)):03d}"


def parse_budget_tag(tag: str) -> float:
    """tag ``bm###`` -> K (float). Inverse of :func:`budget_tag`.

    Examples:
        >>> parse_budget_tag("bm050")
        0.5
    """
    if not isinstance(tag, str):
        raise ValueError(f"parse_budget_tag: tag must be str, got {type(tag).__name__}")
    m = _TAG_RX.match(tag)
    if not m:
        raise ValueError(f"parse_budget_tag: tag {tag!r} invalid, expected 'bm###'")
    K = int(m.group("digits")) / 100.0
    if K < _K_MIN or K > _K_MAX:
        raise ValueError(f"parse_budget_tag: K={K} (from {tag!r}) outside valid range")
    return K


def is_budget_tag(s: str) -> bool:
    """True if s matches the ``bm###`` format (without range validation)."""
    return isinstance(s, str) and bool(_TAG_RX.match(s))
