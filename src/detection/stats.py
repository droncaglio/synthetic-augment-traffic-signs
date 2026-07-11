"""Bootstrap CI of delta-AP, paired by seed, resampling TEST PANORAMAS.

Primary statistic (p3-plano-experimentos): for a (baseline arm, treatment arm) pair,
resample the test panoramas with replacement; for each replica recompute the GLOBAL AP
(not a per-image average) for both arms, per seed, and take the mean delta across seeds.
The percentile CI over replicas is the panorama-sampling uncertainty; "CI excludes 0"
is the guardrail for declaring a difference.

A "run" here is a mapping {panorama_id: [detections]} where a detection is
{"class_id", "conf", "box"=(cx,cy,w,h)} (panorama-normalized, already reconstructed +
globally NMS'd). Ground truth is {panorama_id: [{"class_id","box"}]}.
"""
from __future__ import annotations

import random
from statistics import mean, median

from detection.ap_by_size import Detection, GroundTruth, compute_ap_by_size


def ap_small_overall(ap_result: dict) -> float:
    v = ap_result["overall"]["small"]["ap50"]
    return 0.0 if v is None or v != v else v  # NaN/None -> 0


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0,100]) of an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100) * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac


def _metric_on_sample(run: dict, gts_by_pid: dict, class_names: list[str],
                      sampled_pids: list[str], metric) -> float:
    """Recompute AP metric over a (possibly resampled) list of panoramas.

    Each slot gets a fresh image_id so duplicated panoramas do not merge.
    """
    dets: list[Detection] = []
    gts: list[GroundTruth] = []
    for new_id, pid in enumerate(sampled_pids):
        for d in run.get(pid, []):
            cx, cy, w, h = d["box"]
            dets.append(Detection(new_id, d["class_id"], d["conf"], cx, cy, w, h))
        for g in gts_by_pid.get(pid, []):
            cx, cy, w, h = g["box"]
            gts.append(GroundTruth(new_id, g["class_id"], cx, cy, w, h))
    return metric(compute_ap_by_size(dets, gts, class_names))


def paired_seed_deltas(baseline_runs: list[dict], treatment_runs: list[dict],
                       gts_by_pid: dict, class_names: list[str],
                       panorama_ids: list[str], metric=ap_small_overall) -> dict:
    """Observed per-seed delta-AP on the full test (no resampling)."""
    deltas = [
        _metric_on_sample(t, gts_by_pid, class_names, panorama_ids, metric)
        - _metric_on_sample(b, gts_by_pid, class_names, panorama_ids, metric)
        for b, t in zip(baseline_runs, treatment_runs)
    ]
    return {
        "deltas": deltas,
        "mean": mean(deltas) if deltas else float("nan"),
        "median": median(deltas) if deltas else float("nan"),
        "n_positive": sum(1 for d in deltas if d > 0),
        "n_seeds": len(deltas),
    }


def bootstrap_delta_ap(baseline_runs: list[dict], treatment_runs: list[dict],
                       gts_by_pid: dict, class_names: list[str],
                       panorama_ids: list[str], metric=ap_small_overall,
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap CI of mean-across-seeds delta-AP, resampling panoramas."""
    rng = random.Random(seed)
    n = len(panorama_ids)
    replica_deltas: list[float] = []
    for _ in range(n_boot):
        sample = [panorama_ids[rng.randrange(n)] for _ in range(n)]
        per_seed = [
            _metric_on_sample(t, gts_by_pid, class_names, sample, metric)
            - _metric_on_sample(b, gts_by_pid, class_names, sample, metric)
            for b, t in zip(baseline_runs, treatment_runs)
        ]
        replica_deltas.append(mean(per_seed) if per_seed else float("nan"))
    replica_deltas.sort()
    return {
        "mean": mean(replica_deltas),
        "ci_low": _percentile(replica_deltas, 2.5),
        "ci_high": _percentile(replica_deltas, 97.5),
        "n_boot": n_boot,
    }


def ci_excludes_zero(boot: dict) -> bool:
    """True if the 95% CI does not contain 0 (guardrail to declare a difference)."""
    return boot["ci_low"] > 0 or boot["ci_high"] < 0
