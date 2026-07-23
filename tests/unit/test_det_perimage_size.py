"""Per-image (variable-size) normalization for DFG-style datasets.

Locks in that reconstruct + gts_by_pid normalize by an ISOTROPIC per-image reference
R = max(W, H) when the panorama dims are known, and fall back to the fixed panorama_size
otherwise (TT100K byte-identical). Isotropic (single R for both axes) is what keeps IoU
undistorted on non-square images.
"""
from detection.reconstruct import _pano_ref, map_det_to_panorama
from detection.report import gts_by_pid


def test_pano_ref_prefers_max_dim_when_present():
    entry = {"size": 640, "x_off": 0, "y_off": 0, "pano_w": 1920, "pano_h": 1080}
    assert _pano_ref(entry, 2048) == 1920.0            # max(W,H), not the scalar
    # retrocompat: no dims -> fixed panorama_size
    assert _pano_ref({"size": 640, "x_off": 0, "y_off": 0}, 2048) == 2048.0


def test_map_det_normalizes_by_max_dim():
    entry = {"size": 640, "x_off": 512, "y_off": 0, "pano_w": 1920, "pano_h": 1080}
    # tile-local center (0.5,0.5), size (0.1,0.1) -> panorama px (832, 320), ref=1920
    cx, cy, w, h = map_det_to_panorama((0.5, 0.5, 0.1, 0.1), entry, panorama_size=2048)
    assert abs(cx - 832 / 1920) < 1e-9
    assert abs(cy - 320 / 1920) < 1e-9
    assert abs(w - 64 / 1920) < 1e-9 and abs(h - 64 / 1920) < 1e-9


def test_gt_and_det_share_the_same_reference():
    """A GT box and a detection covering the SAME pixels must reconstruct to the same
    normalized box (else IoU matching silently fails on non-square images)."""
    W, H = 1920, 1080
    # GT: a sign at panorama px x[832..896], y[320..384] (64x64) on a 1920x1080 image.
    rec = {"id": "img", "width": W, "height": H,
           "objects": [{"category": "c", "xyxy": [832.0, 320.0, 896.0, 384.0]}]}
    gt = gts_by_pid({"img": rec}, ["img"], {"c": 0})["img"][0]["box"]  # (cx,cy,w,h) norm

    # A detection in the tile at x_off=512 that covers exactly those pixels:
    # tile-local px cx=(832+896)/2-512=352 -> 352/640=0.55; w=64/640=0.1
    entry = {"size": 640, "x_off": 512, "y_off": 0, "pano_w": W, "pano_h": H}
    det = map_det_to_panorama((352 / 640, (320 + 384) / 2 / 640, 64 / 640, 64 / 640),
                              entry, panorama_size=2048)
    for a, b in zip(gt, det):
        assert abs(a - b) < 1e-6, f"GT {gt} vs det {det} diverge -> IoU would be <1"
