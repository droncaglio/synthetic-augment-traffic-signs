"""Unit tests for the sign-gen img2img POC helpers (pure, no GPU/model)."""
import numpy as np

from detection.generators.signgen_i2i import (
    crop_bbox, to_square, resize_back, contact_sheet,
)


def test_crop_bbox_extracts_pixel_region():
    tile = np.zeros((640, 640, 3), np.uint8)
    tile[100:200, 300:400] = 255                       # a 100x100 white square
    # bbox center (350,150)/640, size 100/640 -> exactly that square
    crop = crop_bbox(tile, [350 / 640, 150 / 640, 100 / 640, 100 / 640])
    assert crop.shape == (100, 100, 3)
    assert (crop == 255).all()


def test_to_square_resize_back_roundtrip_preserves_shape():
    crop = np.random.default_rng(0).integers(0, 255, (37, 61, 3), dtype=np.uint8)
    sq, meta = to_square(crop, size=512)
    assert sq.shape == (512, 512, 3)
    assert meta["orig_hw"] == (37, 61)
    # inner region keeps aspect (wider than tall -> nw is the 512 side)
    assert meta["nw"] == 512 and meta["nh"] <= 512
    back = resize_back(sq, meta)
    assert back.shape == crop.shape                    # exact original crop size


def test_to_square_letterbox_is_centered_and_gray():
    crop = np.full((40, 20, 3), 255, np.uint8)         # tall -> pad left/right
    sq, meta = to_square(crop, size=512, pad=114)
    assert meta["nh"] == 512 and meta["nw"] < 512
    assert sq[0, 0].tolist() == [114, 114, 114]        # corner is gray pad
    assert meta["top"] == 0 and meta["left"] > 0       # centered horizontally


def test_contact_sheet_dims_and_type():
    cells = [np.zeros((20, 20, 3), np.uint8), np.full((30, 10, 3), 200, np.uint8)]
    rows = [[(cells[0], "orig"), (cells[1], "s0.3")]]
    sheet = contact_sheet(rows, ["original", "strength 0.3"], cell=160, pad=6,
                          header=26, note=18)
    assert sheet.dtype == np.uint8 and sheet.ndim == 3
    # width = pad + 2*(cell+pad); height = header + 1*(cell+note+pad) + pad
    assert sheet.shape[1] == 6 + 2 * (160 + 6)
    assert sheet.shape[0] == 26 + (160 + 18 + 6) + 6
