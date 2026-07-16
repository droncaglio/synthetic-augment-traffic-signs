"""Unit tests for the Stage-2 arm generators."""
import random

import numpy as np
from PIL import Image

from detection.generators.real_duplicate import RealDuplicate
from detection.generators.bg_photometric import BgPhotometric
from detection.generators.photometric_full import PhotometricFull
from detection.generators.copy_paste import CopyPaste
from detection.generators.copy_paste_mask import CopyPasteMask
from detection.generators.masks import sign_shape, shape_alpha


def _fake_train(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    Image.fromarray(np.full((640, 640, 3), 100, np.uint8)).save(timg / "t0.jpg")
    (tlbl / "t0.txt").write_text("0 0.5 0.5 0.1 0.1")
    return tmp_path


def test_real_duplicate_generates_and_preserves_labels(tmp_path):
    _fake_train(tmp_path)
    sources = [{"class_id": 0, "source_tile": "t0", "bbox": [0.5, 0.5, 0.1, 0.1]}] * 2
    gen = RealDuplicate(tmp_path, seed=1)
    out = tmp_path / "arms" / "real_duplicate"
    manifest = gen.generate(sources, out)

    imgs = sorted((out / "images").glob("*.jpg"))
    assert len(imgs) == 2                                   # 2 sources -> 2 tiles
    # duplicated tile keeps the source label verbatim
    lbl = sorted((out / "labels").glob("*.txt"))[0].read_text().strip()
    assert lbl == "0 0.5 0.5 0.1 0.1"
    # duplicated image is pixel-identical to the source (pure oversampling)
    assert np.array_equal(np.asarray(Image.open(imgs[0])),
                          np.full((640, 640, 3), 100, np.uint8))
    assert manifest["allocated_per_class"] == {0: 2}
    assert manifest["realized_per_class"] == {0: 2}
    assert manifest["n_tiles_written"] == 2


def test_generate_resume_skips_existing_and_keeps_manifest_correct(tmp_path):
    _fake_train(tmp_path)
    sources = [{"class_id": 0, "source_tile": "t0", "bbox": [0.5, 0.5, 0.1, 0.1]}] * 3
    out = tmp_path / "arms" / "real_duplicate"
    RealDuplicate(tmp_path, seed=1).generate(sources[:1], out)   # 1st run: 1 tile
    assert len(list((out / "images").glob("*.jpg"))) == 1

    # resume with the full list -> keeps the existing tile, adds the 2 missing ones,
    # and the manifest re-counts the existing tile from disk (correct totals).
    m = RealDuplicate(tmp_path, seed=1).generate(sources, out, resume=True)
    assert len(list((out / "images").glob("*.jpg"))) == 3
    assert m["n_tiles_written"] == 3
    assert m["realized_per_class"] == {0: 3}


def test_bg_photometric_preserves_sign_perturbs_background(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    # gradient background (so any gamma/contrast perturbation is visible) + flat sign
    grad = np.tile(np.arange(640, dtype=np.uint8), (640, 1))
    arr = np.stack([grad, grad, grad], axis=-1).copy()
    arr[300:340, 300:340] = 200                         # the sign region (distinct, flat)
    Image.fromarray(arr).save(timg / "t0.jpg")
    # sign bbox ~ (320/640, 320/640, 40/640, 40/640)
    (tlbl / "t0.txt").write_text("0 0.5 0.5 0.0625 0.0625")

    loaded = np.asarray(Image.open(timg / "t0.jpg").convert("RGB"))  # JPEG as the gen sees it
    gen = BgPhotometric(tmp_path, seed=3)
    img, labels = gen.make_tile({"class_id": 0, "source_tile": "t0",
                                 "bbox": [0.5, 0.5, 0.0625, 0.0625]}, random.Random(3))
    # sign pixels pixel-exact (vs the loaded tile); background perturbed
    assert np.array_equal(img[300:340, 300:340], loaded[300:340, 300:340])
    assert not np.array_equal(img[:100, :100], loaded[:100, :100])
    assert labels == ["0 0.5 0.5 0.0625 0.0625"]


def test_photometric_full_perturbs_whole_tile_including_sign(tmp_path):
    # same fixture as bg_photometric, but photometric_full must perturb the SIGN too
    # (physically faithful: fog covers the sign). Contrast with bg_photometric above.
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    grad = np.tile(np.arange(640, dtype=np.uint8), (640, 1))
    arr = np.stack([grad, grad, grad], axis=-1).copy()
    arr[300:340, 300:340] = 200
    Image.fromarray(arr).save(timg / "t0.jpg")
    (tlbl / "t0.txt").write_text("0 0.5 0.5 0.0625 0.0625")

    loaded = np.asarray(Image.open(timg / "t0.jpg").convert("RGB"))
    src = {"class_id": 0, "source_tile": "t0", "bbox": [0.5, 0.5, 0.0625, 0.0625]}
    gen = PhotometricFull(tmp_path, seed=3)
    # preserve NOTHING -> mask is all-False
    assert not gen._preserve_mask(["0 0.5 0.5 0.0625 0.0625"], 640, 640, src).any()
    img, labels = gen.make_tile(src, random.Random(3))
    # BOTH sign and background are perturbed (unlike bg_photometric, which froze the sign)
    assert not np.array_equal(img[300:340, 300:340], loaded[300:340, 300:340])
    assert not np.array_equal(img[:100, :100], loaded[:100, :100])
    assert labels == ["0 0.5 0.5 0.0625 0.0625"]                # label still valid


def test_copy_paste_relocates_sign_into_background(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    src = np.full((640, 640, 3), 50, np.uint8)
    src[300:340, 300:340] = 222                         # sign crop (distinct color)
    Image.fromarray(src).save(timg / "src.jpg")
    (tlbl / "src.txt").write_text("0 0.5 0.5 0.0625 0.0625")
    Image.fromarray(np.full((640, 640, 3), 10, np.uint8)).save(timg / "bg.jpg")
    (tlbl / "bg.txt").write_text("")                    # background tile (no labels)

    gen = CopyPaste(tmp_path, seed=5)
    entry = {"class_id": 0, "source_tile": "src", "bbox": [0.5, 0.5, 0.0625, 0.0625],
             "recipient_tile": "bg", "place": [0.25, 0.25, 0.0625, 0.0625]}
    img, labels = gen.make_tile(entry, random.Random(5))
    # sign now around (0.25,0.25) on the background; a new label added
    assert len(labels) == 1 and labels[0].startswith("0 ")
    assert (img == 222).any()                           # pasted sign present (feather center)
    assert (img == 10).any()                            # background base present


def test_assign_placements_realistic_uses_donor_bbox():
    from detection.generators.manifests import assign_placements_realistic
    sources = [{"class_id": 0, "source_tile": "s0", "bbox": [0.5, 0.5, 0.1, 0.1]}]
    donors = [("d0", [0.3, 0.7, 0.05, 0.06]), ("d1", [0.8, 0.2, 0.04, 0.04])]
    a = assign_placements_realistic(sources, donors, seed=1)
    assert a == assign_placements_realistic(sources, donors, seed=1)     # deterministic (pairing)
    e = a[0]
    assert e["recipient_tile"] in ("d0", "d1")
    assert e["place"] == dict(donors)[e["recipient_tile"]]               # place = donor's real bbox
    assert e["class_id"] == 0 and e["source_tile"] == "s0"               # source preserved


def test_copy_paste_realistic_drops_covered_recipient_label(tmp_path):
    # realistic placement: paste over the donor's REAL sign -> its label is dropped; a far one stays
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    src = np.full((640, 640, 3), 50, np.uint8)
    src[300:340, 300:340] = 222
    Image.fromarray(src).save(timg / "src.jpg")
    (tlbl / "src.txt").write_text("0 0.5 0.5 0.0625 0.0625")
    Image.fromarray(np.full((640, 640, 3), 10, np.uint8)).save(timg / "don.jpg")
    (tlbl / "don.txt").write_text("3 0.25 0.25 0.0625 0.0625\n5 0.8 0.8 0.05 0.05")  # 2 real signs
    gen = CopyPaste(tmp_path, seed=5)
    entry = {"class_id": 0, "source_tile": "src", "bbox": [0.5, 0.5, 0.0625, 0.0625],
             "recipient_tile": "don", "place": [0.25, 0.25, 0.0625, 0.0625]}   # cover the class-3
    img, labels = gen.make_tile(entry, random.Random(5))
    assert sorted(int(ln.split()[0]) for ln in labels) == [0, 5]        # 3 dropped, 5 kept, 0 added


def test_copy_paste_none_on_empty_crop(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    Image.fromarray(np.full((640, 640, 3), 50, np.uint8)).save(timg / "src.jpg")
    (tlbl / "src.txt").write_text("0 0.5 0.5 0.06 0.06")
    Image.fromarray(np.full((640, 640, 3), 10, np.uint8)).save(timg / "bg.jpg")
    (tlbl / "bg.txt").write_text("")
    gen = CopyPaste(tmp_path, seed=1)
    entry = {"class_id": 0, "source_tile": "src", "bbox": [0.5, 0.5, 0.0, 0.0],  # zero area
             "recipient_tile": "bg", "place": [0.5, 0.5, 0.05, 0.05]}
    assert gen.make_tile(entry, random.Random(1)) is None



def test_sign_shape_taxonomy():
    assert sign_shape("w57") == "triangle"      # advertência
    assert sign_shape("il60") == "circle"       # velocidade mínima (il antes de i)
    assert sign_shape("i5") == "rectangle"      # indicação retangular
    assert sign_shape("pl60") == "circle"
    assert sign_shape("pn") == "circle"


def test_shape_alpha_circle_and_triangle_drop_corners():
    a = shape_alpha(40, 40, "circle")
    assert a[0, 0] == 0.0 and a[-1, -1] == 0.0          # cantos fora do círculo
    assert a[20, 20] == 1.0                             # centro dentro
    assert shape_alpha(40, 40, "rectangle").sum() > a.sum()   # círculo mais justo que retângulo
    t = shape_alpha(40, 40, "triangle")
    assert t[2, 2] == 0.0                               # topo-esquerda fora do triângulo (ápice p/ cima)
    assert t[-3, 20] == 1.0                             # base (embaixo) dentro


def test_copy_paste_mask_removes_rectangular_halo(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    src = np.full((640, 640, 3), 50, np.uint8)
    src[280:360, 280:360] = 222                          # crop = placa clara (bbox inteira)
    Image.fromarray(src).save(timg / "src.jpg")
    (tlbl / "src.txt").write_text("7 0.5 0.5 0.125 0.125")
    Image.fromarray(np.full((640, 640, 3), 10, np.uint8)).save(timg / "bg.jpg")
    (tlbl / "bg.txt").write_text("")

    gen = CopyPasteMask(tmp_path, seed=5)
    gen._id2name = {7: "w57"}                            # injeta: classe 7 = triângulo
    entry = {"class_id": 7, "source_tile": "src", "bbox": [0.5, 0.5, 0.125, 0.125],
             "recipient_tile": "bg", "place": [0.5, 0.5, 0.125, 0.125]}
    img, labels = gen.make_tile(entry, random.Random(5))
    assert len(labels) == 1 and labels[0].startswith("7 ")
    tw = th = int(round(0.125 * 640))                    # 80
    px1 = int(round(0.5 * 640 - tw / 2)); py1 = int(round(0.5 * 640 - th / 2))
    # canto superior (fora do triângulo) = FUNDO (10), não a placa (222) -> halo removido
    assert img[py1 + 3, px1 + 3].max() < 60
    # centro = placa presente
    assert img[py1 + th // 2, px1 + tw // 2].max() > 150
    # base do triângulo (embaixo) = placa
    assert img[py1 + th - 4, px1 + tw // 2].max() > 150
