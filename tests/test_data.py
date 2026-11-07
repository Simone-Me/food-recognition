import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from src.data import build_transforms, ids_to_multihot, mask_to_multihot
from src.utils import BLANK, load_config, need


def test_mask_to_multihot_area_threshold():
    mask = torch.zeros(10, 10, dtype=torch.uint8)
    mask.view(-1)[:50] = 3      # 50% of the image
    mask.view(-1)[50:52] = 7    # 2% of the image
    class_ids = [3, 7, 9]
    assert mask_to_multihot(mask, class_ids, 0.05).tolist() == [1, 0, 0]
    assert mask_to_multihot(mask, class_ids, 0.0).tolist() == [1, 1, 0]


def test_ids_to_multihot_ignores_unknown_ids():
    assert ids_to_multihot([3, 200], [3, 7]).tolist() == [1, 0]


def test_geometric_augmentation_keeps_mask_aligned():
    # Left half red + class 5, right half black + background: after crop/flip, red pixels must sit on class 5.
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32, 0] = 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :32] = 5
    torch.manual_seed(0)
    basic = build_transforms("basic", 32, mean=(0, 0, 0), std=(1, 1, 1), crop_scale_min=0.3)   # geometry only
    for _ in range(20):
        x, m = basic(Image.fromarray(img), tv_tensors.Mask(torch.from_numpy(mask)))
        if (m == 5).any() and (m == 0).any():
            assert x[0][m == 5].mean() > 0.8 and x[0][m == 0].mean() < 0.2
    strong = build_transforms("strong", 32, mean=(0, 0, 0), std=(1, 1, 1), crop_scale_min=0.3)
    for _ in range(20):
        _, m = strong(Image.fromarray(img), tv_tensors.Mask(torch.from_numpy(mask)))
        assert set(torch.unique(m).tolist()) <= {0, 5}, "mask interpolated: values other than class ids"


def test_eval_transform_shape_and_dtype():
    t = build_transforms("none", 48, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    x = t(Image.fromarray(np.full((60, 90, 3), 128, dtype=np.uint8)))
    assert x.shape == (3, 48, 48) and x.dtype == torch.float32
    assert abs(float(x.mean())) < 0.05     # (128/255 - 0.5) / 0.25 ~ 0.008


def test_need_stops_on_blank(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(f"train:\n  lr: {BLANK}\n  epochs: 3\n")
    cfg = load_config(p)
    assert need(cfg, "train.epochs") == 3
    try:
        need(cfg, "train.lr")
        raise AssertionError("need() should stop on a blank value")
    except SystemExit as e:
        assert "train.lr" in str(e)


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb:\n  c: 2\n  d: 3\n")
    (tmp_path / "exp.yaml").write_text("base: base.yaml\nb:\n  c: 20\n")
    cfg = load_config(tmp_path / "exp.yaml", ["b.d=30"])
    assert cfg["a"] == 1 and cfg["b"] == {"c": 20, "d": 30}
