import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.losses import AsymmetricLoss
from src.metrics import (average_precision, best_thresholds, ece_binary, f1_at_thresholds, fit_temperature,
                         map_macro, topk_accuracy)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 12)) < 0.2).astype(float)
    p = np.clip(0.6 * y + 0.5 * rng.random((300, 12)), 0, 1)
    return p, y


def test_ap_matches_sklearn(data):
    p, y = data
    for k in range(y.shape[1]):
        assert average_precision(p[:, k], y[:, k]) == pytest.approx(average_precision_score(y[:, k], p[:, k]))


def test_ap_with_ties_matches_sklearn(data):
    p, y = data
    q = np.round(p * 8) / 8          # int8-like outputs: many ties
    for k in range(y.shape[1]):
        assert average_precision(q[:, k], y[:, k]) == pytest.approx(average_precision_score(y[:, k], q[:, k]))


def test_map_macro_matches_sklearn_and_skips_empty_classes(data):
    p, y = data
    assert map_macro(p, y) == pytest.approx(average_precision_score(y, p, average="macro"))
    y2 = y.copy()
    y2[:, 0] = 0
    assert np.isfinite(map_macro(p, y2))


def test_thresholds_and_f1(data):
    p, y = data
    thr = best_thresholds(p, y)
    assert thr.shape == (12,)
    assert f1_at_thresholds(p, y, thr)["f1_micro"] >= f1_at_thresholds(p, y, np.full(12, 0.99))["f1_micro"]


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 500, dtype=float)
    assert ece_binary(np.full(1000, 0.5), y) == pytest.approx(0.0, abs=1e-9)
    assert ece_binary(np.where(y == 1, 0.0, 1.0), y) == pytest.approx(1.0)


def test_temperature_recovers_scale():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, size=(4000, 5))
    y = (rng.random(z.shape) < 1 / (1 + np.exp(-z))).astype(float)   # calibrated at T=1
    assert fit_temperature(3 * z, y, "multilabel") == pytest.approx(3.0, rel=0.1)


def test_topk():
    p = np.array([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]])
    assert topk_accuracy(p, np.array([2, 2]), 1) == 0.0
    assert topk_accuracy(p, np.array([2, 2]), 2) == 1.0


def test_asl_reduces_to_bce_without_focusing():
    import torch
    import torch.nn.functional as F

    z = torch.randn(8, 6)
    y = (torch.rand(8, 6) < 0.3).float()
    asl = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)(z, y)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="sum") / 8
    assert asl.item() == pytest.approx(bce.item(), rel=1e-4)
