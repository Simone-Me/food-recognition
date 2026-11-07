import numpy as np

from src.stats import bootstrap_ci, mcnemar_test, paired_bootstrap_delta

acc = lambda pred, y: float((pred == y).mean())


def test_identical_models_delta_contains_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    pred = np.where(rng.random(2000) < 0.8, y, 1 - y)
    d, lo, hi = paired_bootstrap_delta(acc, pred, pred.copy(), y, n_boot=300)
    assert d == 0 and lo <= 0 <= hi


def test_better_model_delta_excludes_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    a = np.where(rng.random(2000) < 0.85, y, 1 - y)
    b = a.copy()
    flip = rng.choice(np.flatnonzero(a == y), 100, replace=False)   # b is wrong on 5% more images
    b[flip] = 1 - b[flip]
    d, lo, hi = paired_bootstrap_delta(acc, a, b, y, n_boot=300)
    assert lo > 0


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    pred = np.where(rng.random(500) < 0.7, y, 1 - y)
    p, lo, hi = bootstrap_ci(acc, pred, y, n_boot=300)
    assert lo <= p <= hi and hi - lo < 0.15


def test_mcnemar():
    a = np.array([1] * 100 + [0] * 0, dtype=bool)
    assert mcnemar_test(a, a) == 1.0
    b = a.copy()
    b[:40] = False
    assert mcnemar_test(a, b) < 0.01
