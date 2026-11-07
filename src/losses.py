"""Multi-label losses for the ingredient head: BCE, BCE + pos_weight, Asymmetric Loss (ASL)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.utils import need

LOSSES = ("bce", "bce_posw", "asl")


class AsymmetricLoss(nn.Module):
    """Ridnik et al., ICCV 2021.

    - gamma_neg > gamma_pos: easy negatives (the ~95% of zeros) are down-weighted more than positives.
    - clip ("probability shifting"): negatives with p < clip give zero loss (also hides some label noise).
    """

    def __init__(self, gamma_neg: float, gamma_pos: float, clip: float, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.float(), targets.float()   # stay in fp32 under autocast
        p = torch.sigmoid(logits)
        p_neg = 1 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
        loss = targets * torch.log(p.clamp(min=self.eps)) + (1 - targets) * torch.log(p_neg.clamp(min=self.eps))
        pt = p * targets + p_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss = loss * torch.pow(1 - pt, gamma)
        return -loss.sum() / logits.shape[0]


class BCE(nn.Module):
    """BCEWithLogitsLoss computed in fp32 (sigmoid + log fused: numerically stable)."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.loss(logits.float(), targets.float())


def pos_weight_from_counts(counts: np.ndarray, n_images: int, max_weight: float) -> torch.Tensor:
    """pos_weight_k = (#negatives / #positives), capped so that very rare classes do not explode."""
    pos = np.maximum(counts, 1)
    return torch.as_tensor(np.minimum((n_images - pos) / pos, max_weight), dtype=torch.float32)


def build_ingredient_loss(cfg: dict, train_counts: np.ndarray, n_train: int) -> nn.Module:
    name = need(cfg, "train.loss")
    if name == "bce":
        return BCE()
    if name == "bce_posw":
        return BCE(pos_weight_from_counts(train_counts, n_train, need(cfg, "train.pos_weight_max")))
    if name == "asl":
        return AsymmetricLoss(need(cfg, "train.asl.gamma_neg"), need(cfg, "train.asl.gamma_pos"),
                              need(cfg, "train.asl.clip"))
    raise ValueError(f"train.loss must be one of {LOSSES}, got {name!r}")


def build_dish_loss(cfg: dict) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=need(cfg, "multitask.label_smoothing"))
