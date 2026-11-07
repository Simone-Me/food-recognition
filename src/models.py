"""Model creation, multi-task model (part D), head/backbone handling, profiling, export wrapper."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn

from src.utils import need

# Screening candidates (§A6). The two finalists are the first two.
CANDIDATES = [
    "tf_efficientnet_lite0",
    "mobilenetv3_large_100",
    "mobilenetv3_small_100",
    "efficientnet_b0",
    "mobilevit_xs",
    "convnext_atto",
]


class MultiTaskModel(nn.Module):
    """Part D: ONE shared backbone, two linear heads (ingredients = sigmoid, dish = softmax)."""

    def __init__(self, name: str, n_ingr: int, n_dish: int, pretrained: bool = True, drop_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.backbone.eval()
        with torch.no_grad():
            dim = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.backbone.train()
        self.drop = nn.Dropout(drop_rate)
        self.ingr_head = nn.Linear(dim, n_ingr)
        self.dish_head = nn.Linear(dim, n_dish)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.drop(self.backbone(x))
        return self.ingr_head(f), self.dish_head(f)


def create_model(cfg: dict, num_ingredients: int, num_dishes: int = 0, pretrained: bool | None = None) -> nn.Module:
    """Single entry point for every model. num_dishes > 0 builds the part-D multi-task model."""
    name = need(cfg, "model.name")
    drop = need(cfg, "model.drop_rate")
    pretrained = cfg["model"].get("pretrained", True) if pretrained is None else pretrained
    if num_dishes:
        return MultiTaskModel(name, num_ingredients, num_dishes, pretrained, drop)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_ingredients, drop_rate=drop)


def heads(model: nn.Module) -> list[nn.Module]:
    if isinstance(model, MultiTaskModel):
        return [model.ingr_head, model.dish_head]
    return [model.get_classifier()]


def ingredient_head(model: nn.Module) -> nn.Linear:
    return model.ingr_head if isinstance(model, MultiTaskModel) else model.get_classifier()


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
    """Discriminative LR: the pretrained backbone moves slower than the freshly initialised heads."""
    head_ids = {id(p) for h in heads(model) for p in h.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return [
        {"params": backbone, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay, "name": "backbone"},
        {"params": head, "lr": lr, "weight_decay": weight_decay, "name": "head"},
    ]


def freeze_backbone(model: nn.Module, frozen: bool) -> None:
    """Linear probing: no gradients in the backbone and BatchNorm statistics kept fixed."""
    head_ids = {id(p) for h in heads(model) for p in h.parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(not frozen)
    if frozen:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def init_bias_from_frequencies(linear: nn.Linear, freqs: np.ndarray) -> None:
    """bias_k = log(p_k / (1 - p_k)): at step 0 the sigmoid already outputs the class prior."""
    p = np.clip(freqs, 1e-4, 1 - 1e-4)
    with torch.no_grad():
        linear.bias.copy_(torch.as_tensor(np.log(p / (1 - p)), dtype=linear.bias.dtype))


def data_config(model: nn.Module) -> dict:
    """mean/std the pretrained weights expect (the normalisation MUST match, also in JS)."""
    m = model.backbone if isinstance(model, MultiTaskModel) else model
    dc = timm.data.resolve_data_config({}, model=m)
    return {"mean": list(dc["mean"]), "std": list(dc["std"]), "default_input_size": list(dc["input_size"])}


def profile(name: str, resolution: int, num_classes: int = 103) -> dict:
    """Params, mult-adds and estimated file sizes, without training."""
    from torchinfo import summary

    m = timm.create_model(name, pretrained=False, num_classes=num_classes).eval()
    s = summary(m, input_size=(1, 3, resolution, resolution), verbose=0)
    params = sum(p.numel() for p in m.parameters())
    return {
        "model": name,
        "resolution": resolution,
        "params_M": round(params / 1e6, 3),
        "mult_adds_G": round(s.total_mult_adds / 1e9, 3),
        "size_fp32_MB": round(params * 4 / 1e6, 2),
        "size_int8_MB_est": round(params / 1e6, 2),
        **{k: v for k, v in data_config(m).items() if k != "default_input_size"},
    }


class ExportWrapper(nn.Module):
    """What goes INSIDE the ONNX graph: model + temperature + sigmoid (ingredients) / softmax (dish).

    Normalisation stays OUTSIDE the graph; its mean/std go to model_meta.json, read by the JS code.
    """

    def __init__(self, model: nn.Module, t_ingr: float = 1.0, t_dish: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("t_ingr", torch.tensor(float(t_ingr)))
        self.register_buffer("t_dish", torch.tensor(float(t_dish)))

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out, tuple):
            ingr, dish = out
            return torch.sigmoid(ingr / self.t_ingr), torch.softmax(dish / self.t_dish, dim=1)
        return torch.sigmoid(out / self.t_ingr)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model FROM THE CONFIG SAVED IN THE CHECKPOINT, then load the weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ckpt["config"], ckpt["num_ingredients"], ckpt["num_dishes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def entropy_bce(p: np.ndarray) -> float:
    """Expected BCE at init when the bias equals the prior p (used by the sanity checks)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


LN2 = math.log(2)
