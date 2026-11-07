"""Shared helpers: seeding, config loading, paths and small I/O utilities."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
BLANK = "___"  # placeholder for a value YOU must choose (see the TP step that uses it)


# ---------------------------------------------------------------- seeding

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA).

    deterministic=True also forces deterministic cuDNN kernels: reproducible but slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: each worker derives its numpy/random seed from torch's."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ---------------------------------------------------------------- config

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Read a yaml config.

    A `base:` key (path relative to the file) is loaded first, then overridden by the file.
    `overrides` are "dotted.key=value" strings from the command line (--set), parsed as yaml.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "base" in cfg:
        cfg = _deep_merge(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides or []:
        key, _, value = item.partition("=")
        set_key(cfg, key.strip(), yaml.safe_load(value))
    cfg["_config_path"] = str(path)
    return cfg


def set_key(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, last = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def need(cfg: dict, dotted: str) -> Any:
    """Return cfg[a][b][c] for "a.b.c"; stop with a clear message if it is still `___`."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing config key '{dotted}' (config: {cfg.get('_config_path')})")
        node = node[part]
    if node == BLANK:
        raise SystemExit(
            f"\n>>> Parameter to fill in: '{dotted}' is still '{BLANK}'.\n"
            f"    Open configs/base.yaml (or the experiment yaml), read the comment next to it,\n"
            f"    and the TP step that introduces it. Then run the command again.\n"
        )
    return node


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}, used to log params to MLflow."""
    out = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------- paths & I/O

def path_from(cfg: dict, name: str) -> Path:
    """Resolve cfg.paths.<name> relative to the repo root."""
    return ROOT / cfg["paths"][name]


def run_dir(cfg_or_run: dict | str) -> Path:
    run = cfg_or_run if isinstance(cfg_or_run, str) else need(cfg_or_run, "run_name")
    d = ROOT / "models" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6
