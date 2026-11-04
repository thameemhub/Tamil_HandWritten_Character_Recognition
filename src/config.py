"""
config.py – Central configuration loader for Tamil Few-Shot HTR.

Reads config.yaml once, exposes a frozen dict-like object with dot access.
All paths are resolved relative to project_root.
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# ── Force UTF-8 on Windows ──────────────────────────────────────
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class _CfgNode(dict):
    """Dict subclass that supports attribute-style access."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")
        if isinstance(val, dict) and not isinstance(val, _CfgNode):
            val = _CfgNode(val)
            self[name] = val
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _deep_node(d: dict) -> _CfgNode:
    node = _CfgNode()
    for k, v in d.items():
        if isinstance(v, dict):
            node[k] = _deep_node(v)
        else:
            node[k] = v
    return node


def _find_config() -> Path:
    """Walk up from this file to find config.yaml."""
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        cand = cur / "config.yaml"
        if cand.exists():
            return cand
        cur = cur.parent
    raise FileNotFoundError("config.yaml not found in parent directories")


def load_config(path: str | Path | None = None) -> _CfgNode:
    """Load and return the configuration as a nested _CfgNode."""
    if path is None:
        path = _find_config()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _deep_node(raw)

    # ── Resolve derived paths ────────────────────────────────────
    proj = Path(cfg.paths.project_root)
    cfg.paths._project = proj
    cfg.paths._dataset_root = Path(cfg.paths.dataset_root)
    cfg.paths._data = proj / cfg.paths.data_dir
    cfg.paths._processed = proj / cfg.paths.processed_dir
    cfg.paths._interim = proj / cfg.paths.interim_dir
    cfg.paths._support_set = proj / cfg.paths.support_set_dir
    cfg.paths._models = proj / cfg.paths.models_dir
    cfg.paths._reports = proj / cfg.paths.reports_dir

    return cfg


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Module-level singleton ───────────────────────────────────────
_CFG: _CfgNode | None = None


def get_config() -> _CfgNode:
    """Return the cached global config (loads on first call)."""
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG

