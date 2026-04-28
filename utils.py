from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def _tolist(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        # All .py files and monthly_modeling_dataset.csv can be placed in the same folder.
        "monthly_dataset_path": "monthly_modeling_dataset.csv",
    },
    "project": {
        "start_month": "2005-01",
        "end_month": "2026-03",
        "train_end_month": "2019-12",
        "rolling_window_months": 180,
        "max_lag_for_selection": 12,
        "main_lag": 1,
        # Robustness that is possible with the already-built monthly dataset.
        "robust_lag_orders": [2, 3],
        "robust_rolling_windows": [154, 165, 187, 198],
        # Oil lookback robustness requires brent_monthly_avg in the dataset.
        # If the column is absent, the code will skip it and write a note.
        "robust_oil_lookbacks": [6, 12, 24, 36],
    },
    "model": {
        "random_seed": 42,
        "run_lstm": False,
        "lasso": {
            "alphas": _tolist(np.logspace(-4, 1, 100)),
        },
        "svr": {
            "kernels": ["linear", "rbf"],
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.05, 0.1, 0.2],
            "gamma": ["scale", 0.01, 0.1],
        },
        "xgboost": {
            "tune": True,
            "n_estimators": [50, 100, 200],
            "max_depth": [1, 2, 3],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1.0, 5.0, 10.0],
            "reg_alpha": [0.0],
            "n_jobs": 1,
        },
        "lstm": {
            "sequence_length": [3, 6, 12],
            "hidden_units": [8, 16, 32],
            "dropout": [0.0, 0.2],
            "learning_rate": [0.0005, 0.001],
            "batch_size": [4, 8],
            "epochs": 200,
            "patience": 15,
            "validation_fraction": 0.2,
            "seeds": [42, 52, 62],
        },
    },
    "evaluation": {
        # Metrics are computed on log realized volatility, not level RV.
        "mape_on_level": False,
        "dmspes_lambdas": [1.0, 0.9],
    },
    "outputs": {
        "base_dir": "outputs",
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load optional YAML config and merge it into DEFAULT_CONFIG.

    The pipeline works without a config file. If config.yaml or config/config.yaml exists,
    only the values written there overwrite the defaults.
    """
    candidates = [Path(path), Path("config/config.yaml")]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            return deep_update(DEFAULT_CONFIG, user_cfg)
    return deepcopy(DEFAULT_CONFIG)


def ensure_dirs(base_dir: str = "outputs") -> None:
    for p in [
        base_dir,
        f"{base_dir}/diagnostics",
        f"{base_dir}/figures",
        f"{base_dir}/predictions",
        f"{base_dir}/metrics",
        f"{base_dir}/shap",
        f"{base_dir}/robustness",
        f"{base_dir}/.mplconfig",
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)
