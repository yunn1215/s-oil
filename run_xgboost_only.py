from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from utils import load_config
    from features import build_design, load_monthly_modeling_dataset, split_sample
    from models import evaluate_metrics, fit_xgboost_cv, oos_r2_against_benchmark
except ImportError as exc:
    raise ImportError(
        "Put this script in the same folder as utils.py, features.py, models.py, "
        "and monthly_modeling_dataset.csv."
    ) from exc


def ensure_output_dirs(out_dir: Path) -> None:
    for sub in ["", "predictions", "metrics"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def safe_predict(model: Any, X_test: pd.DataFrame) -> float:
    try:
        return float(model.predict(X_test)[0])
    except Exception:
        return np.nan


def load_analysis_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    monthly_path = cfg["data"].get("monthly_dataset_path", "monthly_modeling_dataset.csv")
    monthly_df = load_monthly_modeling_dataset(
        monthly_path,
        max_lag=int(cfg["project"].get("max_lag_for_selection", 12)),
    )
    bundle = split_sample(
        monthly_df,
        start_month=cfg["project"].get("start_month", "2005-01"),
        end_month=cfg["project"].get("end_month", "2026-03"),
        train_end_month=cfg["project"].get("train_end_month", "2019-12"),
    )
    return bundle.monthly_df


def run_rolling_xgboost(
    monthly_df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_set: str,
    tag: str,
    out_dir: Path,
) -> pd.DataFrame:
    lag_order = int(cfg["project"].get("main_lag", 1))
    rolling_window = int(cfg["project"].get("rolling_window_months", 180))
    train_end = pd.Period(cfg["project"].get("train_end_month", "2019-12"), freq="M").to_timestamp("M")

    _, _, work, feat_cols = build_design(monthly_df, feature_set=feature_set, lag_order=lag_order, oil_lag=1)
    eval_df = work[work["month"] > train_end].copy().reset_index(drop=True)

    preds: List[Dict[str, Any]] = []
    best_params: List[Dict[str, Any]] = []

    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df), desc=f"XGBoost-{tag}"):
        i = work.index[work["month"] == row.month][0]
        train = work.iloc[i - rolling_window : i].copy()
        test = work.iloc[[i]].copy()
        if len(train) < rolling_window:
            continue

        X_train = train[feat_cols]
        y_train = train["log_rv"]
        X_test = test[feat_cols]
        y_test = float(test["log_rv"].iloc[0])
        month = test["month"].iloc[0]

        xgb = fit_xgboost_cv(X_train, y_train, cfg["model"]["xgboost"], cfg["model"].get("random_seed", 42))
        preds.append({"month": month, "actual": y_test, "XGBoost": safe_predict(xgb, X_test)})

        params = getattr(xgb, "best_params_", {}) or {}
        rec = {"month": month, "feature_set": feature_set}
        rec.update(params)
        best_params.append(rec)

    pred_df = pd.DataFrame(preds).replace([np.inf, -np.inf], np.nan)
    pred_df.to_csv(out_dir / "predictions" / f"oos_predictions_{tag}.csv", index=False)
    pd.DataFrame(best_params).to_csv(out_dir / "metrics" / f"best_params_{tag}.csv", index=False)
    return pred_df


def load_common_benchmark(months: pd.Series) -> np.ndarray | None:
    """Use lag-only MLR from outputs_core if it exists; otherwise return None."""
    path = Path("outputs_core") / "predictions" / "oos_predictions_lag_only_core.csv"
    if not path.exists():
        return None
    core = pd.read_csv(path)
    core["month"] = pd.to_datetime(core["month"], errors="coerce")
    target = pd.DataFrame({"month": pd.to_datetime(months)})
    merged = target.merge(core[["month", "MLR"]], on="month", how="left")
    return merged["MLR"].values


def make_xgboost_result_table(lag_pred: pd.DataFrame, oil_pred: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    if lag_pred.empty or oil_pred.empty:
        result = pd.DataFrame()
        result.to_csv(out_dir / "metrics" / "main_result_table_xgboost_only.csv", index=False)
        return result

    merged = lag_pred.merge(oil_pred, on="month", suffixes=("_lag_only", "_oil_aug"))
    merged = merged.rename(columns={"actual_lag_only": "actual"})
    if "actual_oil_aug" in merged.columns:
        merged = merged.drop(columns=["actual_oil_aug"])

    y = merged["actual"].values
    lag_metrics = evaluate_metrics(y, merged["XGBoost_lag_only"].values, on_level=cfg["evaluation"].get("mape_on_level", False))
    oil_metrics = evaluate_metrics(y, merged["XGBoost_oil_aug"].values, on_level=cfg["evaluation"].get("mape_on_level", False))

    common_benchmark = load_common_benchmark(merged["month"])
    common_r2_lag = np.nan
    common_r2_oil = np.nan
    if common_benchmark is not None:
        common_r2_lag = oos_r2_against_benchmark(y, merged["XGBoost_lag_only"].values, common_benchmark)
        common_r2_oil = oos_r2_against_benchmark(y, merged["XGBoost_oil_aug"].values, common_benchmark)

    result = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "lag_only_log_MSE": lag_metrics["log_MSE"],
                "oil_augmented_log_MSE": oil_metrics["log_MSE"],
                "lag_only_log_MAPE": lag_metrics["log_MAPE"],
                "oil_augmented_log_MAPE": oil_metrics["log_MAPE"],
                "Oil_gain_R2_vs_same_algorithm_lag_only": oos_r2_against_benchmark(
                    y, merged["XGBoost_oil_aug"].values, merged["XGBoost_lag_only"].values
                ),
                "Common_R2_lag_only_vs_lag_only_MLR": common_r2_lag,
                "Common_R2_oil_augmented_vs_lag_only_MLR": common_r2_oil,
                "n_eval": oil_metrics["n_eval"],
            }
        ]
    )
    result.to_csv(out_dir / "metrics" / "main_result_table_xgboost_only.csv", index=False)
    return result


def main() -> None:
    cfg = load_config("config.yaml")
    out_dir = Path("outputs_xgboost")
    ensure_output_dirs(out_dir)

    # For a heavy computer, set model.xgboost.tune: true in config.yaml.
    # For quick fixed-parameter XGBoost, keep tune: false.
    monthly_df = load_analysis_data(cfg)
    lag_pred = run_rolling_xgboost(monthly_df, cfg, "lag_only", "lag_only_xgboost", out_dir)
    oil_pred = run_rolling_xgboost(monthly_df, cfg, "oil_augmented", "oil_augmented_xgboost", out_dir)
    result = make_xgboost_result_table(lag_pred, oil_pred, cfg, out_dir)

    print("XGBoost-only rolling forecast completed.")
    print(f"Saved: {out_dir / 'metrics' / 'main_result_table_xgboost_only.csv'}")
    print(result)


if __name__ == "__main__":
    main()
