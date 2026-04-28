from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import os

import matplotlib

try:
    from .utils import ensure_dirs
except ImportError:
    from utils import ensure_dirs

# Keep matplotlib cache inside outputs when possible.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm

try:
    from .features import (
        OIL_VARS,
        basic_stats_table,
        build_design,
        distribution_tests,
        load_monthly_modeling_dataset,
        recompute_oil_shocks_if_possible,
        split_sample,
    )
    from .models import (
        dmspes_weights,
        evaluate_metrics,
        extract_lasso_nonzero,
        fit_lasso_cv,
        fit_mlr,
        fit_svr_cv,
        fit_xgboost_cv,
        oos_r2_against_benchmark,
        predict_lstm_cv,
    )
except ImportError:
    from features import (
        OIL_VARS,
        basic_stats_table,
        build_design,
        distribution_tests,
        load_monthly_modeling_dataset,
        recompute_oil_shocks_if_possible,
        split_sample,
    )
    from models import (
        dmspes_weights,
        evaluate_metrics,
        extract_lasso_nonzero,
        fit_lasso_cv,
        fit_mlr,
        fit_svr_cv,
        fit_xgboost_cv,
        oos_r2_against_benchmark,
        predict_lstm_cv,
    )

try:
    import shap
except Exception:
    shap = None


INDIVIDUAL_MODELS = ["MLR", "LASSO", "SVR", "XGBoost", "LSTM"]


def _select_lag_by_bic(log_rv: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    rows = []
    s = log_rv.dropna()
    for lag in range(1, max_lag + 1):
        try:
            model = AutoReg(s, lags=lag, old_names=False).fit()
            rows.append({"lag": lag, "bic": model.bic, "aic": model.aic})
        except Exception as exc:
            rows.append({"lag": lag, "bic": np.nan, "aic": np.nan, "error": str(exc)})
    out = pd.DataFrame(rows)
    if "error" not in out.columns:
        out["error"] = ""
    return out


def _best_lag_from_bic(lag_table: pd.DataFrame, default_lag: int = 1) -> int:
    valid = lag_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["bic"])
    if valid.empty:
        return default_lag
    return int(valid.loc[valid["bic"].idxmin(), "lag"])


def _save_acf_pacf(df: pd.DataFrame, out_dir: Path) -> None:
    s = df["log_rv"].dropna()
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(s, lags=24, ax=ax[0])
    plot_pacf(s, lags=24, ax=ax[1], method="ywm")
    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "acf_pacf_log_rv.png", dpi=150)
    plt.close(fig)


def _safe_float_prediction(model: Any, X_test: pd.DataFrame) -> float:
    try:
        return float(model.predict(X_test)[0])
    except Exception:
        return np.nan


def _nan_trimmed_mean(values: np.ndarray) -> float:
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return np.nan
    if arr.size < 3:
        return float(np.nanmean(arr))
    s = np.sort(arr)
    return float(np.nanmean(s[1:-1]))


def _dmspes_pred(indiv_preds: Dict[str, float], raw_w: Dict[str, float]) -> float:
    finite = {k: v for k, v in indiv_preds.items() if np.isfinite(v)}
    if not finite:
        return np.nan
    wsum = sum(raw_w.get(k, 0.0) for k in finite.keys())
    if wsum <= 0:
        return float(np.nanmean(np.asarray(list(finite.values()), dtype=float)))
    return float(sum((raw_w.get(k, 0.0) / wsum) * finite[k] for k in finite.keys()))


def _run_in_sample_mlr(bundle_df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path, lag_order: int) -> None:
    for feature_set in ["lag_only", "oil_augmented"]:
        X, y, _, _ = build_design(bundle_df, feature_set=feature_set, lag_order=lag_order, oil_lag=1)
        if len(y) == 0:
            continue
        X_const = sm.add_constant(X)
        ols = sm.OLS(y, X_const).fit()
        with open(out_dir / "diagnostics" / f"in_sample_mlr_summary_{feature_set}.txt", "w", encoding="utf-8") as f:
            f.write(ols.summary().as_text())


def _append_best_param(records: List[Dict[str, Any]], month: Any, feature_set: str, model_name: str, model: Any) -> None:
    params = getattr(model, "best_params_", {}) or {}
    rec = {"month": month, "feature_set": feature_set, "model": model_name}
    for k, v in params.items():
        rec[k] = v
    records.append(rec)


def _run_rolling_forecast(
    monthly_df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_set: str,
    lag_order: int,
    rolling_window: int,
    tag: str,
    out_dir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    X_all, y_all, work, feat_cols = build_design(monthly_df, feature_set=feature_set, lag_order=lag_order, oil_lag=1)
    train_end = pd.Period(cfg["project"]["train_end_month"], freq="M").to_timestamp("M")
    eval_df = work[work["month"] > train_end].copy().reset_index(drop=True)

    preds: List[Dict[str, Any]] = []
    best_param_records: List[Dict[str, Any]] = []
    lasso_selection_records: List[Dict[str, Any]] = []
    lstm_param_records: List[Dict[str, Any]] = []
    error_hist = pd.DataFrame(columns=INDIVIDUAL_MODELS)

    iterator = eval_df.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(eval_df), desc=f"Rolling-{tag}")

    for row in iterator:
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

        mlr = fit_mlr(X_train, y_train)
        lasso = fit_lasso_cv(X_train, y_train, cfg["model"]["lasso"]["alphas"], cfg["model"]["random_seed"])
        svr = fit_svr_cv(X_train, y_train, cfg["model"]["svr"])
        xgb = fit_xgboost_cv(X_train, y_train, cfg["model"]["xgboost"], cfg["model"]["random_seed"])

        p_mlr = _safe_float_prediction(mlr, X_test)
        p_lasso = _safe_float_prediction(lasso, X_test)
        p_svr = _safe_float_prediction(svr, X_test)
        p_xgb = _safe_float_prediction(xgb, X_test)

        _append_best_param(best_param_records, month, feature_set, "MLR", mlr)
        _append_best_param(best_param_records, month, feature_set, "LASSO", lasso)
        _append_best_param(best_param_records, month, feature_set, "SVR", svr)
        _append_best_param(best_param_records, month, feature_set, "XGBoost", xgb)

        if feature_set == "oil_augmented":
            selected = extract_lasso_nonzero(lasso, feat_cols)
            rec = {"month": month, **selected}
            lasso_selection_records.append(rec)

        p_lstm = np.nan
        if cfg["model"].get("run_lstm", False):
            lstm_res = predict_lstm_cv(
                X_train,
                y_train,
                X_test,
                cfg["model"]["lstm"],
                cfg["model"]["random_seed"],
            )
            p_lstm = lstm_res.prediction
            lstm_param_records.append(
                {
                    "month": month,
                    "feature_set": feature_set,
                    "prediction": p_lstm,
                    "best_val_loss": lstm_res.best_val_loss,
                    **{f"lstm__{k}": v for k, v in lstm_res.best_params.items()},
                    **{f"seed_pred_{k}": v for k, v in lstm_res.seed_predictions.items()},
                }
            )

        indiv = {
            "MLR": p_mlr,
            "LASSO": p_lasso,
            "SVR": p_svr,
            "XGBoost": p_xgb,
            "LSTM": p_lstm,
        }
        indiv_vals = np.asarray(list(indiv.values()), dtype=float)
        mean_comb = float(np.nanmean(indiv_vals)) if np.isfinite(indiv_vals).any() else np.nan
        median_comb = float(np.nanmedian(indiv_vals)) if np.isfinite(indiv_vals).any() else np.nan
        trim_comb = _nan_trimmed_mean(indiv_vals)

        dmspes = {}
        for lam in cfg["evaluation"]["dmspes_lambdas"]:
            w = dmspes_weights(error_hist, list(indiv.keys()), lam=lam)
            dmspes[f"DMSPE_{lam}"] = _dmspes_pred(indiv, w)

        rec = {
            "month": month,
            "actual": y_test,
            **indiv,
            "MeanComb": mean_comb,
            "MedianComb": median_comb,
            "TrimmedMeanComb": trim_comb,
            **dmspes,
        }
        preds.append(rec)

        err_row = {m: (y_test - rec[m]) for m in INDIVIDUAL_MODELS}
        error_hist = pd.concat([error_hist, pd.DataFrame([err_row])], ignore_index=True)

    pred_df = pd.DataFrame(preds).replace([np.inf, -np.inf], np.nan)
    pred_df.to_csv(out_dir / "predictions" / f"oos_predictions_{tag}.csv", index=False)

    if best_param_records:
        pd.DataFrame(best_param_records).to_csv(out_dir / "diagnostics" / f"best_params_{tag}.csv", index=False)
    if lasso_selection_records:
        lasso_sel = pd.DataFrame(lasso_selection_records)
        lasso_sel.to_csv(out_dir / "diagnostics" / f"lasso_selection_by_window_{tag}.csv", index=False)
        freq = lasso_sel.drop(columns=["month"], errors="ignore").mean().rename("selection_frequency").reset_index()
        freq = freq.rename(columns={"index": "feature"})
        freq.to_csv(out_dir / "diagnostics" / f"lasso_selection_frequency_{tag}.csv", index=False)
    if lstm_param_records:
        pd.DataFrame(lstm_param_records).to_csv(out_dir / "diagnostics" / f"lstm_params_{tag}.csv", index=False)

    _save_metrics_for_prediction(pred_df, tag, out_dir, cfg)
    return pred_df


def _save_metrics_for_prediction(pred_df: pd.DataFrame, tag: str, out_dir: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    metrics = []
    if pred_df.empty:
        out = pd.DataFrame(metrics)
        out.to_csv(out_dir / "metrics" / f"oos_metrics_{tag}.csv", index=False)
        return out
    y_true = pred_df["actual"].values
    model_cols = [c for c in pred_df.columns if c not in ["month", "actual"]]
    for m in model_cols:
        ms = evaluate_metrics(y_true, pred_df[m].values, on_level=cfg["evaluation"].get("mape_on_level", False))
        metrics.append({"model": m, **ms})
    out = pd.DataFrame(metrics)
    out.to_csv(out_dir / "metrics" / f"oos_metrics_{tag}.csv", index=False)
    return out


def _make_main_result_table(
    lag_pred: pd.DataFrame,
    oil_pred: pd.DataFrame,
    out_dir: Path,
    cfg: Dict[str, Any],
    save_name: str | None = "main_result_table.csv",
) -> pd.DataFrame:
    if lag_pred.empty or oil_pred.empty:
        result = pd.DataFrame()
        if save_name is not None:
            result.to_csv(out_dir / "metrics" / save_name, index=False)
        return result

    merged = lag_pred.merge(oil_pred, on="month", suffixes=("_lag_only", "_oil_aug"))
    merged = merged.rename(columns={"actual_lag_only": "actual"})
    if "actual_oil_aug" in merged.columns:
        merged = merged.drop(columns=["actual_oil_aug"])

    y = merged["actual"].values
    model_cols = [c for c in lag_pred.columns if c not in ["month", "actual"]]
    rows = []
    common_benchmark = merged["MLR_lag_only"].values

    for m in model_cols:
        lag_col = f"{m}_lag_only"
        oil_col = f"{m}_oil_aug"
        if lag_col not in merged.columns or oil_col not in merged.columns:
            continue

        lag_metrics = evaluate_metrics(y, merged[lag_col].values, on_level=cfg["evaluation"].get("mape_on_level", False))
        oil_metrics = evaluate_metrics(y, merged[oil_col].values, on_level=cfg["evaluation"].get("mape_on_level", False))
        oil_gain_r2 = oos_r2_against_benchmark(y, merged[oil_col].values, merged[lag_col].values)
        common_r2_lag = oos_r2_against_benchmark(y, merged[lag_col].values, common_benchmark)
        common_r2_oil = oos_r2_against_benchmark(y, merged[oil_col].values, common_benchmark)

        rows.append(
            {
                "model": m,
                "lag_only_log_MSE": lag_metrics["log_MSE"],
                "oil_augmented_log_MSE": oil_metrics["log_MSE"],
                "lag_only_log_MAPE": lag_metrics["log_MAPE"],
                "oil_augmented_log_MAPE": oil_metrics["log_MAPE"],
                "Oil_gain_R2_vs_same_algorithm_lag_only": oil_gain_r2,
                "Common_R2_lag_only_vs_lag_only_MLR": common_r2_lag,
                "Common_R2_oil_augmented_vs_lag_only_MLR": common_r2_oil,
                "n_eval": oil_metrics["n_eval"],
            }
        )

    result = pd.DataFrame(rows)
    if save_name is not None:
        result.to_csv(out_dir / "metrics" / save_name, index=False)
    return result


def _run_shap_for_latest_oil_xgb(
    monthly_df: pd.DataFrame,
    cfg: Dict[str, Any],
    lag_order: int,
    rolling_window: int,
    out_dir: Path,
) -> None:
    if shap is None:
        with open(out_dir / "shap" / "shap_note.txt", "w", encoding="utf-8") as f:
            f.write("SHAP is not installed. Skipped SHAP analysis.\n")
        return

    try:
        _, _, work, feat_cols = build_design(monthly_df, feature_set="oil_augmented", lag_order=lag_order, oil_lag=1)
        train_end = pd.Period(cfg["project"]["train_end_month"], freq="M").to_timestamp("M")
        eval_df = work[work["month"] > train_end].copy().reset_index(drop=True)
        if eval_df.empty:
            return
        last_month = eval_df["month"].max()
        i = work.index[work["month"] == last_month][0]
        train = work.iloc[i - rolling_window : i].copy()
        if len(train) < rolling_window:
            return
        X_train = train[feat_cols]
        y_train = train["log_rv"]
        xgb_pipe = fit_xgboost_cv(X_train, y_train, cfg["model"]["xgboost"], cfg["model"]["random_seed"])

        if hasattr(xgb_pipe, "named_steps") and "xgb" in xgb_pipe.named_steps:
            estimator = xgb_pipe.named_steps["xgb"]
            X_for_shap = xgb_pipe.named_steps["scaler"].transform(X_train)
        else:
            with open(out_dir / "shap" / "shap_note.txt", "w", encoding="utf-8") as f:
                f.write("XGBoost estimator was unavailable. SHAP was skipped.\n")
            return

        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_for_shap)
        shap_abs = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": shap_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
        shap_df.to_csv(out_dir / "shap" / "shap_importance_oil_augmented_latest_window.csv", index=False)
    except Exception as exc:
        with open(out_dir / "shap" / "shap_note.txt", "w", encoding="utf-8") as f:
            f.write(f"SHAP analysis failed: {exc}\n")


def _run_robustness(monthly_df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path) -> None:
    rows = []

    # 1) Lag-order robustness: p=2,3 etc.
    for lag_order in cfg["project"].get("robust_lag_orders", []):
        lag_pred = _run_rolling_forecast(
            monthly_df,
            cfg,
            feature_set="lag_only",
            lag_order=int(lag_order),
            rolling_window=int(cfg["project"]["rolling_window_months"]),
            tag=f"robust_lag{lag_order}_lag_only",
            out_dir=out_dir,
            show_progress=False,
        )
        oil_pred = _run_rolling_forecast(
            monthly_df,
            cfg,
            feature_set="oil_augmented",
            lag_order=int(lag_order),
            rolling_window=int(cfg["project"]["rolling_window_months"]),
            tag=f"robust_lag{lag_order}_oil_augmented",
            out_dir=out_dir,
            show_progress=False,
        )
        result = _make_main_result_table(lag_pred, oil_pred, out_dir, cfg, save_name=None)
        if not result.empty:
            result.insert(0, "robustness_type", "lag_order")
            result.insert(1, "robustness_value", lag_order)
            rows.append(result)

    # 2) Rolling-window robustness.
    main_lag = int(cfg["project"].get("main_lag", 1))
    for window in cfg["project"].get("robust_rolling_windows", []):
        lag_pred = _run_rolling_forecast(
            monthly_df,
            cfg,
            feature_set="lag_only",
            lag_order=main_lag,
            rolling_window=int(window),
            tag=f"robust_window{window}_lag_only",
            out_dir=out_dir,
            show_progress=False,
        )
        oil_pred = _run_rolling_forecast(
            monthly_df,
            cfg,
            feature_set="oil_augmented",
            lag_order=main_lag,
            rolling_window=int(window),
            tag=f"robust_window{window}_oil_augmented",
            out_dir=out_dir,
            show_progress=False,
        )
        result = _make_main_result_table(lag_pred, oil_pred, out_dir, cfg, save_name=None)
        if not result.empty:
            result.insert(0, "robustness_type", "rolling_window")
            result.insert(1, "robustness_value", window)
            rows.append(result)

    # 3) Oil-lookback robustness only if brent_monthly_avg exists.
    oil_lookback_rows = []
    for lookback in cfg["project"].get("robust_oil_lookbacks", []):
        recomputed = recompute_oil_shocks_if_possible(monthly_df, int(lookback))
        if recomputed is None:
            continue
        lag_pred = _run_rolling_forecast(
            recomputed,
            cfg,
            feature_set="lag_only",
            lag_order=main_lag,
            rolling_window=int(cfg["project"]["rolling_window_months"]),
            tag=f"robust_oillookback{lookback}_lag_only",
            out_dir=out_dir,
            show_progress=False,
        )
        oil_pred = _run_rolling_forecast(
            recomputed,
            cfg,
            feature_set="oil_augmented",
            lag_order=main_lag,
            rolling_window=int(cfg["project"]["rolling_window_months"]),
            tag=f"robust_oillookback{lookback}_oil_augmented",
            out_dir=out_dir,
            show_progress=False,
        )
        result = _make_main_result_table(lag_pred, oil_pred, out_dir, cfg, save_name=None)
        if not result.empty:
            result.insert(0, "robustness_type", "oil_lookback")
            result.insert(1, "robustness_value", lookback)
            oil_lookback_rows.append(result)

    if rows or oil_lookback_rows:
        pd.concat(rows + oil_lookback_rows, ignore_index=True).to_csv(
            out_dir / "robustness" / "robustness_summary.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(out_dir / "robustness" / "robustness_summary.csv", index=False)

    if not oil_lookback_rows:
        with open(out_dir / "robustness" / "oil_lookback_robustness_note.txt", "w", encoding="utf-8") as f:
            f.write(
                "Oil-shock historical-length robustness was skipped because monthly_modeling_dataset.csv "
                "does not contain brent_monthly_avg. To run this robustness check, add brent_monthly_avg "
                "to the monthly dataset or recompute NPI/ANP/SNP/NPI2 from raw Brent data before running.\n"
            )


def run_pipeline(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg.get("outputs", {}).get("base_dir", "outputs"))
    os.environ.setdefault("MPLCONFIGDIR", str((out_dir / ".mplconfig").resolve()))
    ensure_dirs(str(out_dir))

    monthly_path = cfg["data"].get("monthly_dataset_path", "monthly_modeling_dataset.csv")
    monthly_df = load_monthly_modeling_dataset(monthly_path, max_lag=int(cfg["project"]["max_lag_for_selection"]))

    bundle = split_sample(
        monthly_df,
        start_month=cfg["project"]["start_month"],
        end_month=cfg["project"]["end_month"],
        train_end_month=cfg["project"]["train_end_month"],
    )
    bundle.monthly_df.to_csv(out_dir / "diagnostics" / "monthly_modeling_dataset_used.csv", index=False)

    # In-sample diagnostics.
    ins_cols = ["log_rv", "NPI", "ANP", "SNP", "NPI2"]
    basic_stats_table(bundle.in_sample_df, ins_cols).to_csv(out_dir / "diagnostics" / "in_sample_basic_stats.csv", index=False)
    distribution_tests(bundle.in_sample_df, ins_cols, lb_lags=(5, 12, 22)).to_csv(
        out_dir / "diagnostics" / "in_sample_tests.csv", index=False
    )

    lag_table = _select_lag_by_bic(bundle.in_sample_df["log_rv"], int(cfg["project"]["max_lag_for_selection"]))
    lag_table.to_csv(out_dir / "diagnostics" / "lag_selection_bic.csv", index=False)
    best_bic_lag = _best_lag_from_bic(lag_table, default_lag=int(cfg["project"].get("main_lag", 1)))
    main_lag = int(cfg["project"].get("main_lag", best_bic_lag))
    pd.DataFrame(
        [
            {
                "best_lag_by_bic": best_bic_lag,
                "main_lag_used": main_lag,
                "note": "main_lag_used comes from config/default. Check BIC table before changing it.",
            }
        ]
    ).to_csv(out_dir / "diagnostics" / "lag_choice_summary.csv", index=False)

    _save_acf_pacf(bundle.in_sample_df, out_dir)
    _run_in_sample_mlr(bundle.in_sample_df, cfg, out_dir, lag_order=main_lag)

    # Main out-of-sample analysis: strict ex-ante oil-augmented model.
    window = int(cfg["project"]["rolling_window_months"])
    lag_pred = _run_rolling_forecast(
        bundle.monthly_df,
        cfg,
        feature_set="lag_only",
        lag_order=main_lag,
        rolling_window=window,
        tag="lag_only",
        out_dir=out_dir,
        show_progress=True,
    )
    oil_pred = _run_rolling_forecast(
        bundle.monthly_df,
        cfg,
        feature_set="oil_augmented",
        lag_order=main_lag,
        rolling_window=window,
        tag="oil_augmented_ex_ante",
        out_dir=out_dir,
        show_progress=True,
    )
    _make_main_result_table(lag_pred, oil_pred, out_dir, cfg, save_name="main_result_table.csv")
    _run_shap_for_latest_oil_xgb(bundle.monthly_df, cfg, main_lag, window, out_dir)

    _run_robustness(bundle.monthly_df, cfg, out_dir)
