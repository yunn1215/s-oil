from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


@dataclass
class DataBundle:
    monthly_df: pd.DataFrame
    in_sample_df: pd.DataFrame
    oos_df: pd.DataFrame


REQUIRED_BASE_COLUMNS = ["month", "log_rv", "NPI", "ANP", "SNP", "NPI2"]
OIL_VARS = ["NPI", "ANP", "SNP", "NPI2"]


def load_monthly_modeling_dataset(path: str | Path, max_lag: int = 12) -> pd.DataFrame:
    """Load the already-preprocessed monthly modeling dataset.

    Expected minimum columns:
    month, log_rv, NPI, ANP, SNP, NPI2, and preferably log_rv_lag1 ... log_rv_lag12.
    If lag columns are absent, they are generated from log_rv after sorting by month.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put monthly_modeling_dataset.csv in the same folder "
            "as run_pipeline.py or update data.monthly_dataset_path in config.yaml."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"monthly dataset is missing required columns: {missing}")

    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    for c in df.columns:
        if c != "month":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["month", "log_rv"]).sort_values("month").reset_index(drop=True)

    # Generate missing log RV lag columns if needed. Existing lag columns are preserved,
    # which is useful when lags were created using pre-2005 raw observations.
    for lag in range(1, max_lag + 1):
        col = f"log_rv_lag{lag}"
        if col not in df.columns:
            df[col] = df["log_rv"].shift(lag)

    return df


def split_sample(df: pd.DataFrame, start_month: str, end_month: str, train_end_month: str) -> DataBundle:
    start = pd.Period(start_month, freq="M").to_timestamp("M")
    end = pd.Period(end_month, freq="M").to_timestamp("M")
    train_end = pd.Period(train_end_month, freq="M").to_timestamp("M")

    mdf = df[(df["month"] >= start) & (df["month"] <= end)].copy().reset_index(drop=True)
    in_sample = mdf[mdf["month"] <= train_end].copy().reset_index(drop=True)
    oos = mdf[mdf["month"] > train_end].copy().reset_index(drop=True)
    return DataBundle(monthly_df=mdf, in_sample_df=in_sample, oos_df=oos)


def add_ex_ante_oil_lags(df: pd.DataFrame, oil_lag: int = 1) -> pd.DataFrame:
    """Create NPI_lag1, ANP_lag1, SNP_lag1, NPI2_lag1 for strict ex-ante forecasting."""
    out = df.copy().sort_values("month").reset_index(drop=True)
    for c in OIL_VARS:
        out[f"{c}_lag{oil_lag}"] = out[c].shift(oil_lag)
    return out


def feature_columns(feature_set: str, lag_order: int, oil_lag: int = 1) -> List[str]:
    rv_cols = [f"log_rv_lag{i}" for i in range(1, lag_order + 1)]
    if feature_set == "lag_only":
        return rv_cols
    if feature_set == "oil_augmented":
        oil_cols = [f"{c}_lag{oil_lag}" for c in OIL_VARS]
        return rv_cols + oil_cols
    raise ValueError("feature_set must be 'lag_only' or 'oil_augmented'.")


def build_design(
    df: pd.DataFrame,
    feature_set: str,
    lag_order: int,
    oil_lag: int = 1,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """Build X and y for one feature set.

    oil_augmented always uses lagged oil shocks, e.g. NPI_{t-1}, not NPI_t.
    """
    work = add_ex_ante_oil_lags(df, oil_lag=oil_lag)
    feat_cols = feature_columns(feature_set, lag_order, oil_lag=oil_lag)
    missing = [c for c in feat_cols + ["log_rv"] if c not in work.columns]
    if missing:
        raise ValueError(f"Missing columns for design matrix: {missing}")
    work = work.dropna(subset=feat_cols + ["log_rv"]).copy().reset_index(drop=True)
    return work[feat_cols], work["log_rv"], work, feat_cols


def basic_stats_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        rows.append(
            {
                "variable": c,
                "n": int(s.shape[0]),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
                "skew": s.skew(),
                "kurtosis": s.kurtosis(),
            }
        )
    return pd.DataFrame(rows)


def distribution_tests(df: pd.DataFrame, columns: Sequence[str], lb_lags: Iterable[int] = (5, 12, 22)) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        rec = {"variable": c, "n": int(s.shape[0])}

        if len(s) >= 8:
            jb = jarque_bera(s)
            rec["jb_stat"] = jb.statistic
            rec["jb_pvalue"] = jb.pvalue
        else:
            rec["jb_stat"] = np.nan
            rec["jb_pvalue"] = np.nan

        for lag in lb_lags:
            if len(s) > lag + 1:
                lb = acorr_ljungbox(s, lags=[lag], return_df=True)
                rec[f"lb{lag}_stat"] = lb["lb_stat"].iloc[0]
                rec[f"lb{lag}_pvalue"] = lb["lb_pvalue"].iloc[0]
            else:
                rec[f"lb{lag}_stat"] = np.nan
                rec[f"lb{lag}_pvalue"] = np.nan

        try:
            adf = adfuller(s, autolag="AIC")
            rec["adf_stat"] = adf[0]
            rec["adf_pvalue"] = adf[1]
        except Exception:
            rec["adf_stat"] = np.nan
            rec["adf_pvalue"] = np.nan

        rows.append(rec)
    return pd.DataFrame(rows)


def recompute_oil_shocks_if_possible(df: pd.DataFrame, lookback: int) -> pd.DataFrame | None:
    """Recompute NPI/ANP/SNP/NPI2 if brent_monthly_avg exists.

    The provided monthly_modeling_dataset.csv usually already contains only the final oil shocks.
    In that case, oil-lookback robustness cannot be recomputed and this returns None.
    """
    if "brent_monthly_avg" not in df.columns:
        return None
    out = df.copy().sort_values("month").reset_index(drop=True)
    prev_max = out["brent_monthly_avg"].shift(1).rolling(lookback).max()
    prev_min = out["brent_monthly_avg"].shift(1).rolling(lookback).min()
    out["NPI"] = np.where(out["brent_monthly_avg"] > prev_max, np.log(out["brent_monthly_avg"] / prev_max), 0.0)
    out["ANP"] = np.where(out["brent_monthly_avg"] < prev_min, np.log(out["brent_monthly_avg"] / prev_min), 0.0)
    out["SNP"] = out["NPI"] + out["ANP"]
    out["NPI2"] = np.where(out["brent_monthly_avg"] > prev_max, 1.0, 0.0)
    return out
