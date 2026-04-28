from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


@dataclass
class LSTMResult:
    prediction: float
    best_params: Dict[str, Any]
    best_val_loss: float
    seed_predictions: Dict[int, float]


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def attach_cv_metadata(best_estimator: Any, grid: GridSearchCV) -> Any:
    best_estimator.best_params_ = grid.best_params_
    best_estimator.best_score_ = grid.best_score_
    return best_estimator


def fit_mlr(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    # X is standardized inside each rolling training window to avoid leakage.
    model = Pipeline([("scaler", StandardScaler()), ("mlr", LinearRegression())])
    model.fit(X, y)
    model.best_params_ = {}
    return model


def fit_lasso_cv(X: pd.DataFrame, y: pd.Series, alphas: Iterable[float], seed: int) -> Pipeline:
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(random_state=seed, max_iter=20000))])
    grid = GridSearchCV(
        pipe,
        param_grid={"lasso__alpha": list(alphas)},
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    grid.fit(X, y)
    return attach_cv_metadata(grid.best_estimator_, grid)


def fit_svr_cv(X: pd.DataFrame, y: pd.Series, svr_cfg: Dict[str, Any]) -> Pipeline:
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    param_grid = []
    for kernel in svr_cfg["kernels"]:
        base = {
            "svr__kernel": [kernel],
            "svr__C": _as_list(svr_cfg["C"]),
            "svr__epsilon": _as_list(svr_cfg["epsilon"]),
        }
        if kernel == "rbf":
            base["svr__gamma"] = _as_list(svr_cfg["gamma"])
        if kernel == "poly":
            base["svr__degree"] = _as_list(svr_cfg.get("degree", [2, 3]))
            base["svr__gamma"] = _as_list(svr_cfg["gamma"])
        param_grid.append(base)

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
    grid.fit(X, y)
    return attach_cv_metadata(grid.best_estimator_, grid)


def fit_xgboost_cv(X: pd.DataFrame, y: pd.Series, cfg: Dict[str, Any], seed: int) -> Pipeline:
    """수정됨: 월별 데이터 과적합 방지를 위한 강력한 규제 적용"""
    tscv = TimeSeriesSplit(n_splits=5)

    if XGBRegressor is not None:
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=int(cfg.get("n_jobs", 1)),
            verbosity=0,
            # 아래 두 줄 추가: 트리 복잡도 원천 차단
            importance_type='gain' 
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("xgb", estimator)])
        
        if cfg.get("tune", True):
            # 핵심 수정 부분: 후보군을 아주 보수적으로 강제 재설정
            param_grid = {
                "xgb__n_estimators": [50, 80],         # 트리 개수 축소
                "xgb__max_depth": [1, 2],              # 깊이를 1~2로 제한 (매우 중요)
                "xgb__learning_rate": [0.01, 0.05],    # 학습 속도 저하
                "xgb__subsample": [0.5, 0.7],          # 데이터 샘플링 강화
                "xgb__colsample_bytree": [0.8, 1.0],
                "xgb__reg_lambda": [50, 100, 200],     # L2 규제 대폭 강화
                "xgb__reg_alpha": [1, 10],             # L1 규제 추가 (불필요 변수 제거)
            }
            grid = GridSearchCV(pipe, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
            grid.fit(X, y)
            return attach_cv_metadata(grid.best_estimator_, grid)
        
        # 튜닝 안 할 경우 기본값도 보수적으로 설정
        pipe.set_params(
            xgb__n_estimators=50,
            xgb__max_depth=1,
            xgb__learning_rate=0.01,
            xgb__reg_lambda=100
        )
        pipe.fit(X, y)
        pipe.best_params_ = {}
        return pipe

    # Fallback if xgboost cannot be imported.
    fallback = HistGradientBoostingRegressor(random_state=seed, loss="squared_error")
    pipe = Pipeline([("scaler", StandardScaler()), ("hgb", fallback)])
    param_grid = {
        "hgb__max_iter": _as_list(cfg.get("n_estimators", [50, 100])),
        "hgb__max_depth": _as_list(cfg.get("max_depth", [1, 2, 3])),
        "hgb__learning_rate": _as_list(cfg.get("learning_rate", [0.03, 0.05, 0.1])),
        "hgb__l2_regularization": _as_list(cfg.get("reg_lambda", [1.0, 5.0, 10.0])),
    }
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
    grid.fit(X, y)
    return attach_cv_metadata(grid.best_estimator_, grid)


def extract_lasso_nonzero(model: Pipeline, feature_names: List[str], tol: float = 1e-10) -> Dict[str, int]:
    if "lasso" not in model.named_steps:
        return {f: 0 for f in feature_names}
    coefs = model.named_steps["lasso"].coef_
    return {f: int(abs(c) > tol) for f, c in zip(feature_names, coefs)}


def dmspes_weights(error_hist: pd.DataFrame, model_names: List[str], lam: float) -> Dict[str, float]:
    if error_hist.empty:
        w = 1.0 / len(model_names)
        return {m: w for m in model_names}
    discounted = {}
    for m in model_names:
        e = error_hist[m].dropna().values
        if len(e) == 0:
            discounted[m] = np.inf
            continue
        powers = np.arange(len(e) - 1, -1, -1)
        weights = lam ** powers
        discounted[m] = np.sum(weights * np.square(e)) / np.sum(weights)
    inv = {m: (1.0 / v if np.isfinite(v) and v > 0 else 0.0) for m, v in discounted.items()}
    total = sum(inv.values())
    if total == 0:
        w = 1.0 / len(model_names)
        return {m: w for m in model_names}
    return {m: inv[m] / total for m in model_names}


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, on_level: bool = False) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"log_MSE": np.nan, "log_MAPE": np.nan, "n_eval": 0}

    mse = mean_squared_error(y_true, y_pred)
    if on_level:
        y_true_level = np.exp(y_true)
        y_pred_level = np.exp(y_pred)
        denom = np.where(np.abs(y_true_level) < 1e-12, 1e-12, np.abs(y_true_level))
        mape = np.mean(np.abs((y_true_level - y_pred_level) / denom)) * 100
    else:
        denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
        mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {"log_MSE": mse, "log_MAPE": mape, "n_eval": int(len(y_true))}


def oos_r2_against_benchmark(y_true: np.ndarray, y_pred: np.ndarray, y_pred_benchmark: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_pred_benchmark)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_pred_benchmark = y_pred_benchmark[mask]
    if len(y_true) == 0:
        return np.nan
    sse_model = np.sum(np.square(y_true - y_pred))
    sse_bm = np.sum(np.square(y_true - y_pred_benchmark))
    return 1.0 - (sse_model / sse_bm if sse_bm > 0 else np.nan)


def _lstm_make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(seq_len - 1, len(X)):
        Xs.append(X[i - seq_len + 1 : i + 1, :])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def _lstm_fit_predict_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[float, float]:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.optimizers import Adam

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    seq_len = int(params["sequence_length"])
    if len(X_train) < seq_len + 5:
        return np.nan, np.nan

    scaler = StandardScaler()
    Xs_train_full = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test.reshape(1, -1))[0]

    X_seq, y_seq = _lstm_make_sequences(Xs_train_full, y_train, seq_len)
    if len(X_seq) < 10:
        return np.nan, np.nan

    val_n = max(1, int(np.ceil(len(X_seq) * float(cfg.get("validation_fraction", 0.2)))))
    if len(X_seq) - val_n < 5:
        return np.nan, np.nan
    X_tr, y_tr = X_seq[:-val_n], y_seq[:-val_n]
    X_val, y_val = X_seq[-val_n:], y_seq[-val_n:]

    model = Sequential(
        [
            LSTM(int(params["hidden_units"]), input_shape=(seq_len, X_train.shape[1])),
            Dropout(float(params["dropout"])),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=float(params["learning_rate"])), loss="mse")
    es = EarlyStopping(
        monitor="val_loss",
        patience=int(cfg.get("patience", 15)),
        restore_best_weights=True,
    )
    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=int(cfg.get("epochs", 200)),
        batch_size=int(params["batch_size"]),
        verbose=0,
        callbacks=[es],
        shuffle=False,
    )
    best_val = float(np.nanmin(hist.history.get("val_loss", [np.nan])))

    test_seq = np.vstack([Xs_train_full[-(seq_len - 1) :, :], Xs_test.reshape(1, -1)])
    test_seq = test_seq.reshape(1, seq_len, X_train.shape[1])
    pred = float(model.predict(test_seq, verbose=0).ravel()[0])
    return pred, best_val


def predict_lstm_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cfg: Dict[str, Any],
    default_seed: int,
) -> LSTMResult:
    """Chronological validation for LSTM with shuffle=False.

    If TensorFlow is not available, returns NaN so the rest of the pipeline can continue.
    """
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return LSTMResult(np.nan, {"error": "tensorflow_not_available"}, np.nan, {})

    grid = list(
        ParameterGrid(
            {
                "sequence_length": _as_list(cfg["sequence_length"]),
                "hidden_units": _as_list(cfg["hidden_units"]),
                "dropout": _as_list(cfg["dropout"]),
                "learning_rate": _as_list(cfg["learning_rate"]),
                "batch_size": _as_list(cfg["batch_size"]),
            }
        )
    )
    seeds = [int(s) for s in cfg.get("seeds", [default_seed])]
    validation_seed = seeds[0]

    Xtr = X_train.values.astype(float)
    ytr = y_train.values.astype(float)
    xte = X_test.values.astype(float)[0]

    best_params: Dict[str, Any] = {}
    best_loss = np.inf
    for params in grid:
        _, val_loss = _lstm_fit_predict_once(Xtr, ytr, xte, params, cfg, validation_seed)
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    if not best_params:
        return LSTMResult(np.nan, {"error": "no_valid_lstm_fit"}, np.nan, {})

    seed_predictions: Dict[int, float] = {}
    for seed in seeds:
        pred, _ = _lstm_fit_predict_once(Xtr, ytr, xte, best_params, cfg, seed)
        seed_predictions[seed] = pred

    finite_preds = [v for v in seed_predictions.values() if np.isfinite(v)]
    prediction = float(np.mean(finite_preds)) if finite_preds else np.nan
    return LSTMResult(prediction, best_params, float(best_loss), seed_predictions)
