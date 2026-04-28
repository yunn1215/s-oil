from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCES = [
    Path("outputs_core/metrics/main_result_table_core_no_xgb_lstm.csv"),
    Path("outputs_xgboost/metrics/main_result_table_xgboost_only.csv"),
    Path("outputs_lstm/metrics/main_result_table_lstm_only.csv"),
]


def main() -> None:
    frames = []
    for path in SOURCES:
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        else:
            print(f"Skipped missing file: {path}")

    out_dir = Path("outputs_merged/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frames:
        pd.DataFrame().to_csv(out_dir / "main_result_table_all_models.csv", index=False)
        print("No result files found.")
        return

    result = pd.concat(frames, ignore_index=True)
    order = {"MLR": 0, "LASSO": 1, "SVR": 2, "XGBoost": 3, "LSTM": 4}
    result["_order"] = result["model"].map(order).fillna(999)
    result = result.sort_values("_order").drop(columns="_order")
    result.to_csv(out_dir / "main_result_table_all_models.csv", index=False)
    print(f"Saved: {out_dir / 'main_result_table_all_models.csv'}")
    print(result)


if __name__ == "__main__":
    main()
