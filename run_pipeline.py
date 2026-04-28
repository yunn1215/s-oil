from __future__ import annotations

import traceback

try:
    from .pipeline import run_pipeline
    from .utils import load_config
except ImportError:
    from pipeline import run_pipeline
    from utils import load_config


def main() -> None:
    cfg = load_config("config.yaml")
    run_pipeline(cfg)
    print("Pipeline completed. Check outputs/ directory.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Pipeline failed:", e)
        traceback.print_exc()
