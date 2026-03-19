from __future__ import annotations

import importlib.util
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
BASE_MODULE_PATH = CODE_DIR / "backtest_engine_v3_4_calmar.py"


def load_base():
    spec = importlib.util.spec_from_file_location("bt_v34_best1", BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {BASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(bt) -> None:
    bt.OUTPUT_DIR = bt.FEATURE_DIR / "output" / "composite_momentum_score_regime_fix_calmar_vol20_cap60_looser_dd2"
    bt.METRICS_PATH = bt.OUTPUT_DIR / "backtest_metrics_v3_4_calmar_vol20_cap60_looser_dd2.csv"
    bt.DAILY_PATH = bt.OUTPUT_DIR / "backtest_daily_returns_v3_4_calmar_vol20_cap60_looser_dd2.csv"
    bt.POSITIONS_PATH = bt.OUTPUT_DIR / "backtest_positions_v3_4_calmar_vol20_cap60_looser_dd2.csv"
    bt.ANNUAL_PATH = bt.OUTPUT_DIR / "backtest_annual_summary_v3_4_calmar_vol20_cap60_looser_dd2.csv"
    bt.TEARSHEET_PATH = bt.OUTPUT_DIR / "backtest_tearsheet_v3_4_calmar_vol20_cap60_looser_dd2.png"

    bt.CONVICTION_MODE = "softmax"
    bt.WEIGHT_CAP = 0.60
    bt.VOL_LOOKBACK = 20
    bt.VOL_TARGET_ANNUAL = 0.20
    bt.MAX_VOL_SCALE = 1.75
    bt.MIN_SIGNAL_GROSS = 0.35
    bt.SOFT_DRAWDOWN = 0.06
    bt.HARD_DRAWDOWN = 0.16
    bt.MIN_DRAWDOWN_SCALE = 0.50


def main() -> None:
    bt = load_base()
    configure(bt)
    bt.main()


if __name__ == "__main__":
    main()
