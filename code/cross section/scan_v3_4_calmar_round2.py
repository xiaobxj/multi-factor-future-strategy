from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
MODULE_PATH = CODE_DIR / "backtest_engine_v3_4_calmar.py"
OUTPUT_DIR = CODE_DIR / "output" / "composite_momentum_score_regime_fix_calmar"
SCAN_PATH = OUTPUT_DIR / "parameter_scan_v3_4_calmar_round2.csv"


def load_module():
    spec = importlib.util.spec_from_file_location("bt_v34_round2", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_map(metrics: pd.DataFrame) -> dict[str, float]:
    return dict(zip(metrics["metric"], metrics["value"]))


def main() -> None:
    bt = load_module()
    bt.ensure_dirs()
    panel = bt.load_panel()

    tests = [
        {
            "name": "anchor_softmax_looser_dd",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.15,
            "MAX_VOL_SCALE": 1.25,
            "MIN_SIGNAL_GROSS": 0.25,
            "SOFT_DRAWDOWN": 0.05,
            "HARD_DRAWDOWN": 0.14,
            "MIN_DRAWDOWN_SCALE": 0.45,
        },
        {
            "name": "vol18_max150",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.18,
            "MAX_VOL_SCALE": 1.50,
            "MIN_SIGNAL_GROSS": 0.25,
            "SOFT_DRAWDOWN": 0.05,
            "HARD_DRAWDOWN": 0.14,
            "MIN_DRAWDOWN_SCALE": 0.45,
        },
        {
            "name": "vol20_max175",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.20,
            "MAX_VOL_SCALE": 1.75,
            "MIN_SIGNAL_GROSS": 0.25,
            "SOFT_DRAWDOWN": 0.05,
            "HARD_DRAWDOWN": 0.14,
            "MIN_DRAWDOWN_SCALE": 0.45,
        },
        {
            "name": "vol18_minsig35",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.18,
            "MAX_VOL_SCALE": 1.50,
            "MIN_SIGNAL_GROSS": 0.35,
            "SOFT_DRAWDOWN": 0.05,
            "HARD_DRAWDOWN": 0.14,
            "MIN_DRAWDOWN_SCALE": 0.45,
        },
        {
            "name": "vol20_minsig35",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.20,
            "MAX_VOL_SCALE": 1.75,
            "MIN_SIGNAL_GROSS": 0.35,
            "SOFT_DRAWDOWN": 0.05,
            "HARD_DRAWDOWN": 0.14,
            "MIN_DRAWDOWN_SCALE": 0.45,
        },
        {
            "name": "vol18_looser_dd2",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.18,
            "MAX_VOL_SCALE": 1.50,
            "MIN_SIGNAL_GROSS": 0.25,
            "SOFT_DRAWDOWN": 0.06,
            "HARD_DRAWDOWN": 0.16,
            "MIN_DRAWDOWN_SCALE": 0.50,
        },
        {
            "name": "vol20_looser_dd2",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.20,
            "MAX_VOL_SCALE": 1.75,
            "MIN_SIGNAL_GROSS": 0.25,
            "SOFT_DRAWDOWN": 0.06,
            "HARD_DRAWDOWN": 0.16,
            "MIN_DRAWDOWN_SCALE": 0.50,
        },
        {
            "name": "vol18_cap60",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.60,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.18,
            "MAX_VOL_SCALE": 1.50,
            "MIN_SIGNAL_GROSS": 0.25,
            "SOFT_DRAWDOWN": 0.05,
            "HARD_DRAWDOWN": 0.14,
            "MIN_DRAWDOWN_SCALE": 0.45,
        },
        {
            "name": "vol20_cap60_looser_dd2",
            "CONVICTION_MODE": "softmax",
            "WEIGHT_CAP": 0.60,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.20,
            "MAX_VOL_SCALE": 1.75,
            "MIN_SIGNAL_GROSS": 0.35,
            "SOFT_DRAWDOWN": 0.06,
            "HARD_DRAWDOWN": 0.16,
            "MIN_DRAWDOWN_SCALE": 0.50,
        },
        {
            "name": "linear_vol18_looser_dd2",
            "CONVICTION_MODE": "linear",
            "WEIGHT_CAP": 0.50,
            "VOL_LOOKBACK": 20,
            "VOL_TARGET_ANNUAL": 0.18,
            "MAX_VOL_SCALE": 1.50,
            "MIN_SIGNAL_GROSS": 0.25,
            "SOFT_DRAWDOWN": 0.06,
            "HARD_DRAWDOWN": 0.16,
            "MIN_DRAWDOWN_SCALE": 0.50,
        },
    ]

    tracked_keys = [
        "CONVICTION_MODE",
        "WEIGHT_CAP",
        "VOL_LOOKBACK",
        "VOL_TARGET_ANNUAL",
        "MAX_VOL_SCALE",
        "MIN_SIGNAL_GROSS",
        "SOFT_DRAWDOWN",
        "HARD_DRAWDOWN",
        "MIN_DRAWDOWN_SCALE",
    ]
    original = {key: getattr(bt, key) for key in tracked_keys}
    rows: list[dict[str, float | str]] = []

    for test in tests:
        for key, value in test.items():
            if key != "name":
                setattr(bt, key, value)
        daily, _positions, metrics, _annual = bt.run_backtest(panel=panel.copy())
        m = metric_map(metrics)
        rows.append(
            {
                "name": test["name"],
                "conviction_mode": test["CONVICTION_MODE"],
                "weight_cap": test["WEIGHT_CAP"],
                "vol_lookback": test["VOL_LOOKBACK"],
                "vol_target_annual": test["VOL_TARGET_ANNUAL"],
                "max_vol_scale": test["MAX_VOL_SCALE"],
                "min_signal_gross": test["MIN_SIGNAL_GROSS"],
                "soft_drawdown": test["SOFT_DRAWDOWN"],
                "hard_drawdown": test["HARD_DRAWDOWN"],
                "min_drawdown_scale": test["MIN_DRAWDOWN_SCALE"],
                "annualized_return": m["annualized_return"],
                "annualized_volatility": m["annualized_volatility"],
                "maximum_drawdown": m["maximum_drawdown"],
                "sharpe_ratio": m["sharpe_ratio"],
                "calmar_ratio": m["calmar_ratio"],
                "average_gross_exposure": m["average_gross_exposure"],
                "average_signal_gross_budget": m["average_signal_gross_budget"],
                "average_vol_scale": m["average_vol_scale"],
                "average_drawdown_scale": m["average_drawdown_scale"],
                "average_active_signal_count": m["average_active_signal_count"],
                "avg_turnover": float(daily["turnover"].mean()),
            }
        )

    for key, value in original.items():
        setattr(bt, key, value)

    out = pd.DataFrame(rows).sort_values(["calmar_ratio", "sharpe_ratio"], ascending=False).reset_index(drop=True)
    out.to_csv(SCAN_PATH, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan"))
    print(f"Saved round-2 scan summary to {SCAN_PATH}")


if __name__ == "__main__":
    main()
