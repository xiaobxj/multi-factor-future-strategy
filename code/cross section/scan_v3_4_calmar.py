from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
MODULE_PATH = CODE_DIR / "backtest_engine_v3_4_calmar.py"
OUTPUT_DIR = CODE_DIR / "output" / "composite_momentum_score_regime_fix_calmar"
SCAN_PATH = OUTPUT_DIR / "parameter_scan_v3_4_calmar.csv"
BASELINE_V33_PATH = (
    CODE_DIR / "output" / "composite_momentum_score_regime_fix_controlled" / "backtest_metrics_v3_3_controlled.csv"
)


def load_module():
    spec = importlib.util.spec_from_file_location("bt_v34", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_metric_map(metrics: pd.DataFrame) -> dict[str, float]:
    return dict(zip(metrics["metric"], metrics["value"]))


def baseline_v33_row() -> dict[str, float | str]:
    metrics = pd.read_csv(BASELINE_V33_PATH)
    metric_map = extract_metric_map(metrics)
    return {
        "name": "v3_3_controlled_baseline",
        "conviction_mode": "topk_binary",
        "weight_cap": 0.50,
        "vol_lookback": 20,
        "soft_drawdown": 0.04,
        "hard_drawdown": 0.12,
        "annualized_return": metric_map["annualized_return"],
        "annualized_volatility": metric_map["annualized_volatility"],
        "maximum_drawdown": metric_map["maximum_drawdown"],
        "sharpe_ratio": metric_map["sharpe_ratio"],
        "calmar_ratio": metric_map["calmar_ratio"],
        "average_gross_exposure": metric_map.get("average_gross_exposure"),
        "average_signal_gross_budget": None,
    }


def main() -> None:
    bt = load_module()
    bt.ensure_dirs()
    panel = bt.load_panel()

    tests = [
        {"name": "softmax_default", "CONVICTION_MODE": "softmax", "WEIGHT_CAP": 0.50, "VOL_LOOKBACK": 20, "SOFT_DRAWDOWN": 0.04, "HARD_DRAWDOWN": 0.12},
        {"name": "linear_default", "CONVICTION_MODE": "linear", "WEIGHT_CAP": 0.50, "VOL_LOOKBACK": 20, "SOFT_DRAWDOWN": 0.04, "HARD_DRAWDOWN": 0.12},
        {"name": "power_default", "CONVICTION_MODE": "power", "WEIGHT_CAP": 0.50, "VOL_LOOKBACK": 20, "SOFT_DRAWDOWN": 0.04, "HARD_DRAWDOWN": 0.12},
        {"name": "softmax_cap40", "CONVICTION_MODE": "softmax", "WEIGHT_CAP": 0.40, "VOL_LOOKBACK": 20, "SOFT_DRAWDOWN": 0.04, "HARD_DRAWDOWN": 0.12},
        {"name": "softmax_vol40", "CONVICTION_MODE": "softmax", "WEIGHT_CAP": 0.50, "VOL_LOOKBACK": 40, "SOFT_DRAWDOWN": 0.04, "HARD_DRAWDOWN": 0.12},
        {"name": "softmax_looser_dd", "CONVICTION_MODE": "softmax", "WEIGHT_CAP": 0.50, "VOL_LOOKBACK": 20, "SOFT_DRAWDOWN": 0.05, "HARD_DRAWDOWN": 0.14},
        {"name": "softmax_tighter_dd", "CONVICTION_MODE": "softmax", "WEIGHT_CAP": 0.50, "VOL_LOOKBACK": 20, "SOFT_DRAWDOWN": 0.03, "HARD_DRAWDOWN": 0.10},
    ]

    tracked_keys = ["CONVICTION_MODE", "WEIGHT_CAP", "VOL_LOOKBACK", "SOFT_DRAWDOWN", "HARD_DRAWDOWN"]
    original = {key: getattr(bt, key) for key in tracked_keys}
    rows: list[dict[str, float | str | None]] = [baseline_v33_row()]

    for test in tests:
        for key, value in test.items():
            if key != "name":
                setattr(bt, key, value)
        daily, _positions, metrics, _annual = bt.run_backtest(panel=panel.copy())
        metric_map = extract_metric_map(metrics)
        rows.append(
            {
                "name": test["name"],
                "conviction_mode": test["CONVICTION_MODE"],
                "weight_cap": test["WEIGHT_CAP"],
                "vol_lookback": test["VOL_LOOKBACK"],
                "soft_drawdown": test["SOFT_DRAWDOWN"],
                "hard_drawdown": test["HARD_DRAWDOWN"],
                "annualized_return": metric_map["annualized_return"],
                "annualized_volatility": metric_map["annualized_volatility"],
                "maximum_drawdown": metric_map["maximum_drawdown"],
                "sharpe_ratio": metric_map["sharpe_ratio"],
                "calmar_ratio": metric_map["calmar_ratio"],
                "average_gross_exposure": metric_map["average_gross_exposure"],
                "average_signal_gross_budget": metric_map["average_signal_gross_budget"],
            }
        )

    for key, value in original.items():
        setattr(bt, key, value)

    out = pd.DataFrame(rows).sort_values(["calmar_ratio", "sharpe_ratio"], ascending=False).reset_index(drop=True)
    out.to_csv(SCAN_PATH, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan"))
    print(f"Saved scan summary to {SCAN_PATH}")


if __name__ == "__main__":
    main()
