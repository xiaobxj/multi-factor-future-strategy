from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = FEATURE_DIR / "output" / "composite_momentum_score_regime_fix"
OUTPUT_DIR = FEATURE_DIR / "output" / "composite_momentum_score_regime_fix_calmar"
PANEL_PATH = SOURCE_DIR / "composite_momentum_panel.csv"
METRICS_PATH = OUTPUT_DIR / "backtest_metrics_v3_4_calmar.csv"
DAILY_PATH = OUTPUT_DIR / "backtest_daily_returns_v3_4_calmar.csv"
POSITIONS_PATH = OUTPUT_DIR / "backtest_positions_v3_4_calmar.csv"
ANNUAL_PATH = OUTPUT_DIR / "backtest_annual_summary_v3_4_calmar.csv"
TEARSHEET_PATH = OUTPUT_DIR / "backtest_tearsheet_v3_4_calmar.png"

TRADING_DAYS = 252
BACKTEST_START_DATE = pd.Timestamp("2023-04-21")
COST_RATE = 0.0003
SCORE_THRESHOLD = 1.0
REGIME_GATE = "trend"
HOLD_DAYS = 2

CONVICTION_MODE = "softmax"
CONVICTION_POWER = 1.50
SOFTMAX_TEMPERATURE = 0.35
WEIGHT_CAP = 0.50

MIN_SIGNAL_GROSS = 0.25
MAX_SIGNAL_GROSS = 1.00
TARGET_TREND_ASSETS = 4
TARGET_ACTIVE_ASSETS = 3
MIN_SCORE_SPREAD = 0.10
FULL_SCORE_SPREAD = 0.60
TARGET_AVG_EXCESS_SCORE = 0.45

VOL_TARGET_ANNUAL = 0.15
VOL_LOOKBACK = 20
MIN_VOL_SCALE = 0.35
MAX_VOL_SCALE = 1.25

SOFT_DRAWDOWN = 0.04
HARD_DRAWDOWN = 0.12
MIN_DRAWDOWN_SCALE = 0.45


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cap_and_normalize(raw_weights: pd.Series, target_gross: float, cap: float) -> pd.Series:
    weights = raw_weights.fillna(0.0).clip(lower=0.0)
    result = pd.Series(0.0, index=weights.index, dtype=float)
    if target_gross <= 0 or weights.sum() <= 0:
        return result

    target_gross = float(min(max(target_gross, 0.0), max(cap, 0.0) * max(len(weights), 1)))
    active = weights[weights > 0].copy()
    remaining_gross = target_gross

    while not active.empty and remaining_gross > 1e-12:
        scaled = active / active.sum() * remaining_gross
        capped_mask = scaled >= cap - 1e-12

        if not capped_mask.any():
            result.loc[scaled.index] += scaled
            break

        capped_assets = scaled.index[capped_mask]
        result.loc[capped_assets] = cap
        remaining_gross = max(target_gross - result.sum(), 0.0)
        active = active.loc[~active.index.isin(capped_assets)]

    return result.clip(lower=0.0)


def map_conviction(active_excess_score: pd.Series) -> pd.Series:
    active_excess_score = active_excess_score.clip(lower=0.0).fillna(0.0)
    if active_excess_score.sum() <= 0:
        return pd.Series(0.0, index=active_excess_score.index, dtype=float)

    if CONVICTION_MODE == "linear":
        conviction = active_excess_score
    elif CONVICTION_MODE == "power":
        conviction = active_excess_score.pow(CONVICTION_POWER)
    elif CONVICTION_MODE == "softmax":
        scaled = np.exp(active_excess_score / max(SOFTMAX_TEMPERATURE, 1e-6))
        conviction = pd.Series(scaled, index=active_excess_score.index, dtype=float)
    else:
        raise ValueError(f"Unsupported CONVICTION_MODE: {CONVICTION_MODE}")
    return conviction.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_spread_scale(score_spread: float | None) -> float:
    if score_spread is None or pd.isna(score_spread):
        return 0.5
    span = max(FULL_SCORE_SPREAD - MIN_SCORE_SPREAD, 1e-6)
    return float(np.clip((score_spread - MIN_SCORE_SPREAD) / span, 0.0, 1.0))


def compute_drawdown_scale(current_drawdown: float) -> float:
    dd = abs(min(current_drawdown, 0.0))
    if dd <= SOFT_DRAWDOWN:
        return 1.0
    if dd >= HARD_DRAWDOWN:
        return MIN_DRAWDOWN_SCALE
    ratio = (dd - SOFT_DRAWDOWN) / max(HARD_DRAWDOWN - SOFT_DRAWDOWN, 1e-6)
    return float(1.0 - ratio * (1.0 - MIN_DRAWDOWN_SCALE))


def build_signal_targets(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.sort_values(["trade_date", "product"]).copy()
    work["is_trend_asset"] = ((work["regime"] == REGIME_GATE) & work["vol_20"].gt(0)).astype(float)
    work["entry_excess_score"] = np.where(
        (work["regime"] == REGIME_GATE) & work["vol_20"].gt(0),
        np.maximum(work["optimized_score_global"] - SCORE_THRESHOLD, 0.0),
        0.0,
    )

    work = work.sort_values(["product", "trade_date"]).copy()
    work["hold_excess_score"] = work.groupby("product")["entry_excess_score"].transform(
        lambda s: s.rolling(HOLD_DAYS, min_periods=1).max()
    )
    work["active_signal"] = work["hold_excess_score"] > 0

    work = work.sort_values(["trade_date", "product"]).copy()
    weight_parts: list[pd.DataFrame] = []

    for _, daily in work.groupby("trade_date", sort=True):
        daily = daily.copy()
        trend_count = int(daily["is_trend_asset"].sum())
        active_mask = daily["active_signal"]
        active_count = int(active_mask.sum())

        qualified = daily.loc[daily["entry_excess_score"] > 0, "optimized_score_global"].sort_values(ascending=False)
        score_spread = np.nan
        if len(qualified) >= 2:
            score_spread = float(qualified.iloc[0] - qualified.iloc[1])

        if active_count > 0:
            active_excess = daily.loc[active_mask, "hold_excess_score"]
            conviction = map_conviction(active_excess)
            risk_scaled = conviction / daily.loc[active_mask, "vol_20"].replace(0.0, np.nan)
            risk_scaled = risk_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            trend_scale = float(np.clip(trend_count / max(TARGET_TREND_ASSETS, 1), 0.0, 1.0))
            active_scale = float(np.clip(active_count / max(TARGET_ACTIVE_ASSETS, 1), 0.0, 1.0))
            avg_excess = float(active_excess.mean())
            conviction_scale = float(np.clip(avg_excess / max(TARGET_AVG_EXCESS_SCORE, 1e-6), 0.0, 1.0))
            spread_scale = compute_spread_scale(score_spread)

            breadth_strength = 0.35 * trend_scale + 0.35 * active_scale + 0.15 * conviction_scale + 0.15 * spread_scale
            signal_gross_budget = MIN_SIGNAL_GROSS + (MAX_SIGNAL_GROSS - MIN_SIGNAL_GROSS) * breadth_strength
            target_weights = cap_and_normalize(risk_scaled, target_gross=signal_gross_budget, cap=WEIGHT_CAP)
        else:
            avg_excess = 0.0
            spread_scale = compute_spread_scale(score_spread)
            signal_gross_budget = 0.0
            target_weights = pd.Series(0.0, index=daily.index, dtype=float)

        daily["trend_asset_count"] = trend_count
        daily["active_signal_count"] = active_count
        daily["score_spread_top1_top2"] = score_spread
        daily["avg_active_excess_score"] = avg_excess
        daily["signal_gross_budget"] = signal_gross_budget
        daily["base_target_weight"] = 0.0
        daily.loc[target_weights.index, "base_target_weight"] = target_weights
        weight_parts.append(daily)

    return pd.concat(weight_parts, ignore_index=True).sort_values(["trade_date", "product"]).reset_index(drop=True)


def run_sequential_backtest(weighted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weighted = weighted.sort_values(["trade_date", "product"]).copy()
    prev_weights = {product: 0.0 for product in weighted["product"].unique()}
    trailing_returns: list[float] = []
    net_value = 1.0
    running_peak = 1.0
    position_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for trade_date, daily in weighted.groupby("trade_date", sort=True):
        daily = daily.copy()
        current_drawdown = net_value / running_peak - 1.0

        if len(trailing_returns) >= max(VOL_LOOKBACK // 2, 5):
            realized_vol = float(pd.Series(trailing_returns[-VOL_LOOKBACK:]).std())
            target_daily_vol = VOL_TARGET_ANNUAL / np.sqrt(TRADING_DAYS)
            vol_scale = float(np.clip(target_daily_vol / max(realized_vol, 1e-6), MIN_VOL_SCALE, MAX_VOL_SCALE))
        else:
            realized_vol = np.nan
            vol_scale = 1.0

        drawdown_scale = compute_drawdown_scale(current_drawdown)
        daily["vol_scale"] = vol_scale
        daily["drawdown_scale"] = drawdown_scale
        daily["execution_weight"] = daily["base_target_weight"] * vol_scale * drawdown_scale
        daily["lagged_execution_weight"] = daily["product"].map(prev_weights).astype(float)
        daily["turnover_component"] = (daily["execution_weight"] - daily["lagged_execution_weight"]).abs()
        daily["gross_contribution"] = daily["lagged_execution_weight"] * daily["return"]

        gross_return = float(daily["gross_contribution"].sum())
        turnover = float(daily["turnover_component"].sum())
        transaction_cost = turnover * COST_RATE
        strategy_return = gross_return - transaction_cost
        net_value *= 1.0 + strategy_return
        running_peak = max(running_peak, net_value)
        ending_drawdown = net_value / running_peak - 1.0

        daily["transaction_cost_component"] = daily["turnover_component"] * COST_RATE
        daily["net_contribution"] = daily["gross_contribution"] - daily["transaction_cost_component"]
        daily["gross_exposure"] = float(daily["execution_weight"].sum())

        position_rows.extend(daily.to_dict("records"))
        daily_rows.append(
            {
                "trade_date": trade_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "entry_count": int((daily["entry_excess_score"] > 0).sum()),
                "hold_count": int(daily["active_signal"].sum()),
                "transaction_cost": transaction_cost,
                "strategy_return": strategy_return,
                "net_value": net_value,
                "running_peak": running_peak,
                "drawdown": ending_drawdown,
                "signal_gross_budget": float(daily["signal_gross_budget"].iloc[0]),
                "vol_scale": vol_scale,
                "drawdown_scale": drawdown_scale,
                "gross_exposure": float(daily["execution_weight"].sum()),
                "trend_asset_count": int(daily["trend_asset_count"].iloc[0]),
                "active_signal_count": int(daily["active_signal_count"].iloc[0]),
                "score_spread_top1_top2": daily["score_spread_top1_top2"].iloc[0],
                "avg_active_excess_score": float(daily["avg_active_excess_score"].iloc[0]),
                "realized_vol_lookback": realized_vol,
            }
        )
        prev_weights = dict(zip(daily["product"], daily["execution_weight"]))
        trailing_returns.append(strategy_return)

    daily_df = pd.DataFrame(daily_rows)
    positions_df = pd.DataFrame(position_rows).sort_values(["trade_date", "product"]).reset_index(drop=True)
    return daily_df, positions_df


def summarize_backtest(daily: pd.DataFrame) -> pd.DataFrame:
    strategy_return = daily["strategy_return"]
    annualized_return = float(daily["net_value"].iloc[-1] ** (TRADING_DAYS / len(daily)) - 1.0)
    annualized_volatility = float(strategy_return.std() * np.sqrt(TRADING_DAYS))
    maximum_drawdown = float(-daily["drawdown"].min())
    sharpe_ratio = float(strategy_return.mean() / strategy_return.std() * np.sqrt(TRADING_DAYS))
    calmar_ratio = float(annualized_return / maximum_drawdown) if maximum_drawdown > 0 else np.nan

    return pd.DataFrame(
        {
            "metric": [
                "annualized_return",
                "annualized_volatility",
                "maximum_drawdown",
                "sharpe_ratio",
                "calmar_ratio",
                "average_signal_gross_budget",
                "average_gross_exposure",
                "average_vol_scale",
                "average_drawdown_scale",
                "average_active_signal_count",
            ],
            "value": [
                annualized_return,
                annualized_volatility,
                maximum_drawdown,
                sharpe_ratio,
                calmar_ratio,
                float(daily["signal_gross_budget"].mean()),
                float(daily["gross_exposure"].mean()),
                float(daily["vol_scale"].mean()),
                float(daily["drawdown_scale"].mean()),
                float(daily["active_signal_count"].mean()),
            ],
        }
    )


def summarize_annual(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["year"] = work["trade_date"].dt.year
    rows: list[dict[str, float | int]] = []
    for year, grp in work.groupby("year", sort=True):
        year_curve = (1.0 + grp["strategy_return"]).cumprod()
        year_drawdown = year_curve / year_curve.cummax() - 1.0
        annualized_return = float(year_curve.iloc[-1] ** (TRADING_DAYS / len(grp)) - 1.0)
        maximum_drawdown = float(-year_drawdown.min())
        calmar_ratio = annualized_return / maximum_drawdown if maximum_drawdown > 0 else np.nan
        rows.append(
            {
                "year": int(year),
                "annualized_return": annualized_return,
                "maximum_drawdown": maximum_drawdown,
                "sharpe_ratio": float(grp["strategy_return"].mean() / grp["strategy_return"].std() * np.sqrt(TRADING_DAYS)),
                "calmar_ratio": calmar_ratio,
                "average_gross_exposure": float(grp["gross_exposure"].mean()),
                "average_signal_gross_budget": float(grp["signal_gross_budget"].mean()),
                "average_active_signal_count": float(grp["active_signal_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_tearsheet(daily: pd.DataFrame, metrics: pd.DataFrame) -> plt.Figure:
    metric_map = dict(zip(metrics["metric"], metrics["value"]))
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

    axes[0].plot(daily["trade_date"], daily["net_value"], color="#1f77b4", linewidth=1.6)
    axes[0].set_title(
        "V3.4 Calmar Overlay | "
        f"Mode={CONVICTION_MODE} | "
        f"Cap={WEIGHT_CAP:.0%} | "
        f"VolTarget={VOL_TARGET_ANNUAL:.0%} | "
        f"AnnRet={metric_map['annualized_return']:.2%} | "
        f"Calmar={metric_map['calmar_ratio']:.3f}"
    )
    axes[0].set_ylabel("Net Value")
    axes[0].grid(alpha=0.2)

    axes[1].fill_between(daily["trade_date"], daily["drawdown"], 0.0, color="#d62728", alpha=0.85)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[1].set_title(
        "Drawdown | "
        f"MaxDD={metric_map['maximum_drawdown']:.2%} | "
        f"Sharpe={metric_map['sharpe_ratio']:.3f}"
    )
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.2)

    axes[2].plot(daily["trade_date"], daily["signal_gross_budget"], label="signal_gross_budget", linewidth=1.2)
    axes[2].plot(daily["trade_date"], daily["vol_scale"], label="vol_scale", linewidth=1.2)
    axes[2].plot(daily["trade_date"], daily["drawdown_scale"], label="drawdown_scale", linewidth=1.2)
    axes[2].plot(daily["trade_date"], daily["gross_exposure"], label="final_gross", linewidth=1.2)
    axes[2].set_title("Overlay Diagnostics")
    axes[2].set_ylabel("Scale")
    axes[2].grid(alpha=0.2)
    axes[2].legend()

    axes[3].plot(daily["trade_date"], daily["active_signal_count"], color="#2ca02c", linewidth=1.2, label="active_signal_count")
    axes[3].plot(daily["trade_date"], daily["trend_asset_count"], color="#9467bd", linewidth=1.2, label="trend_asset_count")
    axes[3].set_title(
        "Breadth Diagnostics | "
        f"AvgGross={daily['gross_exposure'].mean():.3f} | "
        f"AvgBudget={daily['signal_gross_budget'].mean():.3f}"
    )
    axes[3].set_xlabel("Trade Date")
    axes[3].set_ylabel("Count")
    axes[3].grid(alpha=0.2)
    axes[3].legend()

    fig.tight_layout()
    return fig


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return panel.loc[panel["trade_date"] >= BACKTEST_START_DATE].copy()


def run_backtest(panel: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel is None:
        panel = load_panel()
    weighted = build_signal_targets(panel)
    daily, positions = run_sequential_backtest(weighted)
    metrics = summarize_backtest(daily)
    annual = summarize_annual(daily)
    return daily, positions, metrics, annual


def main() -> None:
    ensure_dirs()
    daily, positions, metrics, annual = run_backtest()
    daily.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")
    positions.to_csv(POSITIONS_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    annual.to_csv(ANNUAL_PATH, index=False, encoding="utf-8-sig")

    tearsheet = plot_tearsheet(daily, metrics)
    tearsheet.savefig(TEARSHEET_PATH, dpi=150, bbox_inches="tight")
    plt.close(tearsheet)

    for _, row in metrics.iterrows():
        print(f"{row['metric']}: {row['value']:.6f}")
    print(f"Saved daily backtest to {DAILY_PATH}")
    print(f"Saved positions to {POSITIONS_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved annual summary to {ANNUAL_PATH}")
    print(f"Saved tearsheet to {TEARSHEET_PATH}")


if __name__ == "__main__":
    main()
