from __future__ import annotations

from pathlib import Path

import market_state_research as _base
import numpy as np
import pandas as pd


MarketStateConfig = _base.MarketStateConfig
_validate_columns = _base._validate_columns
_prepare_input = _base._prepare_input
compute_true_range = _base.compute_true_range
compute_atr = _base.compute_atr
compute_adx = _base.compute_adx
compute_efficiency_ratio = _base.compute_efficiency_ratio
compute_regression_slope = _base.compute_regression_slope
_safe_divide = _base._safe_divide
summarize_by_time_bucket = _base.summarize_by_time_bucket


def enrich_features(df: pd.DataFrame, config: MarketStateConfig | None = None) -> pd.DataFrame:
    """Return input DataFrame enriched with market-state features."""

    config = config or MarketStateConfig()
    work = _prepare_input(df, config)

    adx_df = compute_adx(work, config.adx_window)
    work = pd.concat([work, adx_df], axis=1)

    work["tr"] = compute_true_range(work)
    work["atr"] = compute_atr(work, config.atr_window)
    work["rolling_vol"] = work["return"].rolling(config.volatility_window).std()

    cumret_window = (1.0 + work["return"]).rolling(config.directional_window).apply(np.prod, raw=True) - 1.0
    work["rolling_cum_return"] = cumret_window
    work["trend_vol_ratio"] = _safe_divide(work["rolling_cum_return"].abs(), work["rolling_vol"])
    work["efficiency_ratio"] = compute_efficiency_ratio(work["close"], config.efficiency_window)

    mid = work["close"].rolling(config.bollinger_window).mean()
    std = work["close"].rolling(config.bollinger_window).std()
    upper = mid + config.bollinger_std * std
    lower = mid - config.bollinger_std * std
    work["bollinger_mid"] = mid
    work["bollinger_upper"] = upper
    work["bollinger_lower"] = lower
    work["bollinger_bandwidth"] = _safe_divide(upper - lower, mid.abs())

    don_high = work["high"].rolling(config.donchian_window).max()
    don_low = work["low"].rolling(config.donchian_window).min()
    work["donchian_width"] = _safe_divide(don_high - don_low, work["close"].abs())
    work["atr_pct"] = _safe_divide(work["atr"], work["close"].abs())

    work["compression_regime"] = (
        (work["bollinger_bandwidth"] < config.compression_bandwidth_threshold)
        | (work["atr_pct"] < config.compression_atr_threshold)
        | (work["donchian_width"] < config.compression_donchian_threshold)
    )

    trend_votes = pd.DataFrame(
        {
            "adx": work["adx"] >= config.adx_trend_threshold,
            "ratio": work["trend_vol_ratio"] >= config.ratio_trend_threshold,
            "er": work["efficiency_ratio"] >= config.er_trend_threshold,
        }
    ).sum(axis=1)
    range_votes = pd.DataFrame(
        {
            "adx": work["adx"] <= config.adx_range_threshold,
            "ratio": work["trend_vol_ratio"] <= config.ratio_range_threshold,
            "er": work["efficiency_ratio"] <= config.er_range_threshold,
            "compression": work["compression_regime"],
        }
    ).sum(axis=1)

    work["trend_score"] = trend_votes
    work["range_score"] = range_votes
    trend_condition = (trend_votes >= 2) & (range_votes < 2)
    range_condition = (range_votes >= 2) & (trend_votes < 2)
    work["regime"] = np.select(
        [trend_condition, range_condition],
        ["trend", "range"],
        default="neutral",
    )

    work["fast_ma"] = work["close"].rolling(config.fast_ma_window).mean()
    work["slow_ma"] = work["close"].rolling(config.slow_ma_window).mean()
    work["ma_gap"] = _safe_divide(work["fast_ma"] - work["slow_ma"], work["slow_ma"].abs())
    work["slope"] = compute_regression_slope(work["close"], config.slope_window)

    roll_ret = (1.0 + work["return"]).rolling(config.segment_return_window).apply(np.prod, raw=True) - 1.0
    work["segment_rolling_return"] = roll_ret

    if config.trend_segment_method == "return":
        trend_signal = np.sign(roll_ret)
        trend_signal = trend_signal.where(roll_ret.abs() > config.segment_return_threshold, 0)
    elif config.trend_segment_method == "slope":
        trend_signal = np.sign(work["slope"])
    else:
        trend_signal = np.sign(work["fast_ma"] - work["slow_ma"])

    trend_direction = pd.Series(trend_signal, index=work.index).fillna(0).astype(int)
    work["trend_direction"] = trend_direction.where(work["regime"] == "trend", 0).astype(int)

    work["rolling_return_std"] = work["return"].rolling(config.large_move_std_window).std()
    work["large_move_by_return"] = work["abs_return"] > (
        config.large_move_return_multiple * work["rolling_return_std"]
    )
    work["large_move_by_tr"] = work["tr"] > (config.large_move_tr_multiple * work["atr"])
    work["large_vol_bar"] = work["large_move_by_return"] | work["large_move_by_tr"]

    total_return = (1.0 + work["return"]).rolling(config.noise_window).apply(np.prod, raw=True) - 1.0
    path_length = work["abs_return"].rolling(config.noise_window).sum()
    work["noise_path_ratio"] = _safe_divide(path_length, total_return.abs())

    bar_range = (work["high"] - work["low"]).replace(0, np.nan)
    upper_wick = work["high"] - np.maximum(work["open"], work["close"])
    lower_wick = np.minimum(work["open"], work["close"]) - work["low"]
    work["upper_wick_ratio"] = _safe_divide(upper_wick.clip(lower=0), bar_range)
    work["lower_wick_ratio"] = _safe_divide(lower_wick.clip(lower=0), bar_range)
    long_wick = (work["upper_wick_ratio"] >= config.wick_ratio_threshold) | (
        work["lower_wick_ratio"] >= config.wick_ratio_threshold
    )
    work["long_wick_freq"] = long_wick.rolling(config.noise_window).mean()

    sign_switch = work["bar_direction"].ne(work["bar_direction"].shift(1)) & work["bar_direction"].notna()
    work["direction_switch_rate"] = sign_switch.rolling(config.switch_window).mean()

    vol_cutoff = work["rolling_vol"].quantile(config.high_vol_noise_quantile)
    work["high_vol_low_er_noise"] = (
        (work["rolling_vol"] >= vol_cutoff) & (work["efficiency_ratio"] <= config.er_range_threshold)
    )
    work["noise_score"] = (
        work["noise_path_ratio"].rank(pct=True)
        + work["long_wick_freq"].rank(pct=True)
        + work["direction_switch_rate"].rank(pct=True)
        + work["high_vol_low_er_noise"].astype(float)
    ) / 4.0

    work["ma60_gap"] = _safe_divide(
        work["close"] - work["close"].rolling(config.slow_ma_window).mean(),
        work["close"].rolling(config.slow_ma_window).mean().abs(),
    )

    return work



def extract_trend_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Extract contiguous trend segments and compute segment statistics."""

    work = df.copy().reset_index(drop=True)
    if "trend_direction" not in work.columns:
        raise ValueError("trend_direction column is required; call enrich_features first.")

    work = work[(work["regime"] == "trend") & (work["trend_direction"] != 0)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "direction",
                "start_time",
                "end_time",
                "bars",
                "cumulative_return",
                "mfe",
                "mae",
            ]
        )

    work["segment_id"] = (work["trend_direction"] != work["trend_direction"].shift(1)).cumsum()
    segments: list[dict[str, object]] = []

    for segment_id, grp in work.groupby("segment_id"):
        grp = grp.sort_values("datetime")
        direction = int(grp["trend_direction"].iloc[0])
        base_price = grp["close"].iloc[0]
        path_return = grp["close"] / base_price - 1.0
        cumulative_return = grp["close"].iloc[-1] / base_price - 1.0

        if direction > 0:
            mfe = path_return.max()
            mae = path_return.min()
            label = "up"
        else:
            short_path = -(path_return)
            mfe = short_path.max()
            mae = short_path.min()
            label = "down"

        segments.append(
            {
                "segment_id": int(segment_id),
                "direction": label,
                "start_time": grp["datetime"].iloc[0],
                "end_time": grp["datetime"].iloc[-1],
                "bars": int(len(grp)),
                "cumulative_return": float(cumulative_return),
                "mfe": float(mfe),
                "mae": float(mae),
            }
        )

    return pd.DataFrame(segments)


_base.enrich_features = enrich_features
_base.extract_trend_segments = extract_trend_segments
summarize_market_state = _base.summarize_market_state
plot_market_state_report = _base.plot_market_state_report



def main() -> None:
    """Example usage on a single futures CSV."""

    config = MarketStateConfig()
    sample_path = Path(__file__).resolve().parents[1] / "data_exact" / "futures" / "daily" / "IF2406.CFX.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    raw = pd.read_csv(sample_path)
    enriched = enrich_features(raw, config=config)
    segments = extract_trend_segments(enriched)
    summary = summarize_market_state(enriched, config=config, segments=segments)
    by_time = summarize_by_time_bucket(enriched)

    output_dir = Path(__file__).resolve().parent / "output" / "market_state_research_regime_fix"
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched.to_csv(output_dir / "enriched_features.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(output_dir / "trend_segments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {k: v for k, v in summary.items() if not isinstance(v, (pd.DataFrame, dict))}
    ]).to_csv(output_dir / "summary_scalar.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["regime_share"], index=[0]).to_csv(
        output_dir / "summary_regime_share.csv", index=False, encoding="utf-8-sig"
    )
    summary["sensitivity_analysis"].to_csv(
        output_dir / "summary_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    for name, table in by_time.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    report = plot_market_state_report(enriched, segments, output_dir / "market_state_report.png")
    if report is not None:
        _base.plt.close(report)

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
