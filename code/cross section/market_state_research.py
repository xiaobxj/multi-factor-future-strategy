from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class MarketStateConfig:
    """Config for market state research."""

    adx_window: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 18.0
    directional_window: int = 20
    volatility_window: int = 20
    ratio_trend_threshold: float = 1.0
    ratio_range_threshold: float = 0.35
    efficiency_window: int = 20
    er_trend_threshold: float = 0.45
    er_range_threshold: float = 0.2
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    atr_window: int = 14
    donchian_window: int = 20
    compression_bandwidth_threshold: float = 0.08
    compression_atr_threshold: float = 0.015
    compression_donchian_threshold: float = 0.03
    fast_ma_window: int = 20
    slow_ma_window: int = 60
    slope_window: int = 20
    trend_segment_method: str = "ma"
    segment_return_window: int = 20
    segment_return_threshold: float = 0.0
    large_move_std_window: int = 60
    large_move_return_multiple: float = 1.8
    large_move_tr_multiple: float = 1.5
    noise_window: int = 20
    wick_ratio_threshold: float = 0.5
    switch_window: int = 20
    high_vol_noise_quantile: float = 0.7
    clip_return_quantile: float = 0.001
    session_bins: tuple[int, ...] = (0, 9, 11, 13, 15, 21, 24)
    session_labels: tuple[str, ...] = (
        "overnight",
        "day_morning",
        "day_mid",
        "day_afternoon",
        "pre_night",
        "night",
    )
    sensitivity_grid: dict[str, list[float]] = field(
        default_factory=lambda: {
            "adx_trend_threshold": [20.0, 25.0, 30.0],
            "adx_range_threshold": [15.0, 18.0, 20.0],
            "er_trend_threshold": [0.35, 0.45, 0.55],
            "er_range_threshold": [0.15, 0.2, 0.25],
        }
    )


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _prepare_input(df: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    """Normalize input schema and sort chronologically."""

    work = df.copy()
    _validate_columns(work, ["open", "high", "low", "close"])

    if "datetime" in work.columns:
        work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    elif "trade_time" in work.columns:
        work["datetime"] = pd.to_datetime(work["trade_time"], errors="coerce")
    elif "timestamp" in work.columns:
        work["datetime"] = pd.to_datetime(work["timestamp"], errors="coerce")
    elif "trade_date" in work.columns:
        work["datetime"] = pd.to_datetime(work["trade_date"], errors="coerce")
    else:
        work["datetime"] = pd.RangeIndex(len(work))

    for col in ["open", "high", "low", "close", "volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    work = work.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    ret = work["close"].pct_change()
    lower = ret.quantile(config.clip_return_quantile)
    upper = ret.quantile(1 - config.clip_return_quantile)
    work["return"] = ret.clip(lower=lower, upper=upper)
    work["log_return"] = np.log(work["close"]).diff()
    work["abs_return"] = work["return"].abs()
    work["bar_direction"] = np.sign(work["close"].diff()).replace(0, np.nan)

    if pd.api.types.is_datetime64_any_dtype(work["datetime"]):
        work["hour"] = work["datetime"].dt.hour
        work["minute"] = work["datetime"].dt.minute
        work["time_bucket"] = work["datetime"].dt.strftime("%H:%M")
        work["session"] = pd.cut(
            work["hour"],
            bins=config.session_bins,
            labels=config.session_labels,
            right=False,
            include_lowest=True,
        ).astype("object")
    else:
        work["hour"] = np.nan
        work["minute"] = np.nan
        work["time_bucket"] = np.nan
        work["session"] = "unknown"

    return work


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute true range."""

    prev_close = df["close"].shift(1)
    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return tr_components.max(axis=1)


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute ATR using Wilder-style smoothing approximation via EMA."""

    tr = compute_true_range(df)
    return tr.ewm(alpha=1 / max(window, 1), adjust=False, min_periods=window).mean()


def compute_adx(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute ADX, +DI, -DI."""

    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    tr = compute_true_range(df)
    atr = tr.ewm(alpha=1 / max(window, 1), adjust=False, min_periods=window).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / max(window, 1), adjust=False, min_periods=window
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / max(window, 1), adjust=False, min_periods=window
    ).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / max(window, 1), adjust=False, min_periods=window).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def compute_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Compute efficiency ratio."""

    direction = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return direction / path.replace(0, np.nan)


def compute_regression_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling OLS slope on price levels."""

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        num = ((x - x_mean) * (values - y_mean)).sum()
        return num / denom if denom != 0 else np.nan

    return series.rolling(window).apply(_slope, raw=True)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


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
    work["regime"] = np.select(
        [trend_votes >= 2, range_votes >= 2],
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

    work["trend_direction"] = pd.Series(trend_signal, index=work.index).fillna(0).astype(int)

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

    work = work[work["trend_direction"] != 0].copy()
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


def summarize_market_state(
    df: pd.DataFrame, config: MarketStateConfig | None = None, segments: pd.DataFrame | None = None
) -> dict[str, object]:
    """Return high-level summary statistics for market state research."""

    config = config or MarketStateConfig()
    work = df.copy()
    if segments is None:
        segments = extract_trend_segments(work)

    regime_share = work["regime"].value_counts(normalize=True, dropna=False).to_dict()
    sensitivity_rows: list[dict[str, float]] = []

    for trend_thr in config.sensitivity_grid.get("adx_trend_threshold", [config.adx_trend_threshold]):
        for range_thr in config.sensitivity_grid.get("adx_range_threshold", [config.adx_range_threshold]):
            trend_ratio = float((work["adx"] >= trend_thr).mean())
            range_ratio = float((work["adx"] <= range_thr).mean())
            sensitivity_rows.append(
                {
                    "adx_trend_threshold": float(trend_thr),
                    "adx_range_threshold": float(range_thr),
                    "trend_share": trend_ratio,
                    "range_share": range_ratio,
                }
            )

    trend_summary = {
        "segment_count": int(len(segments)),
        "avg_duration": float(segments["bars"].mean()) if not segments.empty else np.nan,
        "median_duration": float(segments["bars"].median()) if not segments.empty else np.nan,
        "avg_segment_return": float(segments["cumulative_return"].mean()) if not segments.empty else np.nan,
        "median_segment_return": float(segments["cumulative_return"].median()) if not segments.empty else np.nan,
    }

    up_segments = segments[segments["direction"] == "up"]
    down_segments = segments[segments["direction"] == "down"]

    return {
        "config": asdict(config),
        "regime_share": regime_share,
        "compression_share": float(work["compression_regime"].mean()),
        "large_vol_bar_share": float(work["large_vol_bar"].mean()),
        "avg_noise_score": float(work["noise_score"].mean()),
        "trend_summary": trend_summary,
        "up_duration_mean": float(up_segments["bars"].mean()) if not up_segments.empty else np.nan,
        "down_duration_mean": float(down_segments["bars"].mean()) if not down_segments.empty else np.nan,
        "sensitivity_analysis": pd.DataFrame(sensitivity_rows),
    }


def summarize_by_time_bucket(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize market state by hour, minute, session, and time bucket."""

    work = df.copy()

    def _aggregate(group_key: str) -> pd.DataFrame:
        grouped = (
            work.groupby(group_key, dropna=False)
            .agg(
                bars=("close", "size"),
                big_move_share=("large_vol_bar", "mean"),
                mean_abs_return=("abs_return", "mean"),
                mean_tr=("tr", "mean"),
                noise_score=("noise_score", "mean"),
                switch_rate=("direction_switch_rate", "mean"),
                compression_share=("compression_regime", "mean"),
                trend_share=("regime", lambda x: (x == "trend").mean()),
                range_share=("regime", lambda x: (x == "range").mean()),
            )
            .reset_index()
        )
        return grouped.sort_values(group_key)

    return {
        "by_hour": _aggregate("hour"),
        "by_minute": _aggregate("minute"),
        "by_session": _aggregate("session"),
        "by_time_bucket": _aggregate("time_bucket"),
    }


def plot_market_state_report(
    df: pd.DataFrame,
    segments: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a compact market-state report."""

    work = df.copy()
    segments = segments if segments is not None else extract_trend_segments(work)
    by_time = summarize_by_time_bucket(work)

    regime_map = {"trend": 1, "neutral": 0, "range": -1}
    regime_numeric = work["regime"].map(regime_map)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(work["datetime"], work["close"], color="#1f77b4", linewidth=1.1, label="close")
    ax2 = axes[0, 0].twinx()
    ax2.plot(work["datetime"], regime_numeric, color="#d62728", linewidth=1.0, alpha=0.7, label="regime")
    axes[0, 0].set_title("Price And Regime")
    axes[0, 0].grid(alpha=0.2)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["range", "neutral", "trend"])

    hour_df = by_time["by_hour"].dropna(subset=["hour"])
    axes[0, 1].bar(hour_df["hour"].astype(str), hour_df["mean_abs_return"], color="#ff7f0e")
    axes[0, 1].set_title("Mean Absolute Return By Hour")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[1, 0].bar(hour_df["hour"].astype(str), hour_df["noise_score"], color="#2ca02c")
    axes[1, 0].set_title("Noise Score By Hour")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(axis="y", alpha=0.2)

    if segments.empty:
        axes[1, 1].text(0.5, 0.5, "No trend segments", ha="center", va="center")
    else:
        axes[1, 1].hist(segments["bars"], bins=min(30, len(segments)), color="#9467bd", edgecolor="white")
    axes[1, 1].set_title("Trend Segment Duration Distribution")
    axes[1, 1].grid(alpha=0.2)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


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

    output_dir = Path(__file__).resolve().parent / "output" / "market_state_research"
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
    plot_market_state_report(enriched, segments, output_dir / "market_state_report.png")

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
