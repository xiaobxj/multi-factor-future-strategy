from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import market_state_research as ms


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_exact"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data_exact"
FEATURE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = FEATURE_DIR / "output" / "composite_momentum_score"
NORMAL_DIST = NormalDist()


@dataclass
class CompositeMomentumConfig:
    """Configuration for the composite momentum pipeline."""

    universe: tuple[str, ...] = ("IF", "IM", "TL", "AU", "M", "I")
    sector_map: dict[str, str] = field(
        default_factory=lambda: {
            "IF": "equity_index",
            "IM": "equity_index",
            "TL": "rates",
            "AU": "precious_metals",
            "M": "agriculture",
            "I": "ferrous",
        }
    )
    short_window: int = 20
    medium_window: int = 60
    long_window: int = 120
    atr_window: int = 20
    er_short_window: int = 20
    er_medium_window: int = 60
    vol_short_window: int = 20
    vol_medium_window: int = 60
    ts_norm_window: int = 252
    ts_norm_min_periods: int = 126
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    cross_section_min_obs: int = 3
    feature_corr_prune_threshold: float = 0.85
    forward_windows: tuple[int, ...] = (5, 20)
    long_short_bucket_size: int = 2
    optimized_consistency_weight: float = 1.0
    optimized_trend_weight: float = 0.5
    optimized_neutral_weight: float = 0.0
    sector_score_min_obs: int = 2
    roll_confirm_days: int = 3
    roll_oi_lead_ratio: float = 1.05


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_product_prefix(ts_code: str) -> str:
    symbol = str(ts_code).upper().split(".")[0]
    match = re.match(r"([A-Z]+)", symbol)
    return match.group(1) if match else symbol


def safe_divide(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(a, pd.Series):
        index = a.index
    elif isinstance(b, pd.Series):
        index = b.index
    else:
        index = None
    left = pd.Series(a, index=index)
    right = pd.Series(b, index=index).replace(0, np.nan)
    return left / right


def rolling_compound_return(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def compute_true_range(df: pd.DataFrame) -> pd.Series:
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
    tr = compute_true_range(df)
    return tr.ewm(alpha=1 / max(window, 1), adjust=False, min_periods=window).mean()


def compute_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    direction = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return direction / path.replace(0, np.nan)


def compute_regression_slope(series: pd.Series, window: int) -> pd.Series:
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


def rolling_directional_quality(returns: pd.Series, window: int) -> pd.DataFrame:
    def _total(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.prod(1.0 + values) - 1.0)

    def _return_to_mae(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        path = np.cumprod(1.0 + values) - 1.0
        total = path[-1]
        if np.isclose(total, 0.0):
            return 0.0
        directional_path = np.sign(total) * path
        adverse = max(-directional_path.min(), 1e-6)
        return float(total / adverse)

    def _retention(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        path = np.cumprod(1.0 + values) - 1.0
        total = path[-1]
        if np.isclose(total, 0.0):
            return 0.0
        directional_path = np.sign(total) * path
        favorable = max(directional_path.max(), 1e-6)
        return float(total / favorable)

    def _ulcer_ratio(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        path = np.cumprod(1.0 + values) - 1.0
        total = path[-1]
        if np.isclose(total, 0.0):
            return 0.0
        directional_path = np.sign(total) * path
        drawdowns = np.minimum(directional_path, 0.0)
        ulcer = float(np.sqrt(np.mean(drawdowns**2)))
        return float(total / max(ulcer, 1e-6))

    return pd.DataFrame(
        {
            "window_total_return": returns.rolling(window).apply(_total, raw=True),
            "window_return_to_mae": returns.rolling(window).apply(_return_to_mae, raw=True),
            "window_retention": returns.rolling(window).apply(_retention, raw=True),
            "window_ulcer_ratio": returns.rolling(window).apply(_ulcer_ratio, raw=True),
        },
        index=returns.index,
    )


def _select_sticky_dominant_rows(merged: pd.DataFrame, config: CompositeMomentumConfig) -> pd.DataFrame:
    selected_rows: list[dict[str, object]] = []
    active_ts_code: str | None = None
    pending_ts_code: str | None = None
    pending_streak = 0

    for _, daily in merged.groupby("trade_date", sort=True):
        daily = daily.sort_values(["oi", "vol", "ts_code"], ascending=[False, False, True]).reset_index(drop=True)
        top_row = daily.iloc[0]
        selected_row = top_row
        switch_from_close = np.nan

        if active_ts_code is not None:
            active_match = daily.loc[daily["ts_code"] == active_ts_code]
            if active_match.empty:
                pending_ts_code = top_row["ts_code"]
                pending_streak = 0
            else:
                active_row = active_match.iloc[0]
                if top_row["ts_code"] == active_ts_code:
                    selected_row = active_row
                    pending_ts_code = active_ts_code
                    pending_streak = 0
                else:
                    oi_lead = (top_row["oi"] > 0) and (
                        (active_row["oi"] <= 0)
                        or (top_row["oi"] >= active_row["oi"] * config.roll_oi_lead_ratio)
                    )
                    if oi_lead and pending_ts_code == top_row["ts_code"]:
                        pending_streak += 1
                    elif oi_lead:
                        pending_ts_code = top_row["ts_code"]
                        pending_streak = 1
                    else:
                        pending_ts_code = active_ts_code
                        pending_streak = 0

                    if oi_lead and pending_streak >= config.roll_confirm_days:
                        selected_row = top_row
                        switch_from_close = float(active_row["close"])
                        pending_ts_code = top_row["ts_code"]
                        pending_streak = 0
                    else:
                        selected_row = active_row

        row = selected_row.to_dict()
        row["switch_from_close"] = switch_from_close
        selected_rows.append(row)
        active_ts_code = row["ts_code"]

    return pd.DataFrame(selected_rows).sort_values("trade_date").reset_index(drop=True)


def _apply_roll_adjustment(dominant: pd.DataFrame) -> pd.DataFrame:
    work = dominant.sort_values("trade_date").reset_index(drop=True).copy()
    work["contract_switch"] = work["ts_code"].ne(work["ts_code"].shift(1)).astype(int)
    if not work.empty:
        work.loc[0, "contract_switch"] = 0

    roll_ratios: list[float] = []
    adjustment_factors: list[float] = []
    adjustment_factor = 1.0
    prev_raw_close = np.nan

    for row in work.itertuples(index=False):
        roll_ratio = 1.0
        if row.contract_switch == 1:
            switch_from_close = row.switch_from_close
            if pd.isna(switch_from_close) or np.isclose(switch_from_close, 0.0):
                switch_from_close = prev_raw_close
            if pd.notna(switch_from_close) and pd.notna(row.close) and not np.isclose(row.close, 0.0):
                roll_ratio = float(switch_from_close / row.close)
                adjustment_factor *= roll_ratio
        roll_ratios.append(roll_ratio)
        adjustment_factors.append(adjustment_factor)
        prev_raw_close = row.close

    work["roll_ratio"] = roll_ratios
    work["roll_adjustment_factor"] = adjustment_factors

    for col in ["open", "high", "low", "close"]:
        work[f"raw_{col}"] = work[col]
        work[col] = work[col] * work["roll_adjustment_factor"]

    work["return"] = work["close"].pct_change()
    work["log_return"] = np.log(work["close"]).diff()
    return work


def load_dominant_future_panel(product: str, config: CompositeMomentumConfig) -> pd.DataFrame:
    futures_dir = DATA_DIR / "futures" / "daily"
    frames: list[pd.DataFrame] = []
    needed_cols = ["trade_date", "open", "high", "low", "close", "vol", "oi"]

    for path in sorted(futures_dir.glob("*.csv")):
        ts_code = path.stem
        if extract_product_prefix(ts_code) != product:
            continue
        df = pd.read_csv(path, usecols=lambda col: col in set(needed_cols))
        if df.empty or "close" not in df.columns:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        for col in ["open", "high", "low", "close", "vol", "oi"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df = df.dropna(subset=["trade_date", "open", "high", "low", "close"]).copy()
        if df.empty:
            continue
        df["ts_code"] = ts_code
        frames.append(df.loc[:, ["trade_date", "ts_code", "open", "high", "low", "close", "vol", "oi"]])

    if not frames:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "product",
                "sector",
                "ts_code",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "oi",
                "contract_switch",
                "roll_ratio",
                "roll_adjustment_factor",
                "return",
                "log_return",
            ]
        )

    merged = pd.concat(frames, ignore_index=True)
    merged["oi"] = merged["oi"].fillna(-1)
    merged["vol"] = merged["vol"].fillna(-1)
    merged = merged.sort_values(["trade_date", "oi", "vol", "ts_code"], ascending=[True, False, False, True])

    dominant = _select_sticky_dominant_rows(merged, config)
    dominant = _apply_roll_adjustment(dominant)
    dominant["product"] = product
    dominant["sector"] = config.sector_map.get(product, "other")
    return dominant


def compute_product_features(df: pd.DataFrame, config: CompositeMomentumConfig) -> pd.DataFrame:
    work = df.sort_values("trade_date").reset_index(drop=True).copy()

    work["abs_return"] = work["return"].abs()
    work["sign_return"] = np.sign(work["return"]).fillna(0.0)
    work["atr"] = compute_atr(work, config.atr_window)
    work["vol_20"] = work["return"].rolling(config.vol_short_window).std()
    work["vol_60"] = work["return"].rolling(config.vol_medium_window).std()

    work["cumret_20"] = rolling_compound_return(work["return"], config.short_window)
    work["cumret_60"] = rolling_compound_return(work["return"], config.medium_window)
    work["cumret_120"] = rolling_compound_return(work["return"], config.long_window)

    work["ma_20"] = work["close"].rolling(config.short_window).mean()
    work["ma_60"] = work["close"].rolling(config.medium_window).mean()
    work["ma_120"] = work["close"].rolling(config.long_window).mean()

    work["slope_20"] = compute_regression_slope(np.log(work["close"]), config.short_window)
    work["slope_60"] = compute_regression_slope(np.log(work["close"]), config.medium_window)

    work["er_20"] = compute_efficiency_ratio(work["close"], config.er_short_window)
    work["er_60"] = compute_efficiency_ratio(work["close"], config.er_medium_window)
    work["abs_path_20"] = work["abs_return"].rolling(config.short_window).sum()
    work["abs_path_60"] = work["abs_return"].rolling(config.medium_window).sum()

    work["sign_mean_20"] = work["sign_return"].rolling(config.short_window).mean()
    work["sign_mean_60"] = work["sign_return"].rolling(config.medium_window).mean()
    sign_switch = work["sign_return"].ne(work["sign_return"].shift(1)) & work["sign_return"].ne(0)
    work["switch_rate_20"] = sign_switch.rolling(config.short_window).mean()
    work["switch_rate_60"] = sign_switch.rolling(config.medium_window).mean()
    work["switch_stability_60"] = -work["switch_rate_60"]
    work["consistency_switch_adjusted_20"] = work["sign_mean_20"] * (1.0 - work["switch_rate_20"].fillna(0.0))

    low_20 = work["low"].rolling(config.short_window).min()
    high_20 = work["high"].rolling(config.short_window).max()
    low_60 = work["low"].rolling(config.medium_window).min()
    high_60 = work["high"].rolling(config.medium_window).max()
    prev_high_20 = work["high"].shift(1).rolling(config.short_window).max()
    prev_low_20 = work["low"].shift(1).rolling(config.short_window).min()

    work["range_pct_20"] = 2.0 * safe_divide(work["close"] - low_20, high_20 - low_20) - 1.0
    work["range_pct_60"] = 2.0 * safe_divide(work["close"] - low_60, high_60 - low_60) - 1.0
    breakout_up = safe_divide(work["close"] - prev_high_20, work["atr"])
    breakout_down = safe_divide(work["close"] - prev_low_20, work["atr"])
    work["donchian_breakout_20"] = np.where(work["close"] >= prev_high_20, breakout_up, breakout_down)

    ma_state = 0.5 * np.sign(work["ma_20"] - work["ma_60"]) + 0.5 * np.sign(work["ma_60"] - work["ma_120"])
    ma_gap_stack = safe_divide((work["ma_20"] - work["ma_60"]) + (work["ma_60"] - work["ma_120"]), work["atr"])
    work["ma_stack_soft"] = 0.5 * ma_state + 0.5 * np.tanh(ma_gap_stack / 2.0)

    dq_60 = rolling_directional_quality(work["return"], config.medium_window)
    work["drawdown_return_to_mae_60"] = dq_60["window_return_to_mae"]
    work["drawdown_retention_60"] = dq_60["window_retention"]
    work["drawdown_ulcer_ratio_60"] = dq_60["window_ulcer_ratio"]

    work["strength_ret_20"] = safe_divide(work["cumret_20"], work["vol_20"])
    work["strength_ret_60"] = safe_divide(work["cumret_60"], work["vol_60"])
    work["strength_ret_120"] = safe_divide(work["cumret_120"], work["vol_60"])
    work["strength_ma_gap_60"] = safe_divide(work["close"] - work["ma_60"], work["atr"])
    work["strength_slope_20"] = safe_divide(work["slope_20"], work["vol_20"])

    work["smooth_signed_er_20"] = np.sign(work["cumret_20"]).replace(0, np.nan) * work["er_20"]
    work["smooth_signed_er_60"] = np.sign(work["cumret_60"]).replace(0, np.nan) * work["er_60"]
    work["smooth_path_ratio_20"] = safe_divide(work["cumret_20"], work["abs_path_20"])
    work["smooth_path_ratio_60"] = safe_divide(work["cumret_60"], work["abs_path_60"])

    for horizon in config.forward_windows:
        work[f"forward_return_{horizon}"] = work["close"].shift(-horizon) / work["close"] - 1.0
        work[f"forward_vol_adj_return_{horizon}"] = safe_divide(work[f"forward_return_{horizon}"], work["vol_20"])

    return work


def rolling_winsorize(series: pd.Series, config: CompositeMomentumConfig) -> pd.Series:
    lower = series.rolling(config.ts_norm_window, min_periods=config.ts_norm_min_periods).quantile(config.winsor_lower_q)
    upper = series.rolling(config.ts_norm_window, min_periods=config.ts_norm_min_periods).quantile(config.winsor_upper_q)
    return series.clip(lower=lower, upper=upper)


def percentile_to_normal(series: pd.Series) -> pd.Series:
    clipped = series.clip(lower=1e-4, upper=1 - 1e-4)
    return clipped.map(NORMAL_DIST.inv_cdf)


def ts_normalize_feature(series: pd.Series, config: CompositeMomentumConfig) -> pd.Series:
    winsorized = rolling_winsorize(series, config)
    percentile = winsorized.rolling(config.ts_norm_window, min_periods=config.ts_norm_min_periods).rank(pct=True)
    return percentile_to_normal(percentile)


def grouped_cross_sectional_zscore(
    series: pd.Series,
    group_keys: list[pd.Series] | pd.Series,
    min_obs: int,
    fillna_value: float | None = None,
) -> pd.Series:
    grouped = series.groupby(group_keys)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    count = grouped.transform("count")
    zscore = (series - mean) / std
    zscore = zscore.where(count >= min_obs)
    if fillna_value is not None:
        zscore = zscore.fillna(fillna_value)
    return zscore


def cross_sectional_zscore(series: pd.Series, dates: pd.Series, min_obs: int) -> pd.Series:
    return grouped_cross_sectional_zscore(series, dates, min_obs=min_obs)


def cross_sectional_sector_zscore(
    series: pd.Series,
    dates: pd.Series,
    sectors: pd.Series,
    min_obs: int,
) -> pd.Series:
    return grouped_cross_sectional_zscore(series, [dates, sectors], min_obs=min_obs, fillna_value=0.0)


def normalize_features(panel: pd.DataFrame, feature_cols: list[str], config: CompositeMomentumConfig) -> pd.DataFrame:
    work = panel.copy()
    for col in feature_cols:
        ts_col = f"{col}_ts_norm"
        cs_col = f"{col}_norm"
        work[ts_col] = work.groupby("product", group_keys=False)[col].apply(lambda s: ts_normalize_feature(s, config))
        work[cs_col] = cross_sectional_zscore(work[ts_col], work["trade_date"], config.cross_section_min_obs)
    return work


def prune_correlated_features(df: pd.DataFrame, feature_cols: list[str], threshold: float) -> list[str]:
    available = [col for col in feature_cols if df[col].notna().sum() > 0]
    if len(available) <= 1:
        return available
    corr = df[available].corr().abs()
    selected: list[str] = []
    for col in available:
        if not selected:
            selected.append(col)
            continue
        max_corr = corr.loc[col, selected].max()
        if pd.isna(max_corr) or max_corr < threshold:
            selected.append(col)
    return selected


def compute_inverse_correlation_weights(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    if not feature_cols:
        return pd.Series(dtype=float)
    if len(feature_cols) == 1:
        return pd.Series({feature_cols[0]: 1.0})
    corr = df[feature_cols].corr().abs().fillna(0.0)
    corr = corr.mask(np.eye(len(corr), dtype=bool), 0.0)
    diversification_penalty = corr.sum(axis=1)
    raw_weight = 1.0 / (1.0 + diversification_penalty)
    return raw_weight / raw_weight.sum()


def weighted_row_mean(df: pd.DataFrame, feature_cols: list[str], weights: pd.Series) -> pd.Series:
    if not feature_cols:
        return pd.Series(np.nan, index=df.index)
    values = df[feature_cols]
    aligned_weights = weights.reindex(feature_cols).fillna(0.0)
    weighted = values.mul(aligned_weights, axis=1)
    valid_weight = values.notna().mul(aligned_weights, axis=1).sum(axis=1)
    return weighted.sum(axis=1) / valid_weight.replace(0, np.nan)


def residualize_by_date(
    df: pd.DataFrame,
    target_col: str,
    exposure_cols: list[str],
    min_obs: int,
) -> pd.Series:
    residuals = pd.Series(np.nan, index=df.index, dtype=float)

    for _, grp in df.groupby("trade_date"):
        subset = grp.dropna(subset=[target_col, *exposure_cols]).copy()
        if len(subset) < max(min_obs, len(exposure_cols) + 2):
            continue
        x = subset[exposure_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(subset)), x])
        y = subset[target_col].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals.loc[subset.index] = y - x @ beta

    return residuals


def evaluate_cross_section(
    panel: pd.DataFrame,
    score_col: str,
    target_col: str,
    bucket_size: int,
) -> dict[str, float | int | str]:
    ic_values: list[float] = []
    spread_values: list[float] = []

    for _, grp in panel.groupby("trade_date"):
        subset = grp[[score_col, target_col]].dropna()
        if len(subset) < max(bucket_size * 2, 3):
            continue
        ic = subset[score_col].corr(subset[target_col], method="spearman")
        if pd.notna(ic):
            ic_values.append(float(ic))
        ranked = subset.sort_values(score_col)
        short_mean = ranked.head(bucket_size)[target_col].mean()
        long_mean = ranked.tail(bucket_size)[target_col].mean()
        spread_values.append(float(long_mean - short_mean))

    ic_series = pd.Series(ic_values, dtype=float)
    spread_series = pd.Series(spread_values, dtype=float)
    return {
        "score": score_col,
        "target": target_col,
        "observations": int(ic_series.count()),
        "mean_ic": float(ic_series.mean()) if not ic_series.empty else np.nan,
        "ic_std": float(ic_series.std()) if ic_series.count() > 1 else np.nan,
        "ic_ir": float(ic_series.mean() / ic_series.std()) if ic_series.count() > 1 and ic_series.std() else np.nan,
        "ic_hit_rate": float((ic_series > 0).mean()) if not ic_series.empty else np.nan,
        "mean_long_short": float(spread_series.mean()) if not spread_series.empty else np.nan,
        "ls_std": float(spread_series.std()) if spread_series.count() > 1 else np.nan,
        "ls_ir": float(spread_series.mean() / spread_series.std())
        if spread_series.count() > 1 and spread_series.std()
        else np.nan,
    }



def evaluate_scores(
    panel: pd.DataFrame,
    score_cols: list[str],
    target_cols: list[str],
    bucket_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for score_col in score_cols:
        for target_col in target_cols:
            rows.append(evaluate_cross_section(panel, score_col, target_col, bucket_size))
    return pd.DataFrame(rows).sort_values(["target", "score"]).reset_index(drop=True)

def build_composite_scores(panel: pd.DataFrame, config: CompositeMomentumConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dimension_map = {
        "strength": [
            "strength_ret_20_norm",
            "strength_ret_60_norm",
            "strength_ret_120_norm",
            "strength_ma_gap_60_norm",
            "strength_slope_20_norm",
        ],
        "smoothness": [
            "smooth_signed_er_20_norm",
            "smooth_signed_er_60_norm",
            "smooth_path_ratio_20_norm",
            "smooth_path_ratio_60_norm",
        ],
        "consistency": [
            "sign_mean_20_norm",
            "sign_mean_60_norm",
            "consistency_switch_adjusted_20_norm",
            "switch_stability_60_norm",
        ],
        "breakout": [
            "range_pct_20_norm",
            "range_pct_60_norm",
            "donchian_breakout_20_norm",
            "ma_stack_soft_norm",
        ],
        "drawdown": [
            "drawdown_return_to_mae_60_norm",
            "drawdown_retention_60_norm",
            "drawdown_ulcer_ratio_60_norm",
        ],
    }
    minimal_feature_map = {
        "strength": "strength_ret_60_norm",
        "smoothness": "smooth_signed_er_60_norm",
        "consistency": "consistency_switch_adjusted_20_norm",
        "breakout": "range_pct_60_norm",
        "drawdown": "drawdown_return_to_mae_60_norm",
    }

    work = panel.copy()
    weight_rows: list[dict[str, object]] = []
    dimension_score_cols: list[str] = []
    dimension_score_cols_minimal: list[str] = []

    for dimension, feature_cols in dimension_map.items():
        selected = prune_correlated_features(work, feature_cols, config.feature_corr_prune_threshold)
        weights = compute_inverse_correlation_weights(work, selected)
        raw_col = f"{dimension}_score_raw"
        score_col = f"{dimension}_score"
        work[raw_col] = weighted_row_mean(work, selected, weights)
        work[score_col] = cross_sectional_zscore(work[raw_col], work["trade_date"], config.cross_section_min_obs)
        dimension_score_cols.append(score_col)

        minimal_raw_col = f"{dimension}_score_minimal_raw"
        minimal_col = f"{dimension}_score_minimal"
        minimal_feature = minimal_feature_map[dimension]
        work[minimal_raw_col] = work[minimal_feature]
        work[minimal_col] = cross_sectional_zscore(work[minimal_raw_col], work["trade_date"], config.cross_section_min_obs)
        dimension_score_cols_minimal.append(minimal_col)

        corr = work[selected].corr() if selected else pd.DataFrame()
        for feature in feature_cols:
            selected_flag = feature in selected
            max_selected_corr = np.nan
            if selected_flag and not corr.empty:
                peer_corr = corr.loc[feature, selected].drop(labels=[feature], errors="ignore")
                if not peer_corr.empty:
                    max_selected_corr = float(peer_corr.abs().max())
            weight_rows.append(
                {
                    "dimension": dimension,
                    "feature": feature,
                    "selected": selected_flag,
                    "weight": float(weights.get(feature, 0.0)) if selected_flag else 0.0,
                    "availability": int(work[feature].notna().sum()),
                    "max_selected_corr": max_selected_corr,
                }
            )

    work["composite_score_full_raw"] = work[dimension_score_cols].mean(axis=1)
    work["composite_score_full"] = cross_sectional_zscore(
        work["composite_score_full_raw"], work["trade_date"], config.cross_section_min_obs
    )
    work["composite_score_minimal_raw"] = work[dimension_score_cols_minimal].mean(axis=1)
    work["composite_score_minimal"] = cross_sectional_zscore(
        work["composite_score_minimal_raw"], work["trade_date"], config.cross_section_min_obs
    )

    work["log_vol_20"] = np.log(work["vol_20"].replace(0, np.nan))
    work["composite_score_full_vol_neutral"] = residualize_by_date(
        work,
        target_col="composite_score_full",
        exposure_cols=["log_vol_20"],
        min_obs=config.cross_section_min_obs,
    )
    work["composite_score_full_vol_neutral"] = cross_sectional_zscore(
        work["composite_score_full_vol_neutral"], work["trade_date"], config.cross_section_min_obs
    )

    weight_df = pd.DataFrame(weight_rows).sort_values(
        ["dimension", "selected", "weight"], ascending=[True, False, False]
    )
    return work, weight_df



def annotate_market_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    regime_frames: list[pd.DataFrame] = []

    for product, grp in panel.groupby("product"):
        inp = grp[["trade_date", "open", "high", "low", "close"]].copy()
        enriched = ms.enrich_features(inp)
        regime_part = enriched[
            ["datetime", "regime", "trend_score", "range_score", "compression_regime", "noise_score"]
        ].rename(columns={"datetime": "trade_date"})
        regime_part["product"] = product
        regime_part["regime_signal"] = regime_part["regime"].map({"trend": 1.0, "neutral": 0.0, "range": -1.0})
        regime_frames.append(regime_part)

    return pd.concat(regime_frames, ignore_index=True).sort_values(["trade_date", "product"]).reset_index(drop=True)


def build_optimized_scores(
    panel: pd.DataFrame,
    regime_df: pd.DataFrame,
    config: CompositeMomentumConfig,
) -> pd.DataFrame:
    work = panel.merge(regime_df, on=["trade_date", "product"], how="left")
    work["regime_weight"] = work["regime"].map(
        {"trend": 1.0, "range": 1.0, "neutral": config.optimized_neutral_weight}
    ).fillna(config.optimized_neutral_weight)
    work["trend_following_score"] = work[
        ["strength_score", "smoothness_score", "breakout_score", "drawdown_score"]
    ].mean(axis=1)
    work["trend_state_score"] = work["trend_following_score"] * work["regime_signal"].fillna(0.0) * work["regime_weight"]
    work["composite_score_optimized_raw"] = (
        config.optimized_consistency_weight * work["consistency_score"]
        + config.optimized_trend_weight * work["trend_state_score"]
    )
    work["optimized_score_global"] = cross_sectional_zscore(
        work["composite_score_optimized_raw"], work["trade_date"], config.cross_section_min_obs
    )
    work["optimized_score_sector"] = cross_sectional_sector_zscore(
        work["composite_score_optimized_raw"],
        work["trade_date"],
        work["sector"],
        min_obs=config.sector_score_min_obs,
    )
    # Backward-compatible alias for downstream consumers that still expect the old column.
    work["composite_score_optimized"] = work["optimized_score_global"]
    return work


def summarize_regimes(regime_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        regime_df.groupby("product")["regime"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .pivot(index="product", columns="regime", values="share")
        .fillna(0.0)
        .reset_index()
    )
    return summary

def build_long_short_series(panel: pd.DataFrame, score_col: str, target_col: str, bucket_size: int) -> pd.Series:
    rows: list[tuple[pd.Timestamp, float]] = []

    for trade_date, grp in panel.groupby("trade_date"):
        subset = grp[[score_col, target_col]].dropna().sort_values(score_col)
        if len(subset) < max(bucket_size * 2, 3):
            continue
        short_mean = subset.head(bucket_size)[target_col].mean()
        long_mean = subset.tail(bucket_size)[target_col].mean()
        rows.append((trade_date, float(long_mean - short_mean)))

    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows)).sort_index()


def plot_report(panel: pd.DataFrame, evaluation_df: pd.DataFrame, weight_df: pd.DataFrame) -> plt.Figure:
    latest_date = panel["trade_date"].max()
    latest = panel.loc[panel["trade_date"] == latest_date].sort_values("optimized_score_global")

    optimized_global_ls = build_long_short_series(panel, "optimized_score_global", "forward_return_20", bucket_size=2)
    optimized_sector_ls = build_long_short_series(panel, "optimized_score_sector", "forward_return_20", bucket_size=2)
    full_ls = build_long_short_series(panel, "composite_score_full", "forward_return_20", bucket_size=2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].barh(latest["product"], latest["optimized_score_global"], color="#1f77b4", alpha=0.85)
    axes[0, 0].set_title(f"Latest Global Optimized Score ({latest_date:%Y-%m-%d})")
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].grid(axis="x", alpha=0.2)

    if not optimized_global_ls.empty:
        axes[0, 1].plot(optimized_global_ls.index, (1.0 + optimized_global_ls).cumprod(), label="optimized_global", linewidth=1.5)
    if not optimized_sector_ls.empty:
        axes[0, 1].plot(optimized_sector_ls.index, (1.0 + optimized_sector_ls).cumprod(), label="optimized_sector", linewidth=1.3)
    if not full_ls.empty:
        axes[0, 1].plot(full_ls.index, (1.0 + full_ls).cumprod(), label="full", linewidth=1.1)
    axes[0, 1].set_title("Top-Bottom Spread Equity Curve (20D Forward Return)")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend()

    key_scores = evaluation_df[
        (evaluation_df["target"] == "forward_return_20")
        & (
            evaluation_df["score"].isin(
                [
                    "optimized_score_global",
                    "optimized_score_sector",
                    "consistency_score",
                    "trend_state_score",
                    "composite_score_full",
                ]
            )
        )
    ].copy()
    axes[1, 0].bar(key_scores["score"], key_scores["mean_ic"], color="#ff7f0e")
    axes[1, 0].set_title("Mean Daily Spearman IC")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].grid(axis="y", alpha=0.2)

    selected_weights = weight_df[weight_df["selected"]].copy()
    selected_weights["label"] = (
        selected_weights["dimension"] + ":" + selected_weights["feature"].str.replace("_norm", "", regex=False)
    )
    axes[1, 1].barh(selected_weights["label"], selected_weights["weight"], color="#2ca02c")
    axes[1, 1].set_title("Selected Feature Weights")
    axes[1, 1].grid(axis="x", alpha=0.2)

    fig.tight_layout()
    return fig


def build_panel(config: CompositeMomentumConfig) -> pd.DataFrame:
    product_frames: list[pd.DataFrame] = []

    for product in config.universe:
        raw = load_dominant_future_panel(product, config)
        if raw.empty:
            print(f"[skip] No futures data found for {product}")
            continue
        features = compute_product_features(raw, config)
        product_frames.append(features)
        print(f"[done] {product}: {len(features)} dates")

    if not product_frames:
        raise FileNotFoundError("No valid futures series found for the configured universe.")

    return pd.concat(product_frames, ignore_index=True).sort_values(["trade_date", "product"]).reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    config = CompositeMomentumConfig()
    panel = build_panel(config)

    raw_feature_cols = [
        "strength_ret_20",
        "strength_ret_60",
        "strength_ret_120",
        "strength_ma_gap_60",
        "strength_slope_20",
        "smooth_signed_er_20",
        "smooth_signed_er_60",
        "smooth_path_ratio_20",
        "smooth_path_ratio_60",
        "sign_mean_20",
        "sign_mean_60",
        "consistency_switch_adjusted_20",
        "switch_stability_60",
        "range_pct_20",
        "range_pct_60",
        "donchian_breakout_20",
        "ma_stack_soft",
        "drawdown_return_to_mae_60",
        "drawdown_retention_60",
        "drawdown_ulcer_ratio_60",
    ]
    feature_score_cols = [f"{col}_norm" for col in raw_feature_cols]

    normalized = normalize_features(panel, raw_feature_cols, config)
    scored, weight_df = build_composite_scores(normalized, config)
    regime_df = annotate_market_regimes(panel)
    scored = build_optimized_scores(scored, regime_df, config)

    target_cols = ["forward_return_20"]
    dimension_score_cols = [
        "strength_score",
        "smoothness_score",
        "consistency_score",
        "breakout_score",
        "drawdown_score",
        "trend_following_score",
        "trend_state_score",
        "composite_score_full",
        "composite_score_minimal",
        "composite_score_full_vol_neutral",
        "optimized_score_global",
        "optimized_score_sector",
    ]

    evaluation_df = evaluate_scores(scored, dimension_score_cols, target_cols, config.long_short_bucket_size)
    feature_target_cols = ["forward_return_20"]
    feature_diagnostics_df = evaluate_scores(normalized, feature_score_cols, feature_target_cols, config.long_short_bucket_size)
    regime_summary_df = summarize_regimes(regime_df)

    latest_snapshot = (
        scored.sort_values(["trade_date", "product"])
        .groupby("product", as_index=False)
        .last()[
            [
                "trade_date",
                "product",
                "sector",
                "ts_code",
                "regime",
                "close",
                "vol_20",
                "consistency_score",
                "trend_following_score",
                "trend_state_score",
                "composite_score_full",
                "composite_score_minimal",
                "optimized_score_global",
                "optimized_score_sector",
                "composite_score_optimized",
            ]
        ]
    )

    output_panel_cols = [
        "trade_date",
        "product",
        "sector",
        "ts_code",
        "contract_switch",
        "roll_ratio",
        "roll_adjustment_factor",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "oi",
        "return",
        "vol_20",
        "vol_60",
        "forward_return_5",
        "forward_return_20",
        "forward_vol_adj_return_5",
        "forward_vol_adj_return_20",
        "regime",
        "regime_signal",
        "trend_score",
        "range_score",
        "compression_regime",
        "noise_score",
        "strength_score",
        "smoothness_score",
        "consistency_score",
        "breakout_score",
        "drawdown_score",
        "trend_following_score",
        "trend_state_score",
        "strength_score_minimal",
        "smoothness_score_minimal",
        "consistency_score_minimal",
        "breakout_score_minimal",
        "drawdown_score_minimal",
        "composite_score_full",
        "composite_score_minimal",
        "composite_score_full_vol_neutral",
        "optimized_score_global",
        "optimized_score_sector",
        "composite_score_optimized",
    ]

    scored[output_panel_cols].to_csv(OUTPUT_DIR / "composite_momentum_panel.csv", index=False, encoding="utf-8-sig")
    latest_snapshot.to_csv(OUTPUT_DIR / "latest_snapshot.csv", index=False, encoding="utf-8-sig")
    weight_df.to_csv(OUTPUT_DIR / "dimension_feature_weights.csv", index=False, encoding="utf-8-sig")
    evaluation_df.to_csv(OUTPUT_DIR / "score_evaluation.csv", index=False, encoding="utf-8-sig")
    feature_diagnostics_df.to_csv(OUTPUT_DIR / "feature_diagnostics.csv", index=False, encoding="utf-8-sig")
    regime_summary_df.to_csv(OUTPUT_DIR / "regime_summary.csv", index=False, encoding="utf-8-sig")

    report = plot_report(scored, evaluation_df, weight_df)
    report.savefig(OUTPUT_DIR / "composite_momentum_report.png", dpi=150, bbox_inches="tight")
    plt.close(report)

    print(f"Saved composite panel to {OUTPUT_DIR / 'composite_momentum_panel.csv'}")
    print(f"Saved latest snapshot to {OUTPUT_DIR / 'latest_snapshot.csv'}")
    print(f"Saved evaluation summary to {OUTPUT_DIR / 'score_evaluation.csv'}")
    print(f"Saved feature diagnostics to {OUTPUT_DIR / 'feature_diagnostics.csv'}")
    print(f"Saved regime summary to {OUTPUT_DIR / 'regime_summary.csv'}")
    print(f"Saved report to {OUTPUT_DIR / 'composite_momentum_report.png'}")


if __name__ == "__main__":
    main()
















