from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
CONFIG_MODULE_PATH = CODE_DIR / "backtest_engine_v3_4_calmar_vol20_cap60_looser_dd2.py"


def load_configured_backtest_module():
    spec = importlib.util.spec_from_file_location("bt_v34_report_cfg", CONFIG_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {CONFIG_MODULE_PATH}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    bt = cfg.load_base()
    cfg.configure(bt)
    return bt


def prepare_outputs(bt):
    bt.ensure_dirs()
    return {
        "equity_curve": bt.OUTPUT_DIR / "equity_curve_vol20_cap60_looser_dd2.png",
        "drawdown_curve": bt.OUTPUT_DIR / "drawdown_curve_vol20_cap60_looser_dd2.png",
        "annual_returns": bt.OUTPUT_DIR / "annual_returns_vol20_cap60_looser_dd2.png",
        "asset_attr": bt.OUTPUT_DIR / "asset_attribution_vol20_cap60_looser_dd2.png",
        "sector_attr": bt.OUTPUT_DIR / "sector_attribution_vol20_cap60_looser_dd2.png",
        "overlay_diag": bt.OUTPUT_DIR / "overlay_diagnostics_vol20_cap60_looser_dd2.png",
        "summary_table": bt.OUTPUT_DIR / "performance_summary_vol20_cap60_looser_dd2.csv",
        "asset_table": bt.OUTPUT_DIR / "asset_contribution_summary_vol20_cap60_looser_dd2.csv",
        "sector_table": bt.OUTPUT_DIR / "sector_contribution_summary_vol20_cap60_looser_dd2.csv",
        "annual_table": bt.OUTPUT_DIR / "annual_performance_summary_vol20_cap60_looser_dd2.csv",
        "full_tearsheet": bt.OUTPUT_DIR / "tearsheet_analysis_vol20_cap60_looser_dd2.png",
    }


def build_contribution_tables(daily: pd.DataFrame, positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = positions.copy()
    daily_work = daily[["trade_date", "net_value"]].copy()
    daily_work["prev_net_value"] = daily_work["net_value"].shift(1, fill_value=1.0)
    work = work.merge(daily_work[["trade_date", "prev_net_value"]], on="trade_date", how="left")

    work["gross_contribution_amount"] = work["gross_contribution"] * work["prev_net_value"]
    work["cost_contribution_amount"] = work["transaction_cost_component"] * work["prev_net_value"]
    work["net_contribution_amount"] = work["net_contribution"] * work["prev_net_value"]

    total_pnl = float(work["net_contribution_amount"].sum())
    denom = total_pnl if not np.isclose(total_pnl, 0.0) else np.nan

    asset = (
        work.groupby(["product", "sector"], sort=True)
        .agg(
            gross_pnl_amount=("gross_contribution_amount", "sum"),
            cost_amount=("cost_contribution_amount", "sum"),
            net_pnl_amount=("net_contribution_amount", "sum"),
            avg_weight=("lagged_execution_weight", "mean"),
            avg_execution_weight=("execution_weight", "mean"),
        )
        .reset_index()
        .sort_values("net_pnl_amount", ascending=False)
        .reset_index(drop=True)
    )
    asset["contribution_pct"] = asset["net_pnl_amount"] / denom

    sector = (
        work.groupby("sector", sort=True)
        .agg(
            gross_pnl_amount=("gross_contribution_amount", "sum"),
            cost_amount=("cost_contribution_amount", "sum"),
            net_pnl_amount=("net_contribution_amount", "sum"),
            avg_weight=("lagged_execution_weight", "mean"),
            avg_execution_weight=("execution_weight", "mean"),
        )
        .reset_index()
        .sort_values("net_pnl_amount", ascending=False)
        .reset_index(drop=True)
    )
    sector["contribution_pct"] = sector["net_pnl_amount"] / denom
    return asset, sector


def build_annual_table(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["year"] = work["trade_date"].dt.year
    rows: list[dict[str, float | int]] = []
    for year, grp in work.groupby("year", sort=True):
        curve = (1.0 + grp["strategy_return"]).cumprod()
        dd = curve / curve.cummax() - 1.0
        cal_ret = float((1.0 + grp["strategy_return"]).prod() - 1.0)
        ann_ret = float(curve.iloc[-1] ** (252.0 / len(grp)) - 1.0)
        max_dd = float(-dd.min())
        rows.append(
            {
                "year": int(year),
                "days": int(len(grp)),
                "calendar_return": cal_ret,
                "annualized_return": ann_ret,
                "maximum_drawdown": max_dd,
                "sharpe_ratio": float(grp["strategy_return"].mean() / grp["strategy_return"].std() * np.sqrt(252.0)),
                "calmar_ratio": ann_ret / max_dd if max_dd > 0 else np.nan,
                "average_gross_exposure": float(grp["gross_exposure"].mean()),
                "average_signal_gross_budget": float(grp["signal_gross_budget"].mean()),
                "average_active_signal_count": float(grp["active_signal_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_equity_curve(daily: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily["trade_date"], daily["net_value"], linewidth=1.6)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Net Value")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown_curve(daily: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(daily["trade_date"], daily["drawdown"], 0.0, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title("Drawdown Curve")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_annual_returns(annual: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in annual["calendar_return"]]
    ax.bar(annual["year"].astype(str), annual["calendar_return"], color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title("Annual Calendar Returns")
    ax.set_ylabel("Return")
    ax.grid(axis="y", alpha=0.2)
    for idx, value in enumerate(annual["calendar_return"]):
        ax.text(idx, value, f"{value:.1%}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bar_attribution(df: pd.DataFrame, label_col: str, value_col: str, title: str, output_path: Path) -> None:
    plot_df = df.sort_values(value_col, ascending=True).copy()
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in plot_df[value_col]]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(plot_df[label_col], plot_df[value_col], color=colors, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_diagnostics(daily: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(daily["trade_date"], daily["signal_gross_budget"], label="signal_gross_budget", linewidth=1.2)
    axes[0].plot(daily["trade_date"], daily["vol_scale"], label="vol_scale", linewidth=1.2)
    axes[0].plot(daily["trade_date"], daily["drawdown_scale"], label="drawdown_scale", linewidth=1.2)
    axes[0].plot(daily["trade_date"], daily["gross_exposure"], label="final_gross", linewidth=1.2)
    axes[0].set_title("Overlay Diagnostics")
    axes[0].set_ylabel("Scale")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(daily["trade_date"], daily["active_signal_count"], label="active_signal_count", linewidth=1.2)
    axes[1].plot(daily["trade_date"], daily["trend_asset_count"], label="trend_asset_count", linewidth=1.2)
    axes[1].set_title("Breadth Diagnostics")
    axes[1].set_xlabel("Trade Date")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_tearsheet(daily: pd.DataFrame, annual: pd.DataFrame, asset: pd.DataFrame, sector: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    metric_map = dict(zip(summary["metric"], summary["value"]))
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    ax_eq, ax_dd, ax_ann, ax_asset, ax_sector, ax_overlay = axes.flatten()

    ax_eq.plot(daily["trade_date"], daily["net_value"], linewidth=1.5)
    ax_eq.set_title(
        "Equity Curve | "
        f"AnnRet={metric_map['annualized_return']:.2%} | "
        f"MaxDD={metric_map['maximum_drawdown']:.2%} | "
        f"Calmar={metric_map['calmar_ratio']:.3f}"
    )
    ax_eq.grid(alpha=0.2)

    ax_dd.fill_between(daily["trade_date"], daily["drawdown"], 0.0, alpha=0.85)
    ax_dd.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax_dd.set_title("Drawdown")
    ax_dd.grid(alpha=0.2)

    ann_colors = ["#2ca02c" if value >= 0 else "#d62728" for value in annual["calendar_return"]]
    ax_ann.bar(annual["year"].astype(str), annual["calendar_return"], color=ann_colors, alpha=0.9)
    ax_ann.set_title("Annual Calendar Returns")
    ax_ann.grid(axis="y", alpha=0.2)

    asset_plot = asset.sort_values("net_pnl_amount", ascending=True)
    ax_asset.barh(asset_plot["product"], asset_plot["net_pnl_amount"], alpha=0.9)
    ax_asset.set_title("Asset Attribution")
    ax_asset.grid(axis="x", alpha=0.2)

    sector_plot = sector.sort_values("net_pnl_amount", ascending=True)
    ax_sector.barh(sector_plot["sector"], sector_plot["net_pnl_amount"], alpha=0.9)
    ax_sector.set_title("Sector Attribution")
    ax_sector.grid(axis="x", alpha=0.2)

    ax_overlay.plot(daily["trade_date"], daily["signal_gross_budget"], label="budget", linewidth=1.2)
    ax_overlay.plot(daily["trade_date"], daily["vol_scale"], label="vol", linewidth=1.2)
    ax_overlay.plot(daily["trade_date"], daily["drawdown_scale"], label="dd", linewidth=1.2)
    ax_overlay.plot(daily["trade_date"], daily["gross_exposure"], label="gross", linewidth=1.2)
    ax_overlay.set_title("Overlay")
    ax_overlay.grid(alpha=0.2)
    ax_overlay.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bt = load_configured_backtest_module()
    outputs = prepare_outputs(bt)
    daily, positions, summary, _annual_base = bt.run_backtest()
    asset, sector = build_contribution_tables(daily, positions)
    annual = build_annual_table(daily)

    plot_equity_curve(daily, outputs["equity_curve"])
    plot_drawdown_curve(daily, outputs["drawdown_curve"])
    plot_annual_returns(annual, outputs["annual_returns"])
    plot_bar_attribution(asset, "product", "net_pnl_amount", "Asset Attribution", outputs["asset_attr"])
    plot_bar_attribution(sector, "sector", "net_pnl_amount", "Sector Attribution", outputs["sector_attr"])
    plot_overlay_diagnostics(daily, outputs["overlay_diag"])
    plot_full_tearsheet(daily, annual, asset, sector, summary, outputs["full_tearsheet"])

    summary.to_csv(outputs["summary_table"], index=False, encoding="utf-8-sig")
    asset.to_csv(outputs["asset_table"], index=False, encoding="utf-8-sig")
    sector.to_csv(outputs["sector_table"], index=False, encoding="utf-8-sig")
    annual.to_csv(outputs["annual_table"], index=False, encoding="utf-8-sig")

    print(f"Saved equity curve to {outputs['equity_curve']}")
    print(f"Saved drawdown curve to {outputs['drawdown_curve']}")
    print(f"Saved annual returns chart to {outputs['annual_returns']}")
    print(f"Saved asset attribution chart to {outputs['asset_attr']}")
    print(f"Saved sector attribution chart to {outputs['sector_attr']}")
    print(f"Saved overlay diagnostics to {outputs['overlay_diag']}")
    print(f"Saved full tearsheet to {outputs['full_tearsheet']}")
    print(f"Saved summary tables to {bt.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
