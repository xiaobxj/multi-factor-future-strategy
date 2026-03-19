from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
CONFIG_MODULE_PATH = CODE_DIR / "backtest_engine_v3_4_calmar_vol20_looser_dd2.py"
POSITION_ORDER = ["I", "IM", "IF", "AU", "M", "TL"]


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("bt_v34_trade_cfg", CONFIG_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {CONFIG_MODULE_PATH}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    bt = cfg.load_base()
    cfg.configure(bt)
    return bt


def classify_actions(pre_position: pd.Series, post_position: pd.Series) -> pd.Series:
    conditions = [
        (pre_position == 0.0) & (post_position > 0.0),
        (pre_position > 0.0) & (post_position == 0.0),
        (pre_position > 0.0) & (post_position > pre_position),
        (pre_position > 0.0) & (post_position > 0.0) & (post_position < pre_position),
    ]
    choices = ["open_long", "close_long", "increase_long", "reduce_long"]
    return pd.Series(np.select(conditions, choices, default="rebalance"), index=pre_position.index)


def build_outputs(bt):
    bt.ensure_dirs()
    output_dir = bt.OUTPUT_DIR
    return {
        "daily_positions_csv": output_dir / "daily_positions_v3_4_calmar_vol20_looser_dd2.csv",
        "trade_blotter_csv": output_dir / "trade_blotter_v3_4_calmar_vol20_looser_dd2.csv",
        "daily_positions_heatmap": output_dir / "daily_positions_heatmap_v3_4_calmar_vol20_looser_dd2.png",
        "daily_positions_area": output_dir / "daily_positions_area_v3_4_calmar_vol20_looser_dd2.png",
        "trading_behavior_all": output_dir / "trading_behavior_all_assets_v3_4_calmar_vol20_looser_dd2.png",
    }


def build_position_matrix(positions: pd.DataFrame) -> pd.DataFrame:
    return (
        positions.pivot(index="trade_date", columns="product", values="execution_weight")
        .reindex(columns=POSITION_ORDER)
        .fillna(0.0)
        .sort_index()
    )


def build_trade_blotter(position_matrix: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    post_position = position_matrix.stack().rename("Post_Position").reset_index()
    post_position.columns = ["Date", "Asset", "Post_Position"]

    pre_matrix = position_matrix.shift(1, fill_value=0.0)
    pre_position = pre_matrix.stack().rename("Pre_Position").reset_index()
    pre_position.columns = ["Date", "Asset", "Pre_Position"]

    blotter = pre_position.merge(post_position, on=["Date", "Asset"], how="left")
    blotter["Position_Change"] = blotter["Post_Position"] - blotter["Pre_Position"]
    blotter = blotter.loc[blotter["Position_Change"] != 0.0].copy()
    blotter["Action"] = classify_actions(blotter["Pre_Position"], blotter["Post_Position"])

    price_df = positions[["trade_date", "product", "close", "optimized_score_global", "regime"]].rename(
        columns={
            "trade_date": "Date",
            "product": "Asset",
            "close": "Executed_Price",
            "optimized_score_global": "Signal_Score",
            "regime": "Regime",
        }
    )
    blotter = blotter.merge(price_df, on=["Date", "Asset"], how="left")
    return blotter[
        [
            "Date",
            "Asset",
            "Action",
            "Pre_Position",
            "Post_Position",
            "Position_Change",
            "Executed_Price",
            "Signal_Score",
            "Regime",
        ]
    ].sort_values(["Date", "Asset"]).reset_index(drop=True)


def plot_daily_positions_heatmap(position_matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(position_matrix.T.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_yticks(range(len(position_matrix.columns)))
    ax.set_yticklabels(position_matrix.columns.tolist())
    tick_idx = np.linspace(0, len(position_matrix.index) - 1, num=min(10, len(position_matrix.index)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(position_matrix.index[tick_idx].strftime("%Y-%m-%d"), rotation=45, ha="right")
    ax.set_title("Daily Trading Heatmap (Execution Weights)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_daily_positions_area(position_matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.stackplot(position_matrix.index, *[position_matrix[col].to_numpy() for col in position_matrix.columns], labels=position_matrix.columns)
    ax.set_title("Daily Trading Gross Allocation by Asset")
    ax.set_ylabel("Execution Weight")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trading_behavior_all_assets(positions: pd.DataFrame, output_path: Path, output_dir: Path) -> None:
    assets = [asset for asset in POSITION_ORDER if asset in positions["product"].unique()]
    fig, axes = plt.subplots(len(assets), 1, figsize=(16, 3.5 * len(assets)), sharex=True)
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets):
        asset_df = positions.loc[positions["product"] == asset].sort_values("trade_date").copy()
        asset_df["position_change"] = asset_df["execution_weight"] - asset_df["execution_weight"].shift(1, fill_value=0.0)
        buys = asset_df.loc[asset_df["position_change"] > 0.0]
        sells = asset_df.loc[asset_df["position_change"] < 0.0]

        ax.plot(asset_df["trade_date"], asset_df["close"], linewidth=1.3, label=f"{asset} close")
        ax.scatter(buys["trade_date"], buys["close"], marker="^", s=35, color="#d62728", label="increase", zorder=3)
        ax.scatter(sells["trade_date"], sells["close"], marker="v", s=35, color="#2ca02c", label="decrease", zorder=3)
        ax.set_title(f"Trading Behavior: {asset}")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")

        single_fig, single_ax = plt.subplots(figsize=(16, 5))
        single_ax.plot(asset_df["trade_date"], asset_df["close"], linewidth=1.4, label=f"{asset} close")
        single_ax.scatter(buys["trade_date"], buys["close"], marker="^", s=40, color="#d62728", label="position increase", zorder=3)
        single_ax.scatter(sells["trade_date"], sells["close"], marker="v", s=40, color="#2ca02c", label="position decrease", zorder=3)
        single_ax.set_title(f"Trading Behavior Audit: {asset}")
        single_ax.set_xlabel("Trade Date")
        single_ax.set_ylabel("Adjusted Close")
        single_ax.grid(alpha=0.2)
        single_ax.legend()
        single_fig.tight_layout()
        single_fig.savefig(output_dir / f"trading_behavior_{asset}_v3_4_calmar_vol20_looser_dd2.png", dpi=150, bbox_inches="tight")
        plt.close(single_fig)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bt = load_backtest_module()
    outputs = build_outputs(bt)
    daily, positions, _metrics, _annual = bt.run_backtest()
    positions = positions.sort_values(["trade_date", "product"]).copy()
    position_matrix = build_position_matrix(positions)
    blotter = build_trade_blotter(position_matrix, positions)

    position_matrix.reset_index().rename(columns={"trade_date": "Date"}).to_csv(
        outputs["daily_positions_csv"], index=False, encoding="utf-8-sig"
    )
    blotter.to_csv(outputs["trade_blotter_csv"], index=False, encoding="utf-8-sig")
    plot_daily_positions_heatmap(position_matrix, outputs["daily_positions_heatmap"])
    plot_daily_positions_area(position_matrix, outputs["daily_positions_area"])
    plot_trading_behavior_all_assets(positions, outputs["trading_behavior_all"], bt.OUTPUT_DIR)

    print(f"Saved daily positions to {outputs['daily_positions_csv']}")
    print(f"Saved trade blotter to {outputs['trade_blotter_csv']}")
    print(f"Saved daily trading heatmap to {outputs['daily_positions_heatmap']}")
    print(f"Saved daily trading area chart to {outputs['daily_positions_area']}")
    print(f"Saved trading behavior charts to {bt.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
