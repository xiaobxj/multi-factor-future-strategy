from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import composite_momentum_score as cms


CODE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CODE_DIR / "output" / "composite_momentum_score_regime_fix"
TARGET_DIR = CODE_DIR / "output" / "composite_momentum_score_regime_fix_calmar_vol20_looser_dd2"
PANEL_PATH = SOURCE_DIR / "composite_momentum_panel.csv"
WEIGHTS_PATH = SOURCE_DIR / "dimension_feature_weights.csv"
REPORT_PATH = TARGET_DIR / "composite_momentum_report_v3_4_calmar_vol20_looser_dd2.png"
LATEST_PATH = TARGET_DIR / "latest_snapshot_v3_4_calmar_vol20_looser_dd2.csv"
START_DATE = pd.Timestamp("2023-04-21")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.loc[panel["trade_date"] >= START_DATE].copy()

    evaluation_df = cms.evaluate_scores(
        panel,
        [
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
        ],
        ["forward_return_20"],
        bucket_size=2,
    )
    weight_df = pd.read_csv(WEIGHTS_PATH)
    latest_snapshot = (
        panel.sort_values(["trade_date", "product"])
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

    fig = cms.plot_report(panel, evaluation_df, weight_df)
    fig.savefig(REPORT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    latest_snapshot.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved composite momentum report to {REPORT_PATH}")
    print(f"Saved latest snapshot to {LATEST_PATH}")


if __name__ == "__main__":
    main()
