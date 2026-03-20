# Multi-factor Future Strategy

Cross-sectional futures research on a heterogeneous China futures universe:

- `IF`, `IM`: equity index futures
- `TL`: 30-year government bond futures
- `AU`: precious metals
- `M`: agriculture (soybean meal)
- `I`: ferrous

The repository focuses on a momentum-style, regime-aware cross-sectional framework with:

- dual normalization: time-series normalization followed by cross-sectional normalization
- five feature dimensions: strength, smoothness, consistency, breakout, drawdown quality
- dominant contract panel construction with roll adjustment
- regime labeling and regime-aware score construction
- controlled portfolio overlays for volatility targeting and drawdown management

## Current focus

The latest calibrated public variants included here are:

- `backtest_engine_v3_4_calmar_vol20_looser_dd2.py`
- `backtest_engine_v3_4_calmar_vol20_cap60_looser_dd2.py`

The best calibrated configuration currently packaged in this repo is:

- annualized return: `13.47%`
- annualized volatility: `14.95%`
- maximum drawdown: `11.31%`
- Sharpe: `0.920`
- Calmar: `1.191`

These results correspond to the `vol20_cap60_looser_dd2` variant and a backtest start date of `2023-04-21`.

## Repository layout

```text
Multi-factor future strategy/
├─ README.md
├─ docs/
│  ├─ REPO_GUIDE.md
│  └─ REPORT_INDEX.md
└─ code/
   └─ cross section/
      ├─ composite_momentum_score.py
      ├─ market_state_research_regime_fix.py
      ├─ backtest_engine_v3_4_calmar.py
      ├─ backtest_engine_v3_4_calmar_vol20_looser_dd2.py
      ├─ backtest_engine_v3_4_calmar_vol20_cap60_looser_dd2.py
      ├─ trade_execution_logger_v3_4_calmar_vol20_looser_dd2.py
      ├─ composite_momentum_report_v3_4_calmar_vol20_looser_dd2.py
      ├─ report_v3_4_calmar_vol20_cap60_looser_dd2.py
      └─ output/
```

## Key scripts

- `code/cross section/composite_momentum_score.py`
  Builds the dominant future panel, feature set, normalized scores, and regime-aware composite scores.

- `code/cross section/backtest_engine_v3_4_calmar.py`
  Base continuous-allocation backtest framework with signal gross budgeting, volatility targeting, and drawdown overlay.

- `code/cross section/backtest_engine_v3_4_calmar_vol20_looser_dd2.py`
  Fixed calibrated version with `Calmar > 1`.

- `code/cross section/backtest_engine_v3_4_calmar_vol20_cap60_looser_dd2.py`
  Higher-conviction calibrated version with the strongest packaged result set.

- `code/cross section/trade_execution_logger_v3_4_calmar_vol20_looser_dd2.py`
  Exports daily trading matrices, trade blotter, and trading behavior charts.

- `code/cross section/composite_momentum_report_v3_4_calmar_vol20_looser_dd2.py`
  Regenerates the composite momentum summary report for the selected calibrated variant.

## Included outputs

This repo includes selected research artifacts for public review:

- score panel and diagnostics for `composite_momentum_score_regime_fix`
- calibrated backtest outputs for `vol20_looser_dd2`
- calibrated backtest outputs and attribution package for `vol20_cap60_looser_dd2`

Raw upstream futures downloads are not packaged here as a complete public data distribution.

## Suggested entry points

If you are reviewing the project for the first time, start with:

1. `README.md`
2. `docs/REPO_GUIDE.md`
3. `docs/REPORT_INDEX.md`
4. `code/cross section/backtest_engine_v3_4_calmar.py`

## Notes

- The repo currently emphasizes the cross-sectional strategy branch rather than the full TSM research tree.
- Some legacy or local-only research folders are intentionally left outside the curated public package.
