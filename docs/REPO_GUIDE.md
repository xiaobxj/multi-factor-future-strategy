# Repository Guide

This repository is a curated public subset of a larger futures research workspace.

## What is included

- cross-sectional factor construction
- regime-aware score generation
- calibrated `v3_4_calmar` overlay framework
- selected backtest outputs, charts, and attribution reports

## What is intentionally not included

- the full upstream raw data distribution
- every historical experiment variant
- local-only TSM research folders
- cache folders and transient outputs

## Main code map

- `code/cross section/composite_momentum_score.py`
  Core panel, factor, normalization, and score construction.

- `code/cross section/market_state_research_regime_fix.py`
  Regime labeling logic used by the cross-sectional panel.

- `code/cross section/backtest_engine_v3_4_calmar.py`
  Base continuous-allocation framework.

- `code/cross section/backtest_engine_v3_4_calmar_vol20_looser_dd2.py`
  Fixed calibrated configuration with looser drawdown overlay.

- `code/cross section/backtest_engine_v3_4_calmar_vol20_cap60_looser_dd2.py`
  Strongest packaged configuration with higher single-asset cap.

## Output map

- `code/cross section/output/composite_momentum_score_regime_fix/`
  Factor panel, score diagnostics, and snapshot files.

- `code/cross section/output/composite_momentum_score_regime_fix_calmar_vol20_looser_dd2/`
  Daily trading files, behavior charts, and composite momentum report for the `vol20_looser_dd2` variant.

- `code/cross section/output/composite_momentum_score_regime_fix_calmar_vol20_cap60_looser_dd2/`
  Full attribution pack and tearsheet for the highest-Calmar packaged variant.

## Recommended reading order

1. `README.md`
2. `docs/REPORT_INDEX.md`
3. `code/cross section/backtest_engine_v3_4_calmar.py`
4. The fixed variant script you want to inspect
