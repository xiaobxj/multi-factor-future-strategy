# Multi-factor Future Strategy

This package contains the finalized multi-factor futures strategy research workflow and backtest artifacts.

Selected production candidate:
- Factor: `optimized_score_global`
- Best backtest version: `backtest_engine_v3_2.py`
- Core execution rule: long-only, top-1 selection, `score >= 1.0`, `regime == trend`, 2-day holding persistence

Included code:
- factor construction and regime annotation
- IC time-series stability analysis
- backtest evolution from baseline to V3.2

Included results:
- composite momentum panel and diagnostics
- IC stability outputs
- backtest outputs from baseline, V2, V3, V3.1, and best V3.2

Excluded:
- V3.3 experimental files were intentionally removed after comparison showed no improvement over V3.2
