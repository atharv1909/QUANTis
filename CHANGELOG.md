# QUANTIS 2.0 — 10-Bug Fix Changelog

| Bug | Category | Issue Description | Fixed In File(s) | Fix Summary & Runtime Log |
|---|---|---|---|---|
| **1** | Critical | Gate diversity loss unscaled and dominated MSE task loss | `training.py` | Scaled `entropy_term` and `balance_term` relative to `mse_loss.detach()`. Logs `|ent|/|mse|` ratio every epoch (target < 0.2). |
| **2** | Critical | DSR unit mismatch (annualized Sharpe passed into daily DSR) | `evaluation.py` | Divided `Sharpe` by `sqrt(252)` to convert to daily frequency before passing to `deflated_sharpe_ratio()`. |
| **3** | High | Config decorative — `LGBMConfig` never wired up | `config.py`, `models.py`, `training.py` | Added `to_lgbm_params(seed)` to `LGBMConfig`. Removed hardcoded fallback dict from `LightGBMExpert.__init__`. Wired in `train_lgbm`. Logs LGBM params per fold. |
| **4** | High | LightGBM non-deterministic across identical runs | `config.py`, `training.py` | `to_lgbm_params` includes `seed`, `bagging_seed`, `feature_fraction_seed`, `deterministic=True`, `force_row_wise=True`. Seeded `random`, `np`, `torch`, and set `cudnn.deterministic=True`. Logs determinism confirmation. |
| **5** | High | Data re-downloaded live each session without versioning | `data_pipeline.py` | Added SHA-256 content hashing of processed parquet. Logs `🔒 Data hash: {hash}`. |
| **6** | Medium | `EMBARGO_DAYS=21` dead code | `training.py` | Evaluates and asserts actual trading-day gaps between train, val, and test splits against `EMBARGO_DAYS`. Logs gap per fold. |
| **7** | Medium | Point-in-time universe violation | `config.py`, `data_pipeline.py` | Added explicit survivorship bias warning log at pipeline startup. Defined `STABLE_CORE_TICKERS` (~40 names continuous since 2010) for robustness checks. |
| **8** | Medium | `TATAMOTORS.NS` silently dropped | `config.py`, `data_pipeline.py` | Defined `TICKER_ISSUES` explaining the Oct 2024 Tata Motors demerger. Logs explicit note on download failure. |
| **9** | Medium | PBO & Benjamini-Hochberg claimed in docstring but missing | `evaluation.py` | Implemented `probability_of_backtest_overfitting` (CSCV method) and `benjamini_hochberg` (FDR correction). Updated module docstring. |
| **10** | Low | Wasted market feature dimension (0.0 placeholder) | `data_pipeline.py`, `training.py` | Replaced constant `0.0` with `vix_pctile` in `MarketFeatureBuilder.build()` and `build_daily_batch`. |
