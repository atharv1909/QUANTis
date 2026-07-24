"""
QUANTIS 2.0 — Configuration
All hyperparameters, feature lists, tickers, and paths in one place.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ============================================================
# PATHS — Kaggle vs Local auto-detection
# ============================================================
IS_KAGGLE = os.path.exists("/kaggle")
DATA_DIR = "/kaggle/working/data" if IS_KAGGLE else "./data"
CHECKPOINT_DIR = "/kaggle/working/checkpoints" if IS_KAGGLE else "./checkpoints"
RESULTS_DIR = "/kaggle/working/results" if IS_KAGGLE else "./results"

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# NIFTY 50 TICKERS (yfinance format)
# ============================================================
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
    "M&M.NS", "HCLTECH.NS", "NTPC.NS", "TATAMOTORS.NS", "POWERGRID.NS",
    "JSWSTEEL.NS", "TECHM.NS", "INDUSINDBK.NS", "ONGC.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "TATASTEEL.NS",
    "GRASIM.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "BRITANNIA.NS",
    "EICHERMOT.NS", "BPCL.NS", "HINDALCO.NS", "APOLLOHOSP.NS", "TRENT.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
    "TATACONSUM.NS", "BEL.NS",
]

# Sector mapping for GNN graph construction
SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "Financials", "ICICIBANK.NS": "Financials",
    "KOTAKBANK.NS": "Financials", "AXISBANK.NS": "Financials",
    "SBIN.NS": "Financials", "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials", "INDUSINDBK.NS": "Financials",
    "SBILIFE.NS": "Financials", "HDFCLIFE.NS": "Financials",
    "HINDUNILVR.NS": "Consumer", "ITC.NS": "Consumer",
    "NESTLEIND.NS": "Consumer", "BRITANNIA.NS": "Consumer",
    "TATACONSUM.NS": "Consumer", "TRENT.NS": "Consumer",
    "ASIANPAINT.NS": "Consumer", "TITAN.NS": "Consumer",
    "SUNPHARMA.NS": "Pharma", "DIVISLAB.NS": "Pharma",
    "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "APOLLOHOSP.NS": "Pharma",
    "LT.NS": "Industrials", "NTPC.NS": "Utilities",
    "POWERGRID.NS": "Utilities", "COALINDIA.NS": "Materials",
    "TATAMOTORS.NS": "Auto", "MARUTI.NS": "Auto",
    "EICHERMOT.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "HEROMOTOCO.NS": "Auto", "M&M.NS": "Auto",
    "BHARTIARTL.NS": "Telecom", "ADANIENT.NS": "Conglomerate",
    "ADANIPORTS.NS": "Infra", "JSWSTEEL.NS": "Materials",
    "TATASTEEL.NS": "Materials", "HINDALCO.NS": "Materials",
    "ULTRACEMCO.NS": "Materials", "GRASIM.NS": "Materials",
    "BEL.NS": "Defence",
}

# ============================================================
# FEATURE CONFIGURATION
# ============================================================
FEATURE_COLS = [
    # Returns (4)
    "ret_1d", "ret_5d", "ret_10d", "ret_21d",
    # Momentum oscillators (6)
    "rsi", "macd_hist", "stoch_k", "cci", "roc", "williams_r",
    # Volatility (4)
    "boll_pctb", "boll_width", "atr", "adx",
    # Volume (3)
    "obv_norm", "mfi", "vol_sma_ratio",
    # Ratios (5)
    "vol_ratio_5_20", "vol_ratio_10_60",
    "sma_5_20", "sma_10_60", "sma_20_60",
]
# India-specific features (added separately if available)
INDIA_FEATURES = ["vix_level", "vix_pctile", "fii_dii_net_zscore"]

D_FEATURES = len(FEATURE_COLS)       # 22 base features
SEQ_LEN = 60                          # 60 trading days lookback
LABEL_HORIZON = 5                     # predict 5-day forward return
EMBARGO_DAYS = 21                     # gap between train and test

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
@dataclass
class TKANConfig:
    d_input: int = D_FEATURES
    d_hidden: int = 64
    seq_len: int = SEQ_LEN
    gru_layers: int = 1
    dropout: float = 0.15
    grid_size: int = 5
    spline_order: int = 3

@dataclass
class LGBMConfig:
    n_estimators: int = 1000
    num_leaves: int = 31
    max_depth: int = 6
    learning_rate: float = 0.01
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 5
    lambda_l1: float = 0.5
    lambda_l2: float = 2.0
    min_child_samples: int = 50

@dataclass
class GNNConfig:
    d_input: int = D_FEATURES
    d_hidden: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15
    corr_threshold: float = 0.5
    mom_threshold: float = 0.02

@dataclass
class GateConfig:
    n_experts: int = 3
    d_regime: int = 3         # bull / bear / sideways
    d_market_embed: int = 32
    d_market_input: int = 6   # market-level features for embedding

@dataclass
class TrainingConfig:
    n_epochs: int = 30
    batch_size: int = 50       # all stocks in one batch (cross-sectional)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    grad_clip: float = 1.0
    n_seeds: int = 5
    device: str = "cuda"
    # Gate diversity regularization (prevents expert collapse)
    gate_entropy_weight: float = 0.3   # maximize entropy → spread expert usage
    gate_balance_weight: float = 0.1   # penalize if any expert > 60% average weight

@dataclass
class ConformalConfig:
    target_coverage: float = 0.90
    n_regimes: int = 3

# ============================================================
# WALK-FORWARD FOLDS
# ============================================================
WALK_FORWARD_FOLDS = [
    {
        "name": "fold_1",
        "train_start": "2010-01-01", "train_end": "2017-12-31",
        "val_start": "2018-02-01", "val_end": "2018-12-31",
        "test_start": "2019-02-01", "test_end": "2019-12-31",
    },
    {
        "name": "fold_2",
        "train_start": "2010-01-01", "train_end": "2019-12-31",
        "val_start": "2020-02-01", "val_end": "2020-06-30",
        "test_start": "2020-07-22", "test_end": "2021-06-30",
    },
    {
        "name": "fold_3",
        "train_start": "2010-01-01", "train_end": "2021-06-30",
        "val_start": "2021-08-01", "val_end": "2022-06-30",
        "test_start": "2022-07-22", "test_end": "2023-06-30",
    },
    {
        "name": "fold_4",
        "train_start": "2010-01-01", "train_end": "2023-06-30",
        "val_start": "2023-08-01", "val_end": "2024-03-31",
        "test_start": "2024-04-22", "test_end": "2025-03-31",
    },
]

# Evaluation time windows (for regime/period breakdown)
EVAL_WINDOWS = {
    "pre_covid":  ("2018-01-01", "2020-01-31"),
    "covid_crash": ("2020-02-01", "2020-06-30"),
    "post_covid": ("2020-07-01", "2021-12-31"),
    "tightening": ("2022-01-01", "2024-12-31"),
}

# ============================================================
# TRANSACTION COSTS (NSE India)
# ============================================================
NSE_BROKERAGE_BPS = 3        # 0.03% brokerage
NSE_STT_BPS = 10             # 0.1% Securities Transaction Tax
NSE_IMPACT_BPS = 10          # ~10 bps market impact estimate
TOTAL_COST_BPS = NSE_BROKERAGE_BPS + NSE_STT_BPS + NSE_IMPACT_BPS  # 23 bps one-way
