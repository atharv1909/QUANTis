# QUANTIS 2.0 — Step-by-Step Execution Guide (v2)

## 📁 Project Files

```
quantis/
├── config.py           ← All hyperparameters, tickers, paths, feature lists
├── data_pipeline.py    ← Downloads real NIFTY 50 data, computes features/labels
├── models.py           ← T-KAN, GNN, LightGBM, HMM, Hybrid Gate, Conformal Gate
├── training.py         ← Walk-forward training loop with gate diversity loss
├── evaluation.py       ← IC, Sharpe, multi-seed evaluation (per-seed → mean±std)
├── plotting.py         ← 8 publication-ready figures (300 DPI, multi-seed aware)
└── main.py             ← Entry point (quick test mode)
```

> **v2 changes**: Gate diversity regularization, LightGBM tuning, prediction-specific
> conformal gate, correct multi-seed evaluation, 8 publication figures.
> **You must re-train all 5 seeds** — old seed results are invalid.

---

## Prerequisites

| What | Need API Key? | Notes |
|---|---|---|
| **yfinance** | No | Free. Downloads real OHLCV from Yahoo Finance |
| **India VIX** | No | Via yfinance ticker `^INDIAVIX` |
| **NIFTY 50 stocks** | No | All 50 tickers downloaded via yfinance |

**ZERO API keys needed. Everything uses free public data.**

### Python Dependencies

```bash
!pip install yfinance hmmlearn lightgbm matplotlib seaborn -q
```

---

## Step-by-Step on Kaggle

### Step 0: Upload Code (One Time)

1. Go to kaggle.com > Datasets > New Dataset
2. Name: `quantis-code`
3. Upload ALL 7 `.py` files: `config.py`, `data_pipeline.py`, `models.py`,
   `training.py`, `evaluation.py`, `plotting.py`, `main.py`
4. Create → available at `/kaggle/input/quantis-code/`

> ⚠️ If you previously uploaded an older version, **UPDATE the dataset**
> with all 7 new files. The old code has critical bugs.

---

### Step 1: Data Prep (CPU Session — FREE)

CPU only, no GPU. Unlimited time. Enable Internet.

```python
!pip install yfinance hmmlearn -q
import sys
sys.path.insert(0, "/kaggle/input/quantis-code")
from data_pipeline import Nifty50Pipeline, MarketFeatureBuilder
from config import NIFTY50_TICKERS

pipeline = Nifty50Pipeline(tickers=NIFTY50_TICKERS, start="2008-01-01", end="2025-06-30")
data = pipeline.run(cache=True)
market_df = MarketFeatureBuilder.build(data)

# Verify real prices
print(data.groupby("ticker")["close"].last().sort_values(ascending=False).head(10))

# Save
data.to_parquet("/kaggle/working/nifty50_processed.parquet")
market_df.to_parquet("/kaggle/working/market_features.parquet")
```

Then **Save Version**. This creates a dataset you'll use in all GPU sessions.

---

### Step 2: Quick Test (GPU — ~30 min)

Enable GPU T4. Add `quantis-code` + Step 1 output as Inputs. Enable Internet.

```python
!pip install yfinance hmmlearn lightgbm -q
import sys
sys.path.insert(0, "/kaggle/input/quantis-code")
from main import run_quick_test
predictions = run_quick_test()
```

**What to check**: Gate weights should now show ~30-40% per expert (NOT 97% LightGBM).
LightGBM should train 50+ iterations (NOT stop at iteration 1).

---

### Step 3: Full Training per Seed (GPU — ~6 hrs each)

Run 5 separate GPU sessions, one per seed. Each session:

```python
!pip install yfinance hmmlearn lightgbm -q
import sys
sys.path.insert(0, "/kaggle/input/quantis-code")
import pandas as pd
from training import WalkForwardTrainer
from config import TrainingConfig

data = pd.read_parquet("/kaggle/input/YOUR-STEP1-DATASET/nifty50_processed.parquet")
market_df = pd.read_parquet("/kaggle/input/YOUR-STEP1-DATASET/market_features.parquet")

CURRENT_SEED = 0  # Change per session: 0, 1, 2, 3, 4
config = TrainingConfig(device="cuda")
trainer = WalkForwardTrainer(config)
predictions = trainer.run_all_folds(data, market_df, seed=CURRENT_SEED)

# Quick per-seed evaluation
from evaluation import full_evaluation, print_results_table, display_price_predictions
results = full_evaluation(predictions)
print_results_table(results, f"QUANTIS 2.0 (Seed {CURRENT_SEED})")
print(display_price_predictions(predictions, top_n=20).to_string(index=False))
```

**Save Version** after each seed. The output parquet files are in `/kaggle/working/results/`.

**What to check per seed**:
- LightGBM trains 50+ iterations per fold (NOT stopping at iteration 1)
- Gate weights show meaningful spread (T-KAN ~30-40%, LightGBM ~30-40%, GNN ~20-30%)
- Test IC is positive (>0.01)

---

### Step 4: Combine Seeds + Final Evaluation + Plots (CPU — FREE)

After all 5 seeds are done, download the 5 parquet files and upload them
as a new Kaggle dataset. Then run:

```python
!pip install yfinance hmmlearn lightgbm matplotlib seaborn -q
import sys, os, glob
sys.path.insert(0, "/kaggle/input/quantis-code")
import pandas as pd
import numpy as np

# ---- 1. Load all seed files ----
files = sorted(glob.glob("/kaggle/input/YOUR-SEEDS-DATASET/all_preds_seed*.parquet"))
files = [f for f in files if "partial" not in f]
combined_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"✅ Combined {len(files)} seeds ({len(combined_df):,} total predictions)")

# ---- 2. CORRECT multi-seed evaluation (per-seed → mean±std) ----
from evaluation import multi_seed_evaluation, print_multi_seed_table
multi_results = multi_seed_evaluation(combined_df)
print_multi_seed_table(multi_results, "QUANTIS 2.0")

# ---- 3. Generate 8 publication figures (300 DPI) ----
from plotting import generate_all_plots
figs = generate_all_plots(combined_df)
```

**What to check in the plots**:
- **Fig 1 (Equity)**: QUANTIS line should NOT die after 2021. Per-seed curves visible.
- **Fig 4 (Gate)**: Bars should show ~30-40% per expert, NOT 97% LightGBM.
- **Fig 3 (Conformal)**: IC should INCREASE as coverage decreases (not decrease).
- **Fig 6 (Per-Fold)**: All 4 folds should have positive IC. If one fold is dead, you see it.
- **Fig 7 (Regime)**: Green=Bull, Red=Bear should match actual market moves.
- **Fig 8 (Drawdown)**: Max drawdown should be <50% (not 95%).

---

## Key Changes in v2 (Why You Must Re-train)

### 1. Gate Diversity Regularization (training.py)
- Added entropy loss: spreads expert weights across T-KAN, LightGBM, GNN
- Added load balancing: penalizes any single expert getting >60% weight
- Without this, gate collapsed to 97% LightGBM (MoE novelty was nullified)

### 2. LightGBM Tuning (models.py)
- Learning rate: 0.05 → 0.01 (trains more iterations, better differentiation)
- Num leaves: 256 → 31 (less overfit on validation)
- Early stopping: 50 → 100 (more patient)
- Old LightGBM was stopping at iteration 1 in folds 2-4

### 3. Conformal Gate Fix (models.py)
- Old: confidence = 1/(interval_width) — SAME for all predictions in one regime
- New: confidence = 1/(expected_error) — SPECIFIC per prediction using (regime, |y_pred|) bins
- This makes high-confidence filtering actually select better predictions

### 4. Evaluation Fix (evaluation.py)
- Old: `pd.concat(5 seeds)` → `full_evaluation()` = WRONG (duplicates stocks)
- New: `multi_seed_evaluation()` evaluates each seed independently, reports mean±std

### 5. Plotting Upgrade (plotting.py)
- 5 → 8 figures, all multi-seed aware
- New: Per-fold IC breakdown (Fig 6), Regime timeline (Fig 7), Drawdown (Fig 8)
- All plots de-duplicate across seeds before computing metrics

---

## How Price Predictions Work (Not Hardcoded)

```
predicted_price = current_close × (1 + predicted_return)
               = ₹2893.50 × (1 + 0.023)
               = ₹2960.03
```

- `current_close` = real last known price from yfinance
- `predicted_return` = model output
- `actual_price` = actual future close from yfinance

ZERO hardcoding. Every price is real market data.
