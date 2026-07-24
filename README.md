# QUANTIS 2.0

**Hybrid Gated Mixture-of-Experts for Indian Stock Return Prediction**

> A novel ensemble architecture combining T-KAN (Temporal Kolmogorov-Arnold Networks), LightGBM, and Dynamic GNN experts through a hybrid discrete-continuous gating mechanism with conformal prediction-based abstention.

---

## Architecture

```
                    ┌─────────────┐
                    │  HMM Regime │ (Bull / Bear / Sideways)
                    │  Detector   │
                    └──────┬──────┘
                           │ regime one-hot [3]
    ┌──────────┐    ┌──────┴──────┐    ┌──────────┐
    │  T-KAN   │    │  Hybrid     │    │   GNN    │
    │  Expert  │───▶│  Gate       │◀───│  Expert  │
    │ (Temporal│    │(Discrete +  │    │ (Graph   │
    │  Spline) │    │ Continuous) │    │  Attn)   │
    └──────────┘    └──────┬──────┘    └──────────┘
                           │
    ┌──────────┐    ┌──────┴──────┐
    │ LightGBM │───▶│  Weighted   │
    │  Expert  │    │  Ensemble   │
    │  (CPU)   │    │  Prediction │
    └──────────┘    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  Conformal  │
                    │  Abstention │ (Don't trade when uncertain)
                    │  Gate       │
                    └─────────────┘
```

## Key Novelties

1. **Hybrid Gating**: First to combine discrete HMM regime states with continuous market embeddings for expert weighting (vs. MASTER's continuous-only, ReCAP's discrete-only)
2. **Conformal Abstention**: Prediction-specific confidence via residual binning — abstains from low-confidence trades instead of forcing predictions
3. **T-KAN Expert**: Kolmogorov-Arnold Networks with learnable spline activations replacing fixed activations, integrated with GRU temporal modeling

## Project Structure

```
quantis/
├── config.py           ← All hyperparameters, tickers, paths, feature lists
├── data_pipeline.py    ← Downloads real NIFTY 50 data, computes features/labels
├── models.py           ← T-KAN, GNN, LightGBM, HMM, Hybrid Gate, Conformal Gate
├── training.py         ← Walk-forward training loop with gate diversity loss
├── evaluation.py       ← IC, Sharpe, multi-seed evaluation (per-seed → mean±std)
├── plotting.py         ← 8 publication-ready figures (300 DPI, multi-seed aware)
├── main.py             ← Entry point (quick test / full / eval modes)
└── INSTRUCTIONS.md     ← Step-by-step Kaggle execution guide
```

## Data

- **Universe**: NIFTY 50 stocks (Indian equity market)
- **Source**: Yahoo Finance via `yfinance` (100% free, zero API keys)
- **Period**: 2008-01-01 to 2025-06-30
- **Features**: 22 technical indicators (returns, momentum, volatility, volume)
- **Label**: 5-day forward return
- **No synthetic data. No hardcoded prices.**

## Quick Start

### Prerequisites

```bash
pip install torch yfinance hmmlearn lightgbm matplotlib seaborn
```

### Quick Test (5 stocks, 1 fold, ~30 min on GPU)

```python
from main import run_quick_test
predictions = run_quick_test()
```

### Full Training (Kaggle T4 GPU)

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed step-by-step Kaggle guide.

## Walk-Forward Cross-Validation

| Fold | Train | Validation | Test |
|------|-------|------------|------|
| 1 | 2010–2017 | 2018 | 2019 |
| 2 | 2010–2019 | 2020 H1 | 2020 H2–2021 H1 |
| 3 | 2010–2021 | 2021 H2–2022 H1 | 2022 H2–2023 H1 |
| 4 | 2010–2023 | 2023 H2–2024 Q1 | 2024 Q2–2025 Q1 |

21-day embargo between train/val and val/test to prevent leakage.

## Publication Figures

After training, the `plotting.py` module generates 8 publication-ready figures at 300 DPI:

1. Multi-seed cumulative equity curve with fold boundaries
2. IC distribution & 30-day rolling IC
3. Conformal abstention IC vs coverage trade-off
4. Hybrid gate expert allocation across HMM regimes
5. Predicted vs actual stock prices scatter
6. Per-fold IC breakdown across seeds (diagnostic)
7. HMM regime timeline vs market performance
8. Drawdown underwater chart

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Signal Quality** | IC, Rank IC, ICIR |
| **Portfolio** | Annualized Return, Sharpe, Sortino, Calmar |
| **Risk** | Max Drawdown, Direction Accuracy |
| **Ranking** | Precision@5, NDCG@5 |
| **Statistical** | Deflated Sharpe Ratio, Diebold-Mariano Test |
| **Price** | MAPE (predicted vs actual prices) |

## License

MIT

## Citation

If you use this code, please cite:

```bibtex
@article{quantis2025,
  title={QUANTIS: Hybrid Gated Mixture-of-Experts with Conformal Abstention for Stock Return Prediction},
  author={Atharv},
  year={2025}
}
```
