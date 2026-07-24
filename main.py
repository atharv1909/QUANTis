"""
QUANTIS 2.0 — Main Entry Point
Run this on Kaggle or locally to execute the full pipeline.

Usage:
    # Full run (all seeds — runs across multiple Kaggle sessions):
    python main.py --mode full --seed 0

    # Quick test (2 stocks, 1 fold, 5 epochs):
    python main.py --mode test

    # Evaluate existing predictions:
    python main.py --mode eval
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch

from config import (
    NIFTY50_TICKERS, FEATURE_COLS, D_FEATURES, SEQ_LEN,
    RESULTS_DIR, DATA_DIR, TrainingConfig
)
from data_pipeline import Nifty50Pipeline, MarketFeatureBuilder
from training import WalkForwardTrainer
from evaluation import (
    full_evaluation, print_results_table,
    predictions_to_prices, display_price_predictions
)


def run_data_pipeline(tickers=None, start="2008-01-01", end="2025-06-30",
                      cache=True):
    """Step 1: Download and process real market data."""
    print("\n" + "="*60)
    print("  STEP 1: DATA PIPELINE")
    print("="*60)

    pipeline = Nifty50Pipeline(tickers=tickers, start=start, end=end)
    data = pipeline.run(cache=cache)

    # Build market features for HMM
    market_df = MarketFeatureBuilder.build(data)

    print(f"\n📊 Data Summary:")
    print(f"   Stocks: {data['ticker'].nunique()}")
    print(f"   Date range: {data['date'].min().date()} to "
          f"{data['date'].max().date()}")
    print(f"   Total rows: {len(data):,}")
    print(f"   Features: {len(FEATURE_COLS)}")
    print(f"   Sample prices (not synthetic):")
    sample = data.groupby("ticker").last()[["close"]].head(5)
    for ticker, row in sample.iterrows():
        print(f"     {ticker}: ₹{row['close']:.2f}")

    return data, market_df


def run_training(data, market_df, seed=0, device="auto"):
    """Step 2: Train QUANTIS model."""
    print("\n" + "="*60)
    print(f"  STEP 2: TRAINING (Seed {seed})")
    print("="*60)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {mem:.1f} GB")

    config = TrainingConfig(device=device)
    trainer = WalkForwardTrainer(config)

    t0 = time.time()
    predictions = trainer.run_all_folds(data, market_df, seed=seed)
    elapsed = time.time() - t0

    print(f"\n⏱️  Training completed in {elapsed/60:.1f} minutes")
    print(f"📊 Total predictions: {len(predictions):,}")

    return predictions


def run_evaluation(predictions_input=None):
    """Step 3: Evaluate and display results."""
    print("\n" + "="*60)
    print("  STEP 3: EVALUATION")
    print("="*60)

    # Load predictions
    if isinstance(predictions_input, pd.DataFrame):
        preds = predictions_input.copy()
    elif isinstance(predictions_input, str) and os.path.exists(predictions_input):
        preds = pd.read_parquet(predictions_input)
    else:
        # Try to find non-partial predictions in results dir
        pred_files = sorted([
            f for f in os.listdir(RESULTS_DIR)
            if f.startswith("all_preds") and f.endswith(".parquet")
            and "partial" not in f
        ])
        if not pred_files:
            # Fallback to any parquet in results
            pred_files = sorted([
                f for f in os.listdir(RESULTS_DIR)
                if f.endswith(".parquet") and "partial" not in f
            ])
        if not pred_files:
            print("❌ No prediction files found. Run training first.")
            return
        preds = pd.read_parquet(os.path.join(RESULTS_DIR, pred_files[-1]))

    print(f"📂 Loaded {len(preds):,} predictions")
    preds["date"] = pd.to_datetime(preds["date"])

    # --- Full evaluation ---
    results = full_evaluation(preds)
    print_results_table(results, "QUANTIS 2.0")

    # --- Per-fold breakdown ---
    if "fold" in preds.columns:
        print("\n📋 Per-Fold Breakdown:")
        print("-" * 50)
        for fold in preds["fold"].unique():
            fold_preds = preds[preds["fold"] == fold]
            valid = fold_preds.dropna(subset=["y_true", "y_pred"])
            if len(valid) > 0:
                from evaluation import information_coefficient, rank_ic
                ic = information_coefficient(
                    valid["y_true"].values, valid["y_pred"].values)
                ric = rank_ic(
                    valid["y_true"].values, valid["y_pred"].values)
                print(f"  {fold}: IC={ic:.4f} | RankIC={ric:.4f} | "
                      f"N={len(valid):,}")

    # --- Regime breakdown ---
    if "regime" in preds.columns:
        print("\n📋 Per-Regime Breakdown:")
        print("-" * 50)
        regime_names = {0: "Bull 🟢", 1: "Bear 🔴", 2: "Sideways ⚪"}
        for r in sorted(preds["regime"].unique()):
            r_preds = preds[preds["regime"] == r]
            valid = r_preds.dropna(subset=["y_true", "y_pred"])
            if len(valid) > 0:
                from evaluation import information_coefficient
                ic = information_coefficient(
                    valid["y_true"].values, valid["y_pred"].values)
                name = regime_names.get(int(r), f"Regime {r}")
                print(f"  {name}: IC={ic:.4f} | N={len(valid):,}")

    # --- Conformal gate analysis ---
    if "conformal_trade" in preds.columns:
        print("\n📋 Conformal Gate Analysis:")
        print("-" * 50)
        trade = preds["conformal_trade"]
        coverage = trade.mean()
        print(f"  Trade rate: {coverage:.1%} (target: 90%)")

        if trade.sum() > 0:
            traded = preds[preds["conformal_trade"] == True]
            abstained = preds[preds["conformal_trade"] == False]
            from evaluation import information_coefficient
            ic_traded = information_coefficient(
                traded["y_true"].dropna().values,
                traded["y_pred"].dropna().values)
            print(f"  IC (traded):   {ic_traded:.4f}")
            if len(abstained) > 10:
                ic_abstained = information_coefficient(
                    abstained["y_true"].dropna().values,
                    abstained["y_pred"].dropna().values)
                print(f"  IC (abstained): {ic_abstained:.4f}")
                print(f"  → Abstention improved IC by "
                      f"{ic_traded - ic_abstained:.4f}")

    # --- Gate weight analysis ---
    if "gate_tkan" in preds.columns:
        print("\n📋 Expert Gate Weights (average):")
        print("-" * 50)
        print(f"  T-KAN:    {preds['gate_tkan'].mean():.3f}")
        print(f"  LightGBM: {preds['gate_lgbm'].mean():.3f}")
        print(f"  GNN:      {preds['gate_gnn'].mean():.3f}")

    # --- Display real price predictions ---
    print("\n📋 Price Predictions (Real Prices — Latest Date):")
    print("-" * 60)
    price_display = display_price_predictions(preds, top_n=10)
    if len(price_display) > 0:
        print(price_display.to_string(index=False))
    else:
        print("  No price predictions available.")

    # --- Save formatted results ---
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    import json
    # Convert numpy types for JSON serialization
    json_results = {k: float(v) if not np.isnan(v) else None
                    for k, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")

    return results


def run_quick_test():
    """Quick test with 5 stocks, 1 fold, 5 epochs."""
    print("\n🧪 QUICK TEST MODE — 5 stocks, 2 years, 5 epochs")

    test_tickers = NIFTY50_TICKERS[:5]
    data, market_df = run_data_pipeline(
        tickers=test_tickers, start="2018-01-01", end="2024-12-31",
        cache=False)

    # Override config for quick test
    config = TrainingConfig(
        n_epochs=5,
        patience=3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    from config import WALK_FORWARD_FOLDS
    # Use only the last fold for quick test
    import config as cfg
    original_folds = cfg.WALK_FORWARD_FOLDS
    cfg.WALK_FORWARD_FOLDS = [original_folds[-1]]

    trainer = WalkForwardTrainer(config)
    predictions = trainer.run_all_folds(data, market_df, seed=42)

    # Restore folds
    cfg.WALK_FORWARD_FOLDS = original_folds

    if len(predictions) > 0:
        results = full_evaluation(predictions)
        print_results_table(results, "QUANTIS (Quick Test)")

        print("\n📊 Price predictions from test:")
        price_display = display_price_predictions(predictions)
        if len(price_display) > 0:
            print(price_display.to_string(index=False))

    return predictions


# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUANTIS 2.0")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "full", "eval"],
                        help="test=quick 5-stock test, full=all stocks, "
                             "eval=evaluate existing predictions")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (0-4 for 5-seed experiments)")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda/cpu/auto")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions parquet (for eval mode)")
    args = parser.parse_args()

    print("="*60)
    print("  QUANTIS 2.0 — Hybrid Gated Mixture-of-Experts")
    print("  for Stock Return Prediction")
    print("="*60)
    print(f"  Mode: {args.mode}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)

    if args.mode == "test":
        run_quick_test()

    elif args.mode == "full":
        data, market_df = run_data_pipeline()
        predictions = run_training(data, market_df, seed=args.seed,
                                    device=args.device)
        if len(predictions) > 0:
            run_evaluation(predictions)

    elif args.mode == "eval":
        run_evaluation(args.predictions)
