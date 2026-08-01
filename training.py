"""
QUANTIS 2.0 — Walk-Forward Training Loop
Designed for Kaggle T4 sessions with checkpoint/recovery.

Handles:
- Walk-forward cross-validation with embargo
- Per-fold LightGBM + neural model training
- HMM regime fitting per fold
- Conformal gate calibration
- Checkpoint saving after each fold
- Real price predictions in output
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import random

from config import (
    FEATURE_COLS, D_FEATURES, SEQ_LEN, LABEL_HORIZON,
    EMBARGO_DAYS, WALK_FORWARD_FOLDS, SECTOR_MAP,
    CHECKPOINT_DIR, RESULTS_DIR, TrainingConfig, LGBMConfig
)
from models import (
    QUANTIS, LightGBMExpert, RegimeDetector, ConformalGate
)
from data_pipeline import Nifty50Pipeline, MarketFeatureBuilder
from evaluation import (
    information_coefficient, compute_daily_ics, full_evaluation,
    print_results_table, predictions_to_prices, display_price_predictions
)


class WalkForwardTrainer:
    """Walk-forward trainer for QUANTIS on Kaggle.
    
    Each fold:
    1. Split data (train/val/test) with embargo gap
    2. Fit HMM on train market features
    3. Train LightGBM on train data (CPU)
    4. Train neural experts + gate on GPU
    5. Calibrate conformal gate on validation
    6. Evaluate on test data
    7. Save checkpoint + predictions (with real prices)
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.all_predictions = []
        
    def prepare_fold_data(self, full_df: pd.DataFrame,
                          fold: Dict) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                pd.DataFrame]:
        """Split data for one fold with embargo verification."""
        train = full_df[
            (full_df["date"] >= fold["train_start"]) &
            (full_df["date"] <= fold["train_end"])
        ].copy()
        
        val = full_df[
            (full_df["date"] >= fold["val_start"]) &
            (full_df["date"] <= fold["val_end"])
        ].copy()
        
        test = full_df[
            (full_df["date"] >= fold["test_start"]) &
            (full_df["date"] <= fold["test_end"])
        ].copy()
        
        # Bug 6 fix: verify embargo gap in TRADING days
        all_dates = sorted(full_df["date"].unique())
        train_end = pd.Timestamp(fold["train_end"])
        val_start = pd.Timestamp(fold["val_start"])
        val_end = pd.Timestamp(fold["val_end"])
        test_start = pd.Timestamp(fold["test_start"])
        
        gap_tv = len([d for d in all_dates if train_end < d < val_start])
        gap_vt = len([d for d in all_dates if val_end < d < test_start])
        print(f"  \U0001f4cf Embargo: train\u2192val={gap_tv} trading days, "
              f"val\u2192test={gap_vt} trading days (target={EMBARGO_DAYS})")
        if gap_tv < EMBARGO_DAYS:
            print(f"  \u26a0\ufe0f  WARNING: train\u2192val gap ({gap_tv}) < "
                  f"EMBARGO_DAYS ({EMBARGO_DAYS})")
        if gap_vt < EMBARGO_DAYS:
            print(f"  \u26a0\ufe0f  WARNING: val\u2192test gap ({gap_vt}) < "
                  f"EMBARGO_DAYS ({EMBARGO_DAYS})")
        
        return train, val, test
    
    def train_lgbm(self, train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   seed: int = 42) -> LightGBMExpert:
        """Train LightGBM expert on CPU (zero GPU cost).
        
        Bug 3 fix: params come from LGBMConfig.to_lgbm_params(seed).
        Bug 4 fix: seed propagated for full determinism.
        """
        lgbm_config = LGBMConfig()
        params = lgbm_config.to_lgbm_params(seed=seed)
        print(f"  \U0001f4cb LightGBM params: leaves={params['num_leaves']}, "
              f"depth={params['max_depth']}, lr={params['learning_rate']}, "
              f"seed={params['seed']}, deterministic={params['deterministic']}")
        lgbm = LightGBMExpert(params=params)
        
        # Flatten: each row = one (stock, date) observation
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
        
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["label"].values
        
        # Remove NaN labels
        train_mask = ~np.isnan(y_train)
        val_mask = ~np.isnan(y_val)
        
        lgbm.fit(X_train[train_mask], y_train[train_mask],
                 X_val[val_mask], y_val[val_mask])
        
        return lgbm
    
    def build_daily_batch(self, daily_data: Dict, market_row: pd.Series,
                          regime_detector: RegimeDetector,
                          lgbm: LightGBMExpert,
                          edge_index: np.ndarray,
                          device: str) -> Dict:
        """Construct one day's training batch."""
        sequences = torch.tensor(daily_data["sequences"],
                                  dtype=torch.float32, device=device)
        snapshot = torch.tensor(daily_data["snapshot"],
                                dtype=torch.float32, device=device)
        labels = torch.tensor(daily_data["labels"],
                               dtype=torch.float32, device=device)
        edge_idx = torch.tensor(edge_index, dtype=torch.long, device=device)
        
        # Replace NaN with 0 in features (already dropped most NaNs in pipeline)
        sequences = torch.nan_to_num(sequences, 0.0)
        snapshot = torch.nan_to_num(snapshot, 0.0)
        
        # Market features for embedding
        # Bug 10 fix: replaced 0.0 placeholder with vix_pctile
        market_feats = np.array([
            market_row.get("market_ret", 0),
            market_row.get("market_vol", 0),
            market_row.get("vix_level", 0),
            market_row.get("breadth", 0.5),
            market_row.get("avg_vol_ratio", 1.0),
            market_row.get("vix_pctile", 0.5),
        ], dtype=np.float32)
        market_snap = torch.tensor(market_feats, dtype=torch.float32,
                                    device=device)
        
        # Regime
        regime_feats = np.array([[
            market_row.get("market_ret", 0),
            market_row.get("market_vol", 0),
            market_row.get("vix_level", 0),
        ]], dtype=np.float64)
        regime = regime_detector.predict(regime_feats)[0]
        regime_oh = regime_detector.get_onehot(regime, device)
        
        # LightGBM predictions (CPU)
        lgbm_preds = lgbm.predict_tensor(
            daily_data["snapshot"], device=device)
        
        return {
            "sequences": sequences,
            "snapshot": snapshot,
            "labels": labels,
            "edge_index": edge_idx,
            "market_snap": market_snap,
            "regime_oh": regime_oh,
            "regime_int": regime,
            "lgbm_preds": lgbm_preds,
            "tickers": daily_data["tickers"],
            "current_close": daily_data["current_close"],
            "future_close": daily_data["future_close"],
        }
    
    def train_one_fold(self, fold_idx: int, fold: Dict,
                       full_df: pd.DataFrame, market_df: pd.DataFrame,
                       seed: int) -> Dict:
        """Train one complete walk-forward fold.
        
        Returns dict with test predictions and metrics.
        """
        fold_name = fold["name"]
        device = self.config.device if torch.cuda.is_available() else "cpu"
        
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx+1}/{len(WALK_FORWARD_FOLDS)}: {fold_name} "
              f"| Seed: {seed}")
        print(f"  Train: {fold['train_start']} → {fold['train_end']}")
        print(f"  Val:   {fold['val_start']} → {fold['val_end']}")
        print(f"  Test:  {fold['test_start']} → {fold['test_end']}")
        print(f"{'='*60}")
        
        # Bug 4 fix: full determinism — all RNGs seeded, cudnn deterministic
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"  \U0001f512 Determinism: torch/np/random/lgbm all seeded to {seed}")
        
        # Split data
        train_df, val_df, test_df = self.prepare_fold_data(full_df, fold)
        print(f"  Data: train={len(train_df):,} | val={len(val_df):,} | "
              f"test={len(test_df):,}")
        
        if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
            print(f"  ⚠️  Skipping fold — insufficient data")
            return {}
        
        # Normalize features
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
        train_norm, val_norm = Nifty50Pipeline.normalize_features(
            train_df, val_df, feature_cols)
        _, test_norm = Nifty50Pipeline.normalize_features(
            train_df, test_df, feature_cols)
        
        # ---- Step 1: Fit HMM regime detector (CPU) ----
        print("  📊 Fitting HMM regime detector...")
        train_market = market_df[
            (market_df["date"] >= fold["train_start"]) &
            (market_df["date"] <= fold["train_end"])
        ]
        regime_detector = RegimeDetector(n_states=3)
        if len(train_market) >= 60:
            hmm_features = train_market[
                ["market_ret", "market_vol", "vix_level"]
            ].dropna().values
            if len(hmm_features) >= 60:
                try:
                    regime_detector.fit(hmm_features)
                    print("  ✅ HMM fitted")
                except Exception as e:
                    print(f"  ⚠️  HMM failed: {e}. Using default regimes.")
        
        # ---- Step 2: Train LightGBM (CPU) ----
        print("  🌳 Training LightGBM...")
        t0 = time.time()
        lgbm = self.train_lgbm(train_norm, val_norm, seed=seed)
        print(f"  ✅ LightGBM trained in {time.time()-t0:.1f}s")
        
        # ---- Step 3: Build graph ----
        tickers = sorted(train_norm["ticker"].unique().tolist())
        returns_pivot = train_norm.pivot(
            index="date", columns="ticker", values="ret_1d"
        ).reindex(columns=tickers).dropna(how="all")
        
        edge_index, edge_type = Nifty50Pipeline.build_graph(
            returns_pivot.tail(60), tickers, SECTOR_MAP)
        
        # ---- Step 4: Build sequences ----
        print("  📦 Building sequences...")
        train_batches = Nifty50Pipeline.build_sequences(
            train_norm, feature_cols, SEQ_LEN)
        val_batches = Nifty50Pipeline.build_sequences(
            val_norm, feature_cols, SEQ_LEN)
        test_batches = Nifty50Pipeline.build_sequences(
            test_norm, feature_cols, SEQ_LEN)
        
        print(f"  Train days: {len(train_batches)} | "
              f"Val days: {len(val_batches)} | "
              f"Test days: {len(test_batches)}")
        
        if len(train_batches) < 20 or len(val_batches) < 5:
            print(f"  ⚠️  Skipping fold — too few batches")
            return {}
        
        # ---- Step 5: Initialize model ----
        d_feat = len(feature_cols)
        model = QUANTIS(
            d_features=d_feat, d_hidden=64, seq_len=SEQ_LEN,
            d_market=6, d_embed=32, n_heads=4, dropout=0.15
        ).to(device)
        
        print(f"  🧠 Model params: {model.count_parameters():,}")
        
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3)
        
        # Market features lookup
        mkt_lookup = market_df.set_index("date").to_dict("index")
        
        # ---- Step 6: Training loop ----
        print("  🚀 Training neural experts + gate...")
        best_val_ic = -1
        patience_counter = 0
        train_dates = sorted(train_batches.keys())
        val_dates = sorted(val_batches.keys())
        
        for epoch in range(self.config.n_epochs):
            model.train()
            epoch_losses = []
            epoch_entropy_ratios = []
            
            for date in train_dates:
                batch_data = train_batches[date]
                # Get market features for this date
                mkt_row = mkt_lookup.get(date, {})
                if not mkt_row:
                    mkt_row = {"market_ret": 0, "market_vol": 0.01,
                              "vix_level": 15, "breadth": 0.5,
                              "avg_vol_ratio": 1.0}
                
                try:
                    batch = self.build_daily_batch(
                        batch_data, mkt_row, regime_detector,
                        lgbm, edge_index, device)
                except Exception:
                    continue
                
                labels = batch["labels"]
                valid_mask = ~torch.isnan(labels)
                if valid_mask.sum() < 3:
                    continue
                
                optimizer.zero_grad()
                final_preds, gate_weights, expert_preds = model(
                    batch["sequences"], batch["snapshot"],
                    batch["edge_index"], batch["regime_oh"],
                    batch["market_snap"], batch["lgbm_preds"])
                
                mse_loss = F.mse_loss(
                    final_preds.squeeze()[valid_mask],
                    labels[valid_mask])
                
                # --- Bug 1 fix: Gate diversity regularization SCALED to task loss ---
                # Scale weights relative to mse_loss so regularizer stays <20% of task loss
                gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
                avg_weights = gate_weights.mean(dim=0)  # [3]
                balance_loss = (avg_weights.max() - 1.0 / gate_weights.shape[-1]) ** 2
                
                mse_scale = mse_loss.detach().clamp(min=1e-6)
                entropy_term = self.config.gate_entropy_weight * mse_scale * gate_entropy
                balance_term = self.config.gate_balance_weight * mse_scale * balance_loss
                loss = mse_loss - entropy_term + balance_term
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                          self.config.grad_clip)
                optimizer.step()
                epoch_losses.append(mse_loss.item())  # log pure MSE, not regularized
                epoch_entropy_ratios.append(
                    abs(entropy_term.item()) / (mse_loss.item() + 1e-10))
            
            # ---- Validate ----
            model.eval()
            val_preds_list = []
            val_true_list = []
            
            with torch.no_grad():
                for date in val_dates:
                    batch_data = val_batches[date]
                    mkt_row = mkt_lookup.get(date, {
                        "market_ret": 0, "market_vol": 0.01,
                        "vix_level": 15, "breadth": 0.5,
                        "avg_vol_ratio": 1.0})
                    
                    try:
                        batch = self.build_daily_batch(
                            batch_data, mkt_row, regime_detector,
                            lgbm, edge_index, device)
                    except Exception:
                        continue
                    
                    labels = batch["labels"]
                    valid_mask = ~torch.isnan(labels)
                    if valid_mask.sum() < 3:
                        continue
                    
                    preds, _, _ = model(
                        batch["sequences"], batch["snapshot"],
                        batch["edge_index"], batch["regime_oh"],
                        batch["market_snap"], batch["lgbm_preds"])
                    
                    val_preds_list.append(preds.squeeze()[valid_mask].cpu().numpy())
                    val_true_list.append(labels[valid_mask].cpu().numpy())
            
            if val_preds_list:
                val_preds_all = np.concatenate(val_preds_list)
                val_true_all = np.concatenate(val_true_list)
                val_ic = information_coefficient(val_true_all, val_preds_all)
            else:
                val_ic = -1
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            avg_ent_ratio = np.mean(epoch_entropy_ratios) if epoch_entropy_ratios else 0.0
            scheduler.step(val_ic if not np.isnan(val_ic) else -1)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{self.config.n_epochs} | "
                      f"Loss: {avg_loss:.6f} | Val IC: {val_ic:.4f} | "
                      f"|ent|/|mse|: {avg_ent_ratio:.3f}")
            
            if not np.isnan(val_ic) and val_ic > best_val_ic:
                best_val_ic = val_ic
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(
                    CHECKPOINT_DIR,
                    f"best_{fold_name}_seed{seed}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"    ⏹️  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"  Best Val IC: {best_val_ic:.4f}")
        
        # ---- Step 7: Load best model and test ----
        ckpt_path = os.path.join(CHECKPOINT_DIR,
                                  f"best_{fold_name}_seed{seed}.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path,
                                              map_location=device))
        
        # ---- Step 8: Calibrate conformal gate on validation ----
        print("  🎯 Calibrating conformal gate...")
        conformal = ConformalGate(target_coverage=0.90)
        if val_preds_list:
            # Get regimes for validation data
            val_regimes = []
            for date in val_dates:
                mkt_row = mkt_lookup.get(date, {
                    "market_ret": 0, "market_vol": 0.01,
                    "vix_level": 15})
                regime_feats = np.array([[
                    mkt_row.get("market_ret", 0),
                    mkt_row.get("market_vol", 0),
                    mkt_row.get("vix_level", 0),
                ]], dtype=np.float64)
                r = regime_detector.predict(regime_feats)[0]
                n_stocks = len(val_batches[date]["tickers"])
                val_regimes.extend([r] * n_stocks)
            
            val_regimes = np.array(val_regimes[:len(val_true_all)])
            conformal.calibrate(val_true_all, val_preds_all, val_regimes)
            print(f"  ✅ Conformal: global_q = {conformal.global_q:.6f}")
        
        # ---- Step 9: Test predictions ----
        print("  📝 Generating test predictions...")
        test_dates = sorted(test_batches.keys())
        test_results = []
        
        model.eval()
        with torch.no_grad():
            for date in test_dates:
                batch_data = test_batches[date]
                mkt_row = mkt_lookup.get(date, {
                    "market_ret": 0, "market_vol": 0.01,
                    "vix_level": 15, "breadth": 0.5,
                    "avg_vol_ratio": 1.0})
                
                try:
                    batch = self.build_daily_batch(
                        batch_data, mkt_row, regime_detector,
                        lgbm, edge_index, device)
                except Exception:
                    continue
                
                labels = batch["labels"]
                preds, gate_w, expert_p = model(
                    batch["sequences"], batch["snapshot"],
                    batch["edge_index"], batch["regime_oh"],
                    batch["market_snap"], batch["lgbm_preds"])
                
                preds_np = preds.squeeze().cpu().numpy()
                labels_np = labels.cpu().numpy()
                gate_np = gate_w.cpu().numpy()
                
                # Conformal gate
                trade_mask, confidence, intervals = conformal.apply(
                    preds_np, np.full(len(preds_np), batch["regime_int"]))
                
                for i, ticker in enumerate(batch["tickers"]):
                    test_results.append({
                        "date": date,
                        "ticker": ticker,
                        "y_pred": preds_np[i],
                        "y_true": labels_np[i],
                        "current_close": batch["current_close"][i],
                        "future_close": batch["future_close"][i],
                        "regime": batch["regime_int"],
                        "gate_tkan": gate_np[i, 0],
                        "gate_lgbm": gate_np[i, 1],
                        "gate_gnn": gate_np[i, 2],
                        "conformal_trade": trade_mask[i],
                        "conformal_confidence": confidence[i],
                        "conformal_lower": intervals[i, 0],
                        "conformal_upper": intervals[i, 1],
                        "fold": fold_name,
                        "seed": seed,
                    })
        
        test_pred_df = pd.DataFrame(test_results)
        
        # Save predictions
        pred_path = os.path.join(
            RESULTS_DIR, f"preds_{fold_name}_seed{seed}.parquet")
        test_pred_df.to_parquet(pred_path, index=False)
        print(f"  💾 Saved {len(test_pred_df):,} predictions to {pred_path}")
        
        # Quick evaluation
        if len(test_pred_df) > 0:
            valid = test_pred_df.dropna(subset=["y_true", "y_pred"])
            ic = information_coefficient(
                valid["y_true"].values, valid["y_pred"].values)
            print(f"  📈 Test IC: {ic:.4f}")
            
            # Show price predictions for latest test date
            print(f"\n  📊 Sample price predictions "
                  f"(latest test date):")
            price_display = display_price_predictions(test_pred_df)
            if len(price_display) > 0:
                print(price_display.to_string(index=False))
        
        return {"predictions": test_pred_df, "best_val_ic": best_val_ic}
    
    def run_all_folds(self, full_df: pd.DataFrame,
                      market_df: pd.DataFrame,
                      seed: int = 42,
                      folds: List[Dict] = None) -> pd.DataFrame:
        """Run all walk-forward folds.
        
        Returns concatenated predictions DataFrame.
        """
        all_preds = []
        target_folds = folds or WALK_FORWARD_FOLDS
        
        for fold_idx, fold in enumerate(target_folds):
            result = self.train_one_fold(
                fold_idx, fold, full_df, market_df, seed)
            
            if result and "predictions" in result:
                all_preds.append(result["predictions"])
            
            # Checkpoint after each fold
            if all_preds:
                combined = pd.concat(all_preds, ignore_index=True)
                combined.to_parquet(os.path.join(
                    RESULTS_DIR, f"all_preds_seed{seed}_partial.parquet"),
                    index=False)
                print(f"\n  🔄 Checkpoint: {len(combined):,} predictions saved")
        
        if all_preds:
            final = pd.concat(all_preds, ignore_index=True)
            final.to_parquet(os.path.join(
                RESULTS_DIR, f"all_preds_seed{seed}.parquet"), index=False)
            return final
        
        return pd.DataFrame()
    
    def run_multi_seed(self, full_df: pd.DataFrame,
                       market_df: pd.DataFrame,
                       seeds: List[int] = None) -> pd.DataFrame:
        """Run all folds across multiple seeds.
        
        This is the FULL experiment. On Kaggle, run one seed per session.
        """
        seeds = seeds or list(range(self.config.n_seeds))
        all_seed_preds = []
        
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"  SEED {seed}/{len(seeds)-1}")
            print(f"{'#'*60}")
            
            seed_preds = self.run_all_folds(full_df, market_df, seed)
            if len(seed_preds) > 0:
                all_seed_preds.append(seed_preds)
        
        if all_seed_preds:
            final = pd.concat(all_seed_preds, ignore_index=True)
            final.to_parquet(os.path.join(
                RESULTS_DIR, "all_predictions.parquet"), index=False)
            return final
        
        return pd.DataFrame()
