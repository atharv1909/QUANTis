"""
QUANTIS 2.0 — Data Pipeline
Downloads real NIFTY 50 data via yfinance, computes features, builds labels.
NO synthetic data. NO hardcoded prices. Everything from real market data.

Usage on Kaggle:
    from data_pipeline import Nifty50Pipeline
    pipeline = Nifty50Pipeline()
    dataset = pipeline.run()
"""
import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

warnings.filterwarnings("ignore")

from config import (
    NIFTY50_TICKERS, SECTOR_MAP, FEATURE_COLS, D_FEATURES,
    SEQ_LEN, LABEL_HORIZON, DATA_DIR, TICKER_ISSUES
)


class Nifty50Pipeline:
    """
    End-to-end data pipeline:
        1. Download OHLCV from yfinance (real data)
        2. Download India VIX from yfinance (real data)
        3. Compute 22 technical features per stock
        4. Compute 5-day forward return labels
        5. Build sequences [T, D] per stock for temporal models
        6. Build stock graph for GNN
        7. Validate no data leakage
    """

    def __init__(self, tickers: List[str] = None, start: str = "2008-01-01",
                 end: str = "2025-06-30"):
        self.tickers = tickers or NIFTY50_TICKERS
        self.start = start
        self.end = end
        self.ohlcv: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.vix: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------------
    # Step 1: Download real market data
    # ----------------------------------------------------------------
    def download_ohlcv(self, cache: bool = True) -> pd.DataFrame:
        """Download daily OHLCV for all NIFTY 50 stocks from yfinance.

        Returns:
            DataFrame [N_rows, cols: date/ticker/open/high/low/close/volume]
            Real stock prices — not synthetic, not hardcoded.
        """
        cache_path = os.path.join(DATA_DIR, "nifty50_ohlcv.parquet")
        if cache and os.path.exists(cache_path):
            print(f"📂 Loading cached OHLCV from {cache_path}")
            self.ohlcv = pd.read_parquet(cache_path)
            return self.ohlcv

        import yfinance as yf
        print(f"📥 Downloading OHLCV for {len(self.tickers)} stocks "
              f"({self.start} to {self.end})...")

        frames = []
        failed = []
        for i, ticker in enumerate(self.tickers):
            try:
                df = yf.download(ticker, start=self.start, end=self.end,
                                 auto_adjust=True, progress=False)
                if df.empty:
                    failed.append(ticker)
                    continue
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df["ticker"] = ticker
                df = df.rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"
                })
                # Keep only needed columns
                cols_keep = ["date", "ticker", "open", "high", "low",
                             "close", "volume"]
                available = [c for c in cols_keep if c in df.columns]
                df = df[available].copy()
                frames.append(df)
                if (i + 1) % 10 == 0:
                    print(f"  ✅ Downloaded {i+1}/{len(self.tickers)}")
            except Exception as e:
                failed.append(ticker)
                print(f"  ❌ Failed {ticker}: {e}")

        if failed:
            print(f"⚠️  Failed tickers ({len(failed)}): {failed}")
            for f_ticker in failed:
                if f_ticker in TICKER_ISSUES:
                    print(f"  ℹ️  {f_ticker}: {TICKER_ISSUES[f_ticker]}")
                else:
                    print(f"  ❌ {f_ticker}: Unknown download failure")

        self.ohlcv = pd.concat(frames, ignore_index=True)
        self.ohlcv["date"] = pd.to_datetime(self.ohlcv["date"])
        self.ohlcv = self.ohlcv.sort_values(["date", "ticker"]).reset_index(drop=True)

        # Remove stocks with too little history (< 500 trading days)
        counts = self.ohlcv.groupby("ticker")["date"].count()
        valid_tickers = counts[counts >= 500].index.tolist()
        self.ohlcv = self.ohlcv[self.ohlcv["ticker"].isin(valid_tickers)].copy()
        self.tickers = valid_tickers
        print(f"✅ {len(self.tickers)} stocks with sufficient history")
        print(f"   Date range: {self.ohlcv['date'].min()} to "
              f"{self.ohlcv['date'].max()}")
        print(f"   Total rows: {len(self.ohlcv):,}")

        if cache:
            self.ohlcv.to_parquet(cache_path, index=False)
            print(f"💾 Cached to {cache_path}")

        return self.ohlcv

    def download_vix(self, cache: bool = True) -> pd.DataFrame:
        """Download India VIX from yfinance. Real data."""
        cache_path = os.path.join(DATA_DIR, "india_vix.parquet")
        if cache and os.path.exists(cache_path):
            self.vix = pd.read_parquet(cache_path)
            return self.vix

        import yfinance as yf
        print("📥 Downloading India VIX...")
        vix = yf.download("^INDIAVIX", start=self.start, end=self.end,
                           auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix.reset_index()
        vix = vix.rename(columns={"Date": "date", "Close": "vix_close"})
        self.vix = vix[["date", "vix_close"]].copy()
        self.vix["date"] = pd.to_datetime(self.vix["date"])

        if cache:
            self.vix.to_parquet(cache_path, index=False)
        print(f"✅ India VIX downloaded: {len(self.vix)} days")
        return self.vix

    def download_nifty50_index(self, cache: bool = True) -> pd.DataFrame:
        """Download the real NIFTY 50 index level (^NSEI) from yfinance.

        This is NOT used for training (no lookahead risk — it's an
        aggregate benchmark, not a per-stock feature). It exists purely
        for reporting:
          - Visualization.plot_regime_timeline needs a real index price
            series to overlay HMM regime shading on.
          - Visualization.plot_cumulative_returns needs a real
            "buy and hold NIFTY 50" benchmark curve.

        See generate_report.py, which calls this automatically.
        """
        cache_path = os.path.join(DATA_DIR, "nifty50_index.parquet")
        if cache and os.path.exists(cache_path):
            print(f"📂 Loading cached NIFTY 50 index from {cache_path}")
            return pd.read_parquet(cache_path)

        import yfinance as yf
        print("📥 Downloading NIFTY 50 index (^NSEI)...")
        idx = yf.download("^NSEI", start=self.start, end=self.end,
                           auto_adjust=True, progress=False)
        if idx.empty:
            raise ValueError("yfinance returned no data for ^NSEI")
        if isinstance(idx.columns, pd.MultiIndex):
            idx.columns = idx.columns.get_level_values(0)
        idx = idx.reset_index().rename(
            columns={"Date": "date", "Close": "index_close"})
        idx = idx[["date", "index_close"]].copy()
        idx["date"] = pd.to_datetime(idx["date"])

        if cache:
            idx.to_parquet(cache_path, index=False)
            print(f"💾 Cached to {cache_path}")
        print(f"✅ NIFTY 50 index downloaded: {len(idx)} days")
        return idx

    # ----------------------------------------------------------------
    # Step 2: Compute technical features
    # ----------------------------------------------------------------
    def compute_features(self) -> pd.DataFrame:
        """Compute all 22 technical features per stock.

        CRITICAL LEAKAGE PREVENTION:
        - All features use .shift(0) or past data only (rolling windows)
        - No future information leaks into features
        - Label computation is separate and uses explicit future shift
        """
        if self.ohlcv is None:
            raise ValueError("Call download_ohlcv() first")

        print("🔧 Computing features...")
        df = self.ohlcv.copy()
        g = df.groupby("ticker", group_keys=False)

        # ---- Returns (4 features) ----
        df["ret_1d"] = g["close"].pct_change(1)
        df["ret_5d"] = g["close"].pct_change(5)
        df["ret_10d"] = g["close"].pct_change(10)
        df["ret_21d"] = g["close"].pct_change(21)

        # ---- RSI 14 (1 feature) ----
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = g.apply(lambda x: x["close"].diff().clip(lower=0).rolling(14).mean())
        avg_loss = g.apply(lambda x: (-x["close"].diff()).clip(lower=0).rolling(14).mean())
        # Fix: compute per-group to avoid alignment issues
        rsi_list = []
        for ticker, grp in df.groupby("ticker"):
            d = grp["close"].diff()
            ag = d.clip(lower=0).rolling(14).mean()
            al = (-d).clip(lower=0).rolling(14).mean()
            rs = ag / (al + 1e-10)
            rsi_list.append(100 - (100 / (1 + rs)))
        df["rsi"] = pd.concat(rsi_list)

        # ---- MACD histogram (1 feature) ----
        macd_list = []
        for ticker, grp in df.groupby("ticker"):
            ema12 = grp["close"].ewm(span=12, adjust=False).mean()
            ema26 = grp["close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_list.append(macd_line - signal)
        df["macd_hist"] = pd.concat(macd_list)

        # ---- Bollinger Bands %B and Width (2 features) ----
        bb_list = []
        for ticker, grp in df.groupby("ticker"):
            sma20 = grp["close"].rolling(20).mean()
            std20 = grp["close"].rolling(20).std()
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            pctb = (grp["close"] - lower) / (upper - lower + 1e-10)
            width = (upper - lower) / (sma20 + 1e-10)
            bb_list.append(pd.DataFrame({"boll_pctb": pctb, "boll_width": width},
                                         index=grp.index))
        bb_df = pd.concat(bb_list)
        df["boll_pctb"] = bb_df["boll_pctb"]
        df["boll_width"] = bb_df["boll_width"]

        # ---- ATR 14 (1 feature) ----
        atr_list = []
        for ticker, grp in df.groupby("ticker"):
            h, l, c_prev = grp["high"], grp["low"], grp["close"].shift(1)
            tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()],
                          axis=1).max(axis=1)
            atr_list.append(tr.rolling(14).mean())
        df["atr"] = pd.concat(atr_list)

        # ---- Stochastic %K (1 feature) ----
        stoch_list = []
        for ticker, grp in df.groupby("ticker"):
            low14 = grp["low"].rolling(14).min()
            high14 = grp["high"].rolling(14).max()
            stoch_list.append(
                100 * (grp["close"] - low14) / (high14 - low14 + 1e-10))
        df["stoch_k"] = pd.concat(stoch_list)

        # ---- ADX simplified (1 feature) ----
        adx_list = []
        for ticker, grp in df.groupby("ticker"):
            plus_dm = (grp["high"] - grp["high"].shift(1)).clip(lower=0)
            minus_dm = (grp["low"].shift(1) - grp["low"]).clip(lower=0)
            atr14 = df.loc[grp.index, "atr"]
            plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
            minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
            adx_list.append(dx.rolling(14).mean())
        df["adx"] = pd.concat(adx_list)

        # ---- CCI (1 feature) ----
        cci_list = []
        for ticker, grp in df.groupby("ticker"):
            tp = (grp["high"] + grp["low"] + grp["close"]) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda w: np.mean(np.abs(w - w.mean())),
                                        raw=True)
            cci_list.append((tp - sma_tp) / (0.015 * mad + 1e-10))
        df["cci"] = pd.concat(cci_list)

        # ---- ROC 10 (1 feature) ----
        roc_list = []
        for ticker, grp in df.groupby("ticker"):
            roc_list.append(grp["close"].pct_change(10) * 100)
        df["roc"] = pd.concat(roc_list)

        # ---- OBV normalized (1 feature) ----
        obv_list = []
        for ticker, grp in df.groupby("ticker"):
            sign = np.sign(grp["close"].diff())
            obv = (sign * grp["volume"]).cumsum()
            obv_norm = (obv - obv.rolling(60).mean()) / (obv.rolling(60).std() + 1e-10)
            obv_list.append(obv_norm)
        df["obv_norm"] = pd.concat(obv_list)

        # ---- MFI (1 feature) ----
        mfi_list = []
        for ticker, grp in df.groupby("ticker"):
            tp = (grp["high"] + grp["low"] + grp["close"]) / 3
            mf = tp * grp["volume"]
            tp_diff = tp.diff()
            pos_mf = (mf * (tp_diff > 0).astype(float)).rolling(14).sum()
            neg_mf = (mf * (tp_diff <= 0).astype(float)).rolling(14).sum()
            mfi_list.append(100 - 100 / (1 + pos_mf / (neg_mf + 1e-10)))
        df["mfi"] = pd.concat(mfi_list)

        # ---- Williams %R (1 feature) ----
        wr_list = []
        for ticker, grp in df.groupby("ticker"):
            high14 = grp["high"].rolling(14).max()
            low14 = grp["low"].rolling(14).min()
            wr_list.append(-100 * (high14 - grp["close"]) / (high14 - low14 + 1e-10))
        df["williams_r"] = pd.concat(wr_list)

        # ---- Volume SMA ratio (1 feature) ----
        vsr_list = []
        for ticker, grp in df.groupby("ticker"):
            vsr_list.append(grp["volume"] / (grp["volume"].rolling(20).mean() + 1e-10))
        df["vol_sma_ratio"] = pd.concat(vsr_list)

        # ---- Volatility ratios (2 features) ----
        vr_list = []
        for ticker, grp in df.groupby("ticker"):
            ret = grp["close"].pct_change()
            v5 = ret.rolling(5).std()
            v10 = ret.rolling(10).std()
            v20 = ret.rolling(20).std()
            v60 = ret.rolling(60).std()
            vr_list.append(pd.DataFrame({
                "vol_ratio_5_20": v5 / (v20 + 1e-10),
                "vol_ratio_10_60": v10 / (v60 + 1e-10),
            }, index=grp.index))
        vr_df = pd.concat(vr_list)
        df["vol_ratio_5_20"] = vr_df["vol_ratio_5_20"]
        df["vol_ratio_10_60"] = vr_df["vol_ratio_10_60"]

        # ---- SMA ratios (3 features) ----
        sma_list = []
        for ticker, grp in df.groupby("ticker"):
            sma5 = grp["close"].rolling(5).mean()
            sma10 = grp["close"].rolling(10).mean()
            sma20 = grp["close"].rolling(20).mean()
            sma60 = grp["close"].rolling(60).mean()
            sma_list.append(pd.DataFrame({
                "sma_5_20": sma5 / (sma20 + 1e-10) - 1,
                "sma_10_60": sma10 / (sma60 + 1e-10) - 1,
                "sma_20_60": sma20 / (sma60 + 1e-10) - 1,
            }, index=grp.index))
        sma_df = pd.concat(sma_list)
        df["sma_5_20"] = sma_df["sma_5_20"]
        df["sma_10_60"] = sma_df["sma_10_60"]
        df["sma_20_60"] = sma_df["sma_20_60"]

        # ---- Add VIX features if available ----
        if self.vix is not None and len(self.vix) > 0:
            vix_data = self.vix.copy()
            vix_data["vix_level"] = vix_data["vix_close"]
            vix_data["vix_pctile"] = vix_data["vix_close"].rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            df = df.merge(vix_data[["date", "vix_level", "vix_pctile"]],
                         on="date", how="left")

        # ---- Daily Cross-Sectional Z-Score Standardization ----
        # Normalizes each feature across all stocks on each date t (CS Z-score)
        # Prevents long-term macroeconomic scale drift and enforces stationarity
        print("⚡ Applying daily cross-sectional Z-score standardization...")
        for col in FEATURE_COLS:
            if col in df.columns:
                mean_t = df.groupby("date")[col].transform("mean")
                std_t = df.groupby("date")[col].transform("std").replace(0, 1.0)
                df[col] = (df[col] - mean_t) / (std_t + 1e-8)
                df[col] = df[col].clip(-3.0, 3.0)  # clip extreme outliers

        self.features = df
        print(f"✅ Features computed & CS-standardized: {len(FEATURE_COLS)} features, "
              f"{len(df):,} rows")
        return df

    # ----------------------------------------------------------------
    # Step 3: Compute labels (forward returns)
    # ----------------------------------------------------------------
    def compute_labels(self) -> pd.DataFrame:
        """Compute 5-day forward return as prediction target.

        label_t = close_{t+5} / close_t - 1

        This is the ONLY place future data is used.
        Features at time t use ONLY data up to time t.
        """
        if self.features is None:
            raise ValueError("Call compute_features() first")

        print(f"🏷️  Computing {LABEL_HORIZON}-day forward return labels...")
        df = self.features.copy()

        label_list = []
        for ticker, grp in df.groupby("ticker"):
            fwd_ret = grp["close"].shift(-LABEL_HORIZON) / grp["close"] - 1
            label_list.append(fwd_ret)
        df["label"] = pd.concat(label_list)

        # Also store the actual future close price (for displaying predictions
        # as actual price values, NOT for training)
        price_list = []
        for ticker, grp in df.groupby("ticker"):
            price_list.append(grp["close"].shift(-LABEL_HORIZON))
        df["future_close"] = pd.concat(price_list)

        # Store current close (for converting predicted return → predicted price)
        df["current_close"] = df["close"]

        self.features = df
        n_valid = df["label"].notna().sum()
        print(f"✅ Labels computed: {n_valid:,} valid samples "
              f"(last {LABEL_HORIZON} days per stock have NaN labels)")
        return df

    # ----------------------------------------------------------------
    # Step 4: Normalize features (per walk-forward fold)
    # ----------------------------------------------------------------
    @staticmethod
    def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Z-score normalize using ONLY train statistics.

        No data leakage: test is normalized using train mean/std.
        """
        train_out = train_df.copy()
        test_out = test_df.copy()

        for col in feature_cols:
            if col in train_out.columns:
                mean = train_out[col].mean()
                std = train_out[col].std() + 1e-10
                train_out[col] = (train_out[col] - mean) / std
                test_out[col] = (test_out[col] - mean) / std

        return train_out, test_out

    # ----------------------------------------------------------------
    # Step 5: Build sequences for temporal models
    # ----------------------------------------------------------------
    @staticmethod
    def build_sequences(df: pd.DataFrame, feature_cols: List[str],
                        seq_len: int = SEQ_LEN) -> Dict:
        """Build [N_stocks, T, D] tensor for one trading day.

        For each stock on each date, we look back seq_len days.

        Returns dict of {date: {
            "sequences": np.array [N_stocks, seq_len, D_features],
            "snapshot":  np.array [N_stocks, D_features],
            "labels":    np.array [N_stocks],
            "tickers":   list of ticker names,
            "current_close": np.array [N_stocks],  ← real prices
            "future_close":  np.array [N_stocks],  ← real future prices
        }}
        """
        dates = sorted(df["date"].unique())
        tickers = sorted(df["ticker"].unique())
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # Pivot to wide format for each feature
        pivot_data = {}
        for col in feature_cols:
            pivot_data[col] = df.pivot(index="date", columns="ticker",
                                        values=col).reindex(columns=tickers)

        label_pivot = df.pivot(index="date", columns="ticker",
                                values="label").reindex(columns=tickers)
        close_pivot = df.pivot(index="date", columns="ticker",
                                values="current_close").reindex(columns=tickers)
        future_pivot = df.pivot(index="date", columns="ticker",
                                 values="future_close").reindex(columns=tickers)

        daily_batches = {}
        for i, date in enumerate(dates):
            if i < seq_len:
                continue  # need seq_len days of history

            # Check if this date has labels
            labels = label_pivot.loc[date].values
            if np.all(np.isnan(labels)):
                continue

            # Build sequences: [N_stocks, seq_len, D_features]
            window_dates = dates[i - seq_len: i]
            sequences = np.zeros((len(tickers), seq_len, len(feature_cols)))
            for fi, col in enumerate(feature_cols):
                seq_data = pivot_data[col].loc[window_dates].values  # [T, N]
                sequences[:, :, fi] = seq_data.T  # [N, T]

            # Current snapshot: [N_stocks, D_features] (latest day)
            snapshot = sequences[:, -1, :]  # [N, D]

            daily_batches[date] = {
                "sequences": sequences.astype(np.float32),
                "snapshot": snapshot.astype(np.float32),
                "labels": labels.astype(np.float32),
                "tickers": tickers,
                "current_close": close_pivot.loc[date].values.astype(np.float32),
                "future_close": future_pivot.loc[date].values.astype(np.float32),
            }

        return daily_batches

    # ----------------------------------------------------------------
    # Step 6: Build stock graph for GNN
    # ----------------------------------------------------------------
    @staticmethod
    def build_graph(returns_df: pd.DataFrame, tickers: List[str],
                    sector_map: Dict = None,
                    corr_threshold: float = 0.5,
                    mom_threshold: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Build heterogeneous stock graph with 3 edge types.

        Returns:
            edge_index: np.array [2, E]  — pairs of connected node indices
            edge_type:  np.array [E]     — 0=sector, 1=correlation, 2=momentum
        """
        sector_map = sector_map or SECTOR_MAP
        N = len(tickers)
        edges = []
        types = []

        # Edge Type 0: Same sector
        sectors = [sector_map.get(t, "Unknown") for t in tickers]
        for i in range(N):
            for j in range(i + 1, N):
                if sectors[i] == sectors[j] and sectors[i] != "Unknown":
                    edges.extend([[i, j], [j, i]])
                    types.extend([0, 0])

        # Edge Type 1: High correlation (rolling 60-day)
        if len(returns_df) >= 60:
            corr_matrix = returns_df[tickers].corr().values
            for i in range(N):
                for j in range(i + 1, N):
                    if not np.isnan(corr_matrix[i, j]):
                        if abs(corr_matrix[i, j]) > corr_threshold:
                            edges.extend([[i, j], [j, i]])
                            types.extend([1, 1])

        # Edge Type 2: Similar momentum
        if len(returns_df) >= 20:
            mom = returns_df[tickers].iloc[-20:].sum().values
            for i in range(N):
                for j in range(i + 1, N):
                    if not np.isnan(mom[i]) and not np.isnan(mom[j]):
                        if abs(mom[i] - mom[j]) < mom_threshold:
                            edges.extend([[i, j], [j, i]])
                            types.extend([2, 2])

        # Self-loops (every node connects to itself)
        for i in range(N):
            edges.append([i, i])
            types.append(0)

        edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
        edge_type = np.array(types, dtype=np.int64)      # [E]
        print(f"📊 Graph: {N} nodes, {len(types)} edges "
              f"(sector={types.count(0)}, corr={types.count(1)}, "
              f"mom={types.count(2)})")
        return edge_index, edge_type

    # ----------------------------------------------------------------
    # Step 7: Leakage validation
    # ----------------------------------------------------------------
    def validate_no_leakage(self):
        """Verify features don't use future data."""
        if self.features is None:
            raise ValueError("Call compute_features() first")

        print("🔍 Validating no data leakage...")
        df = self.features
        issues = []

        # Check: for each feature, correlation with FUTURE label should not
        # be suspiciously high (> 0.5 would indicate leakage)
        if "label" in df.columns:
            for col in FEATURE_COLS:
                if col in df.columns:
                    valid = df[[col, "label"]].dropna()
                    if len(valid) > 100:
                        corr = valid[col].corr(valid["label"])
                        if abs(corr) > 0.5:
                            issues.append(f"⚠️  {col} has suspicious "
                                        f"correlation with label: {corr:.3f}")

        if issues:
            for issue in issues:
                print(issue)
            print("❌ LEAKAGE DETECTED — investigate above features")
        else:
            print("✅ No obvious leakage detected (all feature-label "
                  "correlations < 0.5)")

    # ----------------------------------------------------------------
    # Main pipeline
    # ----------------------------------------------------------------
    def run(self, cache: bool = True) -> pd.DataFrame:
        """Execute full pipeline end-to-end.

        Returns:
            DataFrame with all features, labels, and price columns.
        """
        cache_path = os.path.join(DATA_DIR, "nifty50_processed.parquet")
        if cache and os.path.exists(cache_path):
            print(f"📂 Loading fully processed data from {cache_path}")
            self.features = pd.read_parquet(cache_path)
            self.tickers = sorted(self.features["ticker"].unique().tolist())
            return self.features

        self.download_ohlcv(cache=cache)
        try:
            self.download_vix(cache=cache)
        except Exception as e:
            print(f"⚠️  Could not download VIX: {e}. Continuing without it.")
            self.vix = None
        self.compute_features()
        self.compute_labels()
        self.validate_no_leakage()

        # Drop rows with NaN features (warmup period)
        before = len(self.features)
        self.features = self.features.dropna(subset=FEATURE_COLS).copy()
        after = len(self.features)
        print(f"🧹 Dropped {before - after:,} rows with NaN features "
              f"(warmup period)")

        # Bug 7 fix: Survivorship bias explicit warning
        print("⚠️  SURVIVORSHIP BIAS WARNING: Using present-day NIFTY 50 "
              "constituents applied retroactively. TRENT.NS (added 2023) and "
              "BEL.NS (added 2024) appear in pre-2023 training data. "
              "This is an acknowledged limitation.")

        if cache:
            self.features.to_parquet(cache_path, index=False)
            print(f"💾 Saved processed data to {cache_path}")

        # Bug 5 fix: Content hash verification
        import hashlib
        h = hashlib.sha256()
        with open(cache_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        data_hash = h.hexdigest()[:16]
        print(f"🔒 Data hash: {data_hash}")
        print(f"⚠️  Re-use this EXACT parquet file across all 5 seed runs.")

        return self.features


# ================================================================
# Market-level feature builder (for HMM and market embedding)
# ================================================================
class MarketFeatureBuilder:
    """Builds market-level (aggregate) features from stock data.

    These are used for:
    1. HMM regime detection input
    2. Market embedding network input
    """

    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: Full processed DataFrame from Nifty50Pipeline

        Returns:
            market_df: DataFrame [date, market_ret, market_vol,
                                   vix_level, breadth, avg_vol_ratio]
        """
        daily = df.groupby("date").agg(
            market_ret=("ret_1d", "mean"),
            market_vol_raw=("ret_1d", "std"),
            breadth=("ret_1d", lambda x: (x > 0).mean()),
            avg_vol_ratio=("vol_sma_ratio", "mean"),
        ).reset_index()

        # Rolling 20-day volatility
        daily["market_vol"] = daily["market_ret"].rolling(20).std()

        # Add VIX if available
        if "vix_level" in df.columns:
            vix_daily = df.groupby("date")["vix_level"].first().reset_index()
            daily = daily.merge(vix_daily, on="date", how="left")
        else:
            daily["vix_level"] = daily["market_vol"] * 100  # proxy

        daily = daily.dropna()
        if "vix_pctile" not in daily.columns:
            # Bug 10 fix: vix_pctile rank
            daily["vix_pctile"] = daily["vix_level"].rank(pct=True)
            
        return daily[["date", "market_ret", "market_vol", "vix_level",
                       "breadth", "avg_vol_ratio", "vix_pctile"]]


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    pipeline = Nifty50Pipeline(
        tickers=NIFTY50_TICKERS[:5],  # test with 5 stocks
        start="2020-01-01",
        end="2024-12-31"
    )
    data = pipeline.run(cache=False)
    print(f"\nSample data:")
    print(data[["date", "ticker", "close", "ret_1d", "rsi",
                "label", "current_close", "future_close"]].tail(10))
    print(f"\nFeature columns available: {[c for c in FEATURE_COLS if c in data.columns]}")
    print(f"\nData shape: {data.shape}")
