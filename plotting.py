"""
QUANTIS 2.0 — Publication Visualization Suite (v2)
8 publication-ready, 300 DPI figures for AAAI / IJCAI / NeurIPS:

Figure 1: Multi-Seed Cumulative Equity Curve (per-seed lines + mean + ±1σ band)
Figure 2: IC Distribution & 30-Day Rolling IC (per-seed averaged)
Figure 3: Conformal Abstention Gate (IC vs Coverage)
Figure 4: Hybrid Gate Weight Distribution Across HMM Regimes
Figure 5: Predicted vs Actual Stock Prices (scatter)
Figure 6: Per-Fold IC Comparison (diagnostic — catches broken folds)
Figure 7: Regime Timeline + Market Overlay (HMM quality)
Figure 8: Drawdown Underwater Chart (risk diagnostic)

All figures handle multi-seed data CORRECTLY: evaluate per-seed, then
average. No duplicate-stock inflation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless on Kaggle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional

# Style settings for academic publication
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PLOT_DIR = "/kaggle/working/plots" if os.path.exists("/kaggle") else "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Color palette
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_PURPLE = "#9467bd"
C_GRAY = "#7f7f7f"
C_LIGHT_BLUE = "#aec7e8"
C_LIGHT_ORANGE = "#ffbb78"
C_LIGHT_GREEN = "#98df8a"

EXPERT_COLORS = {"T-KAN Expert": C_BLUE, "LightGBM Expert": C_ORANGE, "GNN Expert": C_GREEN}
REGIME_COLORS = {"Bull Market": "#2ecc71", "Bear Market": "#e74c3c", "Sideways / Volatile": "#f39c12"}


def _get_seeds(df: pd.DataFrame) -> list:
    """Get sorted list of seeds in the dataframe."""
    if "seed" in df.columns:
        return sorted(df["seed"].unique())
    return [None]


def _single_seed_df(df: pd.DataFrame, seed) -> pd.DataFrame:
    """Get single-seed slice (or full df if no seed column)."""
    if seed is None or "seed" not in df.columns:
        return df
    return df[df["seed"] == seed].copy()


# ================================================================
# Figure 1: Multi-Seed Cumulative Equity Curve
# ================================================================
def plot_equity_curve(predictions_df: pd.DataFrame, cost_bps: float = 23.0,
                      save_path: str = None) -> plt.Figure:
    """Cumulative return: per-seed thin lines + mean thick line + ±1σ shaded band.
    
    Benchmark is computed from a SINGLE seed to avoid duplication inflation.
    Fold boundaries are shown as vertical dashed lines.
    """
    from evaluation import long_short_returns

    seeds = _get_seeds(predictions_df)
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    all_cum_returns = {}
    
    for i, seed in enumerate(seeds):
        seed_df = _single_seed_df(predictions_df, seed)
        ls_ret = long_short_returns(seed_df, top_k=5, bottom_k=5, cost_bps=cost_bps)
        cum_ret = (1 + ls_ret).cumprod() - 1
        all_cum_returns[seed] = cum_ret
        
        label = f"Seed {seed}" if seed is not None else "QUANTIS 2.0"
        alpha = 0.35 if len(seeds) > 1 else 1.0
        lw = 1.0 if len(seeds) > 1 else 2.2
        ax.plot(cum_ret.index, cum_ret.values * 100, color=C_BLUE,
                alpha=alpha, linewidth=lw, label=label if i == 0 else None)
    
    # Mean + ±1σ band (if multiple seeds)
    if len(seeds) > 1:
        all_dates = sorted(set().union(*(c.index for c in all_cum_returns.values())))
        aligned = pd.DataFrame({s: all_cum_returns[s] for s in seeds}, index=all_dates)
        mean_curve = aligned.mean(axis=1).ffill().bfill()
        std_curve = aligned.std(axis=1).ffill().bfill()
        
        ax.plot(mean_curve.index, mean_curve.values * 100, color=C_BLUE,
                linewidth=2.5, label=f"Mean ({len(seeds)} Seeds)", zorder=5)
        ax.fill_between(mean_curve.index,
                        (mean_curve - std_curve).values * 100,
                        (mean_curve + std_curve).values * 100,
                        color=C_BLUE, alpha=0.15, label="±1σ Band")
    
    # Benchmark from single seed (no duplication)
    ref_df = _single_seed_df(predictions_df, seeds[0])
    bench_daily = ref_df.groupby("date")["y_true"].mean()
    cum_bench = (1 + bench_daily).cumprod() - 1
    ax.plot(cum_bench.index, cum_bench.values * 100,
            color=C_GRAY, linewidth=1.5, linestyle="--",
            label="Market Benchmark (Equal Weight)")
    
    # Fold boundaries
    fold_boundaries = ["2019-02-01", "2020-07-22", "2022-07-22", "2024-04-22"]
    for fb in fold_boundaries:
        fb_date = pd.Timestamp(fb)
        if cum_bench.index.min() <= fb_date <= cum_bench.index.max():
            ax.axvline(fb_date, color=C_GRAY, linestyle=":", alpha=0.5, linewidth=0.8)
    
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_title("Figure 1: Cumulative Strategy Returns vs Market Benchmark", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_path = save_path or os.path.join(PLOT_DIR, "fig1_equity_curve.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 1: {save_path}")
    return fig


# ================================================================
# Figure 2: IC Distribution & 30-Day Rolling IC
# ================================================================
def plot_ic_distribution(predictions_df: pd.DataFrame,
                         save_path: str = None) -> plt.Figure:
    """Rolling IC + histogram. Multi-seed: average daily ICs across seeds."""
    seeds = _get_seeds(predictions_df)
    
    # Compute daily ICs per seed
    seed_ic_series = {}
    for seed in seeds:
        seed_df = _single_seed_df(predictions_df, seed)
        daily_ics = {}
        for d, grp in seed_df.groupby("date"):
            valid = grp.dropna(subset=["y_true", "y_pred"])
            if len(valid) >= 5:
                daily_ics[d] = np.corrcoef(valid["y_true"], valid["y_pred"])[0, 1]
        seed_ic_series[seed] = pd.Series(daily_ics)
    
    # Average across seeds
    all_dates = sorted(set().union(*(s.index for s in seed_ic_series.values())))
    ic_matrix = pd.DataFrame(seed_ic_series, index=all_dates)
    mean_ic = ic_matrix.mean(axis=1).dropna()
    
    rolling_ic = mean_ic.rolling(30).mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5),
                                     gridspec_kw={"width_ratios": [2.5, 1]})
    
    # Time series
    ax1.plot(mean_ic.index, mean_ic.values, color=C_LIGHT_BLUE, alpha=0.4,
             label="Daily IC (seed-averaged)")
    ax1.plot(rolling_ic.index, rolling_ic.values, color=C_BLUE, linewidth=2,
             label="30-Day Rolling IC")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title("Daily & 30-Day Rolling Information Coefficient (IC)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("IC")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Histogram
    sns.histplot(mean_ic.values, ax=ax2, kde=True, color=C_BLUE, bins=30)
    ax2.axvline(mean_ic.mean(), color=C_RED, linestyle="--", linewidth=1.5,
                label=f"Mean IC: {mean_ic.mean():.4f}")
    ax2.set_title("IC Distribution Density")
    ax2.set_xlabel("IC")
    ax2.legend(loc="upper right")
    
    fig.suptitle("Figure 2: Information Coefficient (IC) Signal Stability Analysis",
                 y=1.02)
    save_path = save_path or os.path.join(PLOT_DIR, "fig2_ic_distribution.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 2: {save_path}")
    return fig


# ================================================================
# Figure 3: Conformal Abstention Gate (IC vs Coverage)
# ================================================================
def plot_conformal_abstention(predictions_df: pd.DataFrame,
                               save_path: str = None) -> plt.Figure:
    """IC vs Coverage trade-off. Multi-seed: de-duplicate by averaging y_pred
    across seeds per (date, ticker), keeping single y_true."""
    if "conformal_confidence" not in predictions_df.columns:
        print("⚠️  No conformal confidence column; skipping Fig 3.")
        return None
    
    # De-duplicate: average across seeds
    df = predictions_df.dropna(subset=["y_true", "y_pred", "conformal_confidence"])
    if "seed" in df.columns and df["seed"].nunique() > 1:
        df = df.groupby(["date", "ticker"]).agg({
            "y_true": "first",
            "y_pred": "mean",
            "conformal_confidence": "mean",
        }).reset_index()
    
    quantiles = np.linspace(0.1, 1.0, 10)
    coverages, ics = [], []
    
    for q in quantiles:
        threshold = df["conformal_confidence"].quantile(1 - q)
        subset = df[df["conformal_confidence"] >= threshold]
        if len(subset) > 100:
            daily_ics = []
            for _, g in subset.groupby("date"):
                if len(g) >= 3:
                    ic = np.corrcoef(g["y_true"], g["y_pred"])[0, 1]
                    if not np.isnan(ic):
                        daily_ics.append(ic)
            if daily_ics:
                coverages.append(q * 100)
                ics.append(np.mean(daily_ics))
    
    if not coverages:
        print("⚠️  Not enough data for conformal plot; skipping Fig 3.")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(coverages, ics, marker="o", color=C_GREEN, linewidth=2.2, markersize=7,
            markerfacecolor="white", markeredgewidth=2)
    
    # Shade the "gain" region
    if len(ics) >= 2 and ics[0] > ics[-1]:
        ax.fill_between(coverages, ics[-1], ics, alpha=0.1, color=C_GREEN)
    
    ax.set_title("Figure 3: Conformal Abstention Gate — IC vs Coverage Trade-off",
                 pad=12)
    ax.set_xlabel("Trade Coverage (% of Predictions Executed)")
    ax.set_ylabel("Mean Information Coefficient (IC)")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    if len(ics) > 0:
        ax.annotate(f"Full (100%): IC={ics[-1]:.4f}",
                    xy=(coverages[-1], ics[-1]),
                    xytext=(coverages[-1] - 20, ics[-1] - 0.008),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9)
        ax.annotate(f"Top {coverages[0]:.0f}%: IC={ics[0]:.4f}",
                    xy=(coverages[0], ics[0]),
                    xytext=(coverages[0] + 8, ics[0] + 0.005),
                    arrowprops=dict(arrowstyle="->", color=C_GREEN),
                    fontsize=9)
    
    save_path = save_path or os.path.join(PLOT_DIR, "fig3_conformal_gate.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 3: {save_path}")
    return fig


# ================================================================
# Figure 4: Hybrid Gate Weight Distribution Across Regimes
# ================================================================
def plot_hybrid_gate_regimes(predictions_df: pd.DataFrame,
                              save_path: str = None) -> plt.Figure:
    """Expert weight allocation per regime. Multi-seed: average gate weights
    across seeds per (date, ticker) before computing regime means."""
    gate_cols = ["gate_tkan", "gate_lgbm", "gate_gnn"]
    if not all(c in predictions_df.columns for c in gate_cols):
        print("⚠️  Gate weight columns missing; skipping Fig 4.")
        return None
    if "regime" not in predictions_df.columns:
        print("⚠️  Regime column missing; skipping Fig 4.")
        return None
    
    df = predictions_df.copy()
    
    # De-duplicate across seeds
    if "seed" in df.columns and df["seed"].nunique() > 1:
        df = df.groupby(["date", "ticker"]).agg({
            "gate_tkan": "mean", "gate_lgbm": "mean", "gate_gnn": "mean",
            "regime": "first",
        }).reset_index()
    
    regime_names = {0: "Bull Market", 1: "Bear Market", 2: "Sideways / Volatile"}
    df["regime_name"] = df["regime"].map(regime_names)
    
    regime_weights = df.groupby("regime_name")[gate_cols].mean()
    regime_weights.columns = ["T-KAN Expert", "LightGBM Expert", "GNN Expert"]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Grouped bars (not stacked) for clearer comparison
    x = np.arange(len(regime_weights))
    width = 0.22
    experts = regime_weights.columns.tolist()
    colors = [EXPERT_COLORS[e] for e in experts]
    
    for i, (expert, color) in enumerate(zip(experts, colors)):
        vals = regime_weights[expert].values
        bars = ax.bar(x + (i - 1) * width, vals, width, label=expert,
                      color=color, edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    
    ax.set_title("Figure 4: Hybrid Gate Expert Allocation Across HMM Regimes",
                 pad=12)
    ax.set_xlabel("Market Regime (HMM State)")
    ax.set_ylabel("Average Expert Weight")
    ax.set_xticks(x)
    ax.set_xticklabels(regime_weights.index, rotation=0)
    ax.set_ylim(0, min(1.05, regime_weights.values.max() + 0.15))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    save_path = save_path or os.path.join(PLOT_DIR, "fig4_hybrid_gate_regimes.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 4: {save_path}")
    return fig


# ================================================================
# Figure 5: Predicted vs Actual Stock Prices
# ================================================================
def plot_price_precision(predictions_df: pd.DataFrame,
                          save_path: str = None) -> plt.Figure:
    """Scatter: predicted vs actual prices. Uses single seed to avoid duplicates."""
    from evaluation import predictions_to_prices
    
    # Use first seed only for clean scatter
    seeds = _get_seeds(predictions_df)
    ref_df = _single_seed_df(predictions_df, seeds[0])
    
    prices_df = predictions_to_prices(ref_df).dropna(
        subset=["predicted_price", "actual_price"])
    sample = prices_df.sample(min(2000, len(prices_df)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sample["actual_price"], sample["predicted_price"],
               alpha=0.4, color=C_BLUE, edgecolors="none", s=20)
    
    # Identity line
    max_val = max(sample["actual_price"].max(), sample["predicted_price"].max())
    min_val = min(sample["actual_price"].min(), sample["predicted_price"].min())
    ax.plot([min_val, max_val], [min_val, max_val], color=C_RED,
            linestyle="--", linewidth=1.5, label="Ideal 1:1 Line")
    
    mape = np.mean(np.abs(sample["price_error_pct"]))
    r2 = np.corrcoef(sample["actual_price"], sample["predicted_price"])[0, 1] ** 2
    
    ax.set_title(f"Figure 5: 5-Day Predicted vs Actual Prices "
                 f"(MAPE: {mape:.2f}%, R²: {r2:.4f})", pad=12)
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_path = save_path or os.path.join(PLOT_DIR, "fig5_price_precision.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 5: {save_path}")
    return fig


# ================================================================
# Figure 6: Per-Fold IC Comparison (DIAGNOSTIC — catches broken folds)
# ================================================================
def plot_per_fold_ic(predictions_df: pd.DataFrame,
                      save_path: str = None) -> plt.Figure:
    """Bar chart: IC per fold, colored by seed. Instantly shows if a fold is dead."""
    if "fold" not in predictions_df.columns:
        print("⚠️  No fold column; skipping Fig 6.")
        return None
    
    seeds = _get_seeds(predictions_df)
    folds = sorted(predictions_df["fold"].unique())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(folds))
    n_seeds = len(seeds)
    width = 0.7 / max(n_seeds, 1)
    
    seed_colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE]
    
    for i, seed in enumerate(seeds):
        seed_df = _single_seed_df(predictions_df, seed)
        fold_ics = []
        for fold in folds:
            fold_df = seed_df[seed_df["fold"] == fold].dropna(
                subset=["y_true", "y_pred"])
            if len(fold_df) > 10:
                ic = np.corrcoef(fold_df["y_true"], fold_df["y_pred"])[0, 1]
                fold_ics.append(ic)
            else:
                fold_ics.append(0)
        
        color = seed_colors[i % len(seed_colors)]
        label = f"Seed {seed}" if seed is not None else "IC"
        offset = (i - (n_seeds - 1) / 2) * width
        bars = ax.bar(x + offset, fold_ics, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5, alpha=0.85)
        
        # Value labels
        for bar, v in zip(bars, fold_ics):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
    
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("Figure 6: Per-Fold Information Coefficient (IC) Across Seeds",
                 pad=12)
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Test IC")
    ax.set_xticks(x)
    
    # Make fold labels more descriptive
    fold_labels = []
    fold_periods = {
        "fold_1": "Fold 1\n(2019)",
        "fold_2": "Fold 2\n(2020-21)",
        "fold_3": "Fold 3\n(2022-23)",
        "fold_4": "Fold 4\n(2024-25)",
    }
    for f in folds:
        fold_labels.append(fold_periods.get(f, f))
    ax.set_xticklabels(fold_labels)
    
    ax.legend(loc="upper right", frameon=True, ncol=min(n_seeds, 3))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    save_path = save_path or os.path.join(PLOT_DIR, "fig6_per_fold_ic.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 6: {save_path}")
    return fig


# ================================================================
# Figure 7: Regime Timeline + Market Overlay
# ================================================================
def plot_regime_timeline(predictions_df: pd.DataFrame,
                          save_path: str = None) -> plt.Figure:
    """Shows HMM regime (colored background) overlaid with cumulative market return.
    Verifies regime detection quality visually."""
    if "regime" not in predictions_df.columns:
        print("⚠️  No regime column; skipping Fig 7.")
        return None
    
    # Use single seed
    seeds = _get_seeds(predictions_df)
    ref_df = _single_seed_df(predictions_df, seeds[0])
    
    # Daily regime and market return
    daily = ref_df.groupby("date").agg(
        regime=("regime", "first"),
        market_ret=("y_true", "mean"),
    ).sort_index()
    
    cum_market = (1 + daily["market_ret"]).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 4.5))
    
    # Plot market line
    ax.plot(daily.index, cum_market.values, color="black", linewidth=1.5,
            label="NIFTY 50 (Equal Weight)", zorder=3)
    
    # Colored background per regime
    regime_colors_map = {0: "#2ecc7140", 1: "#e74c3c40", 2: "#f39c1230"}
    regime_labels = {0: "Bull", 1: "Bear", 2: "Sideways"}
    plotted_regimes = set()
    
    dates = daily.index.tolist()
    regimes = daily["regime"].values
    
    i = 0
    while i < len(dates):
        r = regimes[i]
        j = i
        while j < len(dates) and regimes[j] == r:
            j += 1
        color = regime_colors_map.get(r, "#cccccc30")
        label = regime_labels.get(r, "Unknown") if r not in plotted_regimes else None
        ax.axvspan(dates[i], dates[min(j, len(dates) - 1)],
                   color=color, label=label, zorder=0)
        plotted_regimes.add(r)
        i = j
    
    ax.set_title("Figure 7: HMM Regime Detection Timeline vs Market Performance",
                 pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Market Return (1.0 = start)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    
    save_path = save_path or os.path.join(PLOT_DIR, "fig7_regime_timeline.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 7: {save_path}")
    return fig


# ================================================================
# Figure 8: Drawdown Underwater Chart
# ================================================================
def plot_drawdown(predictions_df: pd.DataFrame, cost_bps: float = 23.0,
                   save_path: str = None) -> plt.Figure:
    """Underwater chart showing drawdown over time. Multi-seed: per-seed + mean."""
    from evaluation import long_short_returns
    
    seeds = _get_seeds(predictions_df)
    fig, ax = plt.subplots(figsize=(12, 4))
    
    all_dd = {}
    for i, seed in enumerate(seeds):
        seed_df = _single_seed_df(predictions_df, seed)
        ls_ret = long_short_returns(seed_df, top_k=5, bottom_k=5, cost_bps=cost_bps)
        cum = (1 + ls_ret).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max * 100  # percentage
        all_dd[seed] = drawdown
        
        alpha = 0.3 if len(seeds) > 1 else 1.0
        ax.fill_between(drawdown.index, drawdown.values, 0,
                        color=C_RED, alpha=alpha * 0.3)
        if len(seeds) == 1:
            ax.plot(drawdown.index, drawdown.values, color=C_RED, linewidth=1.5)
    
    # Mean drawdown
    if len(seeds) > 1:
        all_dates = sorted(set().union(*(d.index for d in all_dd.values())))
        dd_matrix = pd.DataFrame(all_dd, index=all_dates).ffill().bfill()
        mean_dd = dd_matrix.mean(axis=1)
        ax.plot(mean_dd.index, mean_dd.values, color=C_RED, linewidth=2,
                label=f"Mean Drawdown ({len(seeds)} seeds)")
        ax.fill_between(mean_dd.index, mean_dd.values, 0,
                        color=C_RED, alpha=0.15)
        
        # Annotate worst drawdown
        worst_date = mean_dd.idxmin()
        worst_val = mean_dd.min()
        ax.annotate(f"Max DD: {worst_val:.1f}%",
                    xy=(worst_date, worst_val),
                    xytext=(worst_date, worst_val - 5),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9, fontweight="bold")
    
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Figure 8: Strategy Drawdown Underwater Chart", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    if len(seeds) > 1:
        ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_path = save_path or os.path.join(PLOT_DIR, "fig8_drawdown.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"📊 Saved Figure 8: {save_path}")
    return fig


# ================================================================
# Generate All Plots
# ================================================================
def generate_all_plots(predictions_df: pd.DataFrame) -> Dict[str, plt.Figure]:
    """Generate and save all 8 paper figures in 300 DPI.
    
    Handles multi-seed data CORRECTLY — no duplicate inflation.
    """
    seeds = _get_seeds(predictions_df)
    n_seeds = len(seeds)
    n_rows = len(predictions_df)
    
    print(f"\n🎨 Generating 8 Publication-Ready Figures (300 DPI)")
    print(f"   Data: {n_rows:,} predictions | {n_seeds} seed(s)")
    print(f"   Output: {PLOT_DIR}/")
    print()
    
    figs = {}
    
    try:
        figs["equity"] = plot_equity_curve(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 1 error: {e}")
    
    try:
        figs["ic_dist"] = plot_ic_distribution(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 2 error: {e}")
    
    try:
        figs["conformal"] = plot_conformal_abstention(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 3 error: {e}")
    
    try:
        figs["gate_regimes"] = plot_hybrid_gate_regimes(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 4 error: {e}")
    
    try:
        figs["price_precision"] = plot_price_precision(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 5 error: {e}")
    
    try:
        figs["per_fold_ic"] = plot_per_fold_ic(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 6 error: {e}")
    
    try:
        figs["regime_timeline"] = plot_regime_timeline(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 7 error: {e}")
    
    try:
        figs["drawdown"] = plot_drawdown(predictions_df)
    except Exception as e:
        print(f"⚠️  Fig 8 error: {e}")
    
    n_ok = len(figs)
    print(f"\n✅ {n_ok}/8 figures generated → {PLOT_DIR}/")
    return figs
