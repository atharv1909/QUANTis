import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import YearLocator, DateFormatter

# Set style for publishable plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Try importing evaluation for portfolio returns
try:
    from evaluation import long_short_returns
except ImportError:
    # Fallback if run independently
    def long_short_returns(df, top_k=5, bottom_k=5, cost_bps=23.0):
        daily_returns = []
        dates = sorted(df['date'].unique())
        cost = cost_bps / 10000.0
        
        for dt in dates:
            day_df = df[df['date'] == dt].dropna(subset=['y_pred', 'y_true'])
            if len(day_df) < top_k + bottom_k:
                daily_returns.append(0.0)
                continue
            
            day_df = day_df.sort_values('y_pred', ascending=False)
            longs = day_df.head(top_k)
            shorts = day_df.tail(bottom_k)
            
            ret_long = longs['y_true'].mean()
            ret_short = shorts['y_true'].mean()
            
            day_ret = 0.5 * ret_long - 0.5 * ret_short - cost * 2
            daily_returns.append(day_ret)
            
        return pd.Series(daily_returns, index=dates)

def generate_visuals():
    print("="*60)
    print("  Generating Publishable Figures for QUANTIS 2.0")
    print("="*60)
    
    os.makedirs("figures", exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("Loading prediction parquets...")
    # Check absolute kaggle path first, then relative path
    search_paths = [
        "/kaggle/working/results/preds_fold_*_seed*.parquet",
        "results/preds_fold_*_seed*.parquet"
    ]
    
    pred_files = []
    for path in search_paths:
        pred_files.extend(glob.glob(path))
        
    if not pred_files:
        print("❌ No prediction files found in results/ directories!")
        return
        
    dfs = [pd.read_parquet(f) for f in pred_files]
    preds = pd.concat(dfs)
    
    # Average across all seeds for each (date, ticker) pair to get the ensemble prediction
    print("Asembling ensemble predictions across all seeds...")
    agg_funcs = {'y_pred': 'mean', 'y_true': 'mean'}
    if 'regime' in preds.columns:
        agg_funcs['regime'] = lambda x: x.mode()[0]
    
    gate_cols = [c for c in preds.columns if c.startswith('gate_')]
    for g in gate_cols:
        agg_funcs[g] = 'mean'
        
    preds = preds.groupby(['date', 'ticker']).agg(agg_funcs).reset_index()
    preds = preds.sort_values("date")
    # Convert dates strictly to datetime
    preds["date"] = pd.to_datetime(preds["date"])
    
    # ---------------------------------------------------------
    # 2. Plot 1: Cumulative Returns with Regimes
    # ---------------------------------------------------------
    print("Plotting Cumulative Returns Tear Sheet...")
    ls_returns = long_short_returns(preds)
    cum_returns = (1 + ls_returns).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cum_returns.index, cum_returns.values, color='#2c3e50', linewidth=2.5, label='QUANTIS 2.0 Strategy')
    
    # Shade regimes
    if "regime" in preds.columns:
        # Get dominant regime for each day
        daily_regime = preds.groupby("date")["regime"].agg(lambda x: x.mode()[0])
        # Find continuous blocks of regimes
        regime_colors = {0: '#2ecc71', 1: '#e74c3c', 2: '#95a5a6'} # Bull, Bear, Sideways
        
        start_idx = 0
        current_regime = daily_regime.iloc[0]
        
        for i in range(1, len(daily_regime)):
            if daily_regime.iloc[i] != current_regime or i == len(daily_regime) - 1:
                end_idx = i
                color = regime_colors.get(current_regime, '#bdc3c7')
                
                # Add a dummy span for legend
                if start_idx == 0:
                    ax.axvspan(daily_regime.index[start_idx], daily_regime.index[end_idx], 
                               alpha=0.15, color=regime_colors.get(0), label="Bull Regime" if current_regime==0 else "")
                    ax.axvspan(daily_regime.index[start_idx], daily_regime.index[start_idx], 
                               alpha=0.15, color=regime_colors.get(1), label="Bear Regime")
                    ax.axvspan(daily_regime.index[start_idx], daily_regime.index[start_idx], 
                               alpha=0.15, color=regime_colors.get(2), label="Sideways")
                               
                ax.axvspan(daily_regime.index[start_idx], daily_regime.index[end_idx], 
                           alpha=0.15, color=color, linewidth=0)
                           
                start_idx = i
                current_regime = daily_regime.iloc[i]

    ax.set_title("QUANTIS 2.0: Cumulative Out-of-Sample Return (2019–2025)", fontweight="bold")
    ax.set_ylabel("Cumulative Wealth (1.0 = Initial)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig("figures/cumulative_returns.png", dpi=300)
    print("✅ Saved figures/cumulative_returns.png")
    
    # ---------------------------------------------------------
    # 3. Plot 2: Dynamic Expert Gating Distribution
    # ---------------------------------------------------------
    print("Plotting Dynamic Expert Gating...")
    gate_cols = [c for c in preds.columns if c.startswith("gate_")]
    if gate_cols:
        daily_gates = preds.groupby("date")[gate_cols].mean()
        
        # Smooth with a 21-day moving average to make the chart readable
        smoothed_gates = daily_gates.rolling(window=21, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels = [c.replace("gate_", "").upper() for c in gate_cols]
        colors = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71'][:len(gate_cols)]
        
        ax.stackplot(smoothed_gates.index, smoothed_gates.values.T, 
                     labels=labels, colors=colors, alpha=0.8)
                     
        ax.set_title("Dynamic MoE Gating Allocation (21-Day Moving Avg)", fontweight="bold")
        ax.set_ylabel("Expert Allocation Weight")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        fig.tight_layout()
        fig.savefig("figures/expert_gating.png", dpi=300)
        print("✅ Saved figures/expert_gating.png")
    else:
        print("⚠️ No gate columns found in predictions.")
        
    # ---------------------------------------------------------
    # 4. Plot 3: Bhavcopy Validation Scatter Plot
    # ---------------------------------------------------------
    print("Plotting Bhavcopy Data Validation...")
    if os.path.exists("results/bhavcopy_validation.csv"):
        bhav_df = pd.read_csv("results/bhavcopy_validation.csv")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(bhav_df["Bhavcopy_Ret_%"], bhav_df["YF_Ret_%"], 
                   color='#2980b9', alpha=0.7, edgecolors='white', s=80)
                   
        # Perfect match line
        min_val = min(bhav_df["Bhavcopy_Ret_%"].min(), bhav_df["YF_Ret_%"].min()) - 1
        max_val = max(bhav_df["Bhavcopy_Ret_%"].max(), bhav_df["YF_Ret_%"].max()) + 1
        
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Match (y=x)')
        
        ax.set_title("Data Integrity Validation: NSE Bhavcopy vs YFinance", fontweight="bold")
        ax.set_xlabel("Official NSE Exchange Daily Return (%)")
        ax.set_ylabel("YFinance Extracted Daily Return (%)")
        
        # Add R2 stat to plot
        from scipy.stats import pearsonr
        r, _ = pearsonr(bhav_df["Bhavcopy_Ret_%"], bhav_df["YF_Ret_%"])
        r2 = r**2
        ax.text(0.05, 0.95, f"Validation $R^2 = {r2:.4f}$", 
                transform=ax.transAxes, fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig("figures/bhavcopy_validation.png", dpi=300)
        print("✅ Saved figures/bhavcopy_validation.png")
    else:
        print("⚠️ Bhavcopy validation CSV not found. Run validate_bhavcopy.py first!")

    print("="*60)
    print("All figures successfully generated in figures/ directory!")
    print("="*60)

if __name__ == "__main__":
    generate_visuals()
