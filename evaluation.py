"""
QUANTIS 2.0 — Evaluation Metrics
IC, ICIR, RankIC, Sharpe, Sortino, Calmar, MDD, Precision@N, NDCG,
statistical tests (DM-test, DSR, PBO, Benjamini-Hochberg FDR), and price prediction output.

NO hardcoded values. Everything computed from actual predictions.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple, List


# ================================================================
# Core Metrics
# ================================================================
def information_coefficient(y_true: np.ndarray,
                            y_pred: np.ndarray) -> float:
    """Pearson correlation between predicted and realized returns.
    
    This is the STANDARD metric in quant finance papers (MASTER, HIST).
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 5:
        return np.nan
    return np.corrcoef(y_true[mask], y_pred[mask])[0, 1]


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (more robust to outliers)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 5:
        return np.nan
    return stats.spearmanr(y_true[mask], y_pred[mask])[0]


def ic_ir(daily_ics: np.ndarray) -> float:
    """IC Information Ratio = mean(daily_IC) / std(daily_IC).
    
    Higher is better. > 0.1 is publishable, > 0.15 is strong.
    """
    valid = daily_ics[~np.isnan(daily_ics)]
    if len(valid) < 10:
        return np.nan
    return np.mean(valid) / (np.std(valid) + 1e-10)


def compute_daily_ics(predictions_df: pd.DataFrame) -> np.ndarray:
    """Compute IC for each day.
    
    Args:
        predictions_df: DataFrame with columns [date, ticker, y_true, y_pred]
    Returns:
        daily_ics: np.array of per-day IC values
    """
    daily_ics = []
    for date, group in predictions_df.groupby("date"):
        ic = information_coefficient(
            group["y_true"].values, group["y_pred"].values)
        daily_ics.append(ic)
    return np.array(daily_ics)


# ================================================================
# Portfolio / Trading Metrics
# ================================================================
def long_short_returns(predictions_df: pd.DataFrame,
                       top_k: int = 5, bottom_k: int = 5,
                       cost_bps: float = 23.0) -> pd.Series:
    """Compute daily long-short portfolio returns.
    
    Strategy: Long top-K predicted stocks, short bottom-K.
    Transaction costs applied on turnover.
    
    Returns:
        pd.Series indexed by date with daily portfolio returns
    """
    daily_returns = {}
    prev_long = set()
    prev_short = set()
    
    for date, group in predictions_df.groupby("date"):
        group = group.dropna(subset=["y_pred", "y_true"])
        n_available = len(group)
        if n_available < 2:
            continue
        
        k_top = min(top_k, max(1, n_available // 2))
        k_bot = min(bottom_k, max(1, n_available // 2))
        
        sorted_g = group.sort_values("y_pred", ascending=False)
        long_tickers = set(sorted_g.head(k_top)["ticker"].values)
        short_tickers = set(sorted_g.tail(k_bot)["ticker"].values)
        
        long_ret = sorted_g.head(k_top)["y_true"].mean()
        short_ret = sorted_g.tail(k_bot)["y_true"].mean()
        ls_ret = long_ret - short_ret
        
        # Turnover cost
        turnover = (len(long_tickers - prev_long) + 
                   len(short_tickers - prev_short))
        total_positions = k_top + k_bot
        turnover_pct = turnover / (total_positions + 1e-10)
        cost = turnover_pct * cost_bps / 10000
        
        daily_returns[date] = ls_ret - cost
        prev_long = long_tickers
        prev_short = short_tickers
    
    return pd.Series(daily_returns).sort_index()


def sharpe_ratio(returns: np.ndarray, annual_factor: float = 252) -> float:
    """Annualized Sharpe ratio (assuming daily returns, risk-free ~0)."""
    if len(returns) < 5:
        return np.nan
    return np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(annual_factor)


def sortino_ratio(returns: np.ndarray, annual_factor: float = 252) -> float:
    """Sortino ratio (penalizes only downside volatility)."""
    if len(returns) < 5:
        return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.std(downside)
    return np.mean(returns) / (downside_std + 1e-10) * np.sqrt(annual_factor)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative returns."""
    if len(returns) == 0:
        return np.nan
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / (running_max + 1e-10)
    if len(drawdowns) == 0:
        return np.nan
    return np.min(drawdowns)  # most negative


def calmar_ratio(returns: np.ndarray, annual_factor: float = 252) -> float:
    """Calmar ratio = annualized return / |max drawdown|."""
    if len(returns) < 5:
        return np.nan
    mdd = abs(max_drawdown(returns))
    if np.isnan(mdd) or mdd < 1e-10:
        return np.nan
    annual_ret = np.mean(returns) * annual_factor
    return annual_ret / mdd


# ================================================================
# Ranking Metrics
# ================================================================
def precision_at_n(y_true: np.ndarray, y_pred: np.ndarray,
                   n: int = 5) -> float:
    """Fraction of top-N predicted stocks that are in actual top-N."""
    if len(y_true) < n:
        return np.nan
    pred_top = set(np.argsort(y_pred)[-n:])
    true_top = set(np.argsort(y_true)[-n:])
    return len(pred_top & true_top) / n


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    if len(y_true) < k:
        return np.nan
    
    pred_order = np.argsort(y_pred)[::-1][:k]
    relevance = y_true[pred_order]
    
    # DCG
    dcg = np.sum((2 ** relevance - 1) / np.log2(np.arange(2, k + 2)))
    
    # Ideal DCG
    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_relevance = y_true[ideal_order]
    idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(np.arange(2, k + 2)))
    
    return dcg / (idcg + 1e-10)


# ================================================================
# Statistical Tests
# ================================================================
def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray,
                          h: int = 1) -> Tuple[float, float]:
    """Diebold-Mariano test for forecast comparison.
    
    H0: Both forecasts have equal predictive accuracy.
    
    Args:
        e1: Forecast errors from model 1 (our model)
        e2: Forecast errors from baseline
        h:  Forecast horizon
    Returns:
        (DM statistic, p-value)
    """
    d = e1**2 - e2**2  # loss differential
    n = len(d)
    d_bar = np.mean(d)
    
    # Autocovariance up to h-1 lags
    gamma = np.zeros(h)
    for k in range(h):
        gamma[k] = np.mean((d[k:] - d_bar) * (d[:n-k] - d_bar))
    
    var_d = (gamma[0] + 2 * np.sum(gamma[1:])) / n
    dm_stat = d_bar / (np.sqrt(var_d) + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def deflated_sharpe_ratio(sharpe: float, n_trials: int,
                           n_obs: int, skew: float = 0,
                           kurtosis: float = 3) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado).
    
    CRITICAL: Expects DAILY (non-annualized) Sharpe ratio to match n_obs frequency!
    If you have an annualized Sharpe, divide by sqrt(252) before calling.
    
    Adjusts for multiple testing. Returns probability that the Sharpe
    is not a statistical fluke given the number of trials.
    """
    max_expected = np.sqrt(2 * np.log(n_trials))  # E[max(Z_1,...,Z_M)]
    
    se = np.sqrt((1 - skew * sharpe +
                  (kurtosis - 1) / 4 * sharpe**2) / n_obs)
    
    dsr = stats.norm.cdf((sharpe - max_expected) / (se + 1e-10))
    return float(dsr)


def probability_of_backtest_overfitting(matrix_returns: np.ndarray,
                                         n_partitions: int = 16) -> float:
    """Probability of Backtest Overfitting (PBO) via CSCV (Bailey et al. 2014).
    
    Combinatorial Cross-Validation to estimate likelihood that chosen strategy
    is overfit across N configurations tested during development.
    
    Args:
        matrix_returns: [T, N] daily returns matrix for N strategy configurations
        n_partitions: Number of time slices (must be even, default 16)
    Returns:
        pbo: float in [0, 1] — fraction of OOS slices where IS-best underperforms median
    """
    if matrix_returns is None or matrix_returns.ndim != 2:
        return np.nan
    T, N = matrix_returns.shape
    if N < 2 or T < n_partitions:
        return np.nan
    
    partition_size = T // n_partitions
    sub_matrices = [matrix_returns[i * partition_size : (i + 1) * partition_size]
                    for i in range(n_partitions)]
    
    from itertools import combinations
    half = n_partitions // 2
    combos = list(combinations(range(n_partitions), half))
    
    if len(combos) > 200:
        np.random.seed(42)
        indices = np.random.choice(len(combos), 200, replace=False)
        combos = [combos[i] for i in indices]
        
    underperformed = []
    for is_indices in combos:
        oos_indices = [i for i in range(n_partitions) if i not in is_indices]
        
        is_data = np.concatenate([sub_matrices[i] for i in is_indices], axis=0)
        oos_data = np.concatenate([sub_matrices[i] for i in oos_indices], axis=0)
        
        is_sharpes = np.mean(is_data, axis=0) / (np.std(is_data, axis=0) + 1e-10)
        best_is_idx = np.argmax(is_sharpes)
        
        oos_sharpes = np.mean(oos_data, axis=0) / (np.std(oos_data, axis=0) + 1e-10)
        best_oos_sharpe = oos_sharpes[best_is_idx]
        median_oos_sharpe = np.median(oos_sharpes)
        
        underperformed.append(best_oos_sharpe < median_oos_sharpe)
        
    return float(np.mean(underperformed))


def benjamini_hochberg(p_values: np.ndarray,
                       alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg False Discovery Rate (FDR) correction.
    
    Args:
        p_values: array of raw p-values across multiple tests (e.g. per-stock IC)
        alpha: target FDR threshold (default 0.05)
    Returns:
        (adjusted_p_values, significant_mask)
    """
    p_vals = np.array(p_values)
    n = len(p_vals)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    
    adj_p = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        q = sorted_p[i] * n / (i + 1)
        cum_min = min(cum_min, q)
        adj_p[sorted_idx[i]] = cum_min
        
    significant = adj_p <= alpha
    return np.clip(adj_p, 0.0, 1.0), significant


# ================================================================
# Price Prediction Display (real prices, not synthetic)
# ================================================================
def predictions_to_prices(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Convert return predictions to actual price predictions.
    
    predicted_price = current_close × (1 + predicted_return)
    actual_price    = current_close × (1 + actual_return) = future_close
    
    This is NOT hardcoded — it uses real market data.
    
    Args:
        predictions_df: DataFrame with columns:
            [date, ticker, y_true, y_pred, current_close, future_close]
    
    Returns:
        DataFrame with additional columns:
            predicted_price, actual_price, price_error, price_error_pct
    """
    df = predictions_df.copy()
    
    # Convert predicted return to predicted price
    df["predicted_price"] = df["current_close"] * (1 + df["y_pred"])
    df["actual_price"] = df["future_close"]  # actual future close
    
    # Price prediction error
    df["price_error"] = df["predicted_price"] - df["actual_price"]
    df["price_error_pct"] = (
        df["price_error"] / (df["actual_price"] + 1e-10) * 100)
    
    return df


def display_price_predictions(predictions_df: pd.DataFrame,
                               date: str = None,
                               top_n: int = 10) -> pd.DataFrame:
    """Display actual vs predicted prices for a specific date.
    
    Shows: ticker, current price, predicted price, actual future price,
           error, and the model's recommendation (buy/sell/hold).
    
    ALL prices are real — from yfinance data, not synthetic.
    """
    df = predictions_to_prices(predictions_df)
    
    if date is not None:
        df = df[df["date"] == pd.Timestamp(date)]
    else:
        # Use latest available date
        df = df[df["date"] == df["date"].max()]
    
    if len(df) == 0:
        print("No predictions available for this date.")
        return pd.DataFrame()
    
    # Add recommendation
    df["signal"] = "HOLD"
    df.loc[df["y_pred"] > 0.005, "signal"] = "📈 BUY"   # >0.5% predicted return
    df.loc[df["y_pred"] < -0.005, "signal"] = "📉 SELL"  # <-0.5% predicted return
    
    # Sort by predicted return (best first)
    df = df.sort_values("y_pred", ascending=False)
    
    display_cols = [
        "ticker", "current_close", "predicted_price", "actual_price",
        "y_pred", "y_true", "price_error_pct", "signal"
    ]
    available = [c for c in display_cols if c in df.columns]
    result = df[available].head(top_n).copy()
    
    # Format for display
    result = result.rename(columns={
        "current_close": "Current ₹",
        "predicted_price": "Predicted ₹ (5d)",
        "actual_price": "Actual ₹ (5d)",
        "y_pred": "Pred Return",
        "y_true": "Actual Return",
        "price_error_pct": "Error %",
        "signal": "Signal",
    })
    
    return result


# ================================================================
# Full Evaluation Suite
# ================================================================
def full_evaluation(predictions_df: pd.DataFrame,
                    baseline_errors: np.ndarray = None,
                    n_trials: int = 5,
                    cost_bps: float = 23.0) -> Dict:
    """Run complete evaluation and return all metrics.
    
    Args:
        predictions_df: DataFrame with [date, ticker, y_true, y_pred,
                                         current_close, future_close]
        baseline_errors: Errors from baseline model (for DM-test)
        n_trials: Number of experiment seeds (for DSR)
        cost_bps: Transaction cost in bps (for portfolio metrics)
    
    Returns:
        Dict with all metrics
    """
    results = {}
    
    # ---- IC metrics ----
    daily_ics = compute_daily_ics(predictions_df)
    results["IC_mean"] = np.nanmean(daily_ics)
    results["IC_std"] = np.nanstd(daily_ics)
    results["ICIR"] = ic_ir(daily_ics)
    
    # Rank IC
    rank_ics = []
    for date, group in predictions_df.groupby("date"):
        ric = rank_ic(group["y_true"].values, group["y_pred"].values)
        rank_ics.append(ric)
    rank_ics = np.array(rank_ics)
    results["RankIC_mean"] = np.nanmean(rank_ics)
    results["RankIC_IR"] = ic_ir(rank_ics)
    
    # ---- Portfolio metrics ----
    ls_returns = long_short_returns(predictions_df, top_k=5, bottom_k=5,
                                     cost_bps=cost_bps)
    ls_arr = ls_returns.values
    results["Annualized_Return"] = np.mean(ls_arr) * 252
    results["Sharpe"] = sharpe_ratio(ls_arr)
    results["Sortino"] = sortino_ratio(ls_arr)
    results["MDD"] = max_drawdown(ls_arr)
    results["Calmar"] = calmar_ratio(ls_arr)
    
    # ---- Ranking metrics (daily average) ----
    p5_list, ndcg5_list = [], []
    for date, group in predictions_df.groupby("date"):
        vals = group.dropna(subset=["y_true", "y_pred"])
        if len(vals) >= 10:
            p5_list.append(precision_at_n(vals["y_true"].values,
                                          vals["y_pred"].values, 5))
            ndcg5_list.append(ndcg_at_k(vals["y_true"].values,
                                         vals["y_pred"].values, 5))
    results["Precision@5"] = np.nanmean(p5_list)
    results["NDCG@5"] = np.nanmean(ndcg5_list)
    
    # ---- Statistical tests ----
    if baseline_errors is not None:
        our_errors = []
        for date, group in predictions_df.groupby("date"):
            vals = group.dropna(subset=["y_true", "y_pred"])
            if len(vals) > 0:
                our_errors.append(
                    np.mean((vals["y_true"].values - vals["y_pred"].values)**2))
        our_errors = np.array(our_errors)
        
        min_len = min(len(our_errors), len(baseline_errors))
        dm_stat, dm_p = diebold_mariano_test(
            our_errors[:min_len], baseline_errors[:min_len])
        results["DM_stat"] = dm_stat
        results["DM_pvalue"] = dm_p
    
    # DSR (Bug 2 fix: divide annualized Sharpe by sqrt(252) for daily frequency)
    daily_sharpe = results["Sharpe"] / np.sqrt(252) if not np.isnan(results["Sharpe"]) else 0.0
    results["DSR"] = deflated_sharpe_ratio(
        daily_sharpe, n_trials, len(ls_arr))
    
    # ---- Price prediction accuracy ----
    prices = predictions_to_prices(predictions_df)
    valid_prices = prices.dropna(subset=["predicted_price", "actual_price"])
    if len(valid_prices) > 0:
        results["MAPE"] = np.mean(
            np.abs(valid_prices["price_error_pct"].values))
        results["Direction_Accuracy"] = np.mean(
            np.sign(valid_prices["y_pred"].values) ==
            np.sign(valid_prices["y_true"].values))
    
    return results


def print_results_table(results: Dict, model_name: str = "QUANTIS"):
    """Pretty-print results as a publishable table."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    
    metrics_format = {
        "IC_mean": ("IC (mean)", ".4f"),
        "IC_std": ("IC (std)", ".4f"),
        "ICIR": ("ICIR", ".4f"),
        "RankIC_mean": ("Rank IC (mean)", ".4f"),
        "RankIC_IR": ("Rank IC IR", ".4f"),
        "Annualized_Return": ("Ann. Return", ".2%"),
        "Sharpe": ("Sharpe Ratio", ".3f"),
        "Sortino": ("Sortino Ratio", ".3f"),
        "MDD": ("Max Drawdown", ".2%"),
        "Calmar": ("Calmar Ratio", ".3f"),
        "Precision@5": ("Precision@5", ".3f"),
        "NDCG@5": ("NDCG@5", ".3f"),
        "MAPE": ("Price MAPE %", ".2f"),
        "Direction_Accuracy": ("Direction Acc", ".2%"),
        "DSR": ("Deflated SR", ".4f"),
        "DM_stat": ("DM Statistic", ".3f"),
        "DM_pvalue": ("DM p-value", ".4f"),
    }
    
    for key, (label, fmt) in metrics_format.items():
        if key in results and results[key] is not None:
            val = results[key]
            if not np.isnan(val):
                print(f"  {label:25s}: {val:{fmt}}")
    
    print(f"{'='*60}\n")


# ================================================================
# Multi-Seed Evaluation (CORRECT method for combined seeds)
# ================================================================
def multi_seed_evaluation(combined_df: pd.DataFrame,
                          cost_bps: float = 23.0) -> Dict:
    """Evaluate predictions across multiple seeds CORRECTLY.
    
    CRITICAL: Do NOT just concatenate seeds and run full_evaluation().
    That duplicates stocks per day and breaks all ranking/portfolio metrics.
    
    Instead: evaluate each seed independently, then report mean ± std.
    
    Args:
        combined_df: DataFrame with 'seed' column from multiple training runs
    Returns:
        Dict with 'mean' and 'std' for every metric, plus per-seed breakdown
    """
    if "seed" not in combined_df.columns:
        # Single seed — just run normal evaluation
        results = full_evaluation(combined_df, n_trials=1, cost_bps=cost_bps)
        return {"mean": results, "std": {k: 0.0 for k in results},
                "per_seed": {0: results}, "n_seeds": 1}
    
    seeds = sorted(combined_df["seed"].unique())
    per_seed_results = {}
    
    for seed in seeds:
        seed_df = combined_df[combined_df["seed"] == seed].copy()
        per_seed_results[seed] = full_evaluation(
            seed_df, n_trials=len(seeds), cost_bps=cost_bps)
    
    # Compute mean ± std across seeds
    all_keys = set()
    for r in per_seed_results.values():
        all_keys.update(r.keys())
    
    mean_results = {}
    std_results = {}
    for key in all_keys:
        vals = [per_seed_results[s].get(key, np.nan) for s in seeds]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if vals:
            mean_results[key] = np.mean(vals)
            std_results[key] = np.std(vals)
        else:
            mean_results[key] = np.nan
            std_results[key] = np.nan
    
    return {
        "mean": mean_results,
        "std": std_results,
        "per_seed": per_seed_results,
        "n_seeds": len(seeds),
    }


def print_multi_seed_table(multi_results: Dict,
                            model_name: str = "QUANTIS 2.0"):
    """Pretty-print multi-seed results as mean ± std."""
    mean = multi_results["mean"]
    std = multi_results["std"]
    n = multi_results["n_seeds"]
    
    print(f"\n{'='*70}")
    print(f"  {model_name} — {n}-Seed Aggregated Results (mean ± std)")
    print(f"{'='*70}")
    
    metrics_format = {
        "IC_mean": ("IC (mean)", ".4f"),
        "IC_std": ("IC (std across days)", ".4f"),
        "ICIR": ("ICIR", ".4f"),
        "RankIC_mean": ("Rank IC (mean)", ".4f"),
        "RankIC_IR": ("Rank IC IR", ".4f"),
        "Annualized_Return": ("Ann. Return", ".2%"),
        "Sharpe": ("Sharpe Ratio", ".3f"),
        "Sortino": ("Sortino Ratio", ".3f"),
        "MDD": ("Max Drawdown", ".2%"),
        "Calmar": ("Calmar Ratio", ".3f"),
        "Precision@5": ("Precision@5", ".3f"),
        "NDCG@5": ("NDCG@5", ".3f"),
        "MAPE": ("Price MAPE %", ".2f"),
        "Direction_Accuracy": ("Direction Acc", ".2%"),
        "DSR": ("Deflated SR", ".4f"),
    }
    
    for key, (label, fmt) in metrics_format.items():
        m = mean.get(key)
        s = std.get(key, 0)
        if m is not None and not np.isnan(m):
            if "%" in fmt:
                # Format percentage with ± std
                print(f"  {label:25s}: {m:{fmt}} ± {s:{fmt}}")
            else:
                print(f"  {label:25s}: {m:{fmt}} ± {s:{fmt}}")
    
    print(f"{'='*70}")
    
    # Per-seed IC breakdown
    print(f"\n  Per-Seed IC Breakdown:")
    for seed, res in multi_results["per_seed"].items():
        ic = res.get("IC_mean", np.nan)
        sharpe = res.get("Sharpe", np.nan)
        print(f"    Seed {seed}: IC={ic:.4f} | Sharpe={sharpe:.3f}")
    print()
