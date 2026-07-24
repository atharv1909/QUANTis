"""
QUANTIS 2.0 — Model Architectures
All neural network experts, gating mechanisms, and the full MoE ensemble.

Expert 1: T-KAN (Temporal Kolmogorov-Arnold Network) — GPU
Expert 2: LightGBM — CPU (separate class, non-differentiable)
Expert 3: Dynamic GNN — GPU
Gate: Hybrid (discrete HMM + continuous embedding)
Conformal Gate: Post-processing abstention mechanism
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ================================================================
# KAN Layer (Efficient B-Spline implementation)
# Avoids pykan's memory issues; works on Kaggle T4
# ================================================================
class KANLinear(nn.Module):
    """Efficient KAN linear layer using B-splines.
    
    Each edge in the network has its own learnable activation function
    (B-spline), instead of a fixed ReLU/GELU.
    
    Based on efficient-kan, reformulated to avoid 3D tensor explosion.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base linear weight (like a standard linear layer)
        self.base_weight = nn.Linear(in_features, out_features)
        
        # B-spline coefficients
        # Each (in, out) pair has grid_size + spline_order coefficients
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * 0.1
        )
        
        # Grid points for B-spline evaluation
        h = 2.0 / grid_size  # grid spacing over [-1, 1]
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)
    
    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions.
        
        Args:
            x: [batch, in_features] — values in roughly [-1, 1]
        Returns:
            bases: [batch, in_features, n_basis]
        """
        x = x.unsqueeze(-1)  # [B, in, 1]
        grid = self.grid      # [n_grid_points]
        
        # Order-0 B-splines (piecewise constant)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # [B, in, n_basis]
        
        # Recursive B-spline evaluation (Cox-de Boor)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)] + 1e-10)
            right = (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k] + 1e-10)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        
        return bases  # [B, in_features, n_basis]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_features]
        Returns:
            out: [batch, out_features]
        """
        # Base linear transformation
        base_out = self.base_weight(x)  # [B, out]
        
        # Spline transformation
        # Clamp input to grid range for numerical stability
        x_clamped = torch.clamp(x, -1.0, 1.0)
        bases = self.b_spline_basis(x_clamped)  # [B, in, n_basis]
        
        # Efficient computation: avoid explicit 3D expansion
        # spline_weight: [out, in, n_basis]
        # bases: [B, in, n_basis]
        # Result should be: [B, out]
        spline_out = torch.einsum("bin,oin->bo", bases, self.spline_weight)
        
        return base_out + spline_out


# ================================================================
# Expert 1: Temporal KAN (T-KAN)
# ================================================================
class TemporalKAN(nn.Module):
    """
    T-KAN: Temporal features → per-stock alpha prediction.
    
    Flow: [B, T=60, D=22] → KAN → GRU → Attention → KAN → [B, 1]
    Params: ~350K | VRAM: ~1 GB
    """
    
    def __init__(self, d_input: int = 22, d_hidden: int = 64,
                 seq_len: int = 60, dropout: float = 0.15,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        
        # KAN layer applied per timestep
        self.kan1 = KANLinear(d_input, d_hidden, grid_size, spline_order)
        
        # GRU for temporal aggregation (causal — no future leak)
        self.gru = nn.GRU(
            input_size=d_hidden,
            hidden_size=d_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        
        # Attention pooling over time
        self.attn_w = nn.Linear(d_hidden, 1)
        
        # Second KAN for compression
        self.kan2 = KANLinear(d_hidden, 32, grid_size=3, spline_order=3)
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(32, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]  — B stocks, T timesteps, D features
        Returns:
            pred:   [B, 1]  — per-stock prediction
            hidden: [B, 32] — hidden repr for gate
        """
        B, T, D = x.shape
        
        # Apply KAN per timestep
        x_flat = x.reshape(B * T, D)        # [B*T, D]
        h_flat = self.kan1(x_flat)           # [B*T, H]
        h = h_flat.reshape(B, T, -1)        # [B, T, H]
        
        # GRU
        gru_out, _ = self.gru(h)             # [B, T, H]
        
        # Attention pooling
        attn_scores = self.attn_w(gru_out)   # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (gru_out * attn_weights).sum(dim=1)  # [B, H]
        
        # Second KAN + head
        context = self.dropout(context)
        hidden = F.relu(self.kan2(context))  # [B, 32]
        pred = self.head(hidden)             # [B, 1]
        
        return pred, hidden


# ================================================================
# Expert 2: LightGBM (CPU-only, non-differentiable)
# ================================================================
class LightGBMExpert:
    """LightGBM expert — runs on CPU. Zero GPU memory.
    
    Predictions are fed as fixed inputs to the gate network.
    Retrained per walk-forward fold.
    """
    
    def __init__(self, params: dict = None):
        self.model = None
        self.params = params or {
            "objective": "regression",
            "metric": "mse",
            "n_estimators": 1000,
            "num_leaves": 15,
            "max_depth": 4,
            "learning_rate": 0.01,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.6,
            "bagging_freq": 5,
            "lambda_l1": 1.0,
            "lambda_l2": 5.0,
            "min_child_samples": 100,
            "verbose": -1,
            "n_jobs": -1,
        }
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """Train LightGBM on flattened tabular data."""
        import lightgbm as lgb
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [lgb.early_stopping(100), lgb.log_evaluation(200)]
        self.model = lgb.train(
            self.params, train_data,
            valid_sets=[val_data], callbacks=callbacks,
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict. Returns [N] array."""
        if self.model is None:
            return np.zeros(X.shape[0])
        return self.model.predict(X)
    
    def predict_tensor(self, X: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Predict and return as torch tensor [N, 1]."""
        preds = self.predict(X)
        return torch.tensor(preds, dtype=torch.float32,
                           device=device).unsqueeze(-1)


# ================================================================
# Expert 3: Dynamic GNN
# ================================================================
class GNNExpert(nn.Module):
    """
    Graph Attention Network over the stock graph.
    
    Uses simple GAT-like attention (no PyG dependency — works on any Kaggle).
    
    Flow: [N, D] → project → GAT×2 → [N, 1]
    Params: ~50K for 50-node graph | VRAM: < 0.5 GB
    """
    
    def __init__(self, d_input: int = 22, d_hidden: int = 64,
                 n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.head_dim = d_hidden // n_heads
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_hidden)
        
        # GAT Layer 1: Multi-head attention
        self.W_q1 = nn.Linear(d_hidden, d_hidden)
        self.W_k1 = nn.Linear(d_hidden, d_hidden)
        self.W_v1 = nn.Linear(d_hidden, d_hidden)
        self.attn_drop1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_hidden)
        
        # GAT Layer 2
        self.W_q2 = nn.Linear(d_hidden, d_hidden)
        self.W_k2 = nn.Linear(d_hidden, d_hidden)
        self.W_v2 = nn.Linear(d_hidden, d_hidden)
        self.attn_drop2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_hidden)
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_hidden, 1)
    
    def graph_attention(self, h, edge_index, W_q, W_k, W_v,
                        attn_drop, ln):
        """One layer of graph attention.
        
        Args:
            h:          [N, d_hidden]
            edge_index: [2, E] — source/target node indices
        Returns:
            h_out: [N, d_hidden]
        """
        N = h.shape[0]
        
        Q = W_q(h)  # [N, d_hidden]
        K = W_k(h)  # [N, d_hidden]
        V = W_v(h)  # [N, d_hidden]
        
        # Reshape for multi-head: [N, n_heads, head_dim]
        Q = Q.view(N, self.n_heads, self.head_dim)
        K = K.view(N, self.n_heads, self.head_dim)
        V = V.view(N, self.n_heads, self.head_dim)
        
        # Build adjacency attention using edge_index
        src, dst = edge_index[0], edge_index[1]  # [E]
        
        # Compute attention scores for edges
        # Q[dst] ⋅ K[src] / sqrt(head_dim)
        attn_scores = (Q[dst] * K[src]).sum(dim=-1) / math.sqrt(self.head_dim)
        # attn_scores: [E, n_heads]
        
        # Softmax over incoming edges per destination node
        # Use scatter operations
        attn_weights = torch.zeros(N, N, self.n_heads, device=h.device)
        attn_weights[dst, src] = attn_scores
        
        # Mask: only edges in edge_index should have non-zero attention
        mask = torch.zeros(N, N, device=h.device)
        mask[dst, src] = 1.0
        attn_weights = attn_weights.masked_fill(
            mask.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)  # [N, N, heads]
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = attn_drop(attn_weights)
        
        # Aggregate: h_out[i] = Σ_j attn[i,j] * V[j]
        # [N, N, heads, 1] * [1, N, heads, head_dim] → sum over j
        h_out = torch.einsum("ijh,jhd->ihd", attn_weights,
                             V)  # [N, heads, head_dim]
        h_out = h_out.reshape(N, self.d_hidden)  # [N, d_hidden]
        
        # Residual + LayerNorm
        h_out = ln(h + h_out)
        
        return h_out
    
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          [N, D] — per-stock features
            edge_index: [2, E] — graph edges
        Returns:
            pred:   [N, 1]
            hidden: [N, d_hidden]
        """
        h = F.relu(self.input_proj(x))  # [N, d_hidden]
        
        h = self.graph_attention(h, edge_index,
                                  self.W_q1, self.W_k1, self.W_v1,
                                  self.attn_drop1, self.ln1)
        h = F.elu(h)
        
        hidden = self.graph_attention(h, edge_index,
                                       self.W_q2, self.W_k2, self.W_v2,
                                       self.attn_drop2, self.ln2)
        hidden = F.elu(hidden)  # [N, d_hidden]
        
        pred = self.head(self.dropout(hidden))  # [N, 1]
        return pred, hidden


# ================================================================
# Market Embedding Network
# ================================================================
class MarketEmbedding(nn.Module):
    """Learned continuous embedding of market state.
    
    Input:  [d_market_input] = market-level features
    Output: [d_embed=32]
    """
    
    def __init__(self, d_input: int = 6, d_embed: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, d_embed),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)


# ================================================================
# Hybrid Gate (Flagship Novelty #1)
# ================================================================
class HybridGate(nn.Module):
    """
    Hybrid discrete + continuous gating.
    
    MASTER uses continuous-only. ReCAP uses discrete-only.
    We combine BOTH — this is our architectural novelty.
    
    Inputs: regime_onehot [3] + market_embedding [32] + expert_preds [3]
    Output: expert_weights [3] via Softmax
    """
    
    def __init__(self, n_experts: int = 3, d_regime: int = 3,
                 d_embed: int = 32):
        super().__init__()
        d_input = d_regime + d_embed + n_experts  # 3+32+3=38
        
        self.gate = nn.Sequential(
            nn.Linear(d_input, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts),
        )
    
    def forward(self, regime_onehot, market_embedding, expert_preds):
        """
        Args:
            regime_onehot:    [B, 3]
            market_embedding: [B, 32]
            expert_preds:     [B, 3]
        Returns:
            weights:    [B, 3]
            final_pred: [B, 1]
        """
        gate_input = torch.cat([regime_onehot, market_embedding,
                                expert_preds], dim=-1)
        logits = self.gate(gate_input)
        weights = torch.softmax(logits, dim=-1)
        final_pred = (weights * expert_preds).sum(dim=-1, keepdim=True)
        return weights, final_pred


# ================================================================
# HMM Regime Detector (CPU)
# ================================================================
class RegimeDetector:
    """3-state Gaussian HMM for market regime detection.
    
    States: 0=Bull, 1=Bear, 2=Sideways (assigned by return mean).
    Fitted on training data ONLY.
    """
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.hmm = None
        self.state_map = {}
    
    def fit(self, market_features: np.ndarray):
        """Fit HMM on market-level features [T, 3].
        
        Columns: [market_return, market_volatility, vix_level]
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            from hmmlearn import GaussianHMM
        
        self.hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.hmm.fit(market_features)
        
        # Map states: highest mean return → Bull(0), lowest → Bear(1)
        means = self.hmm.means_[:, 0]
        sorted_idx = np.argsort(means)[::-1]
        self.state_map = {
            sorted_idx[0]: 0,  # bull
            sorted_idx[2]: 1,  # bear
            sorted_idx[1]: 2,  # sideways
        }
    
    def predict(self, market_features: np.ndarray) -> np.ndarray:
        """Returns regime labels [T]."""
        if self.hmm is None:
            return np.full(len(market_features), 2)  # default sideways
        raw = self.hmm.predict(market_features)
        return np.array([self.state_map.get(s, 2) for s in raw])
    
    def get_onehot(self, regime: int, device: str = "cpu") -> torch.Tensor:
        """Returns one-hot [3] tensor for a single regime."""
        oh = torch.zeros(self.n_states, device=device)
        oh[regime] = 1.0
        return oh


# ================================================================
# Conformal Gate (Flagship Novelty #2) — Post-processing
# Prediction-specific confidence via residual binning
# ================================================================
class ConformalGate:
    """Conformal prediction abstention gate with prediction-specific confidence.
    
    "Don't trade when the model is too uncertain."
    Neither MASTER nor HIST has this mechanism.
    
    Key fix: confidence is computed per-prediction using (regime, |y_pred|)
    bins, NOT uniform per-regime. This makes high-confidence filtering
    actually select better predictions.
    """
    
    def __init__(self, target_coverage: float = 0.90, n_bins: int = 10):
        self.target_coverage = target_coverage
        self.n_bins = n_bins
        self.global_q = None
        self.regime_q = {}
        # Prediction-specific: (regime, magnitude_bin) → median absolute error
        self.bin_edges = None
        self.residual_lookup = {}  # (regime, bin_idx) → expected_error
        self.global_residual_lookup = {}  # bin_idx → expected_error (fallback)
    
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray,
                  regimes: np.ndarray = None):
        """Calibrate on validation set residuals with prediction-specific binning."""
        residuals = np.abs(y_true - y_pred)
        pred_magnitude = np.abs(y_pred)
        
        # Global quantile (for intervals)
        self.global_q = np.quantile(residuals, self.target_coverage)
        
        # Build magnitude bins from validation predictions
        self.bin_edges = np.quantile(pred_magnitude,
                                      np.linspace(0, 1, self.n_bins + 1))
        self.bin_edges[-1] = np.inf  # catch outliers
        bin_indices = np.digitize(pred_magnitude, self.bin_edges[1:])  # 0 to n_bins-1
        
        # Global: bin_idx → median residual
        for b in range(self.n_bins):
            mask_b = bin_indices == b
            if mask_b.sum() > 5:
                self.global_residual_lookup[b] = np.median(residuals[mask_b])
            else:
                self.global_residual_lookup[b] = np.median(residuals)
        
        # Per-regime: (regime, bin_idx) → median residual
        if regimes is not None:
            for r in range(3):
                r_mask = regimes == r
                if r_mask.sum() > 10:
                    self.regime_q[r] = np.quantile(
                        residuals[r_mask], self.target_coverage)
                else:
                    self.regime_q[r] = self.global_q
                
                for b in range(self.n_bins):
                    mask_rb = r_mask & (bin_indices == b)
                    if mask_rb.sum() > 3:
                        self.residual_lookup[(r, b)] = np.median(residuals[mask_rb])
                    else:
                        self.residual_lookup[(r, b)] = self.global_residual_lookup.get(
                            b, np.median(residuals))
    
    def apply(self, y_pred: np.ndarray, regimes: np.ndarray = None,
              threshold_mult: float = 1.0):
        """
        Returns:
            trade_mask: [N] bool — True = trade, False = abstain
            confidence: [N] float — prediction-specific confidence
            intervals:  [N, 2] — (lower, upper) prediction bounds
        """
        N = len(y_pred)
        pred_magnitude = np.abs(y_pred)
        
        # Compute per-prediction expected error
        if self.bin_edges is not None:
            bin_indices = np.digitize(pred_magnitude, self.bin_edges[1:])
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        else:
            bin_indices = np.zeros(N, dtype=int)
        
        expected_errors = np.full(N, self.global_q)
        for i in range(N):
            b = int(bin_indices[i])
            if regimes is not None:
                r = int(regimes[i]) if not np.isnan(regimes[i]) else 2
                expected_errors[i] = self.residual_lookup.get(
                    (r, b), self.global_residual_lookup.get(b, self.global_q))
            else:
                expected_errors[i] = self.global_residual_lookup.get(
                    b, self.global_q)
        
        # Prediction-specific confidence (higher = more confident)
        confidence = 1.0 / (expected_errors + 1e-10)
        
        # Interval width varies per prediction
        q_vals = expected_errors  # use expected error as interval half-width
        widths = 2 * q_vals
        threshold = threshold_mult * 2 * self.global_q
        
        trade_mask = widths <= threshold
        intervals = np.stack([y_pred - q_vals, y_pred + q_vals], axis=1)
        
        return trade_mask, confidence, intervals


# ================================================================
# Full QUANTIS MoE Ensemble
# ================================================================
class QUANTIS(nn.Module):
    """Complete QUANTIS Mixture-of-Experts ensemble.
    
    3 experts (T-KAN + LightGBM + GNN) + hybrid gate.
    LightGBM runs externally on CPU; predictions passed in.
    
    Total GPU params: ~415K | VRAM: ~2 GB on T4
    """
    
    def __init__(self, d_features: int = 22, d_hidden: int = 64,
                 seq_len: int = 60, d_market: int = 6,
                 d_embed: int = 32, n_heads: int = 4,
                 dropout: float = 0.15):
        super().__init__()
        
        self.tkan = TemporalKAN(
            d_input=d_features, d_hidden=d_hidden,
            seq_len=seq_len, dropout=dropout,
        )
        self.gnn = GNNExpert(
            d_input=d_features, d_hidden=d_hidden,
            n_heads=n_heads, dropout=dropout,
        )
        self.market_emb = MarketEmbedding(d_input=d_market, d_embed=d_embed)
        self.gate = HybridGate(n_experts=3, d_regime=3, d_embed=d_embed)
    
    def forward(self, stock_seq, stock_snap, edge_index,
                regime_onehot, market_snap, lgbm_preds):
        """
        Args:
            stock_seq:     [N, T, D]  — temporal features
            stock_snap:    [N, D]     — latest features (for GNN)
            edge_index:    [2, E]     — graph edges
            regime_onehot: [3]        — current regime
            market_snap:   [d_market] — market features
            lgbm_preds:    [N, 1]     — pre-computed LightGBM predictions
        Returns:
            final_preds:  [N, 1]
            gate_weights: [N, 3]
            expert_preds: [N, 3]
        """
        N = stock_seq.shape[0]
        device = stock_seq.device
        
        # Expert 1: T-KAN
        tkan_pred, _ = self.tkan(stock_seq)      # [N, 1]
        
        # Expert 3: GNN
        gnn_pred, _ = self.gnn(stock_snap, edge_index)  # [N, 1]
        
        # Stack expert predictions
        expert_preds = torch.cat([tkan_pred, lgbm_preds, gnn_pred],
                                 dim=-1)          # [N, 3]
        
        # Market embedding
        mkt_emb = self.market_emb(market_snap)    # [d_embed]
        mkt_emb = mkt_emb.unsqueeze(0).expand(N, -1)  # [N, d_embed]
        
        # Regime (same for all stocks on a day)
        regime_batch = regime_onehot.unsqueeze(0).expand(N, -1)  # [N, 3]
        
        # Gate
        gate_weights, final_preds = self.gate(
            regime_batch, mkt_emb, expert_preds)
        
        return final_preds, gate_weights, expert_preds
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
