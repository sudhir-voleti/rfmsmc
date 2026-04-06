# =============================================================================
# SMC_HMM_BEMMAOR.PY
# Bemmaor & Glady (2012) inspired HMM with correlated NBD-Gamma
# Hybrid: Gemini's anchoring + Grok's numerical stability
# =============================================================================

import os
# os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'
# Force Python backend for cross-platform compatibility
# C backend can cause compilation errors on some systems
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,device=cpu,mode=FAST_COMPILE'

import numpy as np
import pytensor.tensor as pt
import pytensor
from pytensor import scan

import argparse
import time
import pickle
import warnings
from pathlib import Path

import pandas as pd
import pymc as pm
import arviz as az
from sklearn.metrics import adjusted_rand_score, confusion_matrix

warnings.filterwarnings('ignore')
RANDOM_SEED = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_rfm_features(y, mask):
    """Compute RFM features."""
    N, T = y.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        last_purchase = -1
        cum_freq = 0
        cum_spend = 0.0
        for t in range(T):
            if mask[i, t]:
                if y[i, t] > 0:
                    last_purchase = t
                    cum_freq += 1
                    cum_spend += y[i, t]
                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cum_freq
                    M[i, t] = cum_spend / cum_freq if cum_freq > 0 else 0.0
                else:
                    R[i, t] = t + 1
                    F[i, t] = 0
                    M[i, t] = 0.0
    return R, F, M

## ----

def load_empirics_data_from_csv(csv_path, N=None, seed=42):
    """Load empirical data (UCI/CDNOW) - cleaned, no OOS split."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Detect world from filename
    world = "unknown"
    for w in ['uci', 'cdnow', 'UCI', 'CDNOW']:
        if w.lower() in csv_path.name.lower():
            world = w.upper()
            break

    # Map spend column
    if 'spend' in df.columns:
        df['y'] = df['spend']
    elif 'WeeklySpend' in df.columns:
        df['y'] = df['WeeklySpend']
    else:
        raise KeyError(f"No spend column found. Available: {list(df.columns)}")

    # Use sequential week index per customer
    df = df.sort_values(['customer_id', 'WeekStart'])
    df['t'] = df.groupby('customer_id').cumcount()

    # Reshape to panel
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()

    y_full = df.pivot(index='customer_id', columns='t', values='y').values
    true_states_full = np.zeros_like(y_full, dtype=int) - 1   # no true states in empirics

    # Subsample customers if requested
    if N is not None and N < N_actual:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_actual, N, replace=False)
        y_full = y_full[idx, :]
        true_states_full = true_states_full[idx, :]
        N_effective = N
    else:
        N_effective = N_actual

    # Create mask and fill NaNs with 0
    mask = ~np.isnan(y_full)
    y_full = np.where(mask, y_full, 0.0)

    # Compute RFM (or use precomputed if available)
    if all(col in df.columns for col in ['R_weeks', 'F_run', 'M_run']):
        R = df.pivot(index='customer_id', columns='t', values='R_weeks').values[:N_effective, :]
        F = df.pivot(index='customer_id', columns='t', values='F_run').values[:N_effective, :]
        M = df.pivot(index='customer_id', columns='t', values='M_run').values[:N_effective, :]
        R = np.where(np.isnan(R), 0, R)
        F = np.where(np.isnan(F), 0, F)
        M = np.where(np.isnan(M), 0, M)
    else:
        R, F, M = compute_rfm_features(y_full, mask)

    # Standardize RFM using valid observations
    R_valid = R[mask]
    F_valid = F[mask]
    M_valid = np.log1p(M[mask])

    if len(R_valid) > 0 and R_valid.std() > 0:
        R = (R - R_valid.mean()) / (R_valid.std() + 1e-6)
    if len(F_valid) > 0 and F_valid.std() > 0:
        F = (F - F_valid.mean()) / (F_valid.std() + 1e-6)
    if len(M_valid) > 0 and M_valid.std() > 0:
        M_scaled = (np.log1p(M) - M_valid.mean()) / (M_valid.std() + 1e-6)
    else:
        M_scaled = np.log1p(M)

    data = {
        'N': N_effective,
        'T': T_actual,
        'y': y_full.astype(np.float32),
        'mask': mask.astype(bool),
        'R': R.astype(np.float32),
        'F': F.astype(np.float32),
        'M': M_scaled.astype(np.float32),
        'true_states': true_states_full.astype(np.int32),
        'world': world,
        'M_raw': M.astype(np.float32),          # needed for whale metrics
        'source_file': str(csv_path.name),
        'T_total': T_actual
    }

    y_valid = data['y'][data['mask']]
    print(f" Empirics: {world}, N={N_effective}, T={T_actual}, "
          f"zeros={np.mean(y_valid==0):.1%}, mean_spend=${np.mean(y_valid[y_valid>0]):.2f}")

    return data

## ----

def load_simulation_data_from_csv(csv_path, T=104, N=None, seed=RANDOM_SEED):
    """
    Load simulation data from CSV - cleaned, no OOS split.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f" Reading simulation CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Detect world name
    world = "unknown"
    for w in ['Harbor', 'Breeze', 'Fog', 'Cliff']:
        if w.lower() in csv_path.name.lower():
            world = w
            break

    # Flexible column mapping
    col_mapping = {
        'customer_id': ['customer_id', 'cust_id', 'id', 'customer', 'cust'],
        't': ['t', 'time', 'period', 'time_period', 'week', 'Time'],
        'y': ['y', 'spend', 'purchase', 'value', 'spend_value', 'amount'],
        'true_state': ['true_state', 'state', 'true_latent', 'latent', 'truestate']
    }

    actual_cols = {}
    for std_name, variants in col_mapping.items():
        for v in variants:
            if v in df.columns:
                actual_cols[std_name] = v
                break

    if len(actual_cols) < 3:   # y is mandatory, true_state optional
        missing = set(['customer_id', 't', 'y']) - set(actual_cols.keys())
        raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}, missing: {missing}")

    df = df.rename(columns={v: k for k, v in actual_cols.items()})

    # Reshape to panel
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()

    y_full = df.pivot(index='customer_id', columns='t', values='y').values

    # true_states (optional)
    if 'true_state' in df.columns:
        true_states_full = df.pivot(index='customer_id', columns='t', values='true_state').values
    else:
        true_states_full = np.zeros_like(y_full, dtype=int) - 1

    # Pad or truncate to target T
    if T_actual < T:
        pad_width = ((0, 0), (0, T - T_actual))
        y_full = np.pad(y_full, pad_width, mode='constant', constant_values=0)
        true_states_full = np.pad(true_states_full, pad_width, mode='constant', constant_values=-1)
        T_effective = T
    elif T_actual > T:
        y_full = y_full[:, :T]
        true_states_full = true_states_full[:, :T]
        T_effective = T
    else:
        T_effective = T_actual

    # Subsample customers
    if N is not None and N < N_actual:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_actual, N, replace=False)
        y_full = y_full[idx, :]
        true_states_full = true_states_full[idx, :]
        N_effective = N
        print(f" Subsampled: N={N} (from {N_actual})")
    else:
        N_effective = N_actual

    # Create mask and fill NaNs
    mask = ~np.isnan(y_full)
    y_full = np.where(mask, y_full, 0.0)

    # Compute RFM
    R, F, M = compute_rfm_features(y_full, mask)

    # Standardize RFM
    R_valid = R[mask]
    F_valid = F[mask]
    M_valid = np.log1p(M[mask])

    if len(R_valid) > 0 and R_valid.std() > 0:
        R = (R - R_valid.mean()) / (R_valid.std() + 1e-6)
    if len(F_valid) > 0 and F_valid.std() > 0:
        F = (F - F_valid.mean()) / (F_valid.std() + 1e-6)
    if len(M_valid) > 0 and M_valid.std() > 0:
        M_scaled = (np.log1p(M) - M_valid.mean()) / (M_valid.std() + 1e-6)
    else:
        M_scaled = np.log1p(M)

    data = {
        'N': N_effective,
        'T': T_effective,
        'y': y_full.astype(np.float32),
        'mask': mask.astype(bool),
        'R': R.astype(np.float32),
        'F': F.astype(np.float32),
        'M': M_scaled.astype(np.float32),
        'true_states': true_states_full.astype(np.int32),
        'world': world,
        'M_raw': M.astype(np.float32),           # needed for whale metrics
        'source_file': str(csv_path.name),
        'T_total': T_effective
    }

    y_valid = data['y'][data['mask']]
    zero_rate = np.mean(y_valid == 0) if len(y_valid) > 0 else 0.0
    mean_spend = np.mean(y_valid[y_valid > 0]) if (y_valid > 0).any() else 0.0

    print(f" Simulation data: N={N_effective}, T={T_effective}, "
          f"zeros={zero_rate:.1%}, mean_spend=${mean_spend:.2f}")

    return data

# =============================================================================
# FORWARD ALGORITHM
# =============================================================================

def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """Batched forward algorithm."""
    N, T, K = log_emission.shape
    
    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]
    log_Z_init = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)
    log_alpha_norm_init = log_alpha_init - log_Z_init
    
    def forward_step(log_emit_t, log_alpha_prev, log_Z_prev, log_Gamma):
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)
        log_alpha_norm = log_alpha_new - log_Z_t
        return log_alpha_norm, log_Z_t
    
    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)
    
    (log_alpha_norm_seq, log_Z_seq), _ = scan(
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
    )
    
    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],
        log_alpha_norm_seq
    ], axis=0)
    
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    
    return log_marginal, log_alpha_norm_full

## ----

def compute_bemmaor_whale_metrics(data, idata, percentile_threshold=95, n_draws_use=200):
    """
    Whale detection metrics using CLV-based segmentation.
    Whale = top (100-percentile_threshold)% of customers by CLV.
    """
    try:
        N = data['N']
        y_obs = data['y']
        mask = data['mask']
        
        # Ground truth: empirical total spend per customer
        total_spend = np.sum(y_obs * mask, axis=1)
        spend_threshold = np.percentile(total_spend, percentile_threshold)
        true_whales = total_spend >= spend_threshold
        
        # Predicted whales: from model CLV
        post = idata.posterior
        
        # Compute CLV per customer using posterior means
        if 'clv_by_state' in post:
            clv_by_state = post['clv_by_state'].mean(dim=['chain', 'draw']).values
        else:
            # Compute CLV manually
            alpha_h = post['alpha_h'].mean(dim=['chain', 'draw']).values
            gamma_h = post['gamma_h'].mean(dim=['chain', 'draw']).values
            theta = post['theta'].mean(dim=['chain', 'draw']).values
            beta_m = post['beta_m'].mean(dim=['chain', 'draw']).values
            gamma_m_val = post['gamma_m'].mean(dim=['chain', 'draw']).values
            
            if 'Gamma' in post:
                K = post['Gamma'].shape[-1]
                Gamma = post['Gamma'].mean(dim=['chain', 'draw']).values
                pi0 = post['pi0'].mean(dim=['chain', 'draw']).values
                
                # Compute stationary distribution
                eigvals, eigvecs = np.linalg.eig(Gamma.T)
                stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
                stationary = stationary / stationary.sum()
                
                # CLV per state
                lam = np.exp(alpha_h + gamma_h * theta.mean(axis=0)[:, None])
                mu_spend = np.exp(beta_m + gamma_m_val * theta.mean(axis=0)[:, None])
                
                MARGIN = 0.20
                DISCOUNT_WEEKLY = 0.10 / 52
                churn = 1 - np.diag(Gamma)
                
                clv_by_state = (MARGIN * mu_spend * lam / (DISCOUNT_WEEKLY + churn + 1e-10)).mean(axis=0)
            else:
                K = 1
                lam = np.exp(alpha_h + gamma_h * theta.mean())
                mu_spend = np.exp(beta_m + gamma_m_val * theta.mean())
                clv_by_state = np.array([0.20 * mu_spend * lam / (0.10/52 + 1e-10)])
        
        # Assign customers to states (use filtered probs if available, else posterior pred)
        if 'alpha_filtered' in post:
            # Use last period state probabilities
            state_probs = post['alpha_filtered'].mean(dim=['chain', 'draw']).values[:, -1, :]
            cust_state = np.argmax(state_probs, axis=1)
            clv_per_cust = clv_by_state[cust_state]
        else:
            # Fallback: random assignment proportional to pi0
            if 'Gamma' in post:
                pi0 = post['pi0'].mean(dim=['chain', 'draw']).values
            else:
                pi0 = np.array([1.0])
            cust_state = np.random.choice(len(clv_by_state), size=N, p=pi0)
            clv_per_cust = clv_by_state[cust_state]
        
        # Predicted whales
        clv_threshold = np.percentile(clv_per_cust, percentile_threshold)
        pred_whales = clv_per_cust >= clv_threshold
        
        # Compute metrics
        tp = np.sum(true_whales & pred_whales)
        fp = np.sum(~true_whales & pred_whales)
        fn = np.sum(true_whales & ~pred_whales)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        whale_metrics = {
            'whale_precision': float(precision),
            'whale_recall': float(recall),
            'whale_f1': float(f1),
            'whale_threshold_spend': float(spend_threshold),
            'whale_threshold_clv': float(clv_threshold),
            'n_whales_true': int(np.sum(true_whales)),
            'n_whales_pred': int(np.sum(pred_whales)),
            'whale_percentile': percentile_threshold
        }
        
        print(f"  Whale Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"  Whales: True={np.sum(true_whales)}, Pred={np.sum(pred_whales)}")
        
        return whale_metrics
        
    except Exception as e:
        print(f"  Whale metrics failed: {str(e)[:60]}")
        import traceback
        traceback.print_exc()
        return {
            'whale_precision': np.nan, 'whale_recall': np.nan, 'whale_f1': np.nan,
            'whale_threshold_spend': np.nan, 'whale_threshold_clv': np.nan,
            'n_whales_true': 0, 'n_whales_pred': 0, 'whale_percentile': percentile_threshold
        }

## ----

def compute_clv_simple(idata, discount_rate=0.10, ci_levels=[2.5, 97.5]):
    """
    Simplified CLV computation with optional posterior CI.
    Sorts states by CLV magnitude and assigns labels.
    """
    post = idata.posterior
    
    # ── Point estimates (fast path) ────────────────────────────────────────
    alpha_h   = post['alpha_h'].mean(dim=['chain', 'draw']).values
    gamma_h   = post['gamma_h'].mean(dim=['chain', 'draw']).values
    theta_avg = post['theta'].mean(dim=['chain', 'draw']).values.mean(axis=0)  # avg over customers
    
    beta_m    = post['beta_m'].mean(dim=['chain', 'draw']).values
    gamma_m   = post['gamma_m'].mean(dim=['chain', 'draw']).values
    
    log_lam   = alpha_h + gamma_h * theta_avg[:, None]
    lam       = np.exp(np.clip(log_lam, -10, 10))
    
    log_mu    = beta_m + gamma_m * theta_avg[:, None]
    mu        = np.exp(np.clip(log_mu, -10, 10))
    
    E_ik      = lam * mu                                 # (N, K)
    E_k       = E_ik.mean(axis=0)                        # (K,)
    
    Gamma     = post['Gamma'].mean(dim=['chain', 'draw']).values
    delta     = 1 / (1 + discount_rate)
    I         = np.eye(len(Gamma))
    
    clv_k     = E_k @ np.linalg.inv(I - delta * Gamma)
    
    # Sort by CLV magnitude
    order     = np.argsort(clv_k)
    clv_sorted = clv_k[order]
    
    result = {
        'clv_by_state_sorted': clv_sorted,
        'state_labels': ['Dormant', 'Lukewarm', 'Whale'],  # ordered low → high
        'clv_total': clv_k.sum(),
        'discount_rate': discount_rate,
        'order_indices': order.tolist()                    # original state indices
    }
    
    # ── Optional posterior CI (if you have time / want rigor) ──────────────
    if 'draw' in post.dims and post.draw.size > 1:
        clv_draws = []
        for d in post.draw:
            for c in post.chain:
                # Extract this draw/chain
                a_h   = post['alpha_h'].sel(chain=c, draw=d).values
                g_h   = post['gamma_h'].sel(chain=c, draw=d).values
                th    = post['theta'].sel(chain=c, draw=d).values.mean(axis=0)  # avg cust
                b_m   = post['beta_m'].sel(chain=c, draw=d).values
                g_m   = post['gamma_m'].sel(chain=c, draw=d).values
                
                ll    = a_h + g_h * th[:, None]
                l     = np.exp(np.clip(ll, -10, 10))
                lm    = b_m + g_m * th[:, None]
                m     = np.exp(np.clip(lm, -10, 10))
                
                E     = (l * m).mean(axis=0)
                clv   = E @ np.linalg.inv(I - delta * Gamma)
                clv_draws.append(clv)
        
        clv_draws = np.array(clv_draws)
        ci_low    = np.percentile(clv_draws, ci_levels[0], axis=0)
        ci_high   = np.percentile(clv_draws, ci_levels[1], axis=0)
        
        result['clv_ci_low']  = ci_low[order]
        result['clv_ci_high'] = ci_high[order]
    
    return result

def compute_hmm_clv_local(idata, discount_rate=0.10):
    """Compute CLV from Bemmaor HMM posterior - bulletproof version."""
    try:
        post = idata.posterior
        
        # Check required variables exist
        required = ['alpha_h', 'gamma_h', 'theta', 'beta_m', 'gamma_m']
        missing = [v for v in required if v not in post]
        if missing:
            print(f"    Missing vars: {missing}")
            return None
        
        # Extract with error handling
        try:
            alpha_h = post['alpha_h'].mean(dim=['chain', 'draw']).values
            gamma_h = post['gamma_h'].mean(dim=['chain', 'draw']).values
            theta = post['theta'].mean(dim=['chain', 'draw']).values
            beta_m = post['beta_m'].mean(dim=['chain', 'draw']).values
            gamma_m = post['gamma_m'].mean(dim=['chain', 'draw']).values
        except Exception as e:
            print(f"    Extraction error: {e}")
            return None
        
        # Handle shapes
        if theta.ndim >= 2:
            theta_mean = theta.mean(axis=0)  # Average over customers
        else:
            theta_mean = theta
        
        # Ensure 1D arrays
        if np.isscalar(alpha_h):
            alpha_h = np.array([alpha_h])
            beta_m = np.array([beta_m])
            K = 1
        else:
            K = len(alpha_h)
        
        # Compute lambda and mu
        try:
            log_lam = alpha_h + gamma_h * theta_mean
            lam = np.exp(np.clip(log_lam, -10, 10))
            
            log_mu = beta_m + gamma_m * theta_mean
            mu = np.exp(np.clip(log_mu, -10, 10))
            
            E_k = lam * mu
        except Exception as e:
            print(f"    Computation error: {e}")
            return None
        
        # Matrix perpetuity
        try:
            if K > 1 and 'Gamma' in post:
                Gamma = post['Gamma'].mean(dim=['chain', 'draw']).values
            else:
                Gamma = np.array([[1.0]])
            
            delta = 1 / (1 + discount_rate)
            I = np.eye(K)
            inv_mat = np.linalg.inv(I - delta * Gamma)
            clv_k = E_k @ inv_mat
            
            # Sort by CLV
            order = np.argsort(clv_k)
            
            return {
                'clv_by_state': clv_k[order].tolist(),
                'clv_total': float(np.sum(clv_k)),
                'discount_rate': discount_rate
            }
            
        except Exception as e:
            print(f"    Matrix error: {e}")
            return None
            
    except Exception as e:
        print(f"    CLV outer error: {e}")
        return None


# =============================================================================
# BEMMAOR HMM MODEL
# =============================================================================

def make_bemmaor_hmm(data, K, use_gam=True, pilot=False):
    """
    Bemmaor & Glady (2012) HMM with correlated NBD-Gamma.
    use_gam: if True, use GAM splines; if False, use GLM (linear)
    """
    y = data['y']
    mask = data['mask']
    N, T = data['N'], data['T']

    if pilot:
        print(f"[PILOT] Building Bemmaor model: N={N}, T={T}, K={K}")

    with pm.Model(coords={
        "customer": np.arange(N),
        "time": np.arange(T),
        "state": np.arange(K)
    }) as model:

        # =====================================================================
        # 1. LATENT DYNAMICS
        # =====================================================================
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype=np.float32))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K) * 1.1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            log_Gamma = pt.log(Gamma)

        # =====================================================================
        # 2. SHARED LATENT FACTOR (Anchored)
        # =====================================================================
        theta = pm.Normal("theta", mu=0, sigma=1, shape=(N, 1), initval=0.0)

        # gamma_m anchored positive: theta>0 -> higher spend
        gamma_m = pm.HalfNormal("gamma_m", sigma=1.0, initval=0.5)
        # gamma_h free: theta>0 -> ? frequency (can be negative)
        gamma_h = pm.Normal("gamma_h", mu=0, sigma=1.0, initval=0.0)

        # =====================================================================
        # 3. NBD PART (Zero/Frequency, Correlated)
        # =====================================================================
        # Log-space parameterization for numerical stability
        log_r = pm.Normal("log_r", 0, 1, shape=K if K > 1 else None, initval=0.0)
        r_nb = pm.Deterministic("r_nb", pt.exp(log_r))

        # Lambda (mean frequency) parameterization
        if K == 1:
            alpha_h = pm.Normal("alpha_h", 0, 1, initval=0.0)
            log_lam = alpha_h + gamma_h * theta
        else:
            alpha_h = pm.Normal("alpha_h", 0, 1, shape=K, initval=0.0)
            log_lam = alpha_h[None, None, :] + gamma_h * theta[:, :, None]

        lam = pt.exp(pt.clip(log_lam, -10, 10))

        # NBD P(y=0) = (r/(r+lam))^r
        if K == 1:
            log_p_zero_nbd = r_nb * (pt.log(r_nb) - pt.log(r_nb + lam.squeeze()))
        else:
            r_exp = r_nb[None, None, :]
            lam_exp = lam
            log_p_zero_nbd = r_exp * (pt.log(r_exp) - pt.log(r_exp + lam_exp))

        # =====================================================================
        # 4. GAMMA PART (Spend, Correlated)
        # =====================================================================
        # Log-space shape for numerical stability
        if K == 1:
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1, initval=0.0)
            beta_m = beta_m_raw
            log_alpha_gamma = pm.Normal("log_alpha_gamma", 0, 1, initval=0.0)
        else:
            # Ordered intercepts for identifiability
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1, shape=K, initval=0.0)
            beta_m = pm.Deterministic("beta_m", pt.sort(beta_m_raw))
            log_alpha_gamma = pm.Normal("log_alpha_gamma", 0, 1, shape=K, initval=0.0)

        alpha_gamma = pm.Deterministic("alpha_gamma", pt.exp(log_alpha_gamma))

        # mu parameterized with anchored gamma_m
        if K == 1:
            log_mu = beta_m + gamma_m * theta.squeeze()
        else:
            log_mu = beta_m[None, None, :] + gamma_m * theta[:, :, None]

        mu = pt.exp(pt.clip(log_mu, -10, 10))
        beta_gamma = alpha_gamma / mu

        # =====================================================================
        # 5. EMISSION LIKELIHOOD
        # =====================================================================
        if K == 1:
            log_zero = log_p_zero_nbd

            y_clipped = pt.clip(y, 1e-10, 1e10)
            log_gamma = ((alpha_gamma - 1) * pt.log(y_clipped) -
                        beta_gamma * y +
                        alpha_gamma * pt.log(beta_gamma) -
                        pt.gammaln(alpha_gamma))

            # P(y>0) = 1 - P(y=0)
            log_pos = pt.log1p(-pt.exp(log_zero) + 1e-10) + log_gamma
            log_emission = pt.where(y == 0, log_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
            logp_cust = pt.sum(log_emission, axis=1)

        else:
            y_exp = y[..., None]
            mask_exp = mask[..., None]

            log_zero = log_p_zero_nbd

            # P(y>0) = 1 - P(y=0)
            log_p_pos = pt.log1p(-pt.exp(log_zero) + 1e-10)

            y_clipped = pt.clip(y_exp, 1e-10, 1e10)
            alpha_exp = alpha_gamma[None, None, :]
            beta_exp = beta_gamma

            log_gamma = ((alpha_exp - 1) * pt.log(y_clipped) -
                        beta_exp * y_exp +
                        alpha_exp * pt.log(beta_exp) -
                        pt.gammaln(alpha_exp))

            log_pos = log_p_pos + log_gamma
            log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)
            log_emission = pt.where(mask_exp, log_emission, 0.0)

            if pilot:
                print(f"[PILOT] Running forward algorithm...")

            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)

            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered,
                           dims=("customer", "time", "state"))

        # =====================================================================
        # 6. LIKELIHOOD
        # =====================================================================
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        pm.Potential("loglike", pt.sum(logp_cust))

        return model

# =============================================================================
# SMC RUNNER - CLEANED
# =============================================================================
def run_smc_bemmaor(data, K, draws, chains, seed, out_dir):
    """Run SMC with BEMMAOR model - cleaned version (no OOS/PPC)."""
    cores = min(chains, 4)
    t0 = time.time()

    try:
        with make_bemmaor_hmm(data, K) as model:
            print(f"\nModel: K={K}, BEMMAOR, world={data['world']}")
            print(f"SMC: draws={draws}, chains={chains}, cores={cores}")

            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                return_inferencedata=True
            )

            elapsed = (time.time() - t0) / 60

            # --- CLV Computation ---
            print(" Computing CLV...")
            clv_results = compute_hmm_clv_local(idata, discount_rate=0.10)

            if clv_results and isinstance(clv_results, dict):
                clv_by_state = clv_results.get('clv_by_state', [])
                clv_total = clv_results.get('clv_total', np.nan)
                clv_ratio = np.nan
                if len(clv_by_state) >= 2:
                    clv_arr = np.array(clv_by_state)
                    clv_ratio = float(np.max(clv_arr) / (np.min(clv_arr) + 1e-6))
                print(f" CLV by state: {clv_by_state}")
                print(f" Total CLV: {clv_total:.2f}")
                print(f" CLV ratio: {clv_ratio:.1f}x")
            else:
                print(" CLV computation failed gracefully")
                clv_results = {'clv_by_state': [], 'clv_total': np.nan}
                clv_ratio = np.nan

            # --- Log Evidence ---
            log_ev = np.nan
            try:
                lm = idata.sample_stats.log_marginal_likelihood.values
                if hasattr(lm, 'dtype') and lm.dtype == object:
                    chain_finals = []
                    for c in range(lm.shape[1] if lm.ndim > 1 else len(lm)):
                        if lm.ndim > 1:
                            chain_data = lm[-1, c] if lm.shape[0] > 0 else lm[0, c]
                        else:
                            chain_data = lm[c]
                        if isinstance(chain_data, (list, np.ndarray)):
                            valid = [float(x) for x in chain_data
                                     if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                            if valid:
                                chain_finals.append(valid[-1])
                        elif isinstance(chain_data, (int, float, np.floating)) and np.isfinite(chain_data):
                            chain_finals.append(float(chain_data))
                    log_ev = float(np.mean(chain_finals)) if chain_finals else np.nan
                else:
                    flat = np.array(lm).flatten()
                    valid = flat[np.isfinite(flat)]
                    log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
            except Exception as e:
                print(f" Warning: log-evidence extraction failed: {e}")

            print(f" log_ev={log_ev:.2f}, time={elapsed:.1f}min")

            # --- Whale Detection Metrics ---
            print(" Computing whale detection...")
            whale_metrics = compute_bemmaor_whale_metrics(data, idata, percentile_threshold=95)

            # --- Diagnostics ---
            diagnostics = {'ess_min': np.nan, 'rhat_max': np.nan}
            try:
                ess = az.ess(idata)
                rhat = az.rhat(idata)
                ess_vals = [ess[v].values for v in ess.data_vars if hasattr(ess[v].values, 'size')]
                rhat_vals = [rhat[v].values for v in rhat.data_vars if hasattr(rhat[v].values, 'size')]
                if ess_vals:
                    diagnostics['ess_min'] = float(min([v.min() for v in ess_vals if v.size > 0]))
                if rhat_vals:
                    diagnostics['rhat_max'] = float(max([v.max() for v in rhat_vals if v.size > 0]))
                print(f" ESS: min={diagnostics['ess_min']:.0f}")
                print(f" R-hat: max={diagnostics['rhat_max']:.3f}")
            except Exception as e:
                print(f" Diagnostics warning: {e}")

            # Build final result dict
            res = {
                'K': K,
                'model_type': 'BEMMAOR',
                'world': data['world'],
                'N': data['N'],
                'T': data['T'],
                'margin': 0.20,
                'discount_annual': 0.10,
                'discount_weekly': 0.10 / 52,
                'log_evidence': log_ev,
                'draws': draws,
                'chains': chains,
                'time_min': elapsed,
                'ess_min': diagnostics['ess_min'],
                'rhat_max': diagnostics['rhat_max'],
                'clv_by_state': clv_results.get('clv_by_state', []),
                'clv_total': clv_results.get('clv_total', np.nan),
                'clv_ratio': clv_ratio,
                **whale_metrics,
            }

            # Save results
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pkl_name = f"smc_K{K}_BEMMAOR_N{data['N']}_T{data['T']}_D{draws}.pkl"
            pkl_path = out_dir / pkl_name

            with open(pkl_path, 'wb') as f:
                pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)

            print(f" Saved PKL: {pkl_path}")
            return pkl_path, res, idata

    except Exception as e:
        elapsed = (time.time() - t0) / 60
        print(f" FAILED after {elapsed:.1f}min: {str(e)}")
        raise

# =============================================================================
# MAIN - CLEANED
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Bemmaor HMM: Correlated NBD-Gamma (JRSS-C Simulation)')

    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to simulation or empirical CSV')
    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Number of hidden states')
    parser.add_argument('--T', type=int, default=52,
                        help='Target time periods (for simulation data)')
    parser.add_argument('--N', type=int, default=None,
                        help='Number of customers to subsample (None = use all)')
    parser.add_argument('--draws', type=int, default=1000,
                        help='Number of SMC draws per chain')
    parser.add_argument('--chains', type=int, default=4,
                        help='Number of SMC chains')
    parser.add_argument('--out_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print("BEMMAOR HMM: Correlated NBD-Gamma (JRSS-C Phase Transition Study)")
    print("=" * 70)

    # Load data - now simplified (no train_ratio)
    if 'uci' in args.csv_path.lower() or 'cdnow' in args.csv_path.lower():
        data = load_empirics_data_from_csv(Path(args.csv_path), args.N, args.seed)
    else:
        data = load_simulation_data_from_csv(Path(args.csv_path), args.T, args.N, args.seed)

    print(f"\nConfiguration: K={args.K}, N={data['N']}, T={data['T']}, world={data['world']}")
    print("=" * 70)

    # Create output directory
    out_dir = Path(args.out_dir) / data['world'].lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run SMC
    pkl_path, res, idata = run_smc_bemmaor(
        data=data,
        K=args.K,
        draws=args.draws,
        chains=args.chains,
        seed=args.seed,
        out_dir=out_dir
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Log-Evidence: {res['log_evidence']:.2f}")
    print(f"Runtime: {res['time_min']:.1f} minutes")
    print(f"ESS min: {res['ess_min']:.0f}")
    print(f"Output: {pkl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
