#!/usr/bin/env python3
"""
smc_hmm_hurdle.py - Production HMM-Hurdle-Gamma with Batched Forward Algorithm
GitHub: https://github.com/sudhir-voleti/smcpaper/blob/main/scripts/smc_hmm_hurdle.py

Usage:
    # PILOT MODE: Quick validation (N=50, T=52, fast)
    python smc_hmm_hurdle.py --world Cliff --K 3 --N 50 --T 52 --draws 200 --pilot

    # PRODUCTION: Paper results (N=200, T=104)
    python smc_hmm_hurdle.py --world Harbor --K 2 --N 200 --T 104
    python smc_hmm_hurdle.py --world Cliff --K 3 --N 200 --T 104

    # EXTRACTION: Post-process results
    python smc_hmm_hurdle.py --extract --pkl_file results/smc_Cliff_K3_GAM_N200_T104_D1000.pkl
"""

# =============================================================================
# 0. ENVIRONMENT
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

print(f"PyTensor: floatX={pytensor.config.floatX}, device={pytensor.config.device}")

# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import time
import pickle
import warnings
from pathlib import Path
import json

import pandas as pd
import pymc as pm
import arviz as az
from patsy import dmatrix
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score, confusion_matrix

warnings.filterwarnings('ignore')
RANDOM_SEED = 42

# =============================================================================
# 2. DATA LOADING (with flexible subsetting)
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

def load_simulation_data(world: str, data_dir: Path, T: int = 104, N: int = None,
                         pilot: bool = False, seed: int = RANDOM_SEED):
    """
    Cleaned version: No train/test split, no OOS logic.
    Loads simulation CSV with true states.
    """
    if pilot:
        print(f" [PILOT MODE] Loading {world} (N={N or 'all'}, T={T})")

    # Try multiple filename patterns
    patterns = [
        data_dir / f"hmm_{world}_N1000_T52.csv",
        data_dir / f"{world}_N1000_T52.csv",
        data_dir / f"hmm_{world}_N1000_T{T}.csv",
        data_dir / f"{world.lower()}_N1000_T52.csv",
    ]

    csv_path = None
    for p in patterns:
        if isinstance(p, Path) and p.exists():
            csv_path = p
            if pilot:
                print(f" [PILOT] Found: {p.name}")
            break

    if csv_path is None:
        # Fallback: try glob
        csv_files = list(data_dir.glob(f"*{world}*.csv"))
        if csv_files:
            csv_path = csv_files[0]
            if pilot:
                print(f" [PILOT] Found via glob: {csv_path.name}")
        else:
            raise FileNotFoundError(f"No simulation data found for {world} in {data_dir}")

    df = pd.read_csv(csv_path)

    # Flexible column mapping
    col_mapping = {
        'customer_id': ['customer_id', 'cust_id', 'id', 'customer'],
        't': ['t', 'time', 'period', 'time_period', 'week'],
        'y': ['y', 'spend', 'purchase', 'value'],
        'true_state': ['true_state', 'state', 'true_latent', 'latent']
    }

    actual_cols = {}
    for std_name, variants in col_mapping.items():
        for v in variants:
            if v in df.columns:
                actual_cols[std_name] = v
                break

    if 'customer_id' not in actual_cols or 't' not in actual_cols or 'y' not in actual_cols:
        raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")

    df = df.rename(columns={v: k for k, v in actual_cols.items()})

    # Reshape to panel
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()

    if pilot:
        print(f" [PILOT] Raw data: N={N_actual}, T={T_actual}")

    y_full = df.pivot(index='customer_id', columns='t', values='y').values

    if 'true_state' in df.columns:
        true_states_full = df.pivot(index='customer_id', columns='t', values='true_state').values
    else:
        true_states_full = np.full_like(y_full, -1, dtype=int)

    # Pad or truncate
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
        if pilot:
            print(f" [PILOT] Subsampled to N={N}")
    else:
        N_effective = N_actual

    # Mask and fill NaNs
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
        'M_raw': M.astype(np.float32),
        'source_file': str(csv_path.name),
        'T_full': T_effective
    }

    y_valid = y_full[mask]
    zero_rate = np.mean(y_valid == 0) if len(y_valid) > 0 else 0.0
    mean_spend = np.mean(y_valid[y_valid > 0]) if len(y_valid[y_valid > 0]) > 0 else 0.0

    print(f" Loaded simulation: N={N_effective}, T={T_effective}, "
          f"zeros={zero_rate:.1%}, mean_spend=${mean_spend:.2f}")

    return data

# =============================================================================
# 3. GAM BASIS
# =============================================================================

def create_bspline_basis(x, df=3, degree=3):
    """Create B-spline basis matrix."""
    x = np.asarray(x, dtype=np.float32).flatten()
    n_knots = df - degree + 1

    if n_knots > 1:
        knots = np.quantile(x, np.linspace(0, 1, n_knots)[1:-1]).tolist()
    else:
        knots = []

    formula = f"bs(x, knots={list(knots)}, degree={degree}, include_intercept=False)"
    basis = dmatrix(formula, {"x": x}, return_type='matrix')
    return np.asarray(basis, dtype=np.float32)


# =============================================================================
# 4. FORWARD ALGORITHM (Grok-Corrected)
# =============================================================================

def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """
    Batched forward algorithm with proper scaling.
    """
    N, T, K = log_emission.shape
    
    # Initial step (t=0)
    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]
    log_Z_init     = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)
    log_alpha_norm_init = log_alpha_init - log_Z_init
    
    # Scan step
    def forward_step(log_emit_t, log_alpha_prev, log_Z_prev, log_Gamma):
        """
        Single forward step.
        log_emit_t: (N, K) - emissions at time t
        log_alpha_prev: (N, K) - normalized log-alpha from t-1
        log_Z_prev: (N, 1) - previous normalization constant (unused but passed by scan)
        log_Gamma: (K, K) - transition matrix
        """
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)
        
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)
        log_alpha_norm = log_alpha_new - log_Z_t
        
        return log_alpha_norm, log_Z_t
    
    # Emission sequence from t=1 onward
    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)  # (T-1, N, K)
    
    # Run scan
    (log_alpha_norm_seq, log_Z_seq), updates = scan(
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
    )
    
    # Full normalized alpha sequence (T, N, K)
    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],
        log_alpha_norm_seq
    ], axis=0)
    
    # Correct marginal likelihood: sum all normalization constants
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    
    return log_marginal, log_alpha_norm_full


## ----

def compute_hurdle_clv(idata, discount_rate=0.10, ci_levels=[2.5, 97.5]):
    """
    Compute CLV from Hurdle-Gamma HMM with posterior CIs.
    
    For Hurdle: CLV = (1-pi) * mu / (discount + churn)
    where pi = hurdle probability, mu = Gamma mean (spend)
    """
    try:
        post = idata.posterior
        
        # Check required variables
        required = ['beta0', 'alpha_gamma']
        missing = [v for v in required if v not in post]
        if missing:
            print(f"    Missing vars for CLV: {missing}")
            return None
        
        # Get posterior means
        beta0 = post['beta0'].mean(dim=['chain', 'draw']).values
        alpha_gamma = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
        
        # Handle K=1 vs K>1
        if np.isscalar(beta0):
            K = 1
            beta0 = np.array([beta0])
            alpha_gamma = np.array([alpha_gamma])
        else:
            K = len(beta0)
        
        # Compute mu (Gamma mean = exp(beta0) for intercept-only, or average over covariates)
        # For simplicity, use exp(beta0) as baseline mean
        mu = np.exp(beta0)


        # Get pi (hurdle probability) - compute from alpha0_h if not stored
        if 'pi' in post:
            pi_vals = post['pi'].mean(dim=['chain', 'draw']).values
            if pi_vals.ndim > 2:
                pi = pi_vals.mean(axis=(0, 1))
            else:
                pi = pi_vals.mean()
        else:
            # Compute from alpha0_h (approximate, assumes mean covariate effects = 0)
            alpha0_h_mean = post['alpha0_h'].mean(dim=['chain', 'draw']).values
            if np.isscalar(alpha0_h_mean):
                pi = 1.0 / (1.0 + np.exp(-float(alpha0_h_mean)))
                pi = np.array([pi])
            else:
                pi = 1.0 / (1.0 + np.exp(-alpha0_h_mean))
        
        # Ensure pi matches K states
        if len(pi) != K:
            pi = np.array([pi.mean()] * K)

        
        # Expected spend per period = (1-pi) * mu
        expected_spend = (1 - pi) * mu
        
        # Churn rate from Gamma diagonal
        if K > 1 and 'Gamma' in post:
            Gamma = post['Gamma'].mean(dim=['chain', 'draw']).values
            churn_rate = 1.0 - np.diag(Gamma)
        else:
            churn_rate = np.zeros(K)
        
        # CLV formula
        delta = discount_rate
        clv_k = expected_spend / (delta + churn_rate + 1e-10)
        
        # Floor at $1
        clv_k = np.maximum(clv_k, 1.0)
        
        # Sort by CLV
        order = np.argsort(clv_k)
        clv_sorted = clv_k[order]
        
        result = {
            'clv_by_state_sorted': clv_sorted,
            'state_labels': ['Dormant', 'Regular', 'Whale'] if K >= 3 else ['Low', 'High'],
            'clv_total': float(np.sum(clv_k)),
            'clv_ratio': float(np.max(clv_k) / (np.min(clv_k) + 1e-6)),
            'discount_rate': discount_rate,
            'order_indices': order.tolist(),
            'pi_by_state': pi[order].tolist(),
            'mu_by_state': mu[order].tolist()
        }
        
        # Posterior CI via sampling
        if 'draw' in post.dims and post.draw.size > 1:
            n_chains, n_draws = post.sizes['chain'], post.sizes['draw']
            clv_draws = []
            
            for c in range(n_chains):
                for d in range(n_draws):
                    # Extract draw
                    beta0_d = post['beta0'].isel(chain=c, draw=d).values
                    if np.isscalar(beta0_d):
                        beta0_d = np.array([beta0_d])
                    
                    alpha_d = post['alpha_gamma'].isel(chain=c, draw=d).values
                    if np.isscalar(alpha_d):
                        alpha_d = np.array([alpha_d])
                    
                    mu_d = np.exp(beta0_d)
    

                    # Pi for this draw
                    if 'pi' in post:
                        pi_d = post['pi'].isel(chain=c, draw=d).values
                        if pi_d.ndim > 2:
                            pi_d = pi_d.mean(axis=(0, 1))
                    else:
                        alpha0_h_d = post['alpha0_h'].isel(chain=c, draw=d).values
                        if np.isscalar(alpha0_h_d):
                            pi_d = 1.0 / (1.0 + np.exp(-float(alpha0_h_d)))
                            pi_d = np.array([pi_d])
                        else:
                            pi_d = 1.0 / (1.0 + np.exp(-alpha0_h_d))

                    # Ensure pi_d is 1D array
                    if np.isscalar(pi_d):
                        pi_d = np.array([pi_d])
                    elif pi_d.ndim > 1:
                        pi_d = pi_d.flatten()
                    
                    if len(pi_d) != K:
                        pi_d = np.array([pi_d.mean()] * K)
                    
                    exp_spend_d = (1 - pi_d) * mu_d
                    
                    # Gamma and churn
                    if K > 1 and 'Gamma' in post:
                        Gamma_d = post['Gamma'].isel(chain=c, draw=d).values
                        churn_d = 1.0 - np.diag(Gamma_d)
                    else:
                        churn_d = np.zeros(K)
                    
                    clv_d = exp_spend_d / (delta + churn_d + 1e-10)
                    clv_d = np.maximum(clv_d, 1.0)
                    clv_draws.append(clv_d)
            
            clv_draws = np.array(clv_draws)
            ci_low = np.percentile(clv_draws, ci_levels[0], axis=0)
            ci_high = np.percentile(clv_draws, ci_levels[1], axis=0)
            
            result['clv_ci_low'] = ci_low[order]
            result['clv_ci_high'] = ci_high[order]
        
        return result
        
    except Exception as e:
        print(f"    CLV computation error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# 5. HMM MODEL
# =============================================================================


## ----

def compute_hurdle_whale_metrics(data, idata, use_gam, gam_df, percentile_threshold=95, n_draws_use=200):
    """
    Whale detection metrics for Hurdle-Gamma model using CLV-based segmentation.
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
        
        # Get posterior
        post = idata.posterior
        n_chains, n_draws_total = post.sizes['chain'], post.sizes['draw']
        K = post['Gamma'].shape[-1] if 'Gamma' in post else 1
        
        # Precompute GAM bases if needed
        if use_gam:
            R_flat = data['R'].flatten()
            F_flat = data['F'].flatten()
            M_flat = data['M'].flatten()
            basis_R = create_bspline_basis(R_flat, df=gam_df).reshape(N, data['T'], -1)
            basis_F = create_bspline_basis(F_flat, df=gam_df).reshape(N, data['T'], -1)
            basis_M = create_bspline_basis(M_flat, df=gam_df).reshape(N, data['T'], -1)
        
        # Compute CLV per customer using posterior means
        clv_per_cust = np.zeros(N)
        
        # Get posterior mean parameters
        if K > 1:
            Gamma = post['Gamma'].mean(dim=['chain', 'draw']).values
            pi0 = post['pi0'].mean(dim=['chain', 'draw']).values
            
            # Compute stationary distribution
            eigvals, eigvecs = np.linalg.eig(Gamma.T)
            stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
            stationary = stationary / stationary.sum()
            
            if use_gam:
                alpha0_h = post['alpha0_h'].mean(dim=['chain', 'draw']).values
                w_R_h = post['w_R_h'].mean(dim=['chain', 'draw']).values
                w_F_h = post['w_F_h'].mean(dim=['chain', 'draw']).values
                w_M_h = post['w_M_h'].mean(dim=['chain', 'draw']).values
                beta0 = post['beta0'].mean(dim=['chain', 'draw']).values
                w_R = post['w_R'].mean(dim=['chain', 'draw']).values
                w_F = post['w_F'].mean(dim=['chain', 'draw']).values
                w_M = post['w_M'].mean(dim=['chain', 'draw']).values
                alpha_gamma = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
            else:
                alpha0_h = post['alpha0_h'].mean(dim=['chain', 'draw']).values
                alphaR_h = post['alphaR_h'].mean(dim=['chain', 'draw']).values
                alphaF_h = post['alphaF_h'].mean(dim=['chain', 'draw']).values
                alphaM_h = post['alphaM_h'].mean(dim=['chain', 'draw']).values
                beta0 = post['beta0'].mean(dim=['chain', 'draw']).values
                betaR = post['betaR'].mean(dim=['chain', 'draw']).values
                betaF = post['betaF'].mean(dim=['chain', 'draw']).values
                betaM = post['betaM'].mean(dim=['chain', 'draw']).values
                alpha_gamma = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
            
            # CLV per state
            MARGIN = 0.20
            DISCOUNT_WEEKLY = 0.10 / 52
            churn = 1 - np.diag(Gamma)
            
            # Average over time and customers for state-specific params
            clv_by_state = np.zeros(K)
            for k in range(K):
                # Expected pi (purchase prob) and mu (spend) for state k
                if use_gam:
                    # Average over all time periods and customers
                    pi_vals = []
                    mu_vals = []
                    for i in range(min(100, N)):  # Sample 100 customers
                        for t in range(data['T']):
                            if not mask[i, t]:
                                continue
                            eff_R_h = np.dot(basis_R[i, t], w_R_h[k])
                            eff_F_h = np.dot(basis_F[i, t], w_F_h[k])
                            eff_M_h = np.dot(basis_M[i, t], w_M_h[k])
                            logit_pi = alpha0_h[k] + eff_R_h + eff_F_h + eff_M_h
                            pi = 1 / (1 + np.exp(-np.clip(logit_pi, -10, 10)))
                            pi_vals.append(pi)
                            
                            eff_R = np.dot(basis_R[i, t], w_R[k])
                            eff_F = np.dot(basis_F[i, t], w_F[k])
                            eff_M = np.dot(basis_M[i, t], w_M[k])
                            log_mu = beta0[k] + eff_R + eff_F + eff_M
                            mu = np.exp(np.clip(log_mu, -10, 10))
                            mu_vals.append(mu)
                    
                    avg_pi = np.mean(pi_vals) if pi_vals else 0.5
                    avg_mu = np.mean(mu_vals) if mu_vals else 1.0
                else:
                    # Use mean covariate values
                    R_mean = data['R'][mask].mean()
                    F_mean = data['F'][mask].mean()
                    M_mean = data['M'][mask].mean()
                    
                    logit_pi = alpha0_h[k] + alphaR_h[k] * R_mean + alphaF_h[k] * F_mean + alphaM_h[k] * M_mean
                    avg_pi = 1 / (1 + np.exp(-np.clip(logit_pi, -10, 10)))
                    
                    log_mu = beta0[k] + betaR[k] * R_mean + betaF[k] * F_mean + betaM[k] * M_mean
                    avg_mu = np.exp(np.clip(log_mu, -10, 10))
                
                # CLV formula for hurdle: (pi * mu) / (discount + churn)
                expected_spend = avg_pi * avg_mu
                clv_by_state[k] = (MARGIN * expected_spend) / (DISCOUNT_WEEKLY + churn[k] + 1e-10)
            
            # Assign customers to most likely state
            if 'alpha_filtered' in post:
                state_probs = post['alpha_filtered'].mean(dim=['chain', 'draw']).values[:, -1, :]
                cust_state = np.argmax(state_probs, axis=1)
            else:
                cust_state = np.random.choice(K, size=N, p=stationary)
            
            clv_per_cust = clv_by_state[cust_state]
            
        else:
            # K=1: Single CLV for all
            if use_gam:
                alpha0_h = post['alpha0_h'].mean(dim=['chain', 'draw']).values
                w_R_h = post['w_R_h'].mean(dim=['chain', 'draw']).values
                w_F_h = post['w_F_h'].mean(dim=['chain', 'draw']).values
                w_M_h = post['w_M_h'].mean(dim=['chain', 'draw']).values
                beta0 = post['beta0'].mean(dim=['chain', 'draw']).values
                w_R = post['w_R'].mean(dim=['chain', 'draw']).values
                w_F = post['w_F'].mean(dim=['chain', 'draw']).values
                w_M = post['w_M'].mean(dim=['chain', 'draw']).values
                alpha_gamma = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
            else:
                alpha0_h = post['alpha0_h'].mean(dim=['chain', 'draw']).values
                alphaR_h = post['alphaR_h'].mean(dim=['chain', 'draw']).values
                alphaF_h = post['alphaF_h'].mean(dim=['chain', 'draw']).values
                alphaM_h = post['alphaM_h'].mean(dim=['chain', 'draw']).values
                beta0 = post['beta0'].mean(dim=['chain', 'draw']).values
                betaR = post['betaR'].mean(dim=['chain', 'draw']).values
                betaF = post['betaF'].mean(dim=['chain', 'draw']).values
                betaM = post['betaM'].mean(dim=['chain', 'draw']).values
                alpha_gamma = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
            
            # Compute average CLV
            MARGIN = 0.20
            DISCOUNT_WEEKLY = 0.10 / 52
            
            if use_gam:
                pi_vals = []
                mu_vals = []
                for i in range(min(100, N)):
                    for t in range(data['T']):
                        if not mask[i, t]:
                            continue
                        eff_R_h = np.dot(basis_R[i, t], w_R_h)
                        eff_F_h = np.dot(basis_F[i, t], w_F_h)
                        eff_M_h = np.dot(basis_M[i, t], w_M_h)
                        logit_pi = alpha0_h + eff_R_h + eff_F_h + eff_M_h
                        pi = 1 / (1 + np.exp(-np.clip(logit_pi, -10, 10)))
                        pi_vals.append(pi)
                        
                        eff_R = np.dot(basis_R[i, t], w_R)
                        eff_F = np.dot(basis_F[i, t], w_F)
                        eff_M = np.dot(basis_M[i, t], w_M)
                        log_mu = beta0 + eff_R + eff_F + eff_M
                        mu = np.exp(np.clip(log_mu, -10, 10))
                        mu_vals.append(mu)
                
                avg_pi = np.mean(pi_vals) if pi_vals else 0.5
                avg_mu = np.mean(mu_vals) if mu_vals else 1.0
            else:
                R_mean = data['R'][mask].mean()
                F_mean = data['F'][mask].mean()
                M_mean = data['M'][mask].mean()
                
                logit_pi = alpha0_h + alphaR_h * R_mean + alphaF_h * F_mean + alphaM_h * M_mean
                avg_pi = 1 / (1 + np.exp(-np.clip(logit_pi, -10, 10)))
                
                log_mu = beta0 + betaR * R_mean + betaF * F_mean + betaM * M_mean
                avg_mu = np.exp(np.clip(log_mu, -10, 10))
            
            clv_single = (MARGIN * avg_pi * avg_mu) / DISCOUNT_WEEKLY
            clv_per_cust = np.full(N, clv_single)
        
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

def make_hurdle_hmm(data, K, use_gam=True, gam_df=3, pilot=False):
    """HMM-Hurdle-Gamma with batched forward algorithm."""
    y = data['y']
    R, F, M = data['R'], data['F'], data['M']
    mask = data['mask']
    N, T = data['N'], data['T']

    if pilot:
        print(f"  [PILOT] Building model: N={N}, T={T}, K={K}")

    # GAM basis
    if use_gam:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()

        basis_R = create_bspline_basis(R_flat, df=gam_df).reshape(N, T, -1)
        basis_F = create_bspline_basis(F_flat, df=gam_df).reshape(N, T, -1)
        basis_M = create_bspline_basis(M_flat, df=gam_df).reshape(N, T, -1)

        n_basis_R = basis_R.shape[2]
        n_basis_F = basis_F.shape[2]
        n_basis_M = basis_M.shape[2]
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={
        "customer": np.arange(N),
        "time": np.arange(T),
        "state": np.arange(K)
    }) as model:

        # Latent dynamics
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype=np.float32))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.eye(K)*1.1 + 1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            log_Gamma = pt.log(Gamma)

        # Hurdle parameters
        if use_gam:
            alpha0_h = pm.Normal("alpha0_h", 0, 1, shape=K if K > 1 else None)
            w_R_h = pm.Normal("w_R_h", 0, 1, shape=(K, n_basis_R) if K > 1 else n_basis_R)
            w_F_h = pm.Normal("w_F_h", 0, 1, shape=(K, n_basis_F) if K > 1 else n_basis_F)
            w_M_h = pm.Normal("w_M_h", 0, 1, shape=(K, n_basis_M) if K > 1 else n_basis_M)

            if K == 1:
                logit_pi = (alpha0_h + 
                           pt.tensordot(basis_R, w_R_h, axes=([2], [0])) +
                           pt.tensordot(basis_F, w_F_h, axes=([2], [0])) +
                           pt.tensordot(basis_M, w_M_h, axes=([2], [0])))
            else:
                logit_pi = (alpha0_h[None, None, :] +
                           pt.tensordot(basis_R, w_R_h, axes=([2], [1])) +
                           pt.tensordot(basis_F, w_F_h, axes=([2], [1])) +
                           pt.tensordot(basis_M, w_M_h, axes=([2], [1])))
        else:
            alpha0_h = pm.Normal("alpha0_h", 0, 1, shape=K if K > 1 else None)
            alphaR_h = pm.Normal("alphaR_h", 0, 1, shape=K if K > 1 else None)
            alphaF_h = pm.Normal("alphaF_h", 0, 1, shape=K if K > 1 else None)
            alphaM_h = pm.Normal("alphaM_h", 0, 1, shape=K if K > 1 else None)

            if K == 1:
                logit_pi = alpha0_h + alphaR_h * R + alphaF_h * F + alphaM_h * M
            else:
                logit_pi = (alpha0_h + 
                           alphaR_h * R[..., None] + 
                           alphaF_h * F[..., None] + 
                           alphaM_h * M[..., None])

        pi = pt.clip(pt.sigmoid(logit_pi), 1e-6, 1 - 1e-6)

        # Gamma parameters (with alpha > 0.1 constraint)
        if K == 1:
            beta0 = pm.Normal("beta0", 0, 1)
            alpha_raw = pm.Beta("alpha_raw", alpha=2, beta=2)
            alpha_gamma = pm.Deterministic("alpha_gamma", 0.1 + alpha_raw * 2.0)
        else:
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0 = pm.Deterministic("beta0", pt.sort(beta0_raw))
            alpha_raw = pm.Beta("alpha_raw", alpha=2, beta=2, shape=K)
            alpha_gamma = pm.Deterministic("alpha_gamma", 0.1 + alpha_raw * 2.0)

        # Mean spend
        if use_gam:
            w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R) if K > 1 else n_basis_R)
            w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F) if K > 1 else n_basis_F)
            w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M) if K > 1 else n_basis_M)

            if K == 1:
                log_mu = (beta0 + 
                         pt.tensordot(basis_R, w_R, axes=([2], [0])) +
                         pt.tensordot(basis_F, w_F, axes=([2], [0])) +
                         pt.tensordot(basis_M, w_M, axes=([2], [0])))
            else:
                log_mu = (beta0[None, None, :] +
                         pt.tensordot(basis_R, w_R, axes=([2], [1])) +
                         pt.tensordot(basis_F, w_F, axes=([2], [1])) +
                         pt.tensordot(basis_M, w_M, axes=([2], [1])))
        else:
            betaR = pm.Normal("betaR", 0, 1, shape=K if K > 1 else None)
            betaF = pm.Normal("betaF", 0, 1, shape=K if K > 1 else None)
            betaM = pm.Normal("betaM", 0, 1, shape=K if K > 1 else None)

            if K == 1:
                log_mu = beta0 + betaR * R + betaF * F + betaM * M
            else:
                log_mu = (beta0 + 
                         betaR * R[..., None] + 
                         betaF * F[..., None] + 
                         betaM * M[..., None])

        mu = pt.exp(pt.clip(log_mu, -10, 10))

        if K == 1:
            beta_gamma = pm.Deterministic("beta_gamma", alpha_gamma / mu)
        else:
            beta_gamma = pm.Deterministic("beta_gamma", alpha_gamma[None, None, :] / mu)

        # Emissions
        if K == 1:
            log_zero = pt.log(1 - pi)
            y_clipped = pt.clip(y, 1e-10, 1e10)
            log_gamma = ((alpha_gamma - 1) * pt.log(y_clipped) - 
                        beta_gamma * y + 
                        alpha_gamma * pt.log(beta_gamma) - 
                        pt.gammaln(alpha_gamma))
            log_pos = pt.log(pi) + log_gamma
            log_emission = pt.where(y == 0, log_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
            logp_cust = pt.sum(log_emission, axis=1)

        else:
            pi_exp = pi[..., None] if pi.ndim == 2 else pi
            alpha_exp = alpha_gamma[None, None, :] if alpha_gamma.ndim == 1 else alpha_gamma
            beta_exp = beta_gamma

            y_exp = y[..., None]
            mask_exp = mask[..., None]

            log_zero = pt.log(1 - pi_exp)
            y_clipped = pt.clip(y_exp, 1e-10, 1e10)
            log_gamma = ((alpha_exp - 1) * pt.log(y_clipped) - 
                        beta_exp * y_exp + 
                        alpha_exp * pt.log(beta_exp) - 
                        pt.gammaln(alpha_exp))
            log_pos = pt.log(pi_exp) + log_gamma

            log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)
            log_emission = pt.where(mask_exp, log_emission, 0.0)

            if pilot:
                print(f"  [PILOT] Running batched forward algorithm...")

            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)

            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered,
                           dims=("customer", "time", "state"))

        # Add log_likelihood deterministic for WAIC/LOO computation (both K=1 and K>1)
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        
        # Viterbi MAP states from filtered probabilities (K>1 only)
        if K > 1:
            viterbi_map = pt.argmax(alpha_filtered, axis=2)
            pm.Deterministic("viterbi", viterbi_map, dims=("customer", "time"))
        
        pm.Potential("loglike", pt.sum(logp_cust))

    return model

## ----

def compute_hurdle_clv(idata, discount_rate=0.10, ci_levels=[2.5, 97.5]):
    """
    Compute CLV from Hurdle-Gamma HMM with posterior CIs.
    
    For Hurdle: CLV = (1-pi) * mu / (discount + churn)
    where pi = hurdle probability, mu = Gamma mean (spend)
    """
    try:
        post = idata.posterior
        
        # Check required variables
        required = ['beta0', 'alpha_gamma']
        missing = [v for v in required if v not in post]
        if missing:
            print(f"    Missing vars for CLV: {missing}")
            return None
        
        # Get posterior means
        beta0 = post['beta0'].mean(dim=['chain', 'draw']).values
        alpha_gamma = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
        
        # Handle K=1 vs K>1
        if np.isscalar(beta0):
            K = 1
            beta0 = np.array([beta0])
            alpha_gamma = np.array([alpha_gamma])
        else:
            K = len(beta0)
        
        # Compute mu (Gamma mean = exp(beta0) for intercept-only, or average over covariates)
        # For simplicity, use exp(beta0) as baseline mean
        mu = np.exp(beta0)


        # Get pi (hurdle probability) - compute from alpha0_h if not stored
        if 'pi' in post:
            pi_vals = post['pi'].mean(dim=['chain', 'draw']).values
            if pi_vals.ndim > 2:
                pi = pi_vals.mean(axis=(0, 1))
            else:
                pi = pi_vals.mean()
        else:
            # Compute from alpha0_h (approximate, assumes mean covariate effects = 0)
            alpha0_h_mean = post['alpha0_h'].mean(dim=['chain', 'draw']).values
            if np.isscalar(alpha0_h_mean):
                pi = 1.0 / (1.0 + np.exp(-float(alpha0_h_mean)))
                pi = np.array([pi])
            else:
                pi = 1.0 / (1.0 + np.exp(-alpha0_h_mean))
        
        # Ensure pi matches K states
        if len(pi) != K:
            pi = np.array([pi.mean()] * K)

        
        # Expected spend per period = (1-pi) * mu
        expected_spend = (1 - pi) * mu
        
        # Churn rate from Gamma diagonal
        if K > 1 and 'Gamma' in post:
            Gamma = post['Gamma'].mean(dim=['chain', 'draw']).values
            churn_rate = 1.0 - np.diag(Gamma)
        else:
            churn_rate = np.zeros(K)
        
        # CLV formula
        delta = discount_rate
        clv_k = expected_spend / (delta + churn_rate + 1e-10)
        
        # Floor at $1
        clv_k = np.maximum(clv_k, 1.0)
        
        # Sort by CLV
        order = np.argsort(clv_k)
        clv_sorted = clv_k[order]
        
        result = {
            'clv_by_state_sorted': clv_sorted,
            'state_labels': ['Dormant', 'Regular', 'Whale'] if K >= 3 else ['Low', 'High'],
            'clv_total': float(np.sum(clv_k)),
            'clv_ratio': float(np.max(clv_k) / (np.min(clv_k) + 1e-6)),
            'discount_rate': discount_rate,
            'order_indices': order.tolist(),
            'pi_by_state': pi[order].tolist(),
            'mu_by_state': mu[order].tolist()
        }
        
        # Posterior CI via sampling
        if 'draw' in post.dims and post.draw.size > 1:
            n_chains, n_draws = post.sizes['chain'], post.sizes['draw']
            clv_draws = []
            
            for c in range(n_chains):
                for d in range(n_draws):
                    # Extract draw
                    beta0_d = post['beta0'].isel(chain=c, draw=d).values
                    if np.isscalar(beta0_d):
                        beta0_d = np.array([beta0_d])
                    
                    alpha_d = post['alpha_gamma'].isel(chain=c, draw=d).values
                    if np.isscalar(alpha_d):
                        alpha_d = np.array([alpha_d])
                    
                    mu_d = np.exp(beta0_d)
    

                    # Pi for this draw
                    if 'pi' in post:
                        pi_d = post['pi'].isel(chain=c, draw=d).values
                        if pi_d.ndim > 2:
                            pi_d = pi_d.mean(axis=(0, 1))
                    else:
                        alpha0_h_d = post['alpha0_h'].isel(chain=c, draw=d).values
                        if np.isscalar(alpha0_h_d):
                            pi_d = 1.0 / (1.0 + np.exp(-float(alpha0_h_d)))
                            pi_d = np.array([pi_d])
                        else:
                            pi_d = 1.0 / (1.0 + np.exp(-alpha0_h_d))

                    if pi_d.ndim > 2:
                        pi_d = pi_d.mean(axis=(0, 1))
                    if np.isscalar(pi_d):
                        pi_d = np.array([pi_d])
                    
                    if len(pi_d) != K:
                        pi_d = np.array([pi_d.mean()] * K)
                    
                    exp_spend_d = (1 - pi_d) * mu_d
                    
                    # Gamma and churn
                    if K > 1 and 'Gamma' in post:
                        Gamma_d = post['Gamma'].isel(chain=c, draw=d).values
                        churn_d = 1.0 - np.diag(Gamma_d)
                    else:
                        churn_d = np.zeros(K)
                    
                    clv_d = exp_spend_d / (delta + churn_d + 1e-10)
                    clv_d = np.maximum(clv_d, 1.0)
                    clv_draws.append(clv_d)
            
            clv_draws = np.array(clv_draws)
            ci_low = np.percentile(clv_draws, ci_levels[0], axis=0)
            ci_high = np.percentile(clv_draws, ci_levels[1], axis=0)
            
            result['clv_ci_low'] = ci_low[order]
            result['clv_ci_high'] = ci_high[order]
        
        return result
        
    except Exception as e:
        print(f"    CLV computation error: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# 6. SMC RUNNER
# =============================================================================

def run_smc_hurdle(data, K, draws, chains, seed, out_dir, use_gam=True, gam_df=3):
    """Cleaned SMC runner for Hurdle model - no OOS, no PPC."""
    cores = min(chains, 4)
    t0 = time.time()

    metrics = {
        'log_evidence': np.nan,
        'ess_min': np.nan,
        'rhat_max': np.nan,
        'clv_by_state': [],
        'clv_total': np.nan,
        'clv_ratio': np.nan,
        'elapsed_min': 0.0,
        'error': None
    }

    idata = None

    try:
        with make_hurdle_hmm(data, K, use_gam=use_gam, gam_df=gam_df) as model:
            glm_gam = "GAM" if use_gam else "GLM"
            print(f"\nModel: K={K}, HURDLE-{glm_gam}, world={data.get('world', 'unknown')}")
            print(f"SMC: draws={draws}, chains={chains}, cores={cores}")

            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                return_inferencedata=True
            )

            elapsed = (time.time() - t0) / 60
            metrics['elapsed_min'] = elapsed

            # Log Evidence
            try:
                lm = idata.sample_stats.log_marginal_likelihood.values
                valid = []
                if lm.dtype == object:
                    for c in range(lm.shape[1] if lm.ndim > 1 else len(lm)):
                        chain_data = lm[-1, c] if lm.ndim > 1 else lm[c]
                        if isinstance(chain_data, (list, np.ndarray)):
                            valid.extend([float(x) for x in chain_data if np.isfinite(float(x))])
                        elif np.isfinite(float(chain_data)):
                            valid.append(float(chain_data))
                else:
                    flat = np.array(lm).flatten()
                    valid = flat[np.isfinite(flat)].tolist()
                metrics['log_evidence'] = float(np.mean(valid)) if valid else np.nan
                print(f" Log-evidence: {metrics['log_evidence']:.2f}")
            except Exception as e:
                print(f" Log-evidence extraction failed: {e}")

            # CLV
            print(" Computing CLV...")
            try:
                clv_results = compute_hurdle_clv(idata, discount_rate=0.10)
                if clv_results and isinstance(clv_results, dict):
                    clv_key = next((k for k in ['clv_by_state', 'clv_by_state_sorted'] if k in clv_results), None)
                    if clv_key:
                        clv_arr = np.atleast_1d(clv_results[clv_key])
                        metrics['clv_by_state'] = [float(x) for x in clv_arr if np.isfinite(float(x))]
                    metrics['clv_total'] = float(clv_results.get('clv_total', np.nan))
                    metrics['clv_ratio'] = float(clv_results.get('clv_ratio', np.nan))
                    print(f" CLV total: {metrics['clv_total']:.2f}, ratio: {metrics['clv_ratio']:.1f}x")
            except Exception as e:
                print(f" CLV failed: {e}")

            # ESS Diagnostics
            try:
                ess = az.ess(idata)
                rhat = az.rhat(idata)
                metrics['ess_min'] = float(ess.to_array().min())
                metrics['rhat_max'] = float(rhat.to_array().max())
                print(f" ESS min: {metrics['ess_min']:.0f} | R-hat max: {metrics['rhat_max']:.3f}")
            except Exception as e:
                print(f" Diagnostics warning: {e}")

            # Whale metrics (kept as bonus)
            print(" Computing whale detection...")
            try:
                whale_metrics = compute_hurdle_whale_metrics(data, idata, use_gam=use_gam, gam_df=gam_df)
                metrics.update(whale_metrics)
            except Exception as e:
                print(f" Whale metrics failed: {e}")
                metrics.update({
                    'whale_precision': np.nan, 'whale_recall': np.nan, 'whale_f1': np.nan,
                    'whale_threshold_spend': np.nan, 'whale_threshold_clv': np.nan,
                    'n_whales_true': 0, 'n_whales_pred': 0
                })

    except Exception as e:
        elapsed = (time.time() - t0) / 60
        metrics['error'] = str(e)
        print(f" CRASH after {elapsed:.1f}min: {str(e)[:100]}...")
        import traceback
        traceback.print_exc()

    # Build result dict
    res = {
        'K': K,
        'model_type': 'HURDLE',
        'glm_gam': 'GAM' if use_gam else 'GLM',
        'world': data.get('world', 'unknown'),
        'N': data['N'],
        'T': data['T'],
        'elapsed_min': metrics['elapsed_min'],
        'log_evidence': metrics['log_evidence'],
        'ess_min': metrics['ess_min'],
        'rhat_max': metrics.get('rhat_max', np.nan),
        'clv_by_state': metrics['clv_by_state'],
        'clv_total': metrics['clv_total'],
        'clv_ratio': metrics['clv_ratio'],
        **{k: v for k, v in metrics.items() if k.startswith('whale_')},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_name = f"smc_K{K}_{'GAM' if use_gam else 'GLM'}_N{data['N']}_T{data['T']}_D{draws}.pkl"
    pkl_path = out_dir / pkl_name

    with open(pkl_path, 'wb') as f:
        pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)

    print(f"\nSAVED: {pkl_path}")
    return pkl_path, res, idata

# =============================================================================
# 7. POST-PROCESSING & EXTRACTION
# =============================================================================

def extract_viterbi_states(idata, data):
    """
    Viterbi decoding from posterior means.

    Uses MAP estimate of parameters to find most likely state sequence.
    """
    if 'alpha_filtered' not in idata.posterior:
        print("WARNING: alpha_filtered not found in idata")
        return None

    # Get posterior means
    post = idata.posterior
    Gamma_mean = post['Gamma'].mean(dim=['chain', 'draw']).values
    pi0_mean = post['pi0'].mean(dim=['chain', 'draw']).values

    # Get emission parameters
    alpha_mean = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
    beta_mean = post['beta_gamma'].mean(dim=['chain', 'draw']).values

    y = data['y']
    N, T = data['N'], data['T']
    K = len(pi0_mean)

    viterbi_states = np.zeros((N, T), dtype=int)

    for i in range(N):
        # Log emission probs for customer i
        log_emit = np.zeros((T, K))

        for t in range(T):
            if not data['mask'][i, t]:
                continue

            y_it = y[i, t]
            for k in range(K):
                # Hurdle emission
                pi_ik = post['pi'].mean(dim=['chain', 'draw']).values[i, t, k] if 'pi' in post else 0.5

                if y_it == 0:
                    log_emit[t, k] = np.log(1 - pi_ik + 1e-10)
                else:
                    # Gamma logpdf
                    from scipy.stats import gamma
                    log_gamma = gamma.logpdf(y_it, alpha_mean[k], scale=1/beta_mean[i,t,k] if beta_mean.ndim > 1 else 1/beta_mean[k])
                    log_emit[t, k] = np.log(pi_ik + 1e-10) + log_gamma

        # Viterbi
        log_delta = np.log(pi0_mean + 1e-10) + log_emit[0]
        psi = np.zeros((T, K), dtype=int)

        for t in range(1, T):
            for k in range(K):
                scores = log_delta + np.log(Gamma_mean[:, k] + 1e-10)
                psi[t, k] = np.argmax(scores)
                log_delta[k] = np.max(scores) + log_emit[t, k]

        # Backtrack
        states_i = np.zeros(T, dtype=int)
        states_i[-1] = np.argmax(log_delta)
        for t in range(T-2, -1, -1):
            states_i[t] = psi[t+1, states_i[t+1]]

        viterbi_states[i] = states_i

    return viterbi_states


def compute_state_recovery_metrics(idata, data):
    """
    Compute state recovery metrics against true states.

    Returns ARI, confusion matrix, per-state accuracy.
    """
    if 'true_states' not in data:
        print("WARNING: No true_states in data")
        return None

    true_states = data['true_states']

    # Try to get Viterbi from model (if stored) or compute post-hoc
    if 'viterbi' in idata.posterior:
        viterbi = idata.posterior['viterbi'].mean(dim=['chain', 'draw']).values.astype(int)
    else:
        print("Computing Viterbi post-hoc...")
        viterbi = extract_viterbi_states(idata, data)

    if viterbi is None:
        return None

    # Flatten
    mask = data['mask']
    true_flat = true_states[mask]
    viterbi_flat = viterbi[mask]

    # ARI
    ari = adjusted_rand_score(true_flat, viterbi_flat)

    # Confusion matrix
    K = len(np.unique(true_flat))
    conf_mat = confusion_matrix(true_flat, viterbi_flat, labels=range(K))

    # Per-state accuracy
    per_state_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)

    metrics = {
        'ari': ari,
        'confusion_matrix': conf_mat.tolist(),
        'per_state_accuracy': per_state_acc.tolist(),
        'overall_accuracy': np.mean(viterbi_flat == true_flat)
    }

    return metrics


def quick_extract(pkl_path):
    """
    Quick extraction and summary of a .pkl result file.

    Usage:
        python smc_hmm_hurdle.py --extract --pkl_file results/smc_...pkl
    """
    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        return

    print(f"\nLoading: {pkl_path.name}")
    print("=" * 60)

    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    idata = result['idata']
    res = result['res']
    data = result.get('data', {})

    # Basic info
    print(f"Model: K={res['K']}, {res['glm_gam']}, world={res['world']}")
    print(f"Data: N={res['N']}, T={res['T']}, zeros={res['zero_rate']:.1%}")
    print(f"SMC: draws={res['draws']}, chains={res['chains']}, time={res['time_min']:.1f}min")
    print(f"Fit: log_ev={res['log_evidence']:.2f}")

    if 'ess_min' in res and not np.isnan(res['ess_min']):
        print(f"Diagnostics: ESS={res['ess_min']:.0f}, R-hat={res['rhat_max']:.3f}")

    # Posterior summary
    print("\nPosterior Summary:")
    print("-" * 60)

    if 'beta0' in idata.posterior:
        beta0 = idata.posterior['beta0']
        print(f"beta0 (state intercepts):")
        for k in range(beta0.shape[-1]):
            mean = float(beta0.mean(dim=['chain', 'draw']).values[k])
            sd = float(beta0.std(dim=['chain', 'draw']).values[k])
            print(f"  State {k}: {mean:.3f} (±{sd:.3f})")

    if 'alpha_gamma' in idata.posterior:
        alpha = idata.posterior['alpha_gamma']
        print(f"\nalpha (Gamma shape):")
        for k in range(alpha.shape[-1]):
            mean = float(alpha.mean(dim=['chain', 'draw']).values[k])
            sd = float(alpha.std(dim=['chain', 'draw']).values[k])
            print(f"  State {k}: {mean:.3f} (±{sd:.3f})")

    if 'Gamma' in idata.posterior:
        Gamma = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
        print(f"\nTransition matrix (mean):")
        print(f"  From\To   Dormant  Lukewarm  Whale")
        for i in range(Gamma.shape[0]):
            row = "  ".join([f"{Gamma[i,j]:.3f}" for j in range(Gamma.shape[1])])
            print(f"  State {i}:  {row}")

    # State recovery
    if 'true_states' in data:
        print("\nState Recovery:")
        print("-" * 60)
        metrics = compute_state_recovery_metrics(idata, data)
        if metrics:
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"  Per-state accuracy: {[f'{a:.2%}' for a in metrics['per_state_accuracy']]}")

    print("=" * 60)

    return result

## ----

def aggregate_results(base_out_dir: str, worlds: list = None):
    """
    Aggregate all summary CSVs across worlds into single table.
    """
    base_path = Path(base_out_dir)
    
    if worlds is None:
        worlds = ['harbor', 'breeze', 'fog', 'cliff']
    
    all_summaries = []
    
    for world in worlds:
        world_dir = base_path / world
        if not world_dir.exists():
            continue
        
        csv_files = list(world_dir.glob("summary_*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['world'] = world
                df['source_file'] = csv_file.name
                all_summaries.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {csv_file}: {e}")
    
    if not all_summaries:
        print("No summary files found")
        return None
    
    combined = pd.concat(all_summaries, ignore_index=True)
    
    if 'log_evidence' in combined.columns:
        combined = combined.sort_values(['world', 'log_evidence'], ascending=[True, False])
    
    master_path = base_path / "all_worlds_summary.csv"
    combined.to_csv(master_path, index=False)
    
    print(f"\nAggregated {len(all_summaries)} runs from {len(set(combined['world']))} worlds")
    print(f"Saved: {master_path}")
    
    # Display
    print("\n" + "="*100)
    print("AGGREGATED RESULTS")
    print("="*100)
    
    display_cols = ['world', 'K', 'glm_gam', 'N', 'T', 'log_evidence', 'time_min', 'ess_min', 'rhat_max']
    display_cols = [c for c in display_cols if c in combined.columns]
    
    for col in ['log_evidence', 'time_min', 'ess_min', 'rhat_max']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    print(combined[display_cols].to_string(index=False))
    print("="*100)
    
    return combined

# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='HMM-Hurdle-Gamma: SMC Estimation (JRSS-C cleaned)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--extract', action='store_true', help='Extraction mode')
    parser.add_argument('--pkl_file', type=str, help='Path to .pkl for extraction')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate results')

    parser.add_argument('--csv_path', type=str, default=None,
                        help='Direct path to CSV (overrides data_dir + world)')
    parser.add_argument('--data_dir', type=str, default='./data/simulation')
    parser.add_argument('--world', type=str, choices=['Harbor', 'Breeze', 'Fog', 'Cliff'])

    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument('--T', type=int, default=104)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--no_gam', action='store_true')
    parser.add_argument('--gam_df', type=int, default=3)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)

    args = parser.parse_args()

    if args.aggregate:
        aggregate_results(args.out_dir)
        return

    if args.extract:
        if not args.pkl_file:
            print("ERROR: --extract requires --pkl_file")
            return
        quick_extract(args.pkl_file)
        return

    print("=" * 70)
    print("HMM-Hurdle-Gamma: SMC Estimation (Cleaned for JRSS-C)")
    print("=" * 70)

    # Load data
    if args.csv_path:
        csv_file = Path(args.csv_path)
        if 'uci' in csv_file.name.lower() or 'cdnow' in csv_file.name.lower():
            data = load_empirics_data_from_csv(csv_file, args.N, args.seed)
        else:
            data = load_simulation_data_from_csv(csv_file, args.T, args.N, args.seed)  # use the cleaned version from BEMMAOR
        world = args.world or "unknown"
    else:
        data = load_simulation_data(args.world, Path(args.data_dir), args.T, args.N, 
                                   pilot=False, seed=args.seed)
        world = args.world

    print(f"Configuration: K={args.K}, N={data['N']}, T={data['T']}, world={world}")

    # Output directory
    world_out_dir = Path(args.out_dir) / world.lower()
    world_out_dir.mkdir(parents=True, exist_ok=True)

    # Run
    pkl_path, res, idata = run_smc_hurdle(
        data=data,
        K=args.K,
        draws=args.draws,
        chains=args.chains,
        seed=args.seed,
        out_dir=world_out_dir,
        use_gam=not args.no_gam,
        gam_df=args.gam_df
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Log-Evidence: {res.get('log_evidence', np.nan):.2f}")
    print(f"Runtime: {res.get('elapsed_min', 0):.1f} min")
    print(f"Output: {pkl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
