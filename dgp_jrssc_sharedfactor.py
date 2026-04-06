
# Fix DGP code to remove --world flag
# Use parameter-based naming instead

#!/usr/bin/env python3
"""
dgp_jrssc_sharedfactor.py
JRSS-C Shared Factor DGP with Persistent θ_i
Generates simulation data for Phase Transition study
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json

RANDOM_SEED = 42

def calibrate_gamma_m(target_rho, gamma_h=1.0, sigma_alpha=0.5, sigma_beta=0.5):
    """Calibrate γ_m to achieve target implied correlation ρ."""
    if target_rho == 0:
        return 0.0
    
    gamma_m = target_rho * gamma_h / np.sqrt(1 - target_rho**2)
    
    for _ in range(10):
        denom = np.sqrt((gamma_h**2 + sigma_alpha**2) * (gamma_m**2 + sigma_beta**2))
        rho_implied = gamma_h * gamma_m / denom
        gamma_m = gamma_m * (target_rho / (rho_implied + 1e-10))
    
    return gamma_m

def compute_hmm_entropy_rate(Gamma, pi0):
    """Compute HMM entropy rate H_0."""
    K = len(pi0)
    H = 0.0
    for s in range(K):
        for sprime in range(K):
            if Gamma[s, sprime] > 0:
                H -= pi0[s] * Gamma[s, sprime] * np.log(Gamma[s, sprime])
    return H

def simulate_customer_sharedfactor(customer_id, T, true_states, theta_i, 
                                   alpha_h, beta_m, gamma_h, gamma_m,
                                   r_nb, alpha_gamma):
    """Simulate single customer with shared factor θ_i."""
    K = len(alpha_h)
    y = np.zeros(T)
    z = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)
    f = np.zeros(T, dtype=int)
    
    cum_freq = 0
    last_purchase = -1
    
    for t in range(T):
        s = true_states[t]
        
        # Frequency (NBD) channel with shared factor
        log_lambda = alpha_h[s] + gamma_h * theta_i
        lambda_it = np.exp(np.clip(log_lambda, -10, 10))
        
        # NBD probability of purchase
        p_purchase = 1.0 - (r_nb[s] / (r_nb[s] + lambda_it)) ** r_nb[s]
        
        # Simulate purchase
        z[t] = 1 if np.random.random() < p_purchase else 0
        
        if z[t] == 1:
            # Magnitude (Gamma) channel with shared factor
            log_mu = beta_m[s] + gamma_m * theta_i
            mu_it = np.exp(np.clip(log_mu, -10, 10))
            
            # Gamma rate parameter
            beta_gamma = alpha_gamma[s] / mu_it
            y[t] = np.random.gamma(alpha_gamma[s], 1.0 / beta_gamma)
            
            cum_freq += 1
            last_purchase = t
        else:
            y[t] = 0.0
        
        # RFM features
        f[t] = cum_freq
        if last_purchase >= 0:
            r[t] = t - last_purchase
        else:
            r[t] = t + 1
    
    return y, z, r, f

def simulate_full_panel(N, T, pi0, psi, rho, 
                       gamma_h=1.0, sigma_alpha=0.5, sigma_beta=0.5,
                       seed=RANDOM_SEED):
    """
    Generate full simulation panel with shared factor θ_i.
    
    Treatment factors:
    - pi0: sparsity (target zero rate)
    - psi: dispersion (controls r_nb)
    - rho: target correlation (calibrated via γ_m)
    """
    rng = np.random.default_rng(seed)
    K = 2
    
    # Calibrate γ_m for target ρ
    gamma_m = calibrate_gamma_m(rho, gamma_h, sigma_alpha, sigma_beta)
    
    print(f"Calibration: target ρ={rho:.2f} → γ_m={gamma_m:.3f} (γ_h={gamma_h})")
    
    # HMM transition matrix
    Gamma = np.array([[0.9, 0.1],
                      [0.2, 0.8]], dtype=np.float32)
    
    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(Gamma.T)
    pi_stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    pi_stationary = pi_stationary / pi_stationary.sum()
    pi_stationary = pi_stationary.flatten()
    
    # State-specific parameters
    r_nb = np.array([psi, psi])
    
    # Calibrate α_h to hit target sparsity π₀
    target_lambda_s0 = -np.log(pi0) if pi0 > 0.5 else 1.0
    alpha_h_s0 = np.log(target_lambda_s0) - 0.5 * gamma_h**2
    
    target_lambda_s1 = 0.5
    alpha_h_s1 = np.log(target_lambda_s1) - 0.5 * gamma_h**2
    
    alpha_h = np.array([alpha_h_s0, alpha_h_s1])
    
    # Magnitude parameters
    beta_m = np.array([2.0, 4.0])
    alpha_gamma = np.array([2.0, 3.0])
    
    # Generate data
    records = []
    
    for i in range(N):
        theta_i = rng.normal(0, 1)
        
        # Sample state sequence
        true_states = np.zeros(T, dtype=int)
        true_states[0] = rng.choice(K, p=pi_stationary)
        for t in range(1, T):
            true_states[t] = rng.choice(K, p=Gamma[true_states[t-1]])
        
        # Simulate customer
        y, z, r, f = simulate_customer_sharedfactor(
            i, T, true_states, theta_i,
            alpha_h, beta_m, gamma_h, gamma_m,
            r_nb, alpha_gamma
        )
        
        for t in range(T):
            records.append({
                'customer_id': i,
                't': t,
                'y': y[t],
                'r': r[t],
                'f': f[t],
                'true_state': true_states[t],
                'theta': theta_i,
                'z': z[t]
            })
    
    df = pd.DataFrame(records)
    
    # Compute implied ρ verification
    theta_sample = df.groupby('customer_id')['theta'].first().values
    log_lambda_s0 = alpha_h[0] + gamma_h * theta_sample
    log_mu_s0 = beta_m[0] + gamma_m * theta_sample
    
    var_log_lambda = gamma_h**2 + 0.1
    var_log_mu = gamma_m**2 + 0.1
    cov_log = gamma_h * gamma_m
    
    rho_implied = cov_log / np.sqrt(var_log_lambda * var_log_mu)
    
    H_0 = compute_hmm_entropy_rate(Gamma, pi_stationary)
    snr = (gamma_h**2 * gamma_m**2) / (psi * pi0)
    
    metadata = {
        'N': N, 'T': T, 'K': K,
        'pi0_target': pi0,
        'pi0_actual': float((df['y'] == 0).mean()),
        'psi': psi,
        'rho_target': rho,
        'rho_implied': float(rho_implied),
        'gamma_h': gamma_h,
        'gamma_m': float(gamma_m),
        'alpha_h': alpha_h.tolist(),
        'beta_m': beta_m.tolist(),
        'r_nb': r_nb.tolist(),
        'alpha_gamma': alpha_gamma.tolist(),
        'Gamma': Gamma.tolist(),
        'HMM_entropy_H0': float(H_0),
        'SNR': float(snr),
        'seed': seed
    }
    
    print(f"DGP Summary: π₀_target={pi0:.2f}, actual={metadata['pi0_actual']:.2%}")
    print(f"             ρ_target={rho:.2f}, implied={rho_implied:.3f}")
    print(f"             H₀={H_0:.3f}, SNR={snr:.3f}")
    
    return df, metadata

def main():
    parser = argparse.ArgumentParser(description='JRSS-C Shared Factor DGP')
    parser.add_argument('--N', type=int, default=500, help='Number of customers')
    parser.add_argument('--T', type=int, default=52, help='Time periods')
    parser.add_argument('--pi0', type=float, default=0.95, help='Target sparsity')
    parser.add_argument('--psi', type=float, default=15, help='Dispersion')
    parser.add_argument('--rho', type=float, default=0.4, help='Target correlation')
    parser.add_argument('--out_dir', type=str, default='./results_sharedfactor/dgp')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("="*60)
    print(f"JRSS-C Shared Factor DGP")
    print(f"N={args.N}, T={args.T}, π₀={args.pi0}, ψ={args.psi}, ρ={args.rho}")
    print("="*60)
    
    df, metadata = simulate_full_panel(
        N=args.N, T=args.T, 
        pi0=args.pi0, psi=args.psi, rho=args.rho,
        seed=args.seed
    )
    
    # Save with parameter-based naming (no world!)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean filename: pi0_0.95_psi_15_rho_0.8_rep00.csv
    csv_path = out_dir / f"pi0_{args.pi0:.2f}_psi_{args.psi:.0f}_rho_{args.rho:.1f}_rep00.csv"
    
    df.to_csv(csv_path, index=False)
    print(f"\\nSaved: {csv_path}")
    
    meta_path = csv_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")
    
    print("\\nVerification sample (first 3 customers):")
    print(df[df['customer_id'] < 3][['customer_id', 't', 'y', 'true_state', 'theta']].head(10))

if __name__ == "__main__":
    main()

