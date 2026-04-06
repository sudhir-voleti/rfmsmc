# Create tiny pilot orchestrator (8 runs only)
#!/usr/bin/env python3
"""
JASA Pilot - Tiny Test (8 runs)
N=500, muted OOS/PPC, keeps CLV
"""

import numpy as np
import pickle
import time
import sys
from pathlib import Path
import importlib.util

# Muted paths - update these to your actual file locations
BEMMAOR_PATH = '/Users/sudhirvoleti/jrssc_april/smc_bemmaor_new.py'
HURDLE_PATH = '/Users/sudhirvoleti/jrssc_april/smc_hurdle_new.py'

# PILOT CONFIG: Just 2 conditions × 2 reps × 2 models = 8 runs
PI0_VALUES = [0.90, 0.95]      # High sparsity (where phase boundary matters)
PSI_VALUES = [5]                # Middle volatility
RHO_VALUES = [0.4, 0.6]         # Around rho_crit
N_PILOT = 500                   # Test N=500
T_PILOT = 52
K_PILOT = 2
REPS = 2                        # Just 2 reps for pilot
DRAW_PILOT = 1000
CHAINS_PILOT = 4

def generate_dgp(N=500, T=52, K=2, pi0=0.9, psi=5, rho=0.4, seed=42):
    """Generate DGP with shared factor theta_i."""
    rng = np.random.default_rng(seed)
    
    # Transition matrix with stickiness
    stickiness = 0.85 + 0.1 * (1 / psi)
    Gamma = np.array([
        [stickiness, 1-stickiness],
        [(1-stickiness)*0.7, 1-(1-stickiness)*0.7]
    ])
    
    pi0_vec = np.array([pi0, 1-pi0])
    r_nb = np.array([1.0, 2.0])
    
    # Emission parameters
    alpha_h = np.array([-1.0 - 0.5*(1-pi0), 0.5])
    alpha_gamma = np.array([2.0, 5.0])
    beta_m = np.array([1.0, 2.5])
    
    # Shared factor
    theta = rng.normal(0, 1, size=(N, 1))
    
    # Coupling parameters (rho = gamma_h * gamma_m)
    gamma_h = rho * 0.6
    gamma_m = rho * 1.0
    
    # Generate states
    Z = np.zeros((N, T), dtype=int)
    for i in range(N):
        Z[i, 0] = rng.choice(K, p=pi0_vec)
        for t in range(1, T):
            Z[i, t] = rng.choice(K, p=Gamma[Z[i, t-1], :])
    
    # Generate observations
    Y = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            k = Z[i, t]
            lam = np.exp(alpha_h[k] + gamma_h * theta[i, 0])
            p_zero = (r_nb[k] / (r_nb[k] + lam)) ** r_nb[k]
            
            if rng.random() > p_zero:
                mu_spend = np.exp(beta_m[k] + gamma_m * theta[i, 0])
                beta_gamma = alpha_gamma[k] / mu_spend
                Y[i, t] = rng.gamma(alpha_gamma[k], 1/beta_gamma)
    
    sparsity = np.mean(Y == 0)
    
    return {
        'Y': Y, 'Z': Z, 'Gamma': Gamma, 'pi0': pi0_vec, 'theta': theta,
        'gamma_h': gamma_h, 'gamma_m': gamma_m, 'N': N, 'T': T, 'K': K,
        'seed': seed, 'params': {'pi0': pi0, 'psi': psi, 'rho': rho},
        'sparsity': sparsity, 'true_rho': gamma_h * gamma_m
    }

def compute_rfm_features(y, mask):
    """Compute RFM features."""
    N, T = y.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    
    for i in range(N):
        last_purchase, cum_freq, cum_spend = -1, 0, 0.0
        for t in range(T):
            if mask[i, t]:
                if y[i, t] > 0:
                    last_purchase, cum_freq = t, cum_freq + 1
                    cum_spend += y[i, t]
                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cum_freq
                    M[i, t] = cum_spend / cum_freq if cum_freq > 0 else 0.0
                else:
                    R[i, t], F[i, t], M[i, t] = t + 1, 0, 0.0
    
    return R, F, M

def dgp_to_mksc_format(Y, Z_true=None, world="jasa_pilot"):
    """Convert DGP to MK SC format."""
    N, T_full = Y.shape
    y_train = Y.copy()
    mask_train = ~np.isnan(y_train) & (y_train >= 0)
    y_train = np.where(mask_train, y_train, 0.0).astype(np.float32)
    
    R_train, F_train, M_train = compute_rfm_features(y_train, mask_train)
    M_train_log = np.log1p(M_train)
    
    # Normalize
    R_valid = R_train[mask_train]
    F_valid = F_train[mask_train]
    M_valid = M_train_log[mask_train]
    
    if len(R_valid) > 0 and R_valid.std() > 0:
        R_train = (R_train - R_valid.mean()) / (R_valid.std() + 1e-6)
    if len(F_valid) > 0 and F_valid.std() > 0:
        F_train = (F_train - F_valid.mean()) / (F_valid.std() + 1e-6)
    if len(M_valid) > 0 and M_valid.std() > 0:
        M_train_scaled = (M_train_log - M_valid.mean()) / (M_valid.std() + 1e-6)
    else:
        M_train_scaled = M_train_log
    
    return {
        'N': N, 'T': T_full,
        'y': y_train.astype(np.float32),
        'mask': mask_train.astype(bool),
        'R': R_train.astype(np.float32),
        'F': F_train.astype(np.float32),
        'M': M_train_scaled.astype(np.float32),
        'true_states': Z_true.astype(np.int32) if Z_true is not None else np.zeros((N, T_full), dtype=np.int32) - 1,
        'world': world,
        'M_raw': M_train.astype(np.float32),
        'T_total': T_full,
        'train_ratio': 1.0  # No OOS split
    }

def run_model(model_type, data, K, draws, chains, out_dir, seed=42):
    """Run model with MUTED OOS and PPC."""
    model_path = BEMMAOR_PATH if model_type == "BEMMAOR" else HURDLE_PATH
    spec = importlib.util.spec_from_file_location(f"mksc_{model_type.lower()}", model_path)
    mksc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mksc)
    
    start = time.time()
    
    try:
        # Check if model has lean_run flag or modify to skip OOS/PPC
        if hasattr(mksc, 'run_smc_bemmaor'):
            # Try to pass lean flags
            try:
                pkl_path, res, idata = mksc.run_smc_bemmaor(
                    data, K, draws, chains, seed, out_dir,
                    compute_oos=False,  # MUTE OOS
                    compute_ppc=False,  # MUTE PPC
                    compute_clv=True    # KEEP CLV
                )
            except TypeError:
                # If lean flags not supported, run full and hope it's fast
                print(f"    Warning: Lean flags not supported, running full...")
                pkl_path, res, idata = mksc.run_smc_bemmaor(data, K, draws, chains, seed, out_dir)
        elif hasattr(mksc, 'run_smc_hurdle'):
            try:
                pkl_path, res, idata = mksc.run_smc_hurdle(
                    data, K, draws, chains, seed, out_dir,
                    compute_oos=False,
                    compute_ppc=False,
                    compute_clv=True
                )
            except TypeError:
                print(f"    Warning: Lean flags not supported, running full...")
                pkl_path, res, idata = mksc.run_smc_hurdle(data, K, draws, chains, seed, out_dir)
        else:
            pkl_path, res, idata = mksc.run_smc(data, K, draws, chains, seed, out_dir)
        
        elapsed = (time.time() - start) / 60
        
        return {
            'success': True,
            'pkl_path': str(pkl_path),
            'log_evidence': res.get('log_evidence', np.nan),
            'ess_min': res.get('ess_min', np.nan),
            'time_min': elapsed,
            'model_type': model_type,
            'N': data['N']
        }
        
    except Exception as e:
        elapsed = (time.time() - start) / 60
        print(f"    ERROR: {e}")
        return {'success': False, 'error': str(e), 'time_min': elapsed}

def run_pilot_condition(pi0, psi, rho, rep, base_dir):
    """Run single pilot condition."""
    folder_name = f"pilot_pi0_{pi0:.2f}_psi_{psi}_rho_{rho:.1f}"
    rep_folder = f"rep_{rep:02d}"
    out_dir = Path(base_dir) / folder_name / rep_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Consistent seed
    seed = int(pi0 * 10000 + psi * 100 + rho * 10 + rep + 10000)  # +10000 to differentiate from N250
    world_str = f"pilot_N500_pi0{pi0:.2f}_psi{psi}_rho{rho:.1f}_rep{rep:02d}"
    
    print(f"\\n[{folder_name}/{rep_folder}] N=500, pi0={pi0}, psi={psi}, rho={rho}")
    
    # Generate DGP
    dgp = generate_dgp(N=N_PILOT, T=T_PILOT, K=K_PILOT, pi0=pi0, psi=psi, rho=rho, seed=seed)
    dgp_filename = f"dgp_pi0{pi0:.2f}_psi{psi}_rho{rho:.1f}_rep{rep:02d}_N500.pkl"
    dgp_path = out_dir / dgp_filename
    with open(dgp_path, 'wb') as f:
        pickle.dump(dgp, f)
    print(f"    DGP: sparsity={dgp['sparsity']:.1%}, N={N_PILOT}")
    
    # Convert to MK SC format
    data = dgp_to_mksc_format(dgp['Y'], dgp['Z'], world=world_str)
    
    results = {}
    
    # BEMMAOR
    print(f"    BEMMAOR...")
    res_bem = run_model("BEMMAOR", data, K_PILOT, DRAW_PILOT, CHAINS_PILOT, str(out_dir), seed)
    if res_bem['success']:
        results['BEMMAOR'] = res_bem
        print(f"      ✓ Log-ev={res_bem['log_evidence']:.2f}, ESS={res_bem['ess_min']:.1f}, Time={res_bem['time_min']:.1f}min")
    else:
        print(f"      ✗ Failed: {res_bem.get('error', 'Unknown')}")
    
    # Hurdle
    print(f"    Hurdle...")
    res_hur = run_model("Hurdle", data, K_PILOT, DRAW_PILOT, CHAINS_PILOT, str(out_dir), seed+1000)
    if res_hur['success']:
        results['Hurdle'] = res_hur
        print(f"      ✓ Log-ev={res_hur['log_evidence']:.2f}, ESS={res_hur['ess_min']:.1f}, Time={res_hur['time_min']:.1f}min")
    else:
        print(f"      ✗ Failed: {res_hur.get('error', 'Unknown')}")
    
    return results

def main():
    base_dir = '/Users/sudhirvoleti/jrssc_april/results_jasa_pilot'
    
    print("="*70)
    print("JASA PILOT - Tiny Test (N=500, Muted OOS/PPC)")
    print("="*70)
    print(f"Config: {len(PI0_VALUES)}×{len(PSI_VALUES)}×{len(RHO_VALUES)}×{REPS} = 8 runs total")
    print(f"N={N_PILOT}, T={T_PILOT}, D={DRAW_PILOT}, chains={CHAINS_PILOT}")
    print(f"Base dir: {base_dir}")
    print("="*70)
    
    total_start = time.time()
    all_results = []
    
    for pi0 in PI0_VALUES:
        for psi in PSI_VALUES:
            for rho in RHO_VALUES:
                for rep in range(REPS):
                    results = run_pilot_condition(pi0, psi, rho, rep, base_dir)
                    all_results.append(results)
    
    total_elapsed = (time.time() - total_start) / 60
    
    # Summary
    print("\\n" + "="*70)
    print("PILOT COMPLETE")
    print("="*70)
    print(f"Total time: {total_elapsed:.1f} minutes")
    
    bem_times = [r['BEMMAOR']['time_min'] for r in all_results if 'BEMMAOR' in r and r['BEMMAOR'].get('success')]
    hur_times = [r['Hurdle']['time_min'] for r in all_results if 'Hurdle' in r and r['Hurdle'].get('success')]
    
    if bem_times:
        print(f"BEMMAOR: {len(bem_times)}/4 successful, avg time {np.mean(bem_times):.1f} min")
    if hur_times:
        print(f"Hurdle:  {len(hur_times)}/4 successful, avg time {np.mean(hur_times):.1f} min")
    
    print(f"\\nEstimated full N=500 vertical slice (180 runs):")
    avg_time = np.mean(bem_times + hur_times) if (bem_times or hur_times) else 5.0
    print(f"  ~{avg_time * 180 / 60:.1f} hours with 1 terminal")
    print(f"  ~{avg_time * 180 / 60 / 4:.1f} hours with 4 terminals (parallel)")
    
    print("="*70)

if __name__ == "__main__":
    main()
