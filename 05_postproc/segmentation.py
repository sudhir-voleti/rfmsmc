#!/usr/bin/env python3
"""
segmentation.py
===============
Generate Table 4: State profiles and CLV metrics.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def extract_state_profiles(idata):
    """
    Extract posterior means for each state.
    
    Returns: beta0, phi, p, churn_risk, clv_proxy per state
    """
    post = idata.posterior
    K = post['beta0'].shape[-1]
    
    profiles = []
    for k in range(K):
        # Extract samples
        beta0_samples = post['beta0'].values[..., k].flatten()
        
        # Phi: shared or state-specific
        if 'phi' in post:
            if post['phi'].shape[-1] == 1:
                phi_samples = post['phi'].values[..., 0].flatten()
            else:
                phi_samples = post['phi'].values[..., k].flatten()
        else:
            phi_samples = np.array([np.nan])
        
        # p: check if state-specific
        if 'p' in post:
            if post['p'].shape[-1] == 1:
                p_samples = post['p'].values[..., 0].flatten()
            else:
                p_samples = post['p'].values[..., k].flatten()
        else:
            p_samples = np.array([1.5])  # Default fixed
        
        # CLV metrics
        if 'churn_risk' in post:
            churn_samples = post['churn_risk'].values[..., k].flatten()
        else:
            churn_samples = np.array([0.5])
        
        if 'clv_proxy' in post:
            clv_samples = post['clv_proxy'].values[..., k].flatten()
        else:
            clv_samples = np.exp(beta0_samples) / (1 - 0.95 * (1 - churn_samples))
        
        profiles.append({
            'State': k,
            'beta0_mean': np.mean(beta0_samples),
            'beta0_std': np.std(beta0_samples),
            'phi_mean': np.mean(phi_samples),
            'phi_std': np.std(phi_samples),
            'p_mean': np.mean(p_samples),
            'p_std': np.std(p_samples),
            'churn_risk_mean': np.mean(churn_samples),
            'churn_risk_std': np.std(churn_samples),
            'clv_proxy_mean': np.mean(clv_samples),
            'clv_proxy_std': np.std(clv_samples)
        })
    
    return pd.DataFrame(profiles)


def compute_stationary_distribution(Gamma):
    """Compute stationary distribution from transition matrix."""
    K = Gamma.shape[0]
    
    # Solve pi = pi * Gamma
    eigvals, eigvecs = np.linalg.eig(Gamma.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)].real
    stationary = stationary / stationary.sum()
    
    return stationary.flatten()


def generate_segmentation_table(pkl_path):
    """Generate Table 4 from fitted model."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    idata = data['idata']
    res = data['res']
    
    # State profiles
    profiles = extract_state_profiles(idata)
    profiles['Dataset'] = res['dataset'].upper()
    profiles['K'] = res['K']
    
    # Add stationary distribution if K > 1
    if res['K'] > 1 and 'Gamma' in idata.posterior:
        Gamma_mean = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
        stationary = compute_stationary_distribution(Gamma_mean)
        profiles['Stationary_pi'] = stationary
    
    # Reorder columns
    cols = ['Dataset', 'K', 'State'] + [c for c in profiles.columns if c not in ['Dataset', 'K', 'State']]
    profiles = profiles[cols]
    
    return profiles


def export_segmentation_table(profiles, output_dir):
    """Export to CSV and LaTeX."""
    output_dir = Path(output_dir) / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'table4_segmentation.csv'
    profiles.to_csv(csv_path, index=False, float_format='%.3f')
    print(f"CSV saved: {csv_path}")
    
    latex_path = output_dir / 'table4_segmentation.tex'
    with open(latex_path, 'w') as f:
        f.write(profiles.to_latex(index=False, float_format='%.3f'))
    print(f"LaTeX saved: {latex_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file', help='Path to SMC result .pkl')
    parser.add_argument('--out_dir', default='./04_results')
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("TABLE 4: STATE PROFILES & CLV METRICS")
    print(f"{'='*80}")
    
    profiles = generate_segmentation_table(args.pkl_file)
    print(profiles.to_string(index=False))
    
    export_segmentation_table(profiles, args.out_dir)
