#!/usr/bin/env python3
"""
convergence.py
==============
Convergence diagnostics: R-hat, ESS, trace inspection.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def check_convergence(idata, var_names=None):
    """
    Check convergence diagnostics.
    
    Returns R-hat and ESS for key parameters.
    """
    if var_names is None:
        var_names = ['beta0', 'phi', 'Gamma']
    
    # Filter to existing variables
    available = [v for v in var_names if v in idata.posterior]
    
    if not available:
        return {'error': 'No requested variables found'}
    
    summary = az.summary(idata, var_names=available)
    
    results = {
        'max_rhat': summary['r_hat'].max(),
        'min_ess_bulk': summary['ess_bulk'].min(),
        'min_ess_tail': summary['ess_tail'].min(),
        'problematic_params': summary[summary['r_hat'] > 1.1].index.tolist(),
        'low_ess_params': summary[summary['ess_bulk'] < 400].index.tolist()
    }
    
    return results


def generate_convergence_report(pkl_path, out_dir=None):
    """Generate convergence report with plots."""
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
    
    idata = saved['idata']
    res = saved['res']
    
    print(f"\n{'='*60}")
    print(f"Convergence Report: {res['dataset']}, K={res['K']}")
    print(f"{'='*60}")
    
    # Numerical diagnostics
    diag = check_convergence(idata)
    
    print(f"\nR-hat (max): {diag['max_rhat']:.4f}")
    print(f"ESS bulk (min): {diag['min_ess_bulk']:.0f}")
    print(f"ESS tail (min): {diag['min_ess_tail']:.0f}")
    
    if diag['problematic_params']:
        print(f"\n⚠️  Problematic (R-hat > 1.1): {diag['problematic_params']}")
    else:
        print("\n✓ All parameters converged (R-hat < 1.1)")
    
    if diag['low_ess_params']:
        print(f"⚠️  Low ESS (< 400): {diag['low_ess_params']}")
    
    # Save summary
    if out_dir:
        out_dir = Path(out_dir) / 'figures'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Trace plots
        try:
            az.plot_trace(idata, var_names=['beta0', 'phi'])
            plt.savefig(out_dir / f"trace_{res['dataset']}_K{res['K']}.png", dpi=150)
            plt.close()
            print(f"\nTrace plot saved")
        except Exception as e:
            print(f"Plot failed: {e}")
    
    return diag


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file')
    parser.add_argument('--out_dir', default='./04_results')
    args = parser.parse_args()
    
    generate_convergence_report(args.pkl_file, args.out_dir)
