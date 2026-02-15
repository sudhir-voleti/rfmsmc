#!/usr/bin/env python3
"""
smc_runner.py
=============
Run SMC estimation with fixed seed=42 for reproducibility.
Extracts log-evidence, WAIC, LOO from fitted models.
"""

import os
import time
import pickle
import platform
from pathlib import Path
import numpy as np
import pymc as pm
import arviz as az

from ..02_model.model_builder import make_model

# Lock seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'


def run_smc(data, K=3, dataset='unknown', out_dir='./04_results',
            state_specific_p=False, p_fixed=1.5,
            use_gam=True, gam_df=3, emission_type='tweedie',
            p_min=None, p_max=None, phi_sort=True,
            gamma_diag=1.0, shared_phi=True,
            draws=200, chains=4, random_seed=RANDOM_SEED):
    """
    Run SMC estimation.
    
    Parameters
    ----------
    data : dict
        Panel data dictionary
    K : int
        Number of states
    dataset : str
        'cdnow' or 'uci'
    out_dir : str
        Output directory
    [model params...]
    draws : int
        Number of SMC draws (pilot: 200, final: 1000)
    chains : int
        Number of chains
    random_seed : int
        Random seed (default: 42, locked)
        
    Returns
    -------
    tuple
        (pkl_path, results_dict)
    """
    out_dir = Path(out_dir) / 'smc_fits'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()
    
    if IS_APPLE_SILICON:
        print(f"  Apple Silicon: {os.cpu_count()} cores, using {cores}")
    
    # Force seed
    np.random.seed(random_seed)
    
    try:
        with make_model(data, K=K, state_specific_p=state_specific_p,
                       p_fixed=p_fixed, use_gam=use_gam, gam_df=gam_df,
                       emission_type=emission_type, p_min=p_min, p_max=p_max,
                       phi_sort=phi_sort, gamma_diag=gamma_diag,
                       shared_phi=shared_phi) as model:
            
            print(f"  Model: K={K}, {'GAM' if use_gam else 'GLM'}, "
                  f"p={'state-specific' if state_specific_p else p_fixed}, "
                  f"phi={'shared' if shared_phi else 'state'}")
            print(f"  Data: N={data['N']}, T={data['T']}")
            print(f"  SMC: draws={draws}, chains={chains}, seed={random_seed}")
            
            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True,
                threshold=0.8
            )
        
        # Extract metrics
        log_ev = _extract_log_evidence(idata)
        waic_loo = _compute_waic_loo(idata)
        
        elapsed = (time.time() - t0) / 60
        
        res = {
            'dataset': dataset,
            'K': K,
            'N': data['N'],
            'T': data['T'],
            'use_gam': use_gam,
            'gam_df': gam_df if use_gam else None,
            'state_specific_p': state_specific_p,
            'p_fixed': p_fixed if not state_specific_p else None,
            'shared_phi': shared_phi,
            'phi_sort': phi_sort,
            'gamma_diag': gamma_diag,
            'emission_type': emission_type,
            'log_evidence': log_ev,
            'waic': waic_loo.get('waic'),
            'loo': waic_loo.get('loo'),
            'draws': draws,
            'chains': chains,
            'seed': random_seed,
            'time_min': elapsed,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'platform': 'Apple_Silicon' if IS_APPLE_SILICON else 'Other'
        }

        # Build filename
        model_tag = f"K{K}_{'GAM' if use_gam else 'GLM'}"
        phi_tag = "sharedphi" if shared_phi else "statespfc"
        p_tag = "statep" if state_specific_p else f"p{p_fixed}"
        
        pkl_name = f"smc_{dataset}_{model_tag}_{phi_tag}_{p_tag}_N{data['N']}_D{draws}_C{chains}.pkl"
        pkl_path = out_dir / pkl_name
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)
        
        size_mb = pkl_path.stat().st_size / (1024**2)
        print(f"  ✓ log_ev={log_ev:.2f}, waic={res['waic']:.2f}, "
              f"time={elapsed:.1f}min, size={size_mb:.1f}MB")
        
        return pkl_path, res

    except Exception as e:
        import traceback
        print(f"  ✗ CRASH: {str(e)[:80]}")
        traceback.print_exc()
        raise


def _extract_log_evidence(idata):
    """Extract log marginal likelihood from SMC results."""
    log_ev = np.nan
    try:
        lm = idata.sample_stats.log_marginal_likelihood.values
        
        if isinstance(lm, np.ndarray) and lm.dtype == object:
            chain_vals = []
            for c in range(lm.shape[1] if lm.ndim > 1 else 1):
                if lm.ndim > 1:
                    chain_list = lm[-1, c] if lm.shape[0] > 0 else lm[0, c]
                else:
                    chain_list = lm[c] if lm.ndim == 1 else lm[0]
                
                if isinstance(chain_list, list):
                    valid = [float(x) for x in chain_list 
                            if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                    if valid:
                        chain_vals.append(valid[-1])
                elif isinstance(chain_list, (int, float, np.floating)) and np.isfinite(chain_list):
                    chain_vals.append(float(chain_list))
            
            log_ev = float(np.mean(chain_vals)) if chain_vals else np.nan
        else:
            flat = np.array(lm).flatten()
            valid = flat[np.isfinite(flat)]
            log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
            
    except Exception as e:
        print(f"  Warning: log-ev extraction failed: {e}")
    
    return log_ev


def _compute_waic_loo(idata):
    """Compute WAIC and LOO."""
    results = {}
    try:
        waic = az.waic(idata)
        results['waic'] = float(waic.elpd_waic)
    except Exception as e:
        print(f"  WAIC failed: {e}")
        results['waic'] = np.nan
    
    try:
        loo = az.loo(idata)
        results['loo'] = float(loo.elpd_loo)
    except Exception as e:
        print(f"  LOO failed: {e}")
        results['loo'] = np.nan
    
    return results


def quick_test_fit(data, K=2, dataset='test'):
    """
    Quick test fit with minimal draws (N=50, D=50).
    For testing code only, not for paper results.
    """
    print(f"\n{'='*60}")
    print(f"QUICK TEST FIT: {dataset}, K={K}, N={data['N']}")
    print(f"{'='*60}")
    
    return run_smc(
        data=data,
        K=K,
        dataset=dataset,
        out_dir='./04_results',
        shared_phi=True,
        p_fixed=1.5,
        use_gam=True,
        draws=50,  # Minimal for testing
        chains=2,
        random_seed=RANDOM_SEED  # Locked at 42
    )
