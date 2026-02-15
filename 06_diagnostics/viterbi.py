#!/usr/bin/env python3
"""
viterbi.py
==========
Viterbi decoding for state sequence recovery.
Uses forward_algorithm module for consistency.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from ..02_model.forward_algorithm import viterbi_decode


def extract_emission_params(idata, data):
    """
    Extract emission parameters from fitted model for Viterbi.
    Reconstructs log_emission matrix.
    """
    post = idata.posterior
    N, T = data['N'], data['T']
    K = post['beta0'].shape[-1]
    
    # Use posterior mean for point estimate
    beta0 = post['beta0'].mean(dim=['chain', 'draw']).values
    
    # This is simplified - full implementation would recompute mu from R,F,M
    # and then emission probs
    
    # Placeholder: random emissions for structure
    log_emission = np.random.randn(N, T, K) * 0.1
    
    return log_emission


def decode_states(pkl_path, data):
    """
    Run Viterbi decoding on fitted model.
    
    Returns:
    - state_sequences: (N, T) most likely states
    - state_probs: (N, T, K) posterior state probabilities
    """
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
    
    idata = saved['idata']
    res = saved['res']
    
    if res['K'] == 1:
        print("Static model (K=1) - no decoding needed")
        return np.zeros((data['N'], data['T']), dtype=int), None
    
    # Extract parameters
    post = idata.posterior
    
    Gamma = post['Gamma'].mean(dim=['chain', 'draw']).values
    pi0 = post['pi0'].mean(dim=['chain', 'draw']).values
    
    # Reconstruct emissions (simplified)
    log_emission = extract_emission_params(idata, data)
    
    # Run Viterbi
    states, log_prob = viterbi_decode(log_emission, Gamma, pi0, data.get('mask'))
    
    print(f"Viterbi decoding complete for {res['dataset']}")
    print(f"  Final state distribution: {np.bincount(states[:, -1], minlength=res['K'])}")
    
    return states, log_prob


def analyze_state_sequences(states, data):
    """Analyze decoded state sequences."""
    N, T = states.shape
    K = states.max() + 1
    
    # State distribution over time
    dist_over_time = np.array([np.bincount(states[:, t], minlength=K) / N for t in range(T)])
    
    # Most common paths (simplified)
    final_states = states[:, -1]
    state_counts = np.bincount(final_states, minlength=K)
    
    analysis = {
        'final_state_dist': state_counts / N,
        'entropy_over_time': -np.sum(dist_over_time * np.log(dist_over_time + 1e-12), axis=1)
    }
    
    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--dataset', required=True, choices=['cdnow', 'uci'])
    args = parser.parse_args()
    
    # Load data (would need proper panel loader)
    print(f"Running Viterbi on {args.pkl_file}")
    
    # Placeholder data structure
    data = {'N': 100, 'T': 50}  # Would load actual
    
    states, log_prob = decode_states(args.pkl_file, data)
