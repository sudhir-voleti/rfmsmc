#!/usr/bin/env python3
"""
transitions.py
==============
Generate Table 5: Transition matrix and dwell times.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def extract_gamma_matrix(idata):
    """Extract Gamma transition matrix posterior mean."""
    if 'Gamma' not in idata.posterior:
        return None
    
    Gamma = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
    return Gamma


def compute_dwell_times(Gamma):
    """
    Compute expected dwell times from transition matrix.
    Dwell time in state k = 1 / (1 - Gamma[k, k])
    """
    K = Gamma.shape[0]
    dwell_times = 1 / (1 - np.diag(Gamma) + 1e-12)
    return dwell_times


def generate_transition_table(pkl_path):
    """Generate Table 5: Transition analysis."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    idata = data['idata']
    res = data['res']
    
    Gamma = extract_gamma_matrix(idata)
    if Gamma is None:
        print("No transition matrix (K=1 static model)")
        return None
    
    K = Gamma.shape[0]
    
    # Build transition table
    rows = []
    for i in range(K):
        for j in range(K):
            rows.append({
                'Dataset': res['dataset'].upper(),
                'K': res['K'],
                'From_State': i,
                'To_State': j,
                'Probability': Gamma[i, j]
            })
    
    trans_df = pd.DataFrame(rows)
    
    # Dwell times
    dwell = compute_dwell_times(Gamma)
    dwell_df = pd.DataFrame({
        'Dataset': res['dataset'].upper(),
        'K': res['K'],
        'State': range(K),
        'Expected_Dwell_Weeks': dwell,
        'Persistence_Gamma_ii': np.diag(Gamma)
    })
    
    return trans_df, dwell_df


def export_transition_tables(trans_df, dwell_df, output_dir):
    """Export to CSV and LaTeX."""
    output_dir = Path(output_dir) / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transitions
    csv_path = output_dir / 'table5a_transitions.csv'
    trans_df.to_csv(csv_path, index=False, float_format='%.3f')
    print(f"Transitions CSV: {csv_path}")
    
    # Dwell times
    csv_path = output_dir / 'table5b_dwell_times.csv'
    dwell_df.to_csv(csv_path, index=False, float_format='.2f')
    print(f"Dwell times CSV: {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file', help='Path to SMC result .pkl')
    parser.add_argument('--out_dir', default='./04_results')
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("TABLE 5: TRANSITIONS & DWELL TIMES")
    print(f"{'='*80}")
    
    result = generate_transition_table(args.pkl_file)
    if result:
        trans_df, dwell_df = result
        
        print("\nTransition Matrix:")
        print(trans_df.pivot(index='From_State', columns='To_State', values='Probability').to_string())
        
        print("\nDwell Times:")
        print(dwell_df.to_string(index=False))
        
        export_transition_tables(trans_df, dwell_df, args.out_dir)
