#!/usr/bin/env python3
"""
ablation_table.py
=================
Generate Table 3: Model comparison (ablation study).
Compares K=1,2,3,4 with different configurations.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def load_all_results(results_dir):
    """Load all SMC result pickles."""
    results_dir = Path(results_dir) / 'smc_fits'
    pkls = list(results_dir.glob("smc_*.pkl"))
    
    rows = []
    for pkl in sorted(pkls):
        try:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            res = data['res']
            res['file'] = pkl.name
            rows.append(res)
        except Exception as e:
            print(f"Failed to load {pkl}: {e}")
    
    return pd.DataFrame(rows)


def generate_ablation_table(results_dir, dataset=None):
    """
    Generate Table 3: Model comparison by complexity.
    
    Columns: Dataset, K, Type, Phi, p, Log-Ev, WAIC, LOO, Time
    """
    df = load_all_results(results_dir)
    
    if df.empty:
        print("No results found.")
        return pd.DataFrame()
    
    # Filter by dataset
    if dataset:
        df = df[df['dataset'] == dataset]
    
    # Select and rename columns
    table = pd.DataFrame({
        'Dataset': df['dataset'].str.upper(),
        'K': df['K'],
        'Type': df['use_gam'].map({True: 'GAM', False: 'GLM'}),
        'Phi': df['shared_phi'].map({True: 'Shared', False: 'State'}),
        'p': df.apply(lambda r: 'State' if r['state_specific_p'] else f"{r['p_fixed']}", axis=1),
        'N': df['N'],
        'D': df['draws'],
        'Log_Ev': df['log_evidence'],
        'WAIC': df['waic'],
        'LOO': df['loo'],
        'Time_min': df['time_min']
    })
    
    # Sort by complexity
    table['complexity'] = table['K'] * 10 + (table['p'] == 'State').astype(int)
    table = table.sort_values(['Dataset', 'complexity', 'Log_Ev'], 
                              ascending=[True, True, False])
    
    # Compute delta vs K=1 baseline per dataset
    table['Delta'] = np.nan
    for ds in table['Dataset'].unique():
        mask = table['Dataset'] == ds
        baseline = table.loc[mask & (table['K'] == 1), 'Log_Ev']
        if not baseline.empty:
            base_val = baseline.iloc[0]
            table.loc[mask, 'Delta'] = table.loc[mask, 'Log_Ev'] - base_val
    
    table = table.drop('complexity', axis=1)
    
    return table


def export_table(table, output_dir):
    """Export to CSV and LaTeX."""
    output_dir = Path(output_dir) / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = output_dir / 'table3_ablation.csv'
    table.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"CSV saved: {csv_path}")
    
    # LaTeX
    latex_path = output_dir / 'table3_ablation.tex'
    with open(latex_path, 'w') as f:
        f.write(table.to_latex(index=False, float_format='%.2f'))
    print(f"LaTeX saved: {latex_path}")
    
    return csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='./04_results')
    parser.add_argument('--dataset', choices=['cdnow', 'uci', None], default=None)
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("TABLE 3: MODEL COMPARISON (ABLATION)")
    print(f"{'='*80}")
    
    table = generate_ablation_table(args.results_dir, args.dataset)
    
    if not table.empty:
        print(table.to_string(index=False))
        export_table(table, args.results_dir)
