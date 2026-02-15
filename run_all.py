#!/usr/bin/env python3
"""
run_all.py
==========
Master execution script for RFM-HMM-SMC pipeline.
Runs stages 1-6 sequentially or individually.

Usage:
    python run_all.py --stage all --dataset cdnow
    python run_all.py --stage fit --dataset cdnow --K 2
    python run_all.py --stage tables --dataset cdnow
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, update_config


def stage1_data(dataset):
    """Stage 1: Download and build RFM panels."""
    print(f"\n{'='*80}")
    print("STAGE 1: DATA DOWNLOAD & PREPARATION")
    print(f"{'='*80}")
    
    from data.download_and_build import download_cdnow, ingest_cdnow, build_rfm_panel
    from data.download_and_build import ingest_uci
    
    config = load_config()
    data_dir = config['paths']['data_dir']
    
    if dataset in ['cdnow', 'both']:
        print("\nProcessing CDNOW...")
        txt_path = download_cdnow(data_dir)
        weekly = ingest_cdnow(txt_path)
        panel = build_rfm_panel(weekly)
        panel.to_csv(Path(data_dir) / "cdnow_full.csv", index=False)
        print(f"  Saved: {data_dir}/cdnow_full.csv")
        print(f"  Customers: {panel['customer_id'].nunique()}")
        print(f"  Weeks: {panel['WeekStart'].nunique()}")
        print(f"  Zero rate: {(panel['WeeklySpend']==0).mean():.2%}")
    
    if dataset in ['uci', 'both']:
        print("\nProcessing UCI...")
        uci_path = Path(data_dir) / "Online Retail.xlsx"  # or .csv
        if uci_path.exists():
            weekly = ingest_uci(uci_path)
            panel = build_rfm_panel(weekly)
            panel.to_csv(Path(data_dir) / "uci_full.csv", index=False)
            print(f"  Saved: {data_dir}/uci_full.csv")
        else:
            print(f"  UCI data not found at {uci_path}")
            print("  Please download UCI Online Retail dataset manually")


def stage1_stats(dataset):
    """Generate Tables 1 and 2."""
    print(f"\n{'='*80}")
    print("STAGE 1b: DESCRIPTIVE STATISTICS")
    print(f"{'='*80}")
    
    from data.desc_stats import make_table1, make_table2
    import pandas as pd
    
    config = load_config()
    data_dir = config['paths']['data_dir']
    tables_dir = config['paths']['tables_dir']
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    
    datasets = []
    if dataset in ['uci', 'both']:
        path = Path(data_dir) / 'uci_full.csv'
        if path.exists():
            datasets.append(('UCI', pd.read_csv(path, parse_dates=['WeekStart'])))
    
    if dataset in ['cdnow', 'both']:
        path = Path(data_dir) / 'cdnow_full.csv'
        if path.exists():
            datasets.append(('CDNOW', pd.read_csv(path, parse_dates=['WeekStart'])))
    
    # Table 1
    if datasets:
        table1 = pd.concat([make_table1(df, name) for name, df in datasets], ignore_index=True)
        table1.to_csv(Path(tables_dir) / 'table1_desc_stats.csv', index=False)
        print("\nTable 1 (Descriptive Statistics):")
        print(table1.to_string(index=False))
    
    # Table 2
    for name, df in datasets:
        table2 = make_table2(df)
        table2.to_csv(Path(tables_dir) / f'table2_tipping_point_{name.lower()}.csv', index=False)
        print(f"\nTable 2 ({name}):")
        print(table2.to_string(index=False))


def stage2_model_fit(dataset, K=None, quick_test=False):
    """Stage 2-3: Fit HMM model with SMC."""
    print(f"\n{'='*80}")
    print("STAGE 2-3: MODEL FITTING")
    print(f"{'='*80}")
    
    from inference.smc_runner import run_smc, quick_test_fit
    from data.download_and_build import build_panel_data
    import pandas as pd
    
    config = load_config()
    data_dir = config['paths']['data_dir']
    
    # Determine K
    if K is None:
        K = 2 if dataset == 'cdnow' else 3
    
    # Load data
    csv_path = Path(data_dir) / f"{dataset}_full.csv"
    if not csv_path.exists():
        print(f"Data not found: {csv_path}")
        print("Run Stage 1 first: python run_all.py --stage data")
        return
    
    # Build panel
    df = pd.read_csv(csv_path, parse_dates=['WeekStart'])
    
    # Subsample for quick test
    n_cust = 50 if quick_test else None
    
    # Build panel (simplified - would need full implementation)
    from data.download_and_build import build_rfm_panel
    
    # Group to weekly and build RFM
    weekly = df.groupby(['customer_id', 'WeekStart']).agg({
        'WeeklySpend': 'sum',
        'n_transactions': 'sum' if 'n_transactions' in df.columns else 'size'
    }).reset_index()
    
    panel_df = build_rfm_panel(weekly)
    
    if n_cust:
        custs = panel_df['customer_id'].unique()[:n_cust]
        panel_df = panel_df[panel_df['customer_id'].isin(custs)]
    
    # Convert to numpy arrays
    pivot = panel_df.pivot(index='customer_id', columns='WeekStart', values='WeeklySpend')
    y = pivot.fillna(0).values.astype(np.float32)
    mask = ~pivot.isna().values
    
    # Get R, F, M (simplified)
    R = np.zeros_like(y)  # Placeholder
    F = np.zeros_like(y)
    M = np.zeros_like(y)
    
    data = {
        'N': y.shape[0],
        'T': y.shape[1],
        'y': y,
        'mask': mask,
        'R': R, 'F': F, 'M': M
    }
    
    print(f"Panel: N={data['N']}, T={data['T']}")
    
    # Run fit
    if quick_test:
        pkl_path, res = quick_test_fit(data, K=K, dataset=dataset)
    else:
        pkl_path, res = run_smc(
            data=data,
            K=K,
            dataset=dataset,
            shared_phi=True,
            p_fixed=1.5,
            use_gam=True,
            draws=200 if quick_test else 1000,
            chains=4
        )
    
    return pkl_path


def stage5_tables(dataset):
    """Stage 5: Generate result tables."""
    print(f"\n{'='*80}")
    print("STAGE 5: POST-PROCESSING & TABLES")
    print(f"{'='*80}")
    
    from postproc.ablation_table import generate_ablation_table, export_table
    
    config = load_config()
    results_dir = config['paths']['results_dir']
    
    # Table 3: Ablation
    table3 = generate_ablation_table(results_dir, dataset)
    if not table3.empty:
        print("\nTable 3 (Ablation):")
        print(table3.to_string(index=False))
        export_table(table3, results_dir)
    
    # Find best model for segmentation
    best = table3.loc[table3['Log_Ev'].idxmax()]
    print(f"\nBest model: K={best['K']}, Log-Ev={best['Log_Ev']:.2f}")


def stage6_diagnostics(pkl_file):
    """Stage 6: Diagnostics and PPC."""
    print(f"\n{'='*80}")
    print("STAGE 6: DIAGNOSTICS")
    print(f"{'='*80}")
    
    from diagnostics.convergence import generate_convergence_report
    
    generate_convergence_report(pkl_file)


def main():
    parser = argparse.ArgumentParser(
        description='RFM-HMM-SMC: Master execution script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--stage', required=True,
                       choices=['all', 'data', 'stats', 'fit', 'tables', 'diagnostics'],
                       help='Which stage to run')
    parser.add_argument('--dataset', default='cdnow', choices=['cdnow', 'uci', 'both'])
    parser.add_argument('--K', type=int, default=None, help='Number of states (default: 2 cdnow, 3 uci)')
    parser.add_argument('--quick', action='store_true', help='Quick test with N=50, D=50')
    parser.add_argument('--pkl', type=str, help='PKL file for diagnostics stage')
    
    args = parser.parse_args()
    
    # Ensure seed
    import numpy as np
    np.random.seed(42)
    
    if args.stage == 'all':
        stage1_data(args.dataset)
        stage1_stats(args.dataset)
        pkl = stage2_model_fit(args.dataset, args.K, args.quick)
        stage5_tables(args.dataset)
        if pkl:
            stage6_diagnostics(pkl)
    
    elif args.stage == 'data':
        stage1_data(args.dataset)
    
    elif args.stage == 'stats':
        stage1_stats(args.dataset)
    
    elif args.stage == 'fit':
        stage2_model_fit(args.dataset, args.K, args.quick)
    
    elif args.stage == 'tables':
        stage5_tables(args.dataset)
    
    elif args.stage == 'diagnostics':
        if not args.pkl:
            print("Error: --pkl required for diagnostics stage")
            sys.exit(1)
        stage6_diagnostics(args.pkl)
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
