#!/usr/bin/env python3
"""
desc_stats.py
=============
Generate manuscript tables: Table 1 (comparative stats), Table 2 (tipping point).
Matches R: make_manuscript_table_1, make_manuscript_table_2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from scipy import stats


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def calc_stats(df, label):
    """Calculate summary statistics for Table 1."""
    return pd.DataFrame([{
        'Dataset': label,
        'Total_Obs_N': len(df),
        'Unique_Cust': df['customer_id'].nunique(),
        'Mean_Weekly_$': round(df['WeeklySpend'].mean(), 2),
        'Zero_Inflation_Pct': f"{(df['WeeklySpend'] == 0).mean() * 100:.1f}%",
        'Spend_Skewness': round(stats.skew(df['WeeklySpend']), 2),
        'Spend_Kurtosis': round(stats.kurtosis(df['WeeklySpend']), 2)
    }])


def make_table1(rfm_uci=None, rfm_cdnow=None):
    """Table 1: Comparative Summary Statistics."""
    rows = []
    if rfm_uci is not None:
        rows.append(calc_stats(rfm_uci, "UCI Retail"))
    if rfm_cdnow is not None:
        rows.append(calc_stats(rfm_cdnow, "CDNOW"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_table2(rfm_df):
    """Table 2: Threshold Tipping Point Evidence."""
    # Calculate p0 per customer
    cust_stats = rfm_df.groupby('customer_id')['WeeklySpend'].apply(lambda x: (x == 0).mean()).reset_index()
    cust_stats.columns = ['customer_id', 'p0']
    
    # Merge back
    rfm_df = rfm_df.merge(cust_stats, on='customer_id')
    
    # Define regimes
    def classify_regime(p0):
        if p0 < 0.50:
            return "Low Zero (Active)"
        elif p0 < 0.75:
            return "Mid Zero (Intermittent)"
        else:
            return "High Zero (Clumpy/Tipping)"
    
    rfm_df['Regime'] = rfm_df['p0'].apply(classify_regime)
    
    # Aggregate by regime
    table2 = rfm_df.groupby('Regime').agg({
        'WeekStart': 'count',
        'WeeklySpend': 'mean',
        'WeeklySpend': lambda x: (x == 0).mean(),
        'F_run': 'mean'
    }).reset_index()
    
    table2.columns = ['Regime', 'N_Weeks', 'Avg_Spend', 'Pr_Y_eq_0', 'Avg_Freq']
    table2['Avg_Spend'] = table2['Avg_Spend'].round(2)
    table2['Pr_Y_eq_0'] = table2['Pr_Y_eq_0'].round(3)
    table2['Avg_Freq'] = table2['Avg_Freq'].round(2)
    
    return table2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cdnow', 'uci', 'both'], default='both')
    args = parser.parse_args()
    
    config = load_config()
    data_dir = Path(config['paths']['data_dir'])
    tables_dir = Path(config['paths']['tables_dir'])
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    rfm_uci = None
    rfm_cdnow = None
    
    if args.dataset in ['uci', 'both']:
        uci_path = data_dir / 'uci_full.csv'
        if uci_path.exists():
            rfm_uci = pd.read_csv(uci_path, parse_dates=['WeekStart'])
    
    if args.dataset in ['cdnow', 'both']:
        cdnow_path = data_dir / 'cdnow_full.csv'
        if cdnow_path.exists():
            rfm_cdnow = pd.read_csv(cdnow_path, parse_dates=['WeekStart'])
    
    # Table 1
    table1 = make_table1(rfm_uci, rfm_cdnow)
    if not table1.empty:
        table1.to_csv(tables_dir / 'table1_desc_stats.csv', index=False)
        print("Table 1:")
        print(table1.to_string(index=False))
    
    # Table 2 (per dataset)
    for name, df in [('UCI', rfm_uci), ('CDNOW', rfm_cdnow)]:
        if df is not None:
            table2 = make_table2(df)
            table2.to_csv(tables_dir / f'table2_tipping_point_{name.lower()}.csv', index=False)
            print(f"\nTable 2 ({name}):")
            print(table2.to_string(index=False))
