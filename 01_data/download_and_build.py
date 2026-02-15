#!/usr/bin/env python3
"""
download_and_build.py
=====================
Download raw datasets and build RFM panel with lagged features.
Matches R workflow: ingest -> weekly aggregation -> RFM engine
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import yaml


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def ingest_uci(path):
    """
    Ingest UCI Online Retail dataset.
    Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, 
             UnitPrice, CustomerID, Country
    """
    df = pd.read_csv(path, parse_dates=['InvoiceDate'])
    
    # Clean and transform
    df = df[df['CustomerID'].notna()]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]  # Remove cancellations
    
    df['Monetary'] = df['Quantity'] * df['UnitPrice']
    df['customer_id'] = df['CustomerID'].astype(str)
    df['date'] = pd.to_datetime(df['InvoiceDate'])
    df['WeekStart'] = df['date'].dt.to_period('W').dt.start_time
    
    # Aggregate to weekly
    weekly = df.groupby(['customer_id', 'WeekStart']).agg({
        'Monetary': 'sum',
        'InvoiceNo': 'nunique'
    }).reset_index()
    weekly.columns = ['customer_id', 'WeekStart', 'WeeklySpend', 'n_transactions']
    
    return weekly


def ingest_cdnow(path):
    """
    Ingest CDNOW dataset.
    Raw format: id date qty amount (space-delimited, no header)
    """
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['id', 'date', 'qty', 'amount'])
    
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['customer_id'] = df['id'].astype(str)
    df['WeekStart'] = df['date'].dt.to_period('W').dt.start_time
    
    # Aggregate to weekly
    weekly = df.groupby(['customer_id', 'WeekStart']).agg({
        'amount': 'sum',
        'qty': 'count'  # proxy for n_transactions
    }).reset_index()
    weekly.columns = ['customer_id', 'WeekStart', 'WeeklySpend', 'n_transactions']
    
    return weekly


def build_rfm_panel(weekly_df):
    """
    Core RFM Engine: Lagged feature engineering.
    Matches R: F_rolling (4-week MA), M_rolling (4-week MA), R_lagged (recency)
    """
    # Complete panel (fill missing weeks with zeros)
    panel = weekly_df.set_index(['customer_id', 'WeekStart']).unstack(fill_value=0)
    panel = panel.stack().reset_index()
    panel.columns = ['customer_id', 'WeekStart', 'n_transactions', 'WeeklySpend']
    
    # Sort
    panel = panel.sort_values(['customer_id', 'WeekStart'])
    
    # Compute RFM metrics per customer
    def compute_rfm_customer(group):
        group = group.copy()
        
        # Frequency: 4-week rolling mean (lagged)
        group['F_run'] = group['n_transactions'].rolling(window=4, min_periods=1).mean().shift(1)
        
        # Monetary: 4-week rolling mean (lagged)  
        group['M_run'] = group['WeeklySpend'].rolling(window=4, min_periods=1).mean().shift(1)
        
        # Recency: weeks since last activity
        was_active = (group['n_transactions'] > 0).shift(1).fillna(False)
        last_active = group.loc[was_active, 'WeekStart'].shift(1)
        
        # Forward fill last active date
        group['last_active'] = pd.Series(last_active.values, index=group.index).ffill()
        group['last_active'] = group['last_active'].fillna(group['WeekStart'].iloc[0])
        
        group['R_weeks'] = (group['WeekStart'] - group['last_active']).dt.days / 7
        
        # p0: initial period indicator
        group['p0_cust'] = (group['WeekStart'] == group['WeekStart'].min()).astype(float)
        
        return group
    
    panel = panel.groupby('customer_id', group_keys=False).apply(compute_rfm_customer)
    
    # Drop rows with missing lagged features
    panel = panel.dropna(subset=['F_run', 'M_run'])
    
    return panel[['customer_id', 'WeekStart', 'WeeklySpend', 'R_weeks', 'F_run', 'M_run', 'p0_cust']]


def download_cdnow(data_dir):
    """Download CDNOW dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "cdnow.zip"
    txt_path = data_dir / "CDNOW_sample.txt"
    
    if not txt_path.exists():
        if not zip_path.exists():
            print("Downloading CDNOW...")
            url = "http://www.brucehardie.com/datasets/CDNOW_sample.zip"
            urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        print(f"CDNOW ready: {txt_path}")
    
    return txt_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cdnow', 'uci', 'both'], default='both')
    parser.add_argument('--uci_path', type=str, default=None, help='Path to UCI Online Retail.csv')
    args = parser.parse_args()
    
    config = load_config()
    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset in ['cdnow', 'both']:
        print("Processing CDNOW...")
        txt_path = download_cdnow(data_dir)
        weekly = ingest_cdnow(txt_path)
        panel = build_rfm_panel(weekly)
        panel.to_csv(data_dir / "cdnow_full.csv", index=False)
        print(f"  Customers: {panel['customer_id'].nunique()}")
        print(f"  Weeks: {panel['WeekStart'].nunique()}")
        print(f"  Zero rate: {(panel['WeeklySpend']==0).mean():.2%}")
    
    if args.dataset in ['uci', 'both']:
        if args.uci_path:
            print("Processing UCI...")
            weekly = ingest_uci(args.uci_path)
            panel = build_rfm_panel(weekly)
            panel.to_csv(data_dir / "uci_full.csv", index=False)
            print(f"  Customers: {panel['customer_id'].nunique()}")
            print(f"  Weeks: {panel['WeekStart'].nunique()}")
            print(f"  Zero rate: {(panel['WeeklySpend']==0).mean():.2%}")
        else:
            print("UCI skipped: provide --uci_path")
