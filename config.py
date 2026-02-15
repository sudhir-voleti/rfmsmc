#!/usr/bin/env python3
"""
config.py
=========
Configuration loader.
"""

import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    'paths': {
        'data_dir': './data',
        'results_dir': './04_results',
        'tables_dir': './04_results/tables',
        'figures_dir': './04_results/figures'
    },
    'model_defaults': {
        'K': 2,
        'use_gam': True,
        'gam_df': 3,
        'shared_phi': True,
        'p_fixed': 1.5
    },
    'smc_defaults': {
        'draws': 200,
        'chains': 4,
        'seed': 42
    }
}


def load_config(path='config.yaml'):
    """Load configuration from YAML."""
    if Path(path).exists():
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG


def update_config(updates, path='config.yaml'):
    """Update configuration file."""
    config = load_config(path)
    config.update(updates)
    with open(path, 'w') as f:
        yaml.dump(config, f)
