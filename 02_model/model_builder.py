#!/usr/bin/env python3
"""
model_builder.py
================
Build HMM-Tweedie model with configurable options.
Assembles: emissions + forward algorithm + priors.
"""

import os
# Apple Silicon optimization (must be first)
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'
os.environ['PYTENSOR_METAL'] = '0'

import numpy as np
import pytensor.tensor as pt
import pymc as pm
from patsy import dmatrix

from .forward_algorithm import forward_algorithm_pytensor


def create_bspline_basis(x, df=3, degree=3):
    """Create B-spline basis for GAM."""
    x = np.asarray(x, dtype=np.float32).flatten()
    
    n_knots = df - degree + 1
    if n_knots > 1:
        knots = np.quantile(x, np.linspace(0, 1, n_knots)[1:-1]).tolist()
    else:
        knots = []
    
    formula = f"bs(x, knots={list(knots)}, degree={degree}, include_intercept=False)"
    basis = dmatrix(formula, {"x": x}, return_type='matrix')
    
    return np.asarray(basis, dtype=np.float32)


def gamma_logp_det(value, mu, phi):
    """Deterministic Gamma log-density."""
    alpha = mu / phi
    beta = 1.0 / phi
    return ((alpha - 1) * pt.log(value) - value * beta + 
            alpha * pt.log(beta) - pt.gammaln(alpha))


def make_model(data, K=3, state_specific_p=False, p_fixed=1.5,
               use_gam=True, gam_df=3, emission_type='tweedie',
               p_min=None, p_max=None, phi_sort=True,
               gamma_diag=1.0, shared_phi=True):
    """
    Build HMM-Tweedie model.
    
    Parameters
    ----------
    data : dict
        Panel data with N, T, y, mask, R, F, M
    K : int
        Number of states (K=1 for static)
    state_specific_p : bool
        State-specific power parameter
    p_fixed : float
        Fixed p value
    use_gam : bool
        Use GAM (B-splines) vs GLM (linear)
    gam_df : int
        GAM degrees of freedom
    emission_type : str
        'tweedie', 'poisson', or 'nbd'
    p_min, p_max : float
        Constrained p range
    phi_sort : bool
        Sort phi for identifiability
    gamma_diag : float
        Gamma prior concentration (1=uniform, higher=more persistence)
    shared_phi : bool
        Share phi across states (recommended)
        
    Returns
    -------
    pm.Model
        PyMC model object
    """
    N, T = data['N'], data['T']
    y, mask = data['y'], data['mask']
    R, F, M = data['R'], data['F'], data['M']
    
    # Precompute GAM bases
    if use_gam:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()
        
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        
        n_basis_R = basis_R.shape[1]
        n_basis_F = basis_F.shape[1]
        n_basis_M = basis_M.shape[1]
        
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={'customer': np.arange(N), 'state': np.arange(K)}) as model:
        
        # ---- Latent dynamics ----
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet('pi0', a=np.ones(K, dtype=np.float32))
            # Persistence prior
            gamma_off = 1.0
            gamma_a = np.eye(K, dtype=np.float32) * (gamma_diag - gamma_off) + gamma_off
            Gamma = pm.Dirichlet('Gamma', a=gamma_a, shape=(K, K))

        # ---- State parameters ----
        if K == 1:
            beta0_raw = pm.Normal('beta0_raw', 0, 1)
            beta0 = pm.Deterministic('beta0', beta0_raw)
        else:
            beta0_raw = pm.Normal('beta0_raw', 0, 1, shape=K)
            beta0 = pm.Deterministic('beta0', pt.sort(beta0_raw))

        # Phi: shared (recommended) or state-specific
        if K == 1 or shared_phi:
            phi_raw = pm.Exponential('phi_raw', lam=10.0)
            phi = pm.Deterministic('phi', phi_raw)
        else:
            phi_raw = pm.Exponential('phi_raw', lam=10.0, shape=K)
            phi = pm.Deterministic('phi', pt.sort(phi_raw) if phi_sort else phi_raw)

        # ---- RFM effects ----
        if use_gam:
            if K == 1:
                w_R = pm.Normal('w_R', 0, 1, shape=n_basis_R)
                w_F = pm.Normal('w_F', 0, 1, shape=n_basis_F)
                w_M = pm.Normal('w_M', 0, 1, shape=n_basis_M)
            else:
                w_R = pm.Normal('w_R', 0, 1, shape=(K, n_basis_R))
                w_F = pm.Normal('w_F', 0, 1, shape=(K, n_basis_F))
                w_M = pm.Normal('w_M', 0, 1, shape=(K, n_basis_M))
        else:
            if K == 1:
                betaR = pm.Normal('betaR', 0, 1)
                betaF = pm.Normal('betaF', 0, 1)
                betaM = pm.Normal('betaM', 0, 1)
            else:
                betaR = pm.Normal('betaR', 0, 1, shape=K)
                betaF = pm.Normal('betaF', 0, 1, shape=K)
                betaM = pm.Normal('betaM', 0, 1, shape=K)

        # ---- Power parameter p ----
        if p_min is not None and p_max is not None:
            p_raw = pm.Beta('p_raw', alpha=2, beta=2)
            p_val = p_min + p_raw * (p_max - p_min)
            p = pm.Deterministic('p', p_val if K == 1 else pt.stack([p_val] * K))
        elif K == 1:
            p_raw = pm.Beta('p_raw', alpha=2, beta=2)
            p = pm.Deterministic('p', 1.1 + p_raw * 0.8)
        elif state_specific_p:
            p_raw = pm.Beta('p_raw', alpha=2, beta=2, shape=K)
            p = pm.Deterministic('p', 1.1 + pt.sort(p_raw) * 0.8)
        else:
            p = pt.as_tensor_variable(np.array([p_fixed] * K, dtype=np.float32))

        # ---- Compute mu ----
        if use_gam:
            if K == 1:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
            else:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                mu = pt.exp(beta0[None, None, :] + effect_R + effect_F + effect_M)
        else:
            if K == 1:
                mu = pt.exp(beta0 + betaR * R + betaF * F + betaM * M)
            else:
                mu = pt.exp(beta0 + betaR * R[..., None] + 
                           betaF * F[..., None] + betaM * M[..., None])

        mu = pt.clip(mu, 1e-3, 1e6)

        # ---- CLV metrics ----
        if K > 1:
            gamma_diag = pt.diag(Gamma)
        else:
            gamma_diag = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
        
        pm.Deterministic('churn_risk', 1.0 - gamma_diag)
        pm.Deterministic('clv_proxy', pt.exp(beta0) / (1.0 - 0.95 * gamma_diag))

        # ---- Emission ----
        if emission_type == 'tweedie':
            log_emission = _tweedie_emission(y, mu, phi, p, mask, K)
        elif emission_type == 'poisson':
            log_emission = _poisson_emission(y, mu, mask, K)
        elif emission_type == 'nbd':
            log_emission = _nbd_emission(y, mu, mask, K)
        else:
            raise ValueError(f"Unknown emission: {emission_type}")

        # ---- Forward algorithm for marginal likelihood ----
        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            _, logp_cust = forward_algorithm_pytensor(log_emission, Gamma, pi0, mask)

        pm.Potential('loglike', pt.sum(logp_cust))
        pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))

    return model


def _tweedie_emission(y, mu, phi, p, mask, K):
    """ZIG-Tweedie emission."""
    if K == 1:
        exponent = 2.0 - p
        psi = pt.exp(-pt.pow(mu, exponent) / (phi * exponent))
        psi = pt.clip(psi, 1e-12, 1 - 1e-12)
        
        log_zero = pt.log(psi)
        log_pos = pt.log1p(-psi) + gamma_logp_det(y, mu, phi)
        log_emission = pt.where(y == 0, log_zero, log_pos)
        log_emission = pt.where(mask, log_emission, 0.0)
    else:
        p_exp = p[None, None, :]
        phi_exp = phi[None, None, :]
        exponent = 2.0 - p_exp
        
        psi = pt.exp(-pt.pow(mu, exponent) / (phi_exp * exponent))
        psi = pt.clip(psi, 1e-12, 1 - 1e-12)
        
        log_zero = pt.log(psi)
        y_exp = y[..., None]
        log_pos = pt.log1p(-psi) + gamma_logp_det(y_exp, mu, phi_exp)
        log_emission = pt.where(y_exp == 0, log_zero, log_pos)
        log_emission = pt.where(mask[:, :, None], log_emission, 0.0)
    
    return log_emission


def _poisson_emission(y, mu, mask, K):
    """Poisson emission (discrete approximation)."""
    if K == 1:
        y_r = pt.round(pt.clip(y, 0, 1e6))
        log_emission = pm.logp(pm.Poisson.dist(mu=mu), y_r)
        log_emission = pt.where(mask, log_emission, 0.0)
    else:
        y_exp = y[..., None]
        y_r = pt.round(pt.clip(y_exp, 0, 1e6))
        log_emission = pm.logp(pm.Poisson.dist(mu=mu), y_r)
        log_emission = pt.where(mask[:, :, None], log_emission, 0.0)
    return log_emission


def _nbd_emission(y, mu, mask, K):
    """Negative Binomial emission."""
    if K == 1:
        theta = pm.Exponential('theta', lam=1.0)
        y_r = pt.round(pt.clip(y, 0, 1e6))
        log_emission = pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=theta), y_r)
        log_emission = pt.where(mask, log_emission, 0.0)
    else:
        theta = pm.Exponential('theta', lam=1.0, shape=K)
        y_exp = y[..., None]
        y_r = pt.round(pt.clip(y_exp, 0, 1e6))
        log_emission = pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=theta), y_r)
        log_emission = pt.where(mask[:, :, None], log_emission, 0.0)
    return log_emission
