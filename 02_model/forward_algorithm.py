#!/usr/bin/env python3
"""
forward_algorithm.py
====================
Vectorized forward algorithm for HMM marginal likelihood.
Core methods innovation: efficient computation of log p(y | theta).

Implements:
- Forward message passing (alpha recursion)
- Log-sum-exp stabilization for numerical stability
- Vectorized over N customers for GPU/CPU efficiency
"""

import numpy as np
from scipy.special import logsumexp


def forward_algorithm(log_emission, Gamma, pi0, mask=None):
    """
    Compute marginal likelihood via forward algorithm.
    
    Parameters
    ----------
    log_emission : ndarray, shape (N, T, K)
        Log emission probabilities: log p(y_it | s_t=k, theta)
    Gamma : ndarray, shape (K, K)
        Transition matrix: Gamma[j, k] = p(s_t=k | s_{t-1}=j)
    pi0 : ndarray, shape (K,)
        Initial state distribution
    mask : ndarray, shape (N, T), optional
        Boolean mask for valid observations (True = observed)
        
    Returns
    -------
    log_alpha : ndarray, shape (N, T, K)
        Forward messages: log p(s_t=k, y_{1:t} | theta)
    log_evidence : ndarray, shape (N,)
        Log marginal likelihood per customer: log p(y_i | theta)
    """
    N, T, K = log_emission.shape
    
    # Initialize
    log_alpha = np.zeros((N, T, K))
    
    # t=0: log alpha_0 = log pi0 + log emission_0
    log_alpha[:, 0, :] = np.log(pi0 + 1e-12)[None, :] + log_emission[:, 0, :]
    
    # Apply mask at t=0 if provided
    if mask is not None:
        log_alpha[:, 0, :] = np.where(mask[:, 0:1], log_alpha[:, 0, :], 0.0)
    
    # Forward recursion
    log_Gamma = np.log(Gamma + 1e-12)  # (K, K)
    
    for t in range(1, T):
        # Previous message: (N, K)
        prev = log_alpha[:, t-1, :]  # (N, K)
        
        # Transition: log sum_j [alpha_{t-1}(j) * Gamma(j, k)]
        # Vectorized: for each k, sum over j
        # Shape: (N, K) -> (N, K, 1) + (1, K, K) -> logsumexp over dim 1
        prev_expanded = prev[:, :, None]  # (N, K, 1)
        trans = logsumexp(prev_expanded + log_Gamma[None, :, :], axis=1)  # (N, K)
        
        # Add emission
        log_alpha[:, t, :] = trans + log_emission[:, t, :]
        
        # Apply mask
        if mask is not None:
            log_alpha[:, t, :] = np.where(mask[:, t:t+1], log_alpha[:, t, :], 0.0)
    
    # Marginal likelihood: log sum_k alpha_T(k)
    log_evidence = logsumexp(log_alpha[:, -1, :], axis=1)  # (N,)
    
    return log_alpha, log_evidence


def forward_algorithm_pytensor(log_emission, Gamma, pi0, mask=None):
    """
    PyTensor version for PyMC integration (symbolic).
    Same logic as numpy version but returns PyTensor graph.
    """
    import pytensor.tensor as pt
    
    N, T, K = log_emission.shape
    
    # t=0 initialization
    log_alpha_t = pt.log(pi0 + 1e-12)[None, :] + log_emission[:, 0, :]
    
    if mask is not None:
        log_alpha_t = pt.where(mask[:, 0:1], log_alpha_t, 0.0)
    
    # Store all alphas
    log_alphas = [log_alpha_t]
    log_Gamma = pt.log(Gamma + 1e-12)
    
    # Scan over time
    def step(log_alpha_prev, log_emit_t):
        # Transition
        prev_expanded = log_alpha_prev[:, :, None]  # (N, K, 1)
        trans = pt.logsumexp(prev_expanded + log_Gamma[None, :, :], axis=1)
        # Add emission
        log_alpha_t = trans + log_emit_t
        return log_alpha_t
    
    # Iterate manually (PyMC context)
    for t in range(1, T):
        log_alpha_t = step(log_alpha_t, log_emission[:, t, :])
        if mask is not None:
            log_alpha_t = pt.where(mask[:, t:t+1], log_alpha_t, 0.0)
        log_alphas.append(log_alpha_t)
    
    # Stack and compute evidence
    log_alpha = pt.stack(log_alphas, axis=1)  # (N, T, K)
    log_evidence = pt.logsumexp(log_alpha[:, -1, :], axis=1)
    
    return log_alpha, log_evidence


def viterbi_decode(log_emission, Gamma, pi0, mask=None):
    """
    Viterbi decoding: find most likely state sequence.
    
    Parameters
    ----------
    log_emission : ndarray, shape (N, T, K)
    Gamma : ndarray, shape (K, K)
    pi0 : ndarray, shape (K,)
    mask : ndarray, shape (N, T), optional
        
    Returns
    -------
    states : ndarray, shape (N, T)
        Most likely state sequence per customer
    log_prob : ndarray, shape (N,)
        Log probability of most likely path
    """
    N, T, K = log_emission.shape
    
    # Initialize
    log_delta = np.zeros((N, T, K))
    psi = np.zeros((N, T, K), dtype=int)
    
    # t=0
    log_delta[:, 0, :] = np.log(pi0 + 1e-12)[None, :] + log_emission[:, 0, :]
    
    # Recursion
    log_Gamma = np.log(Gamma + 1e-12)
    
    for t in range(1, T):
        for k in range(K):
            # Max over previous states
            scores = log_delta[:, t-1, :] + log_Gamma[:, k][None, :]
            psi[:, t, k] = np.argmax(scores, axis=1)
            log_delta[:, t, k] = np.max(scores, axis=1) + log_emission[:, t, k]
    
    # Backtracking
    states = np.zeros((N, T), dtype=int)
    states[:, -1] = np.argmax(log_delta[:, -1, :], axis=1)
    
    for t in range(T-2, -1, -1):
        for i in range(N):
            states[i, t] = psi[i, t+1, states[i, t+1]]
    
    log_prob = np.max(log_delta[:, -1, :], axis=1)
    
    return states, log_prob
