"""
Microservice: Koopman Spectral Analysis
Architecture Component: Capital Fluid Dynamics
Mathematical/Theoretical Purpose:
Maps non-linear asset dynamics into a linear infinite-dimensional space using Koopman 
Operator Theory. Approximates the finite operator using Bagging Optimized Dynamic Mode 
Decomposition (BOPDMD). Translates the real component of continuous-time eigenvalues 
into an expected return vector representing momentum. Reconstructs a mathematically 
sound covariance matrix by taking the inner product of the dominant Koopman spatial modes, 
bypassing statistically fragile sample covariance estimates.
HARDENING UPDATE: Prevents "SVD did not converge in Linear Least Squares" crashes by 
stripping infinities, adding microscopic matrix jitter, and implementing classical fallbacks.
EIGENVALUE MISMATCH FIX: When BOPDMD auto-selects rank < n_assets (common for small
1-5 asset custom scans), the previous np.resize() cyclically repeated the same eigenvalue
for all assets, making them indistinguishable to the Reliability Gate. Both modes produced
equal weights. Now falls back to per-asset classical mean log-returns in this case.
"""
import numpy as np
import pandas as pd
from pydmd import BOPDMD

class KoopmanFluidDynamics:
    def __init__(self, svd_rank: int = 0, num_trials: int = 10):
        self.svd_rank = svd_rank
        self.num_trials = num_trials
        self.dmd_engine = BOPDMD(svd_rank=self.svd_rank, num_trials=self.num_trials)

    def fit_operator(self, data_matrix: np.ndarray, time_array: np.ndarray) -> None:
        self.dmd_engine.fit(data_matrix, time_array)

    def extract_expected_returns(self) -> np.ndarray:
        eigenvalues = self.dmd_engine.eigs
        continuous_eigenvalues = np.log(eigenvalues)
        real_components = np.real(continuous_eigenvalues)
        return real_components

    def reconstruct_covariance_matrix(self) -> np.ndarray:
        modes = self.dmd_engine.modes
        real_modes = np.real(modes)
        covariance_matrix = np.dot(real_modes, real_modes.T)
        
        d = np.sqrt(np.diag(covariance_matrix))
        covariance_matrix = covariance_matrix / np.outer(d, d)
        
        return covariance_matrix

    def execute_fluid_mapping(self, log_returns: pd.DataFrame) -> dict:
        # CRITICAL FIX 1: Strip infinities and NaNs that break SVD calculations
        clean_returns = log_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        # READ-ONLY FIX: Use .copy() to ensure Numpy owns the memory block and allows in-place addition
        data_matrix = clean_returns.to_numpy().T.copy()
        
        # CRITICAL FIX 2: Add microscopic ambient noise. SVD mathematically fails on perfectly 
        # flat matrices (e.g., when an asset halts trading or data is identically zero).
        data_matrix += np.random.normal(0, 1e-8, data_matrix.shape)
        
        n_assets    = data_matrix.shape[0]
        n_snapshots = data_matrix.shape[1]
        time_array  = np.arange(n_snapshots)

        # Classical fallback values — used both as last-resort and for eigenvalue mismatch blending.
        # Computed here so they are always available regardless of whether BOPDMD succeeds.
        classical_mu    = clean_returns.mean().to_numpy().copy()
        classical_sigma = clean_returns.cov().to_numpy().copy()
        classical_sigma = np.nan_to_num(classical_sigma, nan=0.0)
        np.fill_diagonal(classical_sigma, np.diagonal(classical_sigma) + 1e-6)
        
        try:
            self.fit_operator(data_matrix, time_array)
            mu_vector    = self.extract_expected_returns()
            sigma_matrix = self.reconstruct_covariance_matrix()

            # EIGENVALUE MISMATCH FIX:
            # BOPDMD auto-selects svd_rank. For small universes (1-5 asset custom scans)
            # this often resolves to rank=1 — returning only ONE eigenvalue. The old
            # np.resize() cyclically repeated that single value for all N assets, making
            # every ticker look identical to the Reliability Gate (same mu → same mask →
            # equal-weight fallback for BOTH aggressive and hedged → invisible bug).
            # Fix: when eigenvalue count != asset count, substitute per-asset classical
            # mean log-returns so each ticker carries its real momentum signal.
            if len(mu_vector) != n_assets:
                print(f"    -> [Koopman] Rank mismatch: {len(mu_vector)} eigenvalues "
                      f"for {n_assets} assets. Using per-asset classical returns.")
                mu_vector = classical_mu

        except Exception as e:
            print(f"Koopman DMD SVD Convergence Failure. Utilizing Classical Fallback: {e}")
            # CRITICAL FIX 3: Full classical fallback when SVD completely fails
            mu_vector    = classical_mu
            sigma_matrix = classical_sigma
            
        return {
            "expected_returns": mu_vector,
            "covariance_matrix": sigma_matrix,
            "assets": log_returns.columns.tolist()
        }