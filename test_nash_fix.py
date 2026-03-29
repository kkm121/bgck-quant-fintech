import numpy as np
from nash_equilibrium import NashRobustOptimizer

def test_distribution():
    optimizer = NashRobustOptimizer()
    
    # 10 tickers
    tickers = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
    
    # Expected returns (mu): T1 and T2 are very profitable, others are okay
    mu = np.array([1.0, 0.9, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, -0.1, 0.5])
    
    # Covariance (sigma): Identity for simplicity
    sigma = np.eye(10) * 0.1
    
    # Toxicity: T10 is very profitable but toxic
    toxicity = {"T10": 0.95}
    
    results = optimizer.execute_robust_allocation(mu, sigma, tickers, toxicity)
    
    print("\n--- Allocation Results ---")
    weights = results["weights"]
    for t, w in weights.items():
        if w > 0:
            print(f"{t}: {w*100:.1f}%")
    
    print(f"\nFailsafe Reason: {results['failsafe_reason']}")
    
    # Check if we have 5 assets with weights
    non_zero = [w for w in weights.values() if w > 0]
    print(f"Number of assets with weights: {len(non_zero)}")
    assert len(non_zero) <= 5, "Should have at most 5 assets"
    if len(non_zero) > 1:
        assert all(w >= 0.05 for w in non_zero), "Each non-zero weight should be at least 5-10%"

if __name__ == "__main__":
    test_distribution()
