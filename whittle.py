"""
whittle.py — the core of the "Prioritize" layer.

Computes the Whittle index for a 2-state restless bandit arm (a band that
keeps evolving via its own Markov chain whether or not we sense it), given
its (p01, p11). This is the standard restless-bandit setup for spectrum
sensing: playing an arm reveals its true state and pays that state's reward
(1 if active, 0 if not); resting pays a fixed subsidy lambda and lets the
belief propagate through the Markov chain instead.

Rather than hand-typing a remembered closed-form formula (easy to get
subtly wrong and hard for a reader to double-check), this computes the
index NUMERICALLY: value-iterate the arm's belief-MDP for a grid of
candidate subsidies (lambda), and for each belief value on a discretised
grid, record the lambda at which the optimal action flips from "play" to
"rest". That flip point, by Whittle's definition, IS the index.
"""

import os
import numpy as np

def _interp(grid, values, x):
    return np.interp(x, grid, values)

def compute_whittle_index(p01, p11, n_grid=201, beta=0.95,
                         lambda_lo=-0.5, lambda_hi=1.5, n_lambda=241,
                         n_vi_iters=300):
    """
    Returns (belief_grid, whittle_index) — two arrays of length n_grid,
    covering belief in [0, 1] at the given transition probabilities.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    lambdas = np.linspace(lambda_lo, lambda_hi, n_lambda)

    play_is_better = np.zeros((n_lambda, n_grid), dtype=bool)
    for li, lam in enumerate(lambdas):
        V = np.zeros(n_grid)
        for _ in range(n_vi_iters):
            v_after_active = _interp(grid, V, p11)
            v_after_inactive = _interp(grid, V, p01)
            play_val = grid * (1 + beta*v_after_active) + (1-grid) * (0+beta*v_after_inactive)

            t_omega = grid*p11 + (1-grid)*p01
            rest_val = lam + beta*_interp(grid, V, t_omega)

            V_new = np.maximum(play_val, rest_val)
            if np.max(np.abs(V_new-V)) < 1e-9:
                V = V_new
                break
            V = V_new

        v_after_active = _interp(grid, V, p11)
        v_after_inactive = _interp(grid, V, p01)
        play_val = grid * (1 + beta*v_after_active) + (1-grid) * (0+beta*v_after_inactive)
        t_omega = grid*p11 + (1-grid)*p01
        rest_val = lam + beta*_interp(grid, V, t_omega)
        play_is_better[li] = play_val >= rest_val

    whittle = np.empty(n_grid)
    for gi in range(n_grid):
        column = play_is_better[:, gi]
        flip = np.where(~column)[0]
        whittle[gi] = lambdas[flip[0]] if len(flip) else lambda_hi

    return grid, whittle


class WhittleIndexer:
    """Caches a per-(p01,p11) index table and looks it up by belief."""
    
    def __init__(self, p01_array, p11_array, cache_dir="./whittle_cache"):
        self.tables = []
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        for p01, p11 in zip(p01_array, p11_array):
            grid, index = self._get_cached_index(p01, p11)
            self.tables.append((grid, index))

    def _get_cached_index(self, p01, p11):
        """Load from cache or compute and save to prevent slow recomputation."""
        key = f"p01_{p01:.6f}_p11_{p11:.6f}"
        cache_file = os.path.join(self.cache_dir, f"whittle_{key}.npz")
        
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            return data['grid'], data['index']
        
        # Compute fresh
        grid, index = compute_whittle_index(p01, p11)
        
        # Save to cache
        np.savez(cache_file, grid=grid, index=index)
        
        return grid, index

    def score(self, beliefs):
        return np.array([
            _interp(grid, index, beliefs[i])
            for i, (grid, index) in enumerate(self.tables)
        ])

if __name__ == "__main__":
    # --- Check 1: IID special case ---
    p = 0.4
    grid, index = compute_whittle_index(p01=p, p11=p)
    at_p = np.interp(p, grid, index)
    print(f"IID check: p={p}, Whittle index at belief={p} is {at_p:.3f} (expect ~{p})")
    assert abs(at_p-p) < 0.06, "IID sanity check failed"

    # --- Check 2: monotonicity ---
    grid, index = compute_whittle_index(p01=0.1, p11=0.8)
    diffs = np.diff(index)
    n_violations = int((diffs < -1e-6).sum())
    print(f"Monotonicity check: {n_violations} decreasing steps out of {len(diffs)}")
    assert n_violations <= 2, "index should be (near-)monotonic in belief"

    print("whittle.py self-tests passed")