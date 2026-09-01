"""
metrics.py — figures of merit, matching the vocabulary from the original
problem statement (probability of detection, average intercept rate,
interception ratio, average intercept time error), computed from one
logged episode.

Expects three logged arrays for an episode of length T over n_bands:
sensed_band : int array, shape (T,) -- which band was tuned each step
hit : bool array, shape (T,) -- what the detector reported
true_active : bool array, shape (T, n_bands) -- ground truth, offline only
"""

import numpy as np

def compute_metrics(sensed_band, hit, true_active):
    T, n_bands = true_active.shape
    sensed_band = np.asarray(sensed_band)
    hit = np.asarray(hit)
    
    was_truly_active = true_active[np.arange(T), sensed_band]
    true_positive = was_truly_active & hit
    false_positive = (~was_truly_active) & hit
    
    intercept_rate = true_positive.mean()
    empirical_pd = hit[was_truly_active].mean() if was_truly_active.any() else float("nan")
    empirical_pfa = hit[~was_truly_active].mean() if (~was_truly_active).any() else float("nan")
    
    total_active_cells = true_active.sum() # every truly-active (band, time) instance
    coverage_ratio = true_positive.sum() / total_active_cells if total_active_cells else float("nan")
    
    # Streak analysis: contiguous active runs per band, and whether/when each one was first caught.
    streak_hit_flags = []
    streak_latencies = []
    
    for b in range(n_bands):
        active_col = true_active[:, b]
        t = 0
        while t < T:
            if not active_col[t]:
                t += 1
                continue
            start = t
            while t < T and active_col[t]:
                t += 1
            end = t # exclusive
            
            caught_at = None
            for tt in range(start, end):
                if sensed_band[tt] == b and hit[tt]:
                    caught_at = tt - start
                    break
                    
            streak_hit_flags.append(caught_at is not None)
            if caught_at is not None:
                streak_latencies.append(caught_at)
                
    streak_hit_flags = np.array(streak_hit_flags)
    interception_ratio = streak_hit_flags.mean() if len(streak_hit_flags) else float("nan")
    avg_intercept_latency = float(np.mean(streak_latencies)) if streak_latencies else float("nan")
    
    return {
        "intercept_rate": float(intercept_rate),
        "empirical_pd": float(empirical_pd),
        "empirical_pfa": float(empirical_pfa),
        "coverage_ratio": float(coverage_ratio),
        "interception_ratio": float(interception_ratio),
        "avg_intercept_latency_steps": avg_intercept_latency,
        "n_activity_streaks": int(len(streak_hit_flags)),
    }


def compute_regret_vs_oracle(sensed_band, hit, true_active, oracle_intercepts):
    """
    Compute regret compared to an oracle policy that knows the true state.
    Regret = (oracle_intercepts - our_intercepts) / total_timesteps
    """
    T = len(sensed_band)
    our_intercepts = np.sum(hit & true_active[np.arange(T), sensed_band])
    
    regret = (oracle_intercepts - our_intercepts) / T
    return {
        "regret": float(regret),
        "our_intercepts": int(our_intercepts),
        "oracle_intercepts": int(oracle_intercepts),
        "regret_percentage": float(100 * regret)
    }