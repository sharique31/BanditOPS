"""
real_data_experiment.py — same policy comparison as run_experiment.py,
but the environment's band transition probabilities (p01/p11) come from
real TSRD stare-mode data instead of a synthetic Markov chain.

Pipeline this plugs into:
    real .h5 PDWs
      -> tsrd_adapter.py            (PDW -> band x time activity grid)
      -> aggregate_transitions.py   (per-file counts -> dataset-wide counts)
      -> sanitize_transitions.py    (fallback prior for zero-evidence bands)
      -> env_ready_transitions.npz  <-- THIS SCRIPT LOADS THAT FILE
      -> env.py (SpectrumEnv with p01/p11 override)
      -> policies.py / belief.py / whittle.py  (UNCHANGED)
      -> metrics.py                 (same figures of merit as before)

Everything downstream of the .npz load is identical in spirit to
run_experiment.py, specifically so the two sets of plots/tables are an
apples-to-apples "synthetic vs real-calibrated" comparison for the slide
deck. Pd/Pfa remain the same synthetic detector model as before (TSRD's
ground truth doesn't simulate a physical receiver's imperfect detection --
see the note in aggregate_transitions.py's usage guidance) -- only the
underlying activity pattern is now real.

Usage:
    python real_data_experiment.py --transitions env_ready_transitions.npz
"""

import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import SpectrumEnv
from policies import FixedSweepPolicy, UCBPolicy, BeliefGreedyPolicy, WhittlePolicy
from metrics import compute_metrics
from whittle import WhittleIndexer

PD, PFA = 0.90, 0.05
T = 3000
N_REPLICATES = 8

def load_real_transitions(path):
    data = np.load(path)
    p01, p11 = data["p01"], data["p11"]
    assert len(p01) == len(p11)
    n_bands = len(p01)
    
    # FIX: Gracefully handle bands where p11 <= p01 instead of crashing
    if not np.all(p11 > p01):
        print("WARNING: Some bands have p11 <= p01. Applying fallback prior (0.15, 0.70).")
        bad_bands = np.where(p11 <= p01)[0]
        p01[bad_bands] = 0.15
        p11[bad_bands] = 0.70
        
    return n_bands, p01, p11

def run_one_episode(policy_factory, n_bands, episode_seed, p01, p11):
    env = SpectrumEnv(n_bands=n_bands, pd=PD, pfa=PFA, seed=episode_seed,
                       p01=p01, p11=p11)
    env.reset()
    policy = policy_factory(env)

    sensed = np.zeros(T, dtype=int)
    hit = np.zeros(T, dtype=bool)
    true_active = np.zeros((T, n_bands), dtype=bool)

    for t in range(T):
        band = policy.choose_band(t)
        result = env.step(band)
        policy.observe(result["sensed_band"], result["observed_hit"])

        sensed[t] = band
        hit[t] = result["observed_hit"]
        true_active[t] = result["true_active"]

    return (compute_metrics(sensed, hit, true_active),
            np.cumsum(hit & (true_active[np.arange(T), sensed])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=str, default="env_ready_transitions.npz",
                         help="Output of sanitize_transitions.py")
    args = parser.parse_args()

    n_bands, p01_real, p11_real = load_real_transitions(args.transitions)
    print(f"Loaded real transition probabilities for {n_bands} bands from {args.transitions}")

    policy_specs = [
        ("Fixed sweep (baseline)", lambda env: FixedSweepPolicy(n_bands)),
        ("UCB1", lambda env: UCBPolicy(n_bands)),
        ("Belief-greedy (real transitions)", lambda env: BeliefGreedyPolicy(
            n_bands, PD, PFA, p01=p01_real, p11=p11_real, learn_transitions=False)),
        ("Belief-greedy (learned transitions)", lambda env: BeliefGreedyPolicy(
            n_bands, PD, PFA, learn_transitions=True)),
        ("Whittle index (real transitions)", lambda env: WhittlePolicy(
            n_bands, PD, PFA, p01=p01_real, p11=p11_real, warmup=40)),
    ]

    results = {}
    curves = {}
    t0 = time.time()
    for name, factory in policy_specs:
        per_rep_metrics = []
        per_rep_curves = []
        for rep in range(N_REPLICATES):
            m, curve = run_one_episode(factory, n_bands, episode_seed=1000 + rep,
                                        p01=p01_real, p11=p11_real)
            per_rep_metrics.append(m)
            per_rep_curves.append(curve)
        results[name] = per_rep_metrics
        curves[name] = np.mean(per_rep_curves, axis=0)
        print(f"  done: {name}  ({time.time()-t0:.1f}s elapsed)")

    keys = ["intercept_rate", "coverage_ratio", "interception_ratio",
            "avg_intercept_latency_steps", "empirical_pd", "empirical_pfa"]
    header = f"{'Policy':38s} " + " ".join(f"{k:>14s}" for k in
        ["Intercept rate", "Coverage", "Streaks caught", "Avg latency", "Emp. Pd", "Emp. Pfa"])
    print("\n" + header)
    print("-" * len(header))
    summary_rows = []
    for name, _ in policy_specs:
        ms = results[name]
        row_vals = {k: np.nanmean([m[k] for m in ms]) for k in keys}
        row_std = {k: np.nanstd([m[k] for m in ms]) for k in keys}
        summary_rows.append((name, row_vals, row_std))
        print(f"{name:38s} " + " ".join(f"{row_vals[k]:14.4f}" for k in keys))

    plt.figure(figsize=(9, 5.5))
    for name, _ in policy_specs:
        plt.plot(curves[name], label=name, linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative true intercepts")
    plt.title(f"Cumulative intercepts over time -- REAL-DATA-CALIBRATED "
              f"({n_bands} bands, averaged over {N_REPLICATES} runs)")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("real_data_cumulative_intercepts.png", dpi=150)
    print("\nSaved plot: real_data_cumulative_intercepts.png")

    plt.figure(figsize=(8, 5))
    names = [name for name, _ in policy_specs]
    means = [np.mean([m["intercept_rate"] for m in results[n]]) for n in names]
    stds = [np.std([m["intercept_rate"] for m in results[n]]) for n in names]
    colors = ["#888888"] + ["#1f77b4"] * (len(names) - 1)
    plt.barh(names, means, xerr=stds, color=colors)
    plt.xlabel("Intercept rate (fraction of timesteps)")
    plt.title("Final intercept rate by policy -- REAL-DATA-CALIBRATED")
    plt.tight_layout()
    plt.savefig("real_data_intercept_rate_bar.png", dpi=150)
    print("Saved plot: real_data_intercept_rate_bar.png")

    return summary_rows

if __name__ == "__main__":
    main()