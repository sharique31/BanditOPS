"""
run_experiment.py — runs every policy against the same synthetic
environment structure and reports the figures of merit side by side.

Usage: python run_experiment.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import SpectrumEnv
from policies import FixedSweepPolicy, UCBPolicy, BeliefGreedyPolicy, WhittlePolicy
from metrics import compute_metrics
from whittle import WhittleIndexer

N_BANDS = 20
PD, PFA = 0.90, 0.05
T = 3000
N_REPLICATES = 8
BAND_SEED = 7  # fixed ground-truth band structure shared by every policy/replicate


def run_one_episode(policy_factory, episode_seed):
    env = SpectrumEnv(n_bands=N_BANDS, pd=PD, pfa=PFA, seed=episode_seed, band_seed=BAND_SEED)
    env.reset()
    policy = policy_factory(env)

    sensed = np.zeros(T, dtype=int)
    hit = np.zeros(T, dtype=bool)
    true_active = np.zeros((T, N_BANDS), dtype=bool)

    for t in range(T):
        band = policy.choose_band(t)
        result = env.step(band)
        policy.observe(result["sensed_band"], result["observed_hit"])

        sensed[t] = band
        hit[t] = result["observed_hit"]
        true_active[t] = result["true_active"]

    return compute_metrics(sensed, hit, true_active), np.cumsum(hit & (true_active[np.arange(T), sensed]))


def main():
    # Oracle transition probabilities come straight from a throwaway env
    # built with the same BAND_SEED -- this is "cheating" only in the sense
    # that a real deployment would have to learn these; the learned-transition
    # variant below shows that path instead.
    oracle_env = SpectrumEnv(n_bands=N_BANDS, pd=PD, pfa=PFA, seed=0, band_seed=BAND_SEED)

    policy_specs = [
        ("Fixed sweep (baseline)", lambda env: FixedSweepPolicy(N_BANDS)),
        ("UCB1", lambda env: UCBPolicy(N_BANDS)),
        ("Belief-greedy (oracle transitions)", lambda env: BeliefGreedyPolicy(
            N_BANDS, PD, PFA, p01=oracle_env.p01, p11=oracle_env.p11, learn_transitions=False)),
        ("Belief-greedy (learned transitions)", lambda env: BeliefGreedyPolicy(
            N_BANDS, PD, PFA, learn_transitions=True)),
        ("Whittle index (oracle transitions)", lambda env: WhittlePolicy(
            N_BANDS, PD, PFA, p01=oracle_env.p01, p11=oracle_env.p11, warmup=40)),
    ]

    results = {}
    curves = {}
    t0 = time.time()
    for name, factory in policy_specs:
        per_rep_metrics = []
        per_rep_curves = []
        for rep in range(N_REPLICATES):
            m, curve = run_one_episode(factory, episode_seed=1000 + rep)
            per_rep_metrics.append(m)
            per_rep_curves.append(curve)
        results[name] = per_rep_metrics
        curves[name] = np.mean(per_rep_curves, axis=0)
        print(f"  done: {name}  ({time.time()-t0:.1f}s elapsed)")

    # ---- Print comparison table ----
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

    # ---- Plot: cumulative true intercepts over time ----
    plt.figure(figsize=(9, 5.5))
    for name, _ in policy_specs:
        plt.plot(curves[name], label=name, linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative true intercepts")
    plt.title(f"Cumulative intercepts over time ({N_BANDS} bands, averaged over {N_REPLICATES} runs)")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("cumulative_intercepts.png", dpi=150)
    print("\nSaved plot: cumulative_intercepts.png")

    # ---- Bar chart: final intercept rate with error bars ----
    plt.figure(figsize=(8, 5))
    names = [name for name, _ in policy_specs]
    means = [np.mean([m["intercept_rate"] for m in results[n]]) for n in names]
    stds = [np.std([m["intercept_rate"] for m in results[n]]) for n in names]
    colors = ["#888888"] + ["#1f77b4"] * (len(names) - 1)
    plt.barh(names, means, xerr=stds, color=colors)
    plt.xlabel("Intercept rate (fraction of timesteps)")
    plt.title("Final intercept rate by policy")
    plt.tight_layout()
    plt.savefig("intercept_rate_bar.png", dpi=150)
    print("Saved plot: intercept_rate_bar.png")

    return summary_rows


if __name__ == "__main__":
    main()
