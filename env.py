"""
env.py — Synthetic electronic-warfare spectrum environment.

This is a stand-in for the real Turing Synthetic Radar Dataset (TSRD), built
so the scheduler stack (belief tracking, UCB, Whittle index) can be developed
and demonstrated *before* real TSRD pulse trains are downloaded and reshaped
into this same band-vs-time interface. See README.md for the swap-in plan.

Ground truth per band: a 2-state Markov chain (0=inactive, 1=active), with
p11 > p01 for every band ("positively correlated" / bursty, not coin-flip
noise) — this is what gives an adaptive scheduler something real to exploit.
On top of that, a frequency-agile emitter injects activity into a random
band each timestep, independent of any band's own history — this component
is, by construction, much harder to predict, mirroring the problem
statement's distinction between persistent and frequency-agile threats.

The receiver can tune to exactly ONE band per timestep (the "instantaneous
bandwidth << total bandwidth" constraint). Detection is imperfect: Pd chance
of catching a truly active band, Pfa chance of a false alarm on an inactive
one.
"""

import numpy as np


class SpectrumEnv:
    def __init__(
        self,
        n_bands=20,
        pd=0.90,
        pfa=0.05,
        agile_prob=0.15,
        seed=None,
        band_seed=0,
        p01=None,
        p11=None,
    ):
        """
        p01/p11: optional arrays of length n_bands. If given, these REPLACE
        the synthetic band structure below with real, externally-supplied
        transition probabilities (e.g. aggregated from real TSRD stare-mode
        data via aggregate_transitions.py + sanitize_transitions.py). This
        is the swap-in point referenced in this file's module docstring.
        band_seed is ignored when p01/p11 are supplied, since there's no
        synthetic band structure left to seed.
        """
        self.n_bands = n_bands
        self.pd = pd
        self.pfa = pfa
        self.agile_prob = agile_prob
        self.rng = np.random.default_rng(seed)

        if p01 is not None or p11 is not None:
            assert p01 is not None and p11 is not None, \
                "must supply both p01 and p11 together, not just one"
            self.p11 = np.asarray(p11, dtype=float)
            self.p01 = np.asarray(p01, dtype=float)
            assert len(self.p11) == n_bands and len(self.p01) == n_bands, \
                f"p01/p11 length must match n_bands={n_bands}"
        else:
            # Band transition probabilities are fixed by band_seed so every
            # policy in a comparison run is tested against the *same* ground
            # truth environment structure, varying only the episode's random draws.
            band_rng = np.random.default_rng(band_seed)
            self.p11 = band_rng.uniform(0.55, 0.95, n_bands)  # persistence once active
            gap = band_rng.uniform(0.05, 0.25, n_bands)
            self.p01 = np.clip(self.p11 - gap, 0.02, 0.35)     # chance of turning on

        assert np.all(self.p11 > self.p01), "bands must be positively correlated"

        self.state = None  # true hidden state per band, set in reset()

    def stationary_prob(self):
        """P(active) in steady state, per band: p01 / (p01 + (1 - p11))."""
        return self.p01 / (self.p01 + (1 - self.p11))

    def reset(self):
        stat = self.stationary_prob()
        self.state = (self.rng.random(self.n_bands) < stat).astype(int)
        return self.state.copy()

    def step(self, band_to_sense):
        """
        Advance the true environment by one timestep and report what the
        receiver observes if it tunes to `band_to_sense`.

        Returns: dict with:
          true_active   - bool array, ground truth for EVERY band this step
                           (only available for offline metrics, never given
                           to a policy while it's running)
          sensed_band   - the band the receiver was tuned to
          observed_hit  - True/False, what the detector actually reported
        """
        # 1) Markov evolution of each band's own hidden state
        p_stay_active = np.where(self.state == 1, self.p11, self.p01)
        self.state = (self.rng.random(self.n_bands) < p_stay_active).astype(int)

        # 2) Frequency-agile emitter: independent of any band's own history
        if self.rng.random() < self.agile_prob:
            agile_band = self.rng.integers(self.n_bands)
            self.state[agile_band] = 1

        true_active = self.state.astype(bool).copy()

        # 3) Imperfect detection, only on the single band actually sensed
        if true_active[band_to_sense]:
            observed_hit = self.rng.random() < self.pd
        else:
            observed_hit = self.rng.random() < self.pfa

        return {
            "true_active": true_active,
            "sensed_band": band_to_sense,
            "observed_hit": bool(observed_hit),
        }
