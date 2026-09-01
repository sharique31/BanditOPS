"""
belief.py — the "Believe" layer.

Keeps a live P(active) belief for every band, every timestep, whether or not
that band was actually sensed this turn:
- If sensed: Bayes-update the belief using the detector's Pd/Pfa, given the
  observed hit/miss (a genuine confidence-scored update, not a hard
  reset to 0 or 1 — the detector is imperfect, so a "miss" doesn't mean
  certainly inactive).
- Every band, every timestep: propagate one step through the Markov chain
  (Chapman-Kolmogorov), so un-sensed bands' beliefs still decay toward
  their stationary probability over time instead of going stale.

Transition probabilities (p01, p11) can be supplied (oracle) or estimated
online from accumulated hit/miss history, per the proposal's "estimated
online from observed hit/miss history" design.
"""

import numpy as np

class BeliefTracker:
    def __init__(self, n_bands, pd, pfa, p01=None, p11=None, learn_transitions=False):
        self.n_bands = n_bands
        self.pd = pd
        self.pfa = pfa
        self.learn_transitions = learn_transitions
        
        if learn_transitions:
            # Start from a mildly-informative prior; every band looks the
            # same until we've actually watched it for a while.
            self.p01 = np.full(n_bands, 0.15)
            self.p11 = np.full(n_bands, 0.70)
            
            # Counts for online transition-probability estimation:
            self._from0_total = np.zeros(n_bands)
            self._from0_to1 = np.zeros(n_bands)
            self._from1_total = np.zeros(n_bands)
            self._from1_to1 = np.zeros(n_bands)
            self._last_known_state = np.full(n_bands, -1) # -1 = unknown
        else:
            assert p01 is not None and p11 is not None
            self.p01 = np.asarray(p01, dtype=float)
            self.p11 = np.asarray(p11, dtype=float)
            
        self.belief = np.full(n_bands, 0.3) # neutral starting belief

    def _propagate(self):
        """One Markov step for every band's belief (Chapman-Kolmogorov)."""
        self.belief = self.belief * self.p11 + (1 - self.belief) * self.p01

    def update(self, sensed_band, observed_hit):
        """Call once per timestep with what the environment reported."""
        b = sensed_band
        prior = self.belief[b]
        
        # Bayes update
        if observed_hit:
            post = (self.pd * prior) / (self.pd * prior + self.pfa * (1 - prior) + 1e-12)
        else:
            post = ((1 - self.pd) * prior) / ((1 - self.pd) * prior + (1 - self.pfa) * (1 - prior) + 1e-12)
            
        self.belief[b] = post
        
        if self.learn_transitions:
            self._update_transition_estimate(b, post)
            
        # Advance every band (including the one just sensed) one Markov step
        self._propagate()

    def _update_transition_estimate(self, b, posterior_state_estimate):
        """
        We don't have the *true* state, only a posterior belief. As a
        practical online estimator, we treat a confident posterior
        (>0.8 or <0.2) as a hard label and accumulate empirical transition
        counts between consecutive *confident* observations of the same band.
        """
        confident_now = posterior_state_estimate > 0.8 or posterior_state_estimate < 0.2
        hard_now = int(posterior_state_estimate > 0.5)
        
        if confident_now and self._last_known_state[b] != -1:
            prev = self._last_known_state[b]
            if prev == 0:
                self._from0_total[b] += 1
                self._from0_to1[b] += hard_now
            else:
                self._from1_total[b] += 1
                self._from1_to1[b] += hard_now
                
        if confident_now:
            self._last_known_state[b] = hard_now
            
        # Refresh the estimate actually used for propagation/scoring
        # FIX: Increased threshold to 5 for stability, and strictly enforce p11 > p01
        if self._from0_total[b] >= 5:
            self.p01[b] = self._from0_to1[b] / self._from0_total[b]
        if self._from1_total[b] >= 5:
            self.p11[b] = self._from1_to1[b] / self._from1_total[b]
            
        # Keep positively-correlated / avoid degenerate 0 or 1 estimates
        self.p11[b] = np.clip(self.p11[b], 0.05, 0.97)
        # FIX: Ensure p01 is always strictly less than p11 by at least 0.05
        self.p01[b] = np.clip(self.p01[b], 0.01, max(0.02, self.p11[b] - 0.05))