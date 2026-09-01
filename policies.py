"""
policies.py — the "Prioritize" + "Schedule" layers.

Four policies, sharing one interface (`choose_band(t) -> band_index`,
`observe(sensed_band, hit)` to update internal state):

  FixedSweepPolicy   - the open-loop baseline: round-robins every band in a
                        fixed order, exactly what the problem statement
                        describes as today's approach.
  UCBPolicy          - classic UCB1 over empirical hit-rate per band. Used
                        as the proposal's "cold start" method.
  BeliefGreedyPolicy - myopic: always sense the band with the highest
                        current belief. Simple, and a well-known strong
                        baseline for restless bandits.
  WhittlePolicy      - senses the band with the highest Whittle index
                        given its current belief; falls back to UCB for an
                        initial warm-up window per the proposal's rollout
                        plan ("implement UCB first, upgrade to Whittle
                        once Markov estimates are solid").
"""

import numpy as np
from belief import BeliefTracker
from whittle import WhittleIndexer

class FixedSweepPolicy:
    name = "Fixed sweep (baseline)"

    def __init__(self, n_bands, **kwargs):
        self.n_bands = n_bands
        self._next = 0

    def choose_band(self, t):
        b = self._next
        self._next = (self._next + 1) % self.n_bands
        return b

    def observe(self, sensed_band, hit):
        pass  # the baseline doesn't adapt -- that's the whole point


class UCBPolicy:
    name = "UCB1"

    def __init__(self, n_bands, c=1.4, **kwargs):
        self.n_bands = n_bands
        self.c = c
        self.counts = np.zeros(n_bands)
        self.sums = np.zeros(n_bands)
        self.t = 0

    def choose_band(self, t):
        self.t += 1
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited):
            return int(unvisited[0])
        means = self.sums / self.counts
        bonus = self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(means + bonus))

    def observe(self, sensed_band, hit):
        self.counts[sensed_band] += 1
        self.sums[sensed_band] += float(hit)


class BeliefGreedyPolicy:
    name = "Belief-greedy"

    def __init__(self, n_bands, pd, pfa, p01=None, p11=None, learn_transitions=False, **kwargs):
        self.tracker = BeliefTracker(n_bands, pd, pfa, p01, p11, learn_transitions)

    def choose_band(self, t):
        return int(np.argmax(self.tracker.belief))

    def observe(self, sensed_band, hit):
        self.tracker.update(sensed_band, hit)


class WhittlePolicy:
    name = "Whittle index"

    def __init__(self, n_bands, pd, pfa, p01, p11, warmup=None, learn_transitions=False, indexer=None, **kwargs):
        self.tracker = BeliefTracker(n_bands, pd, pfa, p01, p11, learn_transitions)
        
        # Building the table is the expensive part (value iteration per band).
        # Callers comparing multiple replicates should build one indexer and pass
        # it in rather than paying that cost every episode.
        self.indexer = indexer if indexer is not None else WhittleIndexer(p01, p11)
        self.warmup = warmup if warmup is not None else n_bands
        self._ucb_warmup = UCBPolicy(n_bands)
        self.n_bands = n_bands
        
        # FIX: Explicitly initialize the timestep counter
        self._t = 0

    def choose_band(self, t):
        self._t += 1
        if self._t <= self.warmup:
            return self._ucb_warmup.choose_band(t)
        scores = self.indexer.score(self.tracker.belief)
        return int(np.argmax(scores))

    def observe(self, sensed_band, hit):
        self.tracker.update(sensed_band, hit)
        if self._t <= self.warmup:
            self._ucb_warmup.observe(sensed_band, hit)