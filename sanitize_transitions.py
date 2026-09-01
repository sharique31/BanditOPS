"""
sanitize_transitions.py

Post-processes aggregate_transitions.py's output for bands with
insufficient real evidence (zero observed state-1 timesteps, or zero
observed state-0 timesteps) across the whole aggregated dataset.

Rather than leaving these at p01=p11=0.0 (which violates env.py's
p11 > p01 assumption and can't be used to build a Whittle table),
this falls back explicitly to the same mildly-informative prior
belief.py already uses when learn_transitions=True (p01=0.15, p11=0.70).

This does NOT touch bands that have real, sufficient data -- only the
flagged ones. Every fallback is recorded in the output JSON/NPZ so
nothing is silently invented.

Usage:
    python sanitize_transitions.py --in aggregated_transitions.json \
                                    --out env_ready_transitions
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Matches belief.py's BeliefTracker prior exactly -- see that file's
# "mildly-informative prior" comment.
FALLBACK_P01 = 0.15
FALLBACK_P11 = 0.70


def sanitize(data: dict, fallback_p01: float, fallback_p11: float) -> dict:
    p01 = np.array(data["p01"], dtype=np.float64)
    p11 = np.array(data["p11"], dtype=np.float64)
    n_bands = data["n_bands"]

    no_zero_obs = set(data.get("bands_with_no_zero_obs", []))
    no_one_obs = set(data.get("bands_with_no_one_obs", []))
    insufficient = sorted(no_zero_obs | no_one_obs)

    # Also catch the general non-indexable case (p11 <= p01) even if it
    # arose some other way, so nothing slips through to env.py's assert.
    non_indexable = set(np.where(p11 <= p01)[0].tolist())
    insufficient = sorted(set(insufficient) | non_indexable)

    fallback_applied = []
    for b in insufficient:
        fallback_applied.append({
            "band": b,
            "original_p01": float(p01[b]),
            "original_p11": float(p11[b]),
            "reason": (
                "no_state1_observations" if b in no_one_obs else
                "no_state0_observations" if b in no_zero_obs else
                "p11<=p01_non_indexable"
            ),
        })
        p01[b] = fallback_p01
        p11[b] = fallback_p11

    assert np.all(p11 > p01), "sanitized p11/p01 still violate p11 > p01 somewhere"

    return {
        "n_bands": n_bands,
        "band_edges_mhz": data["band_edges_mhz"],
        "p01": p01.tolist(),
        "p11": p11.tolist(),
        "fallback_applied": fallback_applied,
        "fallback_prior": {"p01": fallback_p01, "p11": fallback_p11},
        "source_n_files": data.get("n_files"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="out_prefix", required=True)
    parser.add_argument("--fallback-p01", type=float, default=FALLBACK_P01)
    parser.add_argument("--fallback-p11", type=float, default=FALLBACK_P11)
    args = parser.parse_args()

    with open(args.infile) as f:
        data = json.load(f)

    result = sanitize(data, args.fallback_p01, args.fallback_p11)

    out_prefix = Path(args.out_prefix)
    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez(
        out_prefix.with_suffix(".npz"),
        p01=np.array(result["p01"]),
        p11=np.array(result["p11"]),
        band_edges=np.array(result["band_edges_mhz"]),
    )

    print(f"Bands given fallback prior ({args.fallback_p01}/{args.fallback_p11}): "
          f"{[f['band'] for f in result['fallback_applied']]}")
    for f in result["fallback_applied"]:
        print(f"  band {f['band']:2d}: {f['reason']} "
              f"(was p01={f['original_p01']:.4f}, p11={f['original_p11']:.4f})")
    print(f"\nSaved: {out_prefix.with_suffix('.json')}")
    print(f"Saved: {out_prefix.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
