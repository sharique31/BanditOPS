"""
aggregate_transitions.py

Aggregates per-band Markov transition counts (N00, N01, N10, N11) across
MANY TSRD .h5 files into one robust, dataset-wide (p01, p11) pair per band.

Why not just average each file's p01/p11?
    A band that's barely active in a short/sparse file (few state-0 or
    state-1 observations) would get equal VOTING WEIGHT to a band with
    thousands of observations in a longer file, even though its estimate
    is far noisier. Aggregating raw counts first weights every file by
    how much evidence it actually contributed -- exactly the fix
    requested for tsrd_adapter.py's current per-file-only estimates.

This script does NOT reimplement PDW-to-grid conversion or the
transition-count math -- both already exist, are correct, and are
unit-tested indirectly via tsrd_adapter.py's own CLI. This script only
calls process_h5_file() per file and sums the counts it already returns
under result["transition_counts"].

Usage:
    python aggregate_transitions.py --dir data/stare/test_stare --bands 20
    python aggregate_transitions.py --dir data/scan/test_scan --pattern "config_*.h5"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tsrd_adapter import (
    DEFAULT_N_BANDS,
    DEFAULT_TIMESTEP_US,
    process_h5_file,
)


def find_h5_files(directory: str | Path, pattern: str = "*.h5") -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found:\n{directory.resolve()}")
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in:\n{directory.resolve()}"
        )
    return files


def aggregate_counts(
    per_file_counts: List[Dict[str, np.ndarray]],
    n_bands: int,
) -> Dict[str, np.ndarray]:
    """
    Sum raw N00/N01/N10/N11 across files (NOT an average of ratios),
    then compute the dataset-wide p01/p11 from the summed counts.
    """
    total_n00 = np.zeros(n_bands, dtype=np.float64)
    total_n01 = np.zeros(n_bands, dtype=np.float64)
    total_n10 = np.zeros(n_bands, dtype=np.float64)
    total_n11 = np.zeros(n_bands, dtype=np.float64)

    for counts in per_file_counts:
        total_n00 += counts["n00"]
        total_n01 += counts["n01"]
        total_n10 += counts["n10"]
        total_n11 += counts["n11"]

    zero_total = total_n00 + total_n01
    one_total = total_n10 + total_n11

    p01 = np.zeros(n_bands, dtype=np.float64)
    p11 = np.zeros(n_bands, dtype=np.float64)

    valid_zero = zero_total > 0
    valid_one = one_total > 0

    p01[valid_zero] = total_n01[valid_zero] / zero_total[valid_zero]
    p11[valid_one] = total_n11[valid_one] / one_total[valid_one]

    return {
        "p01": p01,
        "p11": p11,
        "n00": total_n00,
        "n01": total_n01,
        "n10": total_n10,
        "n11": total_n11,
        "zero_total": zero_total,
        "one_total": one_total,
        "bands_with_no_zero_obs": np.where(~valid_zero)[0],
        "bands_with_no_one_obs": np.where(~valid_one)[0],
    }


def process_directory(
    directory: str | Path,
    pattern: str = "*.h5",
    n_bands: int = DEFAULT_N_BANDS,
    timestep_us: float = DEFAULT_TIMESTEP_US,
    max_timesteps: int | None = None,
) -> Dict:
    files = find_h5_files(directory, pattern)

    per_file_counts = []
    per_file_summary = []
    reference_band_edges = None

    for file_path in files:
        result = process_h5_file(
            file_path=file_path,
            n_bands=n_bands,
            timestep_us=timestep_us,
            max_timesteps=max_timesteps,
        )

        # Guard: every file must be using the SAME band edges, or summed
        # counts would silently mix incompatible bands together.
        if reference_band_edges is None:
            reference_band_edges = result["band_edges"]
        elif not np.allclose(reference_band_edges, result["band_edges"], atol=1e-3):
            raise ValueError(
                f"Band edges mismatch in {file_path.name} -- "
                "files must share the same frequency range/band count "
                "to be aggregated together. "
                f"Expected {reference_band_edges}, got {result['band_edges']}."
            )

        per_file_counts.append(result["transition_counts"])
        per_file_summary.append({
            "file": file_path.name,
            "num_pdws": result["num_pdws"],
            "activity_fraction": float(np.mean(result["activity"])),
        })
        print(f"  processed {file_path.name}: "
              f"{result['num_pdws']} PDWs, "
              f"activity_fraction={np.mean(result['activity']):.6f}")

    aggregated = aggregate_counts(per_file_counts, n_bands)

    return {
        "n_bands": n_bands,
        "n_files": len(files),
        "band_edges": reference_band_edges,
        "per_file_summary": per_file_summary,
        **aggregated,
    }


def print_report(result: Dict) -> None:
    print()
    print("=" * 80)
    print(f"AGGREGATED TRANSITION PROBABILITIES  "
          f"({result['n_files']} files, {result['n_bands']} bands)")
    print("=" * 80)

    p01, p11 = result["p01"], result["p11"]
    band_edges = result["band_edges"]

    print("\nBand |  Freq range (MHz)      |     p01 |     p11 | indexable(p11>p01)")
    print("-" * 78)
    non_indexable = []
    for b in range(result["n_bands"]):
        indexable = p11[b] > p01[b]
        if not indexable:
            non_indexable.append(b)
        print(f"{b:4d} | {band_edges[b]:9.1f}-{band_edges[b+1]:9.1f} | "
              f"{p01[b]:7.4f} | {p11[b]:7.4f} | {'yes' if indexable else 'NO'}")

    if len(result["bands_with_no_zero_obs"]):
        print(f"\nWARNING: bands with NO observed state-0 timesteps "
              f"(p01 defaulted to 0, treat with caution): "
              f"{result['bands_with_no_zero_obs'].tolist()}")
    if len(result["bands_with_no_one_obs"]):
        print(f"WARNING: bands with NO observed state-1 timesteps "
              f"(p11 defaulted to 0, treat with caution): "
              f"{result['bands_with_no_one_obs'].tolist()}")
    if non_indexable:
        print(f"\nWARNING: bands where p11 <= p01 (NOT positively correlated -- "
              f"env.py's assert would fail on these as-is): {non_indexable}")
        print("  These need a fallback (e.g. clip, or drop from the Whittle-eligible "
              "set) before feeding into env.py/policies.py.")


def save_results(result: Dict, output_prefix: str | Path) -> None:
    output_prefix = Path(output_prefix)

    np.savez(
        output_prefix.with_suffix(".npz"),
        p01=result["p01"],
        p11=result["p11"],
        band_edges=result["band_edges"],
        n00=result["n00"], n01=result["n01"],
        n10=result["n10"], n11=result["n11"],
    )

    json_payload = {
        "n_bands": result["n_bands"],
        "n_files": result["n_files"],
        "band_edges_mhz": result["band_edges"].tolist(),
        "p01": result["p01"].tolist(),
        "p11": result["p11"].tolist(),
        "counts": {
            "n00": result["n00"].tolist(),
            "n01": result["n01"].tolist(),
            "n10": result["n10"].tolist(),
            "n11": result["n11"].tolist(),
        },
        "bands_with_no_zero_obs": result["bands_with_no_zero_obs"].tolist(),
        "bands_with_no_one_obs": result["bands_with_no_one_obs"].tolist(),
        "per_file_summary": result["per_file_summary"],
    }
    with open(output_prefix.with_suffix(".json"), "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\nSaved: {output_prefix.with_suffix('.npz')}")
    print(f"Saved: {output_prefix.with_suffix('.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate TSRD transition counts across many .h5 files."
    )
    parser.add_argument("--dir", type=str, required=True,
                         help="Directory containing .h5 files.")
    parser.add_argument("--pattern", type=str, default="*.h5",
                         help="Glob pattern for files. Default: *.h5")
    parser.add_argument("--bands", type=int, default=DEFAULT_N_BANDS)
    parser.add_argument("--timestep-us", type=float, default=DEFAULT_TIMESTEP_US)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--out", type=str, default="aggregated_transitions",
                         help="Output file prefix (no extension). "
                              "Writes <prefix>.npz and <prefix>.json")
    args = parser.parse_args()

    result = process_directory(
        directory=args.dir,
        pattern=args.pattern,
        n_bands=args.bands,
        timestep_us=args.timestep_us,
        max_timesteps=args.max_timesteps,
    )
    print_report(result)
    save_results(result, args.out)


if __name__ == "__main__":
    main()
