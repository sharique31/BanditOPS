"""
tsrd_adapter.py

Adapter for converting TSRD HDF5 PDW files into a band-vs-time
boolean activity grid.

Input PDW features:
    [ToA, Frequency, PulseWidth, AoA, Amplitude]

Output:
    activity[timestep, band] = True if at least one PDW falls
    into that time-frequency cell.

Also provides empirical p01 and p11 transition probabilities
for each frequency band.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DEFAULT_N_BANDS = 20
DEFAULT_TIMESTEP_US = 1000.0  # 1 millisecond


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_h5_file(
    file_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load PDW data, labels, and useful metadata from a TSRD HDF5 file.

    Parameters
    ----------
    file_path:
        Path to an HDF5 file.

    Returns
    -------
    data:
        numpy array of shape (N, 5)

    labels:
        numpy array of shape (N,)

    metadata:
        Dictionary containing extracted metadata.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found:\n{file_path.resolve()}"
        )

    with h5py.File(file_path, "r") as h5_file:

        if "data" not in h5_file:
            raise KeyError("HDF5 file does not contain a 'data' dataset.")

        if "labels" not in h5_file:
            raise KeyError("HDF5 file does not contain a 'labels' dataset.")

        data = h5_file["data"][:]
        labels = h5_file["labels"][:].reshape(-1)

        if data.ndim != 2 or data.shape[1] != 5:
            raise ValueError(
                "Expected data with shape (N, 5), "
                f"but got {data.shape}."
            )

        metadata = {}

        if "metadata" in h5_file:
            metadata_group = h5_file["metadata"]

            # Root metadata attributes
            for key, value in metadata_group.attrs.items():
                metadata[key] = decode_value(value)

            # Feature names
            if "feature_names" in metadata_group:
                feature_names = metadata_group["feature_names"][:]
                metadata["feature_names"] = [
                    decode_value(name)
                    for name in feature_names
                ]

            # Receiver metadata
            if "receiver" in metadata_group:
                receiver = metadata_group["receiver"]

                receiver_info = {}

                for key, value in receiver.attrs.items():
                    receiver_info[key] = decode_value(value)

                for dataset_name in receiver.keys():
                    obj = receiver[dataset_name]

                    if isinstance(obj, h5py.Dataset):
                        receiver_info[dataset_name] = obj[:]

                metadata["receiver"] = receiver_info

    return data, labels, metadata


def decode_value(value):
    """
    Decode HDF5 byte strings into normal Python strings.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.bytes_):
        return value.decode("utf-8")

    return value


# ---------------------------------------------------------------------
# FREQUENCY RANGE
# ---------------------------------------------------------------------

def get_frequency_range(
    data: np.ndarray,
    metadata: Dict,
) -> Tuple[float, float]:
    """
    Determine the frequency range for band construction.

    Priority:
        1. metadata.receiver.freq_range_mhz
        2. observed PDW frequency minimum and maximum
    """

    receiver = metadata.get("receiver", {})

    if "freq_range_mhz" in receiver:
        freq_range = np.asarray(
            receiver["freq_range_mhz"],
            dtype=float,
        ).reshape(-1)

        if len(freq_range) >= 2:
            freq_min = float(freq_range[0])
            freq_max = float(freq_range[1])

            if freq_max > freq_min:
                return freq_min, freq_max

    frequencies = data[:, 1].astype(float)

    freq_min = float(np.min(frequencies))
    freq_max = float(np.max(frequencies))

    if freq_max <= freq_min:
        raise ValueError(
            "Invalid frequency range. "
            f"Minimum={freq_min}, Maximum={freq_max}"
        )

    return freq_min, freq_max


# ---------------------------------------------------------------------
# TIME RANGE
# ---------------------------------------------------------------------

def get_time_range_us(
    data: np.ndarray,
    metadata: Dict,
) -> Tuple[float, float]:
    """
    Determine the time range in microseconds.

    Priority:
        1. metadata collection_time_s
        2. observed ToA minimum and maximum
    """

    toa = data[:, 0].astype(float)

    time_start = float(np.min(toa))
    observed_time_end = float(np.max(toa))

    collection_time_s = metadata.get(
        "collection_time_s",
        None,
    )

    if collection_time_s is not None:

        try:
            collection_time_us = (
                float(collection_time_s) * 1_000_000.0
            )

            # Dataset ToA may not start at zero.
            time_end = max(
                time_start + collection_time_us,
                observed_time_end,
            )

            return time_start, time_end

        except (TypeError, ValueError):
            pass

    return time_start, observed_time_end


# ---------------------------------------------------------------------
# BAND EDGES
# ---------------------------------------------------------------------

def create_band_edges(
    freq_min_mhz: float,
    freq_max_mhz: float,
    n_bands: int,
) -> np.ndarray:
    """
    Create equally spaced frequency band edges.
    """

    if n_bands <= 0:
        raise ValueError(
            f"n_bands must be positive. Got {n_bands}."
        )

    if freq_max_mhz <= freq_min_mhz:
        raise ValueError(
            "freq_max_mhz must be greater than freq_min_mhz."
        )

    return np.linspace(
        freq_min_mhz,
        freq_max_mhz,
        n_bands + 1,
        dtype=float,
    )


# ---------------------------------------------------------------------
# PDW -> BAND/TIME ACTIVITY GRID
# ---------------------------------------------------------------------

def pdw_to_activity_grid(
    data: np.ndarray,
    metadata: Dict,
    n_bands: int = DEFAULT_N_BANDS,
    timestep_us: float = DEFAULT_TIMESTEP_US,
    max_timesteps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert PDW data into a boolean band-vs-time activity grid.

    Parameters
    ----------
    data:
        PDW array with columns:
            ToA, Frequency, PulseWidth, AoA, Amplitude

    metadata:
        Metadata dictionary extracted from HDF5.

    n_bands:
        Number of frequency bands.

    timestep_us:
        Width of each time bin in microseconds.

    max_timesteps:
        Optional safety limit. If provided, limits the number
        of generated time bins.

    Returns
    -------
    activity:
        Boolean array of shape:
            (n_timesteps, n_bands)

    band_edges:
        Frequency edges of shape:
            (n_bands + 1,)

    time_edges:
        Time edges in microseconds.
    """

    if timestep_us <= 0:
        raise ValueError(
            "timestep_us must be greater than zero."
        )

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            "Expected PDW data with at least "
            "ToA and Frequency columns."
        )

    toa = data[:, 0].astype(float)
    frequency = data[:, 1].astype(float)

    # ---------------------------------------------------------
    # Frequency bands
    # ---------------------------------------------------------

    freq_min, freq_max = get_frequency_range(
        data,
        metadata,
    )

    band_edges = create_band_edges(
        freq_min,
        freq_max,
        n_bands,
    )

    # ---------------------------------------------------------
    # Time bins
    # ---------------------------------------------------------

    time_start, time_end = get_time_range_us(
        data,
        metadata,
    )

    duration_us = time_end - time_start

    if duration_us <= 0:
        raise ValueError(
            f"Invalid time range: {time_start} to {time_end}"
        )

    n_timesteps = int(
        np.ceil(duration_us / timestep_us)
    )

    n_timesteps = max(n_timesteps, 1)

    if max_timesteps is not None:
        n_timesteps = min(
            n_timesteps,
            int(max_timesteps),
        )

    actual_time_end = (
        time_start +
        n_timesteps * timestep_us
    )

    time_edges = np.linspace(
        time_start,
        actual_time_end,
        n_timesteps + 1,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Create activity grid
    # ---------------------------------------------------------

    activity = np.zeros(
        (n_timesteps, n_bands),
        dtype=bool,
    )

    # Convert frequency to band index
    band_index = np.searchsorted(
        band_edges,
        frequency,
        side="right",
    ) - 1

    # Ensure upper boundary falls in final band
    band_index = np.clip(
        band_index,
        0,
        n_bands - 1,
    )

    # Convert ToA to time index
    time_index = (
        (toa - time_start) / timestep_us
    ).astype(np.int64)

    valid = (
        (time_index >= 0)
        & (time_index < n_timesteps)
        & (band_index >= 0)
        & (band_index < n_bands)
    )

    valid_time_index = time_index[valid]
    valid_band_index = band_index[valid]

    # Multiple PDWs in the same cell still mean True
    activity[
        valid_time_index,
        valid_band_index
    ] = True

    return (
        activity,
        band_edges,
        time_edges,
    )


# ---------------------------------------------------------------------
# TRANSITION PROBABILITIES
# ---------------------------------------------------------------------

def estimate_transition_probabilities(
    activity: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Estimate empirical transition probabilities for each band.

    p01:
        P(next_state = 1 | current_state = 0)

    p11:
        P(next_state = 1 | current_state = 1)

    Also returns:
        counts for all relevant transitions.
    """

    if activity.ndim != 2:
        raise ValueError(
            "activity must have shape "
            "(timesteps, bands)."
        )

    n_timesteps, n_bands = activity.shape

    if n_timesteps < 2:
        raise ValueError(
            "At least two timesteps are required "
            "to estimate transitions."
        )

    current = activity[:-1]
    next_state = activity[1:]

    # Counts
    n00 = np.sum(
        (~current) & (~next_state),
        axis=0,
    )

    n01 = np.sum(
        (~current) & next_state,
        axis=0,
    )

    n10 = np.sum(
        current & (~next_state),
        axis=0,
    )

    n11 = np.sum(
        current & next_state,
        axis=0,
    )

    zero_total = n00 + n01
    one_total = n10 + n11

    p01 = np.zeros(
        n_bands,
        dtype=float,
    )

    p11 = np.zeros(
        n_bands,
        dtype=float,
    )

    valid_zero = zero_total > 0
    valid_one = one_total > 0

    p01[valid_zero] = (
        n01[valid_zero]
        / zero_total[valid_zero]
    )

    p11[valid_one] = (
        n11[valid_one]
        / one_total[valid_one]
    )

    return {
        "p01": p01,
        "p11": p11,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "zero_total": zero_total,
        "one_total": one_total,
    }


# ---------------------------------------------------------------------
# FULL FILE PROCESSING
# ---------------------------------------------------------------------

def process_h5_file(
    file_path: str | Path,
    n_bands: int = DEFAULT_N_BANDS,
    timestep_us: float = DEFAULT_TIMESTEP_US,
    max_timesteps: Optional[int] = None,
) -> Dict:
    """
    Complete processing pipeline for one TSRD HDF5 file.
    """

    data, labels, metadata = load_h5_file(
        file_path
    )

    activity, band_edges, time_edges = (
        pdw_to_activity_grid(
            data=data,
            metadata=metadata,
            n_bands=n_bands,
            timestep_us=timestep_us,
            max_timesteps=max_timesteps,
        )
    )

    transitions = (
        estimate_transition_probabilities(
            activity
        )
    )

    return {
        "file_path": str(file_path),
        "data_shape": data.shape,
        "num_pdws": len(data),
        "labels": labels,
        "metadata": metadata,
        "activity": activity,
        "band_edges": band_edges,
        "time_edges": time_edges,
        "p01": transitions["p01"],
        "p11": transitions["p11"],
        "transition_counts": {
            "n00": transitions["n00"],
            "n01": transitions["n01"],
            "n10": transitions["n10"],
            "n11": transitions["n11"],
        },
    }


# ---------------------------------------------------------------------
# SUMMARY PRINTING
# ---------------------------------------------------------------------

def print_summary(result: Dict) -> None:
    """
    Print a readable summary of adapter output.
    """

    activity = result["activity"]
    band_edges = result["band_edges"]
    p01 = result["p01"]
    p11 = result["p11"]

    print()
    print("=" * 80)
    print("TSRD ADAPTER RESULTS")
    print("=" * 80)

    print(f"\nFile:")
    print(result["file_path"])

    print(f"\nPDW data shape:")
    print(result["data_shape"])

    print(f"\nNumber of PDWs:")
    print(result["num_pdws"])

    print(f"\nActivity grid shape:")
    print(activity.shape)

    print(
        "\nActivity grid meaning:"
    )
    print(
        "Rows    = time bins"
    )
    print(
        "Columns = frequency bands"
    )

    print(
        f"\nTotal active cells: "
        f"{np.sum(activity)}"
    )

    print(
        f"Total cells: "
        f"{activity.size}"
    )

    activity_fraction = (
        np.mean(activity)
        if activity.size > 0
        else 0.0
    )

    print(
        f"Activity fraction: "
        f"{activity_fraction:.6f}"
    )

    print("\nFrequency bands:")

    for band in range(
        len(band_edges) - 1
    ):
        print(
            f"Band {band:02d}: "
            f"{band_edges[band]:10.2f} MHz "
            f"to "
            f"{band_edges[band + 1]:10.2f} MHz"
        )

    print(
        "\nEstimated transition probabilities:"
    )

    print(
        "\nBand |       p01 |       p11"
    )

    print("-" * 35)

    for band in range(len(p01)):
        print(
            f"{band:4d} | "
            f"{p01[band]:9.6f} | "
            f"{p11[band]:9.6f}"
        )

    print()
    print("=" * 80)


# ---------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Convert TSRD HDF5 PDWs into a "
            "band-vs-time activity grid."
        )
    )

    parser.add_argument(
        "file",
        type=str,
        help="Path to an HDF5 file.",
    )

    parser.add_argument(
        "--bands",
        type=int,
        default=DEFAULT_N_BANDS,
        help=(
            "Number of frequency bands. "
            f"Default: {DEFAULT_N_BANDS}"
        ),
    )

    parser.add_argument(
        "--timestep-us",
        type=float,
        default=DEFAULT_TIMESTEP_US,
        help=(
            "Time bin width in microseconds. "
            f"Default: {DEFAULT_TIMESTEP_US}"
        ),
    )

    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help=(
            "Optional maximum number of time bins."
        ),
    )

    args = parser.parse_args()

    result = process_h5_file(
        file_path=args.file,
        n_bands=args.bands,
        timestep_us=args.timestep_us,
        max_timesteps=args.max_timesteps,
    )

    print_summary(result)


if __name__ == "__main__":
    main()