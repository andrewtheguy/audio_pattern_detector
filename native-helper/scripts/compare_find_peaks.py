#!/usr/bin/env python3
"""Benchmark scipy.signal.find_peaks vs native_helper.find_peaks on a numpy array.

Examples:
    uv run --directory native-helper --group dev python scripts/compare_find_peaks.py signal.npy --prominence 0.05
    uv run --directory native-helper --group dev python scripts/compare_find_peaks.py --generate 1000000
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal

import native_helper


def load_array(path: Path, array_name: str | None) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if array_name is not None:
                signal = loaded[array_name]
            elif len(loaded.files) == 1:
                signal = loaded[loaded.files[0]]
            else:
                available = ", ".join(loaded.files)
                raise ValueError(
                    f"{path} contains multiple arrays ({available}); use --array-name"
                )
        finally:
            loaded.close()
    else:
        signal = loaded

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"{path} must be a 1D array, got shape {signal.shape}")
    return np.ascontiguousarray(signal)


def benchmark_find_peaks(
    signal: np.ndarray,
    *,
    height: float | None,
    distance: int | None,
    prominence: float | None,
    warmup: int,
    repeat: int,
) -> None:
    kwargs: dict[str, Any] = {}
    if height is not None:
        kwargs["height"] = height
    if distance is not None:
        kwargs["distance"] = distance
    if prominence is not None:
        kwargs["prominence"] = prominence

    scipy_peaks: np.ndarray | None = None
    rust_peaks: np.ndarray | None = None
    scipy_times: list[float] = []
    rust_times: list[float] = []

    for _ in range(warmup):
        scipy.signal.find_peaks(signal, **kwargs)
        native_helper.find_peaks(signal, **kwargs)

    for _ in range(repeat):
        start = time.perf_counter()
        scipy_peaks, _ = scipy.signal.find_peaks(signal, **kwargs)
        scipy_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        rust_peaks, _ = native_helper.find_peaks(signal, **kwargs)
        rust_times.append(time.perf_counter() - start)

    assert scipy_peaks is not None
    assert rust_peaks is not None

    exact_match = np.array_equal(scipy_peaks, rust_peaks)
    only_scipy = np.setdiff1d(scipy_peaks, rust_peaks, assume_unique=True)
    only_rust = np.setdiff1d(rust_peaks, scipy_peaks, assume_unique=True)
    max_aligned_delta = (
        int(np.max(np.abs(scipy_peaks - rust_peaks)))
        if len(scipy_peaks) == len(rust_peaks) and len(scipy_peaks) > 0
        else None
    )

    scipy_median = statistics.median(scipy_times)
    rust_median = statistics.median(rust_times)

    print(f"signal_len={len(signal)}")
    print(f"scipy_count={len(scipy_peaks)}")
    print(f"rust_count={len(rust_peaks)}")
    print(f"exact_match={exact_match}")
    print(f"only_scipy={len(only_scipy)}")
    print(f"only_rust={len(only_rust)}")
    print(f"max_aligned_delta={max_aligned_delta}")
    print(f"params={kwargs}")
    print(f"scipy_median_sec={scipy_median:.6f}")
    print(f"rust_median_sec={rust_median:.6f}")
    print(f"speedup={scipy_median / rust_median:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Input signal array (.npy/.npz)")
    parser.add_argument("--generate", type=int, metavar="N", help="Generate a random signal of length N")
    parser.add_argument("--array-name", help="Array name to load from an .npz file")
    parser.add_argument("--repeat", type=int, default=7, help="Timed repetitions (default: 7)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (default: 1)")
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--distance", type=int, default=None)
    parser.add_argument("--prominence", type=float, default=None)
    args = parser.parse_args()

    if args.generate is not None:
        signal = np.random.default_rng().standard_normal(args.generate).astype(np.float32)
        print(f"generated={args.generate}")
    elif args.input is not None:
        signal = load_array(Path(args.input), args.array_name)
        print(f"input={args.input}")
    else:
        parser.error("either input file or --generate N is required")

    print(f"dtype={signal.dtype}")
    print()

    benchmark_find_peaks(
        signal,
        height=args.height,
        distance=args.distance,
        prominence=args.prominence,
        warmup=args.warmup,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
