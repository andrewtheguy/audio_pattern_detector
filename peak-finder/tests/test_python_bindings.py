import importlib.machinery
import importlib.util
import os
import pathlib
import unittest

import numpy as np


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _module_path() -> pathlib.Path:
    override = os.environ.get("PEAK_FINDER_PYTHON_MODULE")
    if override:
        path = pathlib.Path(override)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"PEAK_FINDER_PYTHON_MODULE points to a missing file: {path}"
        )

    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES)
    for fallback_suffix in (".so", ".dylib", ".pyd", ".dll"):
        if fallback_suffix not in suffixes:
            suffixes.append(fallback_suffix)

    candidates = []
    for profile in ("debug", "release"):
        target_dir = _repo_root() / "target" / profile
        for stem in ("libpeak_finder", "peak_finder"):
            for suffix in suffixes:
                candidates.append(target_dir / f"{stem}{suffix}")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the built peak_finder extension. "
        "Build it first with `cargo build --features python --lib` or `maturin develop`."
    )


def _load_module():
    module_path = _module_path()
    spec = importlib.util.spec_from_file_location("peak_finder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PythonBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.peak_finder = _load_module()

    def test_basic_peaks(self):
        data = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float32)
        peaks, props = self.peak_finder.find_peaks(data)
        np.testing.assert_array_equal(peaks, np.array([1, 3, 5]))
        self.assertIsInstance(props, dict)
        self.assertEqual(len(props), 0)

    def test_return_types(self):
        data = np.array([0, 1, 0], dtype=np.float32)
        peaks, props = self.peak_finder.find_peaks(data)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertEqual(peaks.dtype, np.int64)
        self.assertIsInstance(props, dict)

    def test_height_filter(self):
        data = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float32)
        peaks, _ = self.peak_finder.find_peaks(data, height=1.5)
        np.testing.assert_array_equal(peaks, np.array([3, 5]))

    def test_distance_filter(self):
        data = np.array([0, 3, 0, 5, 0], dtype=np.float32)
        peaks, _ = self.peak_finder.find_peaks(data, distance=3)
        # Peaks at 1 (3.0) and 3 (5.0) are 2 apart < 3, tallest wins
        np.testing.assert_array_equal(peaks, np.array([3]))

    def test_prominence_filter(self):
        data = np.array([0, 1, 0.5, 2, 0], dtype=np.float32)
        peaks, _ = self.peak_finder.find_peaks(data, prominence=1.0)
        # Peak at 1: prominence 0.5, filtered out. Peak at 3: prominence 1.5, kept.
        np.testing.assert_array_equal(peaks, np.array([3]))

    def test_empty_array(self):
        data = np.array([], dtype=np.float32)
        peaks, props = self.peak_finder.find_peaks(data)
        self.assertEqual(len(peaks), 0)
        self.assertEqual(len(props), 0)

    def test_no_peaks(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        peaks, _ = self.peak_finder.find_peaks(data)
        self.assertEqual(len(peaks), 0)

    def test_compare_with_scipy_height_distance(self):
        """Compare against scipy for a realistic correlation-like signal."""
        from scipy.signal import find_peaks as scipy_find_peaks

        rng = np.random.default_rng(42)
        # Simulate a correlation signal with some clear peaks
        x = np.linspace(0, 10 * np.pi, 500).astype(np.float32)
        data = (np.sin(x) + 0.3 * rng.standard_normal(500)).astype(np.float32)
        data = np.abs(data)  # positive
        data /= np.max(data)  # normalize

        scipy_peaks, _ = scipy_find_peaks(data, height=0.25, distance=20)
        rust_peaks, _ = self.peak_finder.find_peaks(data, height=0.25, distance=20)
        np.testing.assert_array_equal(rust_peaks, scipy_peaks)

    def test_compare_with_scipy_prominence(self):
        """Compare against scipy for prominence filtering."""
        from scipy.signal import find_peaks as scipy_find_peaks

        rng = np.random.default_rng(123)
        data = rng.standard_normal(200).astype(np.float32)
        data = np.abs(data)
        data /= np.max(data)

        scipy_peaks, _ = scipy_find_peaks(data, prominence=0.05)
        rust_peaks, _ = self.peak_finder.find_peaks(data, prominence=0.05)
        np.testing.assert_array_equal(rust_peaks, scipy_peaks)

    def test_non_contiguous_raises(self):
        data = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32)
        with self.assertRaises((ValueError, TypeError)):
            self.peak_finder.find_peaks(data[:, 0])


if __name__ == "__main__":
    unittest.main()
