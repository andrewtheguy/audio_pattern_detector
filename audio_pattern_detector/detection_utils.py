import math

import numpy as np
from numpy.typing import NDArray


def get_pure_tone_frequency(audio_data: NDArray[np.float32], sample_rate: int) -> float | None:
    """Return the dominant frequency if the audio is a pure tone, else None."""
    fft_result = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(audio_data), d=1 / sample_rate)

    magnitude = np.abs(fft_result)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitude = magnitude[:len(freqs) // 2]

    dominant_freq_idx = np.argmax(positive_magnitude)
    dominant_magnitude = positive_magnitude[dominant_freq_idx]
    positive_magnitude_normalized = positive_magnitude / dominant_magnitude

    from native_helper import find_peaks
    peaks, _ = find_peaks(positive_magnitude_normalized, prominence=0.05)

    peak_freqs = positive_freqs[peaks]
    dominant_freq = float(positive_freqs[dominant_freq_idx])
    if len(peaks) == 1 and math.isclose(peak_freqs[0], dominant_freq, rel_tol=0.01):
        return dominant_freq
    return None


def max_distance(sorted_data: list[float]) -> float:
    """Find the maximum distance between consecutive elements in sorted data."""
    max_dist: float = 0
    for i in range(1, len(sorted_data)):
        dist = sorted_data[i] - sorted_data[i - 1]
        max_dist = max(max_dist, dist)
    return max_dist
