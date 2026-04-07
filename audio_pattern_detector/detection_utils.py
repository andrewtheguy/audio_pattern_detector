from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PureToneMetrics:
    """Frequency-domain metrics for validating a pure-tone candidate window."""

    detected_frequency: float
    overall_band_purity: float
    active_frame_ratio: float
    longest_active_run: int
    active_frame_mean_purity: float


def get_pure_tone_frequency(audio_data: NDArray[np.float32], sample_rate: int) -> float | None:
    """Return the dominant frequency if the audio is a pure tone, else None."""
    fft_result = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(len(audio_data), d=1 / sample_rate)

    magnitude = np.abs(fft_result)
    dominant_freq_idx = np.argmax(magnitude)
    dominant_magnitude = magnitude[dominant_freq_idx]
    magnitude_normalized = magnitude / dominant_magnitude

    from native_helper import find_peaks
    peaks, _ = find_peaks(magnitude_normalized, prominence=0.05)

    peak_freqs = freqs[peaks]
    dominant_freq = float(freqs[dominant_freq_idx])
    if len(peaks) == 1 and math.isclose(peak_freqs[0], dominant_freq, rel_tol=0.01):
        return dominant_freq
    return None


def analyze_pure_tone_candidate(
    audio_data: NDArray[np.float32],
    sample_rate: int,
    dominant_frequency: float,
) -> PureToneMetrics:
    """Measure how strongly a candidate window behaves like a single pure tone."""
    if len(audio_data) == 0:
        return PureToneMetrics(
            detected_frequency=0.0,
            overall_band_purity=0.0,
            active_frame_ratio=0.0,
            longest_active_run=0,
            active_frame_mean_purity=0.0,
        )

    target_band_hz = max(40.0, dominant_frequency * 0.08)
    target_lock_hz = max(20.0, dominant_frequency * 0.04)

    windowed_audio = audio_data * np.hanning(len(audio_data))
    spectrum = np.abs(np.fft.rfft(windowed_audio))
    freqs = np.fft.rfftfreq(len(audio_data), d=1 / sample_rate)
    detected_frequency = float(freqs[int(np.argmax(spectrum))])

    total_energy = float(np.sum(spectrum**2))
    if total_energy == 0.0:
        return PureToneMetrics(
            detected_frequency=detected_frequency,
            overall_band_purity=0.0,
            active_frame_ratio=0.0,
            longest_active_run=0,
            active_frame_mean_purity=0.0,
        )

    target_band = np.abs(freqs - dominant_frequency) <= target_band_hz
    overall_band_purity = float(np.sum(spectrum[target_band] ** 2)) / total_energy

    window_len = max(int(round(0.025 * sample_rate)), 32)
    hop = max(window_len // 2, 1)
    frame_window = np.hanning(window_len)

    frame_count = 0
    active_frame_count = 0
    longest_active_run = 0
    current_active_run = 0
    active_frame_purities: list[float] = []

    for start in range(0, len(audio_data) - window_len, hop):
        chunk = audio_data[start:start + window_len] * frame_window
        chunk_spectrum = np.abs(np.fft.rfft(chunk))
        chunk_energy = float(np.sum(chunk_spectrum**2))
        if chunk_energy == 0.0:
            current_active_run = 0
            continue

        chunk_freqs = np.fft.rfftfreq(window_len, d=1 / sample_rate)
        frame_count += 1
        dominant_idx = int(np.argmax(chunk_spectrum))
        frame_dominant_frequency = float(chunk_freqs[dominant_idx])
        frame_target_band = np.abs(chunk_freqs - dominant_frequency) <= target_band_hz
        frame_target_purity = float(np.sum(chunk_spectrum[frame_target_band] ** 2)) / chunk_energy

        is_active = (
            math.isclose(frame_dominant_frequency, dominant_frequency, abs_tol=target_lock_hz)
            and frame_target_purity >= 0.55
        )
        if is_active:
            active_frame_count += 1
            current_active_run += 1
            longest_active_run = max(longest_active_run, current_active_run)
            active_frame_purities.append(frame_target_purity)
        else:
            current_active_run = 0

    active_frame_ratio = active_frame_count / frame_count if frame_count > 0 else 0.0
    active_frame_mean_purity = (
        float(np.mean(active_frame_purities)) if active_frame_purities else 0.0
    )

    return PureToneMetrics(
        detected_frequency=detected_frequency,
        overall_band_purity=overall_band_purity,
        active_frame_ratio=active_frame_ratio,
        longest_active_run=longest_active_run,
        active_frame_mean_purity=active_frame_mean_purity,
    )


def extract_padded_segment(
    audio_data: NDArray[np.float32],
    start: int,
    length: int,
) -> NDArray[np.float32]:
    """Extract a fixed-length segment, padding with zeros when out of bounds."""
    stop = start + length
    left_pad = max(0, -start)
    right_pad = max(0, stop - len(audio_data))
    bounded_start = max(0, start)
    bounded_stop = min(len(audio_data), stop)
    segment = audio_data[bounded_start:bounded_stop]
    if left_pad > 0 or right_pad > 0:
        segment = np.pad(segment, (left_pad, right_pad))
    return np.asarray(segment, dtype=np.float32)


def max_distance(sorted_data: list[float]) -> float:
    """Find the maximum distance between consecutive elements in sorted data."""
    max_dist: float = 0
    for i in range(1, len(sorted_data)):
        dist = sorted_data[i] - sorted_data[i - 1]
        max_dist = max(max_dist, dist)
    return max_dist
