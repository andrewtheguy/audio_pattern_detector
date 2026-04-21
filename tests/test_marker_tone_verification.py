from pathlib import Path

import numpy as np

from audio_pattern_detector.audio_clip import AudioClip
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import DEFAULT_TARGET_SAMPLE_RATE


RTHK_BEEP_PATTERN = "sample_audios/clips/rthk_beep.apd.toml"
HARMONIC_STACK_FUNDAMENTAL = 260.0
SWEEP_START_FREQUENCY = 920.0
SWEEP_END_FREQUENCY = 1160.0


def _active_envelope(active_samples: int) -> np.ndarray:
    return np.hanning(active_samples).astype(np.float32)


def _build_clean_candidate(length: int, sample_rate: int, frequency: float) -> np.ndarray:
    active_samples = length
    signal = np.zeros(length, dtype=np.float32)
    t = np.arange(active_samples, dtype=np.float32) / sample_rate
    signal[:active_samples] = 0.9 * np.sin(2 * np.pi * frequency * t) * _active_envelope(active_samples)
    return signal


def _build_harmonic_stack_candidate(length: int, sample_rate: int) -> np.ndarray:
    active_samples = length
    signal = np.zeros(length, dtype=np.float32)
    t = np.arange(active_samples, dtype=np.float32) / sample_rate
    envelope = _active_envelope(active_samples)
    harmonic_stack = (
        0.50 * np.sin(2 * np.pi * HARMONIC_STACK_FUNDAMENTAL * t)
        + 0.35 * np.sin(2 * np.pi * HARMONIC_STACK_FUNDAMENTAL * 2 * t)
        + 0.30 * np.sin(2 * np.pi * HARMONIC_STACK_FUNDAMENTAL * 3 * t)
        + 0.28 * np.sin(2 * np.pi * HARMONIC_STACK_FUNDAMENTAL * 4 * t)
        + 0.22 * np.sin(2 * np.pi * HARMONIC_STACK_FUNDAMENTAL * 5 * t)
    )
    signal[:active_samples] = harmonic_stack.astype(np.float32) * envelope
    signal /= np.max(np.abs(signal))
    return signal.astype(np.float32)


def _build_swept_candidate(length: int, sample_rate: int) -> np.ndarray:
    active_samples = length
    signal = np.zeros(length, dtype=np.float32)
    instantaneous_frequency = np.linspace(
        SWEEP_START_FREQUENCY,
        SWEEP_END_FREQUENCY,
        active_samples,
        dtype=np.float32,
    )
    phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / sample_rate
    signal[:active_samples] = 0.9 * np.sin(phase) * _active_envelope(active_samples)
    return signal


def _run_verify_marker_tone(
    detector: AudioPatternDetector,
    audio_section: np.ndarray,
    dominant_frequency: float,
) -> bool:
    # peak = len-1 and clip_length = len → match_start = 0, so the
    # entire audio_section is used as the matched segment.
    return detector._verify_marker_tone(
        clip_name="rthk_beep",
        audio_section=audio_section.astype(np.float32),
        peak=len(audio_section) - 1,
        clip_length=len(audio_section),
        dominant_frequency=dominant_frequency,
        sr=DEFAULT_TARGET_SAMPLE_RATE,
        section_ts="00:00:00",
    )


def test_marker_tone_verifier_rejects_harmonic_and_swept_false_positives():
    assert Path(RTHK_BEEP_PATTERN).exists(), f"Pattern file {RTHK_BEEP_PATTERN} not found"

    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)
    detector = AudioPatternDetector(audio_clips=[pattern_clip], debug_mode=False)
    dominant_frequency = float(pattern_clip.strategy_params["dominant_frequency_hz"])

    candidate_length = len(pattern_clip.audio)
    clean_candidate = _build_clean_candidate(candidate_length, DEFAULT_TARGET_SAMPLE_RATE, dominant_frequency)
    harmonic_candidate = _build_harmonic_stack_candidate(candidate_length, DEFAULT_TARGET_SAMPLE_RATE)
    swept_candidate = _build_swept_candidate(candidate_length, DEFAULT_TARGET_SAMPLE_RATE)

    verification_results = [
        _run_verify_marker_tone(detector, clean_candidate, dominant_frequency),
        _run_verify_marker_tone(detector, harmonic_candidate, dominant_frequency),
        _run_verify_marker_tone(detector, swept_candidate, dominant_frequency),
    ]

    assert verification_results == [True, False, False]
