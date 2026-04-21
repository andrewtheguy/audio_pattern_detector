"""Tests for short clip detection through the normal correlation path.

Short clips (< 0.5s) go through the normal path with a 0-100% window,
not the marker tone verification path.
"""

import io

import numpy as np

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import (
    AudioPatternDetector,
    SHORT_CLIP_DURATION_THRESHOLD,
)
from audio_pattern_detector.audio_utils import DEFAULT_TARGET_SAMPLE_RATE
from audio_pattern_detector.detection_utils import get_pure_tone_frequency


SR = DEFAULT_TARGET_SAMPLE_RATE


def _make_chirp(duration: float, f0: float, f1: float, sr: int = SR) -> np.ndarray:
    """Generate a linear chirp signal."""
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    return (0.8 * np.sin(phase) * np.hanning(n)).astype(np.float32)


def _audio_clip_from_array(name: str, audio: np.ndarray) -> AudioClip:
    return AudioClip(name=name, audio=np.asarray(audio, dtype=np.float32), sample_rate=SR)


def _audio_stream_from_array(name: str, audio: np.ndarray) -> AudioStream:
    """Create an AudioStream from a float32 numpy array (raw PCM, no WAV header)."""
    raw = audio.astype(np.float32).tobytes()
    return AudioStream(name=name, audio_stream=io.BytesIO(raw), sample_rate=SR)


# --- Tests ---


def test_short_chirp_does_not_trigger_marker_tone_path():
    """A short chirp clip goes through normal path regardless of FFT analysis."""
    chirp = _make_chirp(0.1, 400, 1200)
    clip = _audio_clip_from_array("my_chirp", chirp)
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)
    # Even if FFT might detect a dominant frequency, clips without strategy metadata
    # still use the normal path.
    assert "my_chirp" not in detector._tone_frequencies


def test_make_chirp_produces_sub_threshold_length():
    """Sanity check: _make_chirp with duration just under the threshold produces a short clip."""
    chirp = _make_chirp(SHORT_CLIP_DURATION_THRESHOLD - 0.01, 400, 1200)
    assert len(chirp) / SR < SHORT_CLIP_DURATION_THRESHOLD


def test_short_chirp_detected_in_audio():
    """A short chirp pattern embedded in silence is detected via the normal path."""
    chirp_duration = 0.1  # seconds, well below 0.5s threshold
    chirp = _make_chirp(chirp_duration, 400, 1200)

    # Build test audio: 2s silence, chirp, 2s silence, chirp, 2s silence
    silence_1 = np.zeros(2 * SR, dtype=np.float32)
    silence_2 = np.zeros(2 * SR, dtype=np.float32)
    silence_3 = np.zeros(2 * SR, dtype=np.float32)
    test_audio = np.concatenate([silence_1, chirp, silence_2, chirp, silence_3])

    clip = _audio_clip_from_array("test_chirp", chirp)
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)

    stream = _audio_stream_from_array("test_audio", test_audio)
    peak_times, total_time = detector.find_clip_in_audio(stream)

    assert peak_times is not None
    assert "test_chirp" in peak_times
    matches = sorted(peak_times["test_chirp"])
    assert len(matches) == 2

    # Chirps placed at 2.0s and 4.1s (2 + 0.1 + 2 = 4.1)
    expected_positions = [2.0 + chirp_duration, 2.0 + chirp_duration + 2.0 + chirp_duration]
    for actual, expected in zip(matches, expected_positions):
        assert abs(actual - expected) < 0.15, f"Expected ~{expected}s, got {actual}s"


def test_short_chirp_no_false_positives_in_noise():
    """Short chirp pattern should not produce false positives in random noise."""
    chirp = _make_chirp(0.1, 400, 1200)

    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(6 * SR) * 0.05).astype(np.float32)

    clip = _audio_clip_from_array("test_chirp", chirp)
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)

    stream = _audio_stream_from_array("noise_audio", noise)
    peak_times, _ = detector.find_clip_in_audio(stream)

    assert peak_times is not None
    assert peak_times.get("test_chirp", []) == []


def test_marker_tone_strategy_triggers_tone_path():
    """A clip with strategy='marker_tone' routes to the tone verifier path."""
    duration = 0.125
    freq = 1000.0
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    tone = (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    clip = AudioClip(
        name="my_marker",
        audio=np.asarray(tone, dtype=np.float32),
        sample_rate=SR,
        strategy="marker_tone",
        strategy_params={"dominant_frequency_hz": freq},
    )
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)

    assert "my_marker" in detector._tone_frequencies, \
        "strategy='marker_tone' should register a dominant frequency"


def test_tone_clip_without_strategy_uses_normal_path():
    """A tone clip without strategy='marker_tone' must NOT trigger the tone path."""
    duration = 0.125
    freq = 1000.0
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    tone = (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    # Verify it IS a pure tone (audio content).
    assert get_pure_tone_frequency(tone, SR) is not None

    # Without strategy metadata, dispatch defaults to the normal path.
    clip = _audio_clip_from_array("other_tone", tone)
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)

    assert "other_tone" not in detector._tone_frequencies, \
        "Clips without strategy='marker_tone' should not route to the tone path"
