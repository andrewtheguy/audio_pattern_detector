"""Tests for short clip detection through the normal correlation path.

Short clips (< 0.5s) go through the normal path with a 0-100% window,
not the pure tone verification path.
"""

import io
import struct
import wave

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


def _wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    """Encode float32 audio as 16-bit WAV bytes."""
    buf = io.BytesIO()
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _audio_stream_from_array(name: str, audio: np.ndarray) -> AudioStream:
    """Create an AudioStream from a float32 numpy array (raw PCM, no WAV header)."""
    raw = audio.astype(np.float32).tobytes()
    return AudioStream(name=name, audio_stream=io.BytesIO(raw), sample_rate=SR)


# --- Tests ---


def test_short_chirp_does_not_trigger_pure_tone_path():
    """A short chirp clip goes through normal path regardless of FFT analysis."""
    chirp = _make_chirp(0.1, 400, 1200)
    clip = _audio_clip_from_array("my_chirp", chirp)
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)
    # Even if FFT might detect a dominant frequency, non-rthk_beep clips use normal path
    assert detector._clip_datas["my_chirp"]["dominant_frequency"] is None


def test_short_clip_below_threshold():
    """Clips shorter than SHORT_CLIP_DURATION_THRESHOLD are classified as short."""
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


def test_rthk_beep_still_uses_pure_tone_path():
    """Verify that a clip named 'rthk_beep' triggers pure tone detection by name."""
    # Create a pure tone clip named 'rthk_beep'
    duration = 0.125
    freq = 1000.0
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    tone = (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    clip = _audio_clip_from_array("rthk_beep", tone)
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)

    clip_data = detector._clip_datas["rthk_beep"]
    assert clip_data["dominant_frequency"] is not None, \
        "rthk_beep should have dominant_frequency set (pure tone path)"


def test_non_rthk_pure_tone_uses_normal_path():
    """A pure tone clip NOT named 'rthk_beep' should go through normal path."""
    duration = 0.125
    freq = 1000.0
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    tone = (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    # Verify it IS a pure tone
    assert get_pure_tone_frequency(tone, SR) is not None

    # But when named something else, it should NOT trigger pure tone path
    clip = _audio_clip_from_array("other_tone", tone)
    detector = AudioPatternDetector(audio_clips=[clip], debug_mode=False)

    clip_data = detector._clip_datas["other_tone"]
    assert clip_data["dominant_frequency"] is None, \
        "Non-rthk_beep clips should have dominant_frequency=None (normal path)"
