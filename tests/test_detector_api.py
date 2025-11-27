"""Tests for direct AudioPatternDetector API.

These tests use AudioPatternDetector directly (not via CLI) to test:
1. Callback-based detection (on_pattern_detected parameter) - equivalent to JSONL mode
2. Memory optimization mode (accumulate_results=False)
3. Combined callback + accumulate scenarios
"""
from pathlib import Path

import pytest

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import ffmpeg_get_float32_pcm, TARGET_SAMPLE_RATE


# --- Helper Functions ---


def run_detector_with_callback(audio_file, pattern_files, accumulate_results=True):
    """Helper to run detector with callback and return events and results."""
    pattern_clips = [AudioClip.from_audio_file(pf) for pf in pattern_files]

    events = []
    def callback(clip_name, timestamp):
        events.append((clip_name, timestamp))

    sr = TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=pattern_clips)
        peak_times, total_time = detector.find_clip_in_audio(
            audio_stream,
            on_pattern_detected=callback,
            accumulate_results=accumulate_results,
        )

    return events, peak_times, total_time


def run_detector_without_callback(audio_file, pattern_files, accumulate_results=True):
    """Helper to run detector without callback."""
    pattern_clips = [AudioClip.from_audio_file(pf) for pf in pattern_files]

    sr = TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=pattern_clips)
        peak_times, total_time = detector.find_clip_in_audio(
            audio_stream,
            on_pattern_detected=None,
            accumulate_results=accumulate_results,
        )

    return peak_times, total_time


# --- Callback Tests (equivalent to JSONL CLI tests) ---


def test_callback_basic():
    """Test on_pattern_detected callback is called correctly."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    assert Path(pattern_file).exists()
    assert Path(audio_file).exists()

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, [pattern_file]
    )

    # Callback should have been called twice (2 beeps)
    assert len(events) == 2, f"Expected 2 callback events, got {len(events)}"

    # Each event should be (clip_name, timestamp) tuple
    for clip_name, timestamp in events:
        assert clip_name == "rthk_beep"
        assert isinstance(timestamp, float)
        assert timestamp >= 0

    # Verify timestamps match expected values
    expected_times = [1.4165, 2.419125]
    for i, (clip_name, actual) in enumerate(events):
        assert abs(actual - expected_times[i]) < 0.01, \
            f"Event {i}: Expected ~{expected_times[i]}s, got {actual}s"


def test_callback_timestamps_monotonic():
    """Test callback events are emitted in timestamp order."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    events, _, _ = run_detector_with_callback(audio_file, [pattern_file])

    assert len(events) >= 2, "Expected at least 2 events"

    # Verify timestamps are monotonically increasing
    timestamps = [ts for _, ts in events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i-1], \
            f"Timestamps not monotonic: {timestamps[i-1]} -> {timestamps[i]}"


def test_callback_multiple_patterns_monotonic():
    """Test multiple patterns emit in timestamp order, not clip order."""
    pattern_files = [
        "sample_audios/clips/cbs_news.wav",      # Found at ~25.9s
        "sample_audios/clips/cbs_news_dada.wav"  # Found at ~1.97s
    ]
    audio_file = "sample_audios/cbs_news_audio_section.wav"

    events, _, _ = run_detector_with_callback(audio_file, pattern_files)

    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    # cbs_news_dada should be emitted first (earlier timestamp)
    # even though cbs_news was passed first in the pattern_files list
    first_clip, first_ts = events[0]
    second_clip, second_ts = events[1]

    assert first_clip == "cbs_news_dada", \
        f"Expected cbs_news_dada first (earlier timestamp), got {first_clip}"
    assert second_clip == "cbs_news", \
        f"Expected cbs_news second, got {second_clip}"

    assert first_ts < second_ts, \
        f"Timestamps not monotonic: {first_ts} should be < {second_ts}"


def test_callback_no_matches():
    """Test callback is not called when no patterns match."""
    pattern_file = "sample_audios/clips/cbs_news.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"  # CBS pattern not in RTHK audio

    events, peak_times, _ = run_detector_with_callback(audio_file, [pattern_file])

    # No matches, so callback should not have been called
    assert len(events) == 0, f"Expected 0 events for no match, got {len(events)}"

    # peak_times should have empty list for the pattern
    assert peak_times is not None
    assert "cbs_news" in peak_times
    assert len(peak_times["cbs_news"]) == 0


def test_callback_interleaved_patterns():
    """Test callback order with interleaved_patterns.wav (clip2, aaa, clip2, aaa)."""
    pattern_files = [
        "sample_audios/test_generated/clips/clip2_early.wav",  # Copy of cbs_news_dada
        "sample_audios/test_generated/clips/aaa_late.wav"      # Copy of cbs_news
    ]
    audio_file = "sample_audios/test_generated/interleaved_patterns.wav"

    # Skip if test files don't exist
    for f in pattern_files + [audio_file]:
        if not Path(f).exists():
            pytest.skip(f"Test file {f} not found")

    events, _, _ = run_detector_with_callback(audio_file, pattern_files)

    # Should have 4 events: clip2, aaa, clip2, aaa in timestamp order
    assert len(events) == 4, f"Expected 4 events, got {len(events)}: {events}"

    # Verify all timestamps are monotonically increasing
    timestamps = [ts for _, ts in events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i-1], \
            f"Timestamps not monotonic at index {i}: {timestamps}"

    # Verify the pattern order matches timestamp order (interleaved)
    clip_names = [name for name, _ in events]
    # Expected order: clip2_early, aaa_late, clip2_early, aaa_late
    assert clip_names[0] == "clip2_early", f"Expected clip2_early first, got {clip_names[0]}"
    assert clip_names[1] == "aaa_late", f"Expected aaa_late second, got {clip_names[1]}"
    assert clip_names[2] == "clip2_early", f"Expected clip2_early third, got {clip_names[2]}"
    assert clip_names[3] == "aaa_late", f"Expected aaa_late fourth, got {clip_names[3]}"


# --- accumulate_results Tests ---


def test_accumulate_results_true():
    """Test accumulate_results=True returns peak_times dict."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    peak_times, total_time = run_detector_without_callback(
        audio_file, [pattern_file], accumulate_results=True
    )

    # peak_times should be a dict with results
    assert peak_times is not None, "peak_times should not be None"
    assert isinstance(peak_times, dict), "peak_times should be a dict"
    assert "rthk_beep" in peak_times, "rthk_beep key should exist"
    assert len(peak_times["rthk_beep"]) == 2, \
        f"Expected 2 matches, got {len(peak_times['rthk_beep'])}"

    # Verify timestamps
    expected_times = [1.4165, 2.419125]
    for i, (actual, expected) in enumerate(zip(sorted(peak_times["rthk_beep"]), expected_times)):
        assert abs(actual - expected) < 0.01, \
            f"Match {i}: Expected ~{expected}s, got {actual}s"


def test_accumulate_results_false_returns_none():
    """Test accumulate_results=False returns None for peak_times."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    peak_times, total_time = run_detector_without_callback(
        audio_file, [pattern_file], accumulate_results=False
    )

    # peak_times should be None when not accumulating
    assert peak_times is None, f"peak_times should be None, got {peak_times}"

    # total_time should still be valid
    assert total_time > 0, "total_time should be positive"


def test_accumulate_results_false_with_callback():
    """Test callback works when accumulate_results=False."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, [pattern_file], accumulate_results=False
    )

    # Callback should still capture events
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    # peak_times should be None
    assert peak_times is None, f"peak_times should be None, got {peak_times}"

    # Verify events have correct data
    expected_times = [1.4165, 2.419125]
    for i, (clip_name, timestamp) in enumerate(events):
        assert clip_name == "rthk_beep"
        assert abs(timestamp - expected_times[i]) < 0.01


def test_accumulate_results_false_no_callback():
    """Test accumulate_results=False without callback (no-op mode)."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    # This is essentially a no-op mode - detection runs but results aren't saved
    peak_times, total_time = run_detector_without_callback(
        audio_file, [pattern_file], accumulate_results=False
    )

    # peak_times should be None
    assert peak_times is None, f"peak_times should be None, got {peak_times}"

    # total_time should still be tracked
    assert total_time > 0, "total_time should be positive"
    assert 4.0 < total_time < 4.2, f"Expected ~4.08s, got {total_time}s"


# --- Combined Callback + Accumulate Tests ---


def test_callback_with_accumulate_true():
    """Test callback and accumulation both capture same matches."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, [pattern_file], accumulate_results=True
    )

    # Both should have captured results
    assert len(events) == 2, f"Callback should have 2 events, got {len(events)}"
    assert peak_times is not None
    assert len(peak_times["rthk_beep"]) == 2, \
        f"peak_times should have 2 matches, got {len(peak_times['rthk_beep'])}"

    # Verify callback and accumulation captured same timestamps
    callback_timestamps = sorted([ts for _, ts in events])
    accumulated_timestamps = sorted(peak_times["rthk_beep"])

    assert len(callback_timestamps) == len(accumulated_timestamps)
    for cb_ts, acc_ts in zip(callback_timestamps, accumulated_timestamps):
        assert abs(cb_ts - acc_ts) < 0.001, \
            f"Callback ({cb_ts}) and accumulated ({acc_ts}) timestamps differ"


def test_callback_with_accumulate_false():
    """Test callback works with accumulate_results=False (streaming mode)."""
    pattern_files = [
        "sample_audios/clips/cbs_news.wav",
        "sample_audios/clips/cbs_news_dada.wav"
    ]
    audio_file = "sample_audios/cbs_news_audio_section.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, pattern_files, accumulate_results=False
    )

    # Callback should work
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    # peak_times should be None (streaming mode)
    assert peak_times is None, "peak_times should be None in streaming mode"

    # Verify events are in timestamp order
    timestamps = [ts for _, ts in events]
    assert timestamps[0] < timestamps[1], \
        f"Events should be in timestamp order: {timestamps}"

    # cbs_news_dada (earlier) should be first
    assert events[0][0] == "cbs_news_dada"
    assert events[1][0] == "cbs_news"


def test_callback_multiple_patterns_with_accumulate():
    """Test multiple patterns with both callback and accumulation."""
    pattern_files = [
        "sample_audios/clips/rthk_beep.wav",
        "sample_audios/clips/cbs_news.wav",
        "sample_audios/clips/cbs_news_dada.wav"
    ]
    audio_file = "sample_audios/cbs_news_audio_section.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, pattern_files, accumulate_results=True
    )

    # CBS patterns should match, RTHK should not
    assert len(events) == 2, f"Expected 2 events (CBS patterns only), got {len(events)}"

    # Verify peak_times structure
    assert peak_times is not None
    assert "rthk_beep" in peak_times
    assert "cbs_news" in peak_times
    assert "cbs_news_dada" in peak_times

    # RTHK should have no matches
    assert len(peak_times["rthk_beep"]) == 0
    # CBS patterns should have 1 match each
    assert len(peak_times["cbs_news"]) == 1
    assert len(peak_times["cbs_news_dada"]) == 1

    # Verify callback captured the same
    callback_names = [name for name, _ in events]
    assert "cbs_news" in callback_names
    assert "cbs_news_dada" in callback_names
    assert "rthk_beep" not in callback_names


def test_callback_with_no_match_accumulate_true():
    """Test callback with no matches and accumulate_results=True."""
    pattern_file = "sample_audios/clips/cbs_news.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, [pattern_file], accumulate_results=True
    )

    # No matches
    assert len(events) == 0, "No callback events expected"
    assert peak_times is not None
    assert "cbs_news" in peak_times
    assert len(peak_times["cbs_news"]) == 0


def test_callback_with_no_match_accumulate_false():
    """Test callback with no matches and accumulate_results=False."""
    pattern_file = "sample_audios/clips/cbs_news.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, [pattern_file], accumulate_results=False
    )

    # No matches
    assert len(events) == 0, "No callback events expected"
    assert peak_times is None, "peak_times should be None"


# --- Edge Cases ---


def test_callback_called_immediately():
    """Test that callback is called as patterns are detected, not at the end."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    # Track when events were added relative to each other
    events_with_order = []
    counter = [0]

    def callback(clip_name, timestamp):
        counter[0] += 1
        events_with_order.append((counter[0], clip_name, timestamp))

    pattern_clip = AudioClip.from_audio_file(pattern_file)

    sr = TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        detector.find_clip_in_audio(
            audio_stream,
            on_pattern_detected=callback,
            accumulate_results=True,
        )

    # Events should have been added in order
    assert len(events_with_order) == 2
    assert events_with_order[0][0] == 1  # First event
    assert events_with_order[1][0] == 2  # Second event


def test_callback_receives_correct_types():
    """Test callback receives correct argument types."""
    pattern_file = "sample_audios/clips/rthk_beep.wav"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    received_types = []

    def callback(clip_name, timestamp):
        received_types.append((type(clip_name).__name__, type(timestamp).__name__))

    pattern_clip = AudioClip.from_audio_file(pattern_file)

    sr = TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        detector.find_clip_in_audio(
            audio_stream,
            on_pattern_detected=callback,
            accumulate_results=True,
        )

    assert len(received_types) == 2
    for clip_type, ts_type in received_types:
        assert clip_type == "str", f"clip_name should be str, got {clip_type}"
        # Timestamp can be Python float or numpy float64
        assert ts_type in ("float", "float64"), \
            f"timestamp should be float or float64, got {ts_type}"
