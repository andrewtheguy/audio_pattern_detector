"""Tests for direct AudioPatternDetector API.

These tests use AudioPatternDetector directly (not via CLI) to test:
1. Callback-based detection (on_pattern_detected parameter) - equivalent to JSONL mode
2. Memory optimization mode (accumulate_results=False)
3. Combined callback + accumulate scenarios
"""
from pathlib import Path


from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector, DEFAULT_SECONDS_PER_CHUNK
from audio_pattern_detector.audio_utils import ffmpeg_get_float32_pcm, DEFAULT_TARGET_SAMPLE_RATE


# --- Helper Functions ---


def run_detector_with_callback(audio_file, pattern_files, accumulate_results=True):
    """Helper to run detector with callback and return events and results."""
    pattern_clips = [AudioClip.from_audio_file(pf) for pf in pattern_files]

    events = []
    def callback(clip_name, timestamp):
        events.append((clip_name, timestamp))

    sr = DEFAULT_TARGET_SAMPLE_RATE
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

    sr = DEFAULT_TARGET_SAMPLE_RATE
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    events, _, _ = run_detector_with_callback(audio_file, [pattern_file])

    assert len(events) >= 2, "Expected at least 2 events"

    # Verify timestamps are monotonically increasing
    timestamps = [ts for _, ts in events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i-1], \
            f"Timestamps not monotonic: {timestamps[i-1]} -> {timestamps[i]}"


def test_callback_multiple_patterns_monotonic():
    """Test multiple patterns emit in timestamp order with single matching pattern."""
    pattern_files = [
        "sample_audios/clips/rthk_beep.apd",  # Found at ~1.4s and ~2.4s
        "sample_audios/clips/cbs_news.wav",    # Not found in RTHK audio
    ]
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    events, _, _ = run_detector_with_callback(audio_file, pattern_files)

    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    # Both events should be rthk_beep (cbs_news doesn't match in this audio)
    for clip_name, _ in events:
        assert clip_name == "rthk_beep", \
            f"Expected rthk_beep, got {clip_name}"

    # Verify timestamps are monotonically increasing
    first_ts = events[0][1]
    second_ts = events[1][1]
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


def test_callback_multiple_patterns_non_matching_ignored():
    """Test callback with two patterns where only one matches the audio."""
    pattern_files = [
        "sample_audios/clips/rthk_beep.apd",  # Not found in CBS audio
        "sample_audios/clips/cbs_news.wav",    # Found at ~25.9s
    ]
    audio_file = "sample_audios/cbs_news_audio_section.wav"

    events, _, _ = run_detector_with_callback(audio_file, pattern_files)

    # Only cbs_news should match
    assert len(events) == 1, f"Expected 1 event, got {len(events)}: {events}"
    assert events[0][0] == "cbs_news", f"Expected cbs_news, got {events[0][0]}"


# --- accumulate_results Tests ---


def test_accumulate_results_true():
    """Test accumulate_results=True returns peak_times dict."""
    pattern_file = "sample_audios/clips/rthk_beep.apd"
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
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
        "sample_audios/clips/天空下的彩虹intro.wav",  # Does not match CBS audio
    ]
    audio_file = "sample_audios/cbs_news_audio_section.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, pattern_files, accumulate_results=False
    )

    # Only cbs_news should match (天空下的彩虹intro doesn't match CBS audio)
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"

    # peak_times should be None (streaming mode)
    assert peak_times is None, "peak_times should be None in streaming mode"

    # The single event should be cbs_news
    assert events[0][0] == "cbs_news"


def test_callback_multiple_patterns_with_accumulate():
    """Test multiple patterns with both callback and accumulation."""
    pattern_files = [
        "sample_audios/clips/rthk_beep.apd",
        "sample_audios/clips/cbs_news.wav",
        "sample_audios/clips/天空下的彩虹intro.wav",  # Does not match CBS audio
    ]
    audio_file = "sample_audios/cbs_news_audio_section.wav"

    events, peak_times, total_time = run_detector_with_callback(
        audio_file, pattern_files, accumulate_results=True
    )

    # Only cbs_news should match; RTHK and rainbow intro should not
    assert len(events) == 1, f"Expected 1 event (cbs_news only), got {len(events)}"

    # Verify peak_times structure
    assert peak_times is not None
    assert "rthk_beep" in peak_times
    assert "cbs_news" in peak_times
    assert "天空下的彩虹intro" in peak_times

    # RTHK and rainbow intro should have no matches
    assert len(peak_times["rthk_beep"]) == 0
    assert len(peak_times["天空下的彩虹intro"]) == 0
    # CBS should have 1 match
    assert len(peak_times["cbs_news"]) == 1

    # Verify callback captured the same
    callback_names = [name for name, _ in events]
    assert "cbs_news" in callback_names
    assert "天空下的彩虹intro" not in callback_names
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    # Track when events were added relative to each other
    events_with_order = []
    counter = [0]

    def callback(clip_name, timestamp):
        counter[0] += 1
        events_with_order.append((counter[0], clip_name, timestamp))

    pattern_clip = AudioClip.from_audio_file(pattern_file)

    sr = DEFAULT_TARGET_SAMPLE_RATE
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
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    audio_file = "sample_audios/rthk_section_with_beep.wav"

    received_types = []

    def callback(clip_name, timestamp):
        received_types.append((type(clip_name).__name__, type(timestamp).__name__))

    pattern_clip = AudioClip.from_audio_file(pattern_file)

    sr = DEFAULT_TARGET_SAMPLE_RATE
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


# --- get_config() Tests ---


def test_get_config_returns_correct_structure():
    """Test get_config returns dict with expected keys."""
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    pattern_clip = AudioClip.from_audio_file(pattern_file)
    detector = AudioPatternDetector(audio_clips=[pattern_clip])

    config = detector.get_config()

    # Verify structure
    assert isinstance(config, dict)
    assert "default_seconds_per_chunk" in config
    assert "min_chunk_size_seconds" in config
    assert "sample_rate" in config
    assert "clips" in config


def test_get_config_default_seconds_per_chunk():
    """Test default_seconds_per_chunk always returns the constant value."""
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    pattern_clip = AudioClip.from_audio_file(pattern_file)

    # Test with default seconds_per_chunk
    detector1 = AudioPatternDetector(audio_clips=[pattern_clip])
    config1 = detector1.get_config()
    assert config1["default_seconds_per_chunk"] == DEFAULT_SECONDS_PER_CHUNK

    # Test with custom seconds_per_chunk (should still return the constant as default)
    detector2 = AudioPatternDetector(audio_clips=[pattern_clip], seconds_per_chunk=30)
    config2 = detector2.get_config()
    assert config2["default_seconds_per_chunk"] == DEFAULT_SECONDS_PER_CHUNK

    # Test with auto mode (None) (should still return the constant as default)
    detector3 = AudioPatternDetector(audio_clips=[pattern_clip], seconds_per_chunk=None)
    config3 = detector3.get_config()
    assert config3["default_seconds_per_chunk"] == DEFAULT_SECONDS_PER_CHUNK


def test_get_config_sample_rate():
    """Test sample_rate is correct."""
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    pattern_clip = AudioClip.from_audio_file(pattern_file)
    detector = AudioPatternDetector(audio_clips=[pattern_clip])

    config = detector.get_config()
    assert config["sample_rate"] == DEFAULT_TARGET_SAMPLE_RATE
    assert config["sample_rate"] == 8000


def test_get_config_min_chunk_size_single_pattern():
    """Test min_chunk_size_seconds for single pattern."""
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    pattern_clip = AudioClip.from_audio_file(pattern_file)
    detector = AudioPatternDetector(audio_clips=[pattern_clip])

    config = detector.get_config()

    # min_chunk_size should be sliding_window * 2
    clip_config = config["clips"]["rthk_beep"]
    expected_min = clip_config["sliding_window_seconds"] * 2
    assert config["min_chunk_size_seconds"] == expected_min


def test_get_config_min_chunk_size_multiple_patterns():
    """Test min_chunk_size_seconds is max of all patterns' minimums."""
    pattern_files = [
        "sample_audios/clips/rthk_beep.apd",            # Short beep
        "sample_audios/clips/cbs_news.wav",             # Longer pattern
        "sample_audios/clips/天空下的彩虹intro.wav",  # Another pattern
    ]
    pattern_clips = [AudioClip.from_audio_file(pf) for pf in pattern_files]
    detector = AudioPatternDetector(audio_clips=pattern_clips)

    config = detector.get_config()

    # Calculate expected min_chunk_size (max of all sliding_window * 2)
    expected_min = 0
    for clip_name, clip_config in config["clips"].items():
        min_for_clip = clip_config["sliding_window_seconds"] * 2
        if min_for_clip > expected_min:
            expected_min = min_for_clip

    assert config["min_chunk_size_seconds"] == expected_min
    # The larger patterns should determine the min
    assert config["min_chunk_size_seconds"] >= 2  # At least 2 seconds


def test_get_config_clips_info():
    """Test clips dict contains correct per-clip info."""
    pattern_file = "sample_audios/clips/rthk_beep.apd"
    pattern_clip = AudioClip.from_audio_file(pattern_file)
    detector = AudioPatternDetector(audio_clips=[pattern_clip])

    config = detector.get_config()

    # Verify clip is in clips dict
    assert "rthk_beep" in config["clips"]
    clip_config = config["clips"]["rthk_beep"]

    # Verify required fields
    assert "duration_seconds" in clip_config
    assert "sliding_window_seconds" in clip_config

    # Verify types
    assert isinstance(clip_config["duration_seconds"], float)
    assert isinstance(clip_config["sliding_window_seconds"], int)

    # Verify reasonable values
    assert clip_config["duration_seconds"] > 0
    assert clip_config["sliding_window_seconds"] >= 1


def test_get_config_clips_multiple_patterns():
    """Test clips dict includes all patterns."""
    pattern_files = [
        "sample_audios/clips/rthk_beep.apd",
        "sample_audios/clips/cbs_news.wav",
        "sample_audios/clips/天空下的彩虹intro.wav",
    ]
    pattern_clips = [AudioClip.from_audio_file(pf) for pf in pattern_files]
    detector = AudioPatternDetector(audio_clips=pattern_clips)

    config = detector.get_config()

    # All patterns should be in clips dict
    assert "rthk_beep" in config["clips"]
    assert "cbs_news" in config["clips"]
    assert "天空下的彩虹intro" in config["clips"]
    assert len(config["clips"]) == 3


def test_get_config_clip_duration():
    """Test clip duration is correctly computed."""
    beep_clip = AudioClip.from_audio_file("sample_audios/clips/rthk_beep.apd")
    detector1 = AudioPatternDetector(audio_clips=[beep_clip])
    config1 = detector1.get_config()
    assert config1["clips"]["rthk_beep"]["duration_seconds"] < 0.5

    rainbow_clip = AudioClip.from_audio_file("sample_audios/clips/天空下的彩虹intro.wav")
    detector2 = AudioPatternDetector(audio_clips=[rainbow_clip])
    config2 = detector2.get_config()
    assert config2["clips"]["天空下的彩虹intro"]["duration_seconds"] >= 0.5


def test_get_config_sliding_window_computed_correctly():
    """Test sliding_window_seconds is ceil of clip duration."""
    import math

    pattern_files = [
        "sample_audios/clips/rthk_beep.apd",
        "sample_audios/clips/cbs_news.wav",
    ]

    for pattern_file in pattern_files:
        pattern_clip = AudioClip.from_audio_file(pattern_file)
        detector = AudioPatternDetector(audio_clips=[pattern_clip])
        config = detector.get_config()

        clip_name = Path(pattern_file).stem
        clip_config = config["clips"][clip_name]

        # sliding_window should be ceil of duration
        expected_sliding_window = math.ceil(clip_config["duration_seconds"])
        assert clip_config["sliding_window_seconds"] == expected_sliding_window, \
            f"{clip_name}: Expected sliding_window {expected_sliding_window}, got {clip_config['sliding_window_seconds']}"
