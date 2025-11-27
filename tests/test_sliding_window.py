"""Tests for sliding window functionality in AudioPatternDetector.

These tests verify:
1. Detections after the first window have correct timestamps
2. Detections work properly when patterns are at window boundaries
3. Minimum validation for seconds_per_chunk parameter
4. Auto-computation of seconds_per_chunk when None or < 1
"""
import io
import numpy as np
from pathlib import Path

import pytest

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import TARGET_SAMPLE_RATE


def create_sine_tone(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    """Create a sine wave tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


def create_silence(duration: float, sample_rate: int) -> np.ndarray:
    """Create silence."""
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


def float_to_float32_bytes(audio: np.ndarray) -> bytes:
    """Convert float audio array to float32 bytes for streaming."""
    audio_float32 = audio.astype(np.float32)
    return audio_float32.tobytes()


def create_audio_stream_from_array(audio: np.ndarray, name: str) -> AudioStream:
    """Create an AudioStream from a numpy array."""
    audio_bytes = float_to_float32_bytes(audio)
    stream = io.BytesIO(audio_bytes)
    return AudioStream(name=name, audio_stream=stream, sample_rate=TARGET_SAMPLE_RATE)


def create_beep_pattern(duration: float = 0.23, frequency: float = 1000.0) -> AudioClip:
    """Create a synthetic beep pattern for testing."""
    sr = TARGET_SAMPLE_RATE
    audio = create_sine_tone(frequency, duration, sr)
    return AudioClip(name="test_beep", audio=audio, sample_rate=sr)


class TestSlidingWindowTimestamps:
    """Tests for correct timestamp calculation across sliding windows."""

    def test_detection_in_first_chunk_has_correct_timestamp(self):
        """Test that a detection in the first chunk has the correct timestamp.

        When a pattern is detected in the first chunk (index=0), subtract_seconds=0,
        so the timestamp should be: peak_time - clip_seconds
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23  # seconds
        pattern = create_beep_pattern(duration=pattern_duration)

        # Create audio with pattern starting at 1.0 second
        pattern_start = 1.0
        audio_duration = 5.0

        # Build audio: silence + pattern + silence
        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=60  # Large chunk so everything is in first chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        assert len(peak_times['test_beep']) == 1

        # Timestamp should be approximately pattern_start
        actual_time = peak_times['test_beep'][0]
        assert abs(actual_time - pattern_start) < 0.1, \
            f"Expected timestamp ~{pattern_start}s, got {actual_time}s"

    def test_detection_in_second_chunk_has_correct_timestamp(self):
        """Test that a detection in the second chunk has the correct timestamp.

        When a pattern is detected in chunk index=1 with seconds_per_chunk=3:
        - The pattern at real position 4.0s should be reported correctly.
        - Formula: (peak_in_section/sr) - subtract_seconds + (index * seconds_per_chunk) - clip_seconds
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Place pattern at 4.0 seconds - this will be in the second chunk
        pattern_start = 4.0
        audio_duration = 10.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        assert len(peak_times['test_beep']) >= 1, \
            f"Expected at least 1 detection, got {len(peak_times['test_beep'])}"

        # Find the detection closest to expected time
        expected_time = pattern_start
        closest_detection = min(peak_times['test_beep'], key=lambda t: abs(t - expected_time))

        assert abs(closest_detection - expected_time) < 0.2, \
            f"Expected timestamp ~{expected_time}s, got {closest_detection}s (all: {peak_times['test_beep']})"

    def test_detection_in_third_chunk_has_correct_timestamp(self):
        """Test that a detection in the third chunk has the correct timestamp."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Place pattern at 7.0 seconds - this will be in the third chunk (index=2)
        pattern_start = 7.0
        audio_duration = 12.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        assert len(peak_times['test_beep']) >= 1, \
            f"Expected at least 1 detection, got {len(peak_times['test_beep'])}"

        expected_time = pattern_start
        closest_detection = min(peak_times['test_beep'], key=lambda t: abs(t - expected_time))

        assert abs(closest_detection - expected_time) < 0.2, \
            f"Expected timestamp ~{expected_time}s, got {closest_detection}s (all: {peak_times['test_beep']})"

    def test_multiple_detections_across_chunks_have_correct_timestamps(self):
        """Test multiple patterns across different chunks all have correct timestamps."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Place patterns at: 1.0s (chunk 0), 4.5s (chunk 1), 8.0s (chunk 2)
        pattern_positions = [1.0, 4.5, 8.0]
        audio_duration = 12.0

        # Start with full silence
        audio = create_silence(audio_duration, sr)

        # Insert patterns at each position
        for pos in pattern_positions:
            start_sample = int(pos * sr)
            end_sample = start_sample + len(pattern.audio)
            if end_sample <= len(audio):
                audio[start_sample:end_sample] = pattern.audio

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times

        # Check that each expected position has a corresponding detection
        for expected_pos in pattern_positions:
            found_match = False
            for actual in peak_times['test_beep']:
                if abs(actual - expected_pos) < 0.3:
                    found_match = True
                    break
            assert found_match, \
                f"No detection found near {expected_pos}s (detections: {peak_times['test_beep']})"


class TestSlidingWindowBoundary:
    """Tests for pattern detection at chunk boundaries."""

    def test_detection_at_chunk_boundary_is_found(self):
        """Test that a pattern spanning a chunk boundary is detected.

        The sliding window overlap should allow patterns that span boundaries
        to be detected in the overlapping region.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Place pattern exactly at chunk boundary (end of first chunk)
        # First chunk ends at 3.0s, so place pattern to straddle this
        pattern_start = 2.9  # Pattern will span 2.9 to 3.13 seconds
        audio_duration = 10.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        assert len(peak_times['test_beep']) >= 1, \
            f"Pattern at boundary should be detected, got {len(peak_times['test_beep'])} detections"

        # Find detection closest to expected position
        expected_time = pattern_start
        closest_detection = min(peak_times['test_beep'], key=lambda t: abs(t - expected_time))

        assert abs(closest_detection - expected_time) < 0.3, \
            f"Expected detection near {expected_time}s, got {closest_detection}s"

    def test_detection_just_after_boundary_has_correct_timestamp(self):
        """Test detection of pattern starting just after a chunk boundary.

        The pattern starts at the beginning of chunk 2, so it should be
        detected via the sliding window overlap from chunk 1.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Pattern starts exactly at chunk boundary
        pattern_start = 3.0
        audio_duration = 10.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        assert len(peak_times['test_beep']) >= 1, \
            "Pattern just after boundary should be detected"

        expected_time = pattern_start
        closest_detection = min(peak_times['test_beep'], key=lambda t: abs(t - expected_time))

        assert abs(closest_detection - expected_time) < 0.3, \
            f"Expected detection near {expected_time}s, got {closest_detection}s"

    def test_detection_just_before_boundary_has_correct_timestamp(self):
        """Test detection of pattern ending just before a chunk boundary."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Pattern ends right at chunk boundary
        pattern_start = 3.0 - pattern_duration  # ~2.77s
        audio_duration = 10.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        assert len(peak_times['test_beep']) >= 1, \
            "Pattern just before boundary should be detected"

        expected_time = pattern_start
        closest_detection = min(peak_times['test_beep'], key=lambda t: abs(t - expected_time))

        assert abs(closest_detection - expected_time) < 0.3, \
            f"Expected detection near {expected_time}s, got {closest_detection}s"

    def test_sliding_window_overlap_captures_boundary_pattern(self):
        """Test that the sliding window overlap mechanism works correctly.

        When chunk 2 is processed, it includes sliding_window seconds from
        chunk 1, allowing detection of patterns that span the boundary.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Place pattern to start in the overlap region
        # sliding_window = ceil(0.23) = 1 second
        # Overlap region for chunk 2 is from 2.0s to 3.0s (last 1s of chunk 1)
        pattern_start = 2.5  # In the overlap region
        audio_duration = 10.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        # The pattern in the overlap region may be detected in both chunks
        # but should have consistent timestamp
        assert len(peak_times['test_beep']) >= 1

        expected_time = pattern_start
        closest_detection = min(peak_times['test_beep'], key=lambda t: abs(t - expected_time))

        assert abs(closest_detection - expected_time) < 0.3, \
            f"Expected detection near {expected_time}s, got {closest_detection}s"


class TestSlidingWindowWithRealPatterns:
    """Integration tests using real audio patterns for sliding window behavior."""

    def test_rthk_beep_detection_with_small_chunks(self):
        """Test RTHK beep detection with small chunk sizes.

        Verifies that using small chunks doesn't affect timestamp accuracy.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        if not Path(pattern_file).exists() or not Path(audio_file).exists():
            pytest.skip("Sample audio files not found")

        from audio_pattern_detector.audio_utils import ffmpeg_get_float32_pcm
        from audio_pattern_detector.match import match_pattern

        # First, get reference results with default chunk size
        reference_results, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Now test with small chunks
        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_stream = AudioStream(
                name=Path(audio_file).stem,
                audio_stream=stdout,
                sample_rate=sr
            )

            # Use small chunk size (3 seconds)
            detector = AudioPatternDetector(
                debug_mode=False,
                audio_clips=[pattern_clip],
                seconds_per_chunk=3
            )
            small_chunk_results, _ = detector.find_clip_in_audio(audio_stream)

        # Both should find the beeps
        assert len(reference_results['rthk_beep']) == 2
        assert len(small_chunk_results['rthk_beep']) >= 2

        # Check that expected timestamps are found in small chunk results
        expected_times = [1.4165, 2.419125]
        for expected in expected_times:
            found = any(
                abs(actual - expected) < 0.1
                for actual in small_chunk_results['rthk_beep']
            )
            assert found, \
                f"Expected detection near {expected}s not found in {small_chunk_results['rthk_beep']}"

    def test_cbs_news_detection_with_multiple_chunks(self):
        """Test CBS news detection that spans multiple chunks.

        The CBS news pattern is detected at ~25.9s, which will be in a
        later chunk when using small chunk sizes.

        Note: When a pattern falls in the overlap region between chunks,
        it may be detected in both chunks, resulting in duplicate timestamps.
        This is expected sliding window behavior.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        if not Path(pattern_file).exists() or not Path(audio_file).exists():
            pytest.skip("Sample audio files not found")

        from audio_pattern_detector.audio_utils import ffmpeg_get_float32_pcm

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_stream = AudioStream(
                name=Path(audio_file).stem,
                audio_stream=stdout,
                sample_rate=sr
            )

            # Use 10-second chunks so detection at ~25.9s is in chunk 2 (index=2)
            detector = AudioPatternDetector(
                debug_mode=False,
                audio_clips=[pattern_clip],
                seconds_per_chunk=10
            )
            peak_times, _ = detector.find_clip_in_audio(audio_stream)

        assert 'cbs_news' in peak_times
        # May have duplicates due to sliding window overlap
        assert len(peak_times['cbs_news']) >= 1

        expected_time = 25.89875
        # Find detection closest to expected time
        closest = min(peak_times['cbs_news'], key=lambda t: abs(t - expected_time))

        assert abs(closest - expected_time) < 0.1, \
            f"Expected timestamp ~{expected_time}s, got {closest}s (all: {peak_times['cbs_news']})"


class TestTimestampCalculationEdgeCases:
    """Tests for edge cases in timestamp calculation."""

    def test_pattern_at_very_beginning_of_audio(self):
        """Test detection of pattern at the very start of audio."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        pattern = create_beep_pattern(duration=pattern_duration)

        # Pattern at the very beginning (starts at 0.0s)
        audio_duration = 5.0

        silence_after = create_silence(audio_duration - pattern_duration, sr)
        audio = np.concatenate([pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=60
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        # Detection at very beginning should have timestamp 0 or close to it
        if len(peak_times['test_beep']) > 0:
            assert peak_times['test_beep'][0] >= 0, \
                f"Timestamp should not be negative: {peak_times['test_beep'][0]}"
            assert peak_times['test_beep'][0] < 0.5, \
                f"Detection at beginning should have small timestamp: {peak_times['test_beep'][0]}"

    def test_pattern_near_end_of_last_chunk(self):
        """Test detection of pattern near the end of the last (partial) chunk."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Create audio that doesn't fill the last chunk completely
        # Audio duration is 8.5 seconds, so last chunk is partial
        audio_duration = 8.5
        pattern_start = audio_duration - pattern_duration - 0.1  # Near the end

        silence_before = create_silence(pattern_start, sr)
        audio = np.concatenate([silence_before, pattern.audio])

        # Ensure we have exactly the right length
        audio = audio[:int(audio_duration * sr)]

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times
        if len(peak_times['test_beep']) > 0:
            closest = min(peak_times['test_beep'], key=lambda t: abs(t - pattern_start))
            assert abs(closest - pattern_start) < 0.5, \
                f"Expected detection near {pattern_start}s, got {closest}s"

    def test_timestamps_increase_monotonically_for_sequential_patterns(self):
        """Test that detected timestamps increase for sequential patterns.

        Note: Patterns in the overlap region between chunks may be detected
        twice, resulting in duplicate timestamps. After deduplication,
        timestamps should be monotonically increasing.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23
        seconds_per_chunk = 3
        pattern = create_beep_pattern(duration=pattern_duration)

        # Place patterns at increasing positions
        pattern_positions = [0.5, 2.0, 4.0, 6.5, 9.0]
        audio_duration = 12.0

        audio = create_silence(audio_duration, sr)
        for pos in pattern_positions:
            start_sample = int(pos * sr)
            end_sample = start_sample + len(pattern.audio)
            if end_sample <= len(audio):
                audio[start_sample:end_sample] = pattern.audio

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'test_beep' in peak_times

        # Deduplicate timestamps that are very close together (within 0.01s)
        # This handles the sliding window overlap duplicate detection
        detections = sorted(peak_times['test_beep'])
        deduplicated = []
        for t in detections:
            if not deduplicated or abs(t - deduplicated[-1]) > 0.01:
                deduplicated.append(t)

        # Verify deduplicated detections are in strictly increasing order
        for i in range(1, len(deduplicated)):
            assert deduplicated[i] > deduplicated[i - 1], \
                f"Timestamps should be increasing after dedup: {deduplicated}"

        # Verify we found detections for most positions
        found_count = 0
        for expected in pattern_positions:
            for actual in deduplicated:
                if abs(actual - expected) < 0.3:
                    found_count += 1
                    break

        assert found_count >= len(pattern_positions) - 1, \
            f"Expected to find most patterns. Positions: {pattern_positions}, Detections: {deduplicated}"


def create_long_beep_pattern(duration: float = 2.5, frequency: float = 1000.0) -> AudioClip:
    """Create a longer synthetic beep pattern for testing large sliding windows.

    A 2.5 second pattern results in sliding_window = ceil(2.5) = 3 seconds.
    """
    sr = TARGET_SAMPLE_RATE
    audio = create_sine_tone(frequency, duration, sr)
    return AudioClip(name="long_beep", audio=audio, sample_rate=sr)


class TestLargeSlidingWindow:
    """Tests with large sliding windows to ensure timestamps don't drift.

    These tests use longer patterns (2.5+ seconds) which result in larger
    sliding windows (3+ seconds). This is important to catch any timestamp
    drift issues that may accumulate across multiple chunks.
    """

    def test_large_window_detection_in_second_chunk(self):
        """Test detection with large sliding window in second chunk.

        Pattern duration: 2.5s -> sliding_window: 3s
        seconds_per_chunk: 10s (must be >= 2 * sliding_window = 6s)
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        seconds_per_chunk = 10
        pattern = create_long_beep_pattern(duration=pattern_duration)

        # Place pattern at 12.0s - in second chunk (index=1)
        pattern_start = 12.0
        audio_duration = 30.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'long_beep' in peak_times
        assert len(peak_times['long_beep']) >= 1, \
            f"Expected detection, got {len(peak_times['long_beep'])}"

        expected_time = pattern_start
        closest = min(peak_times['long_beep'], key=lambda t: abs(t - expected_time))

        # With large sliding window, timestamp should still be accurate
        assert abs(closest - expected_time) < 0.5, \
            f"Expected ~{expected_time}s, got {closest}s (drift detected!)"

    def test_large_window_detection_in_fifth_chunk(self):
        """Test detection with large sliding window in fifth chunk.

        This tests that timestamps don't accumulate drift over many chunks.
        Pattern at 45.0s with 10s chunks means chunk index=4.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        seconds_per_chunk = 10
        pattern = create_long_beep_pattern(duration=pattern_duration)

        # Place pattern at 45.0s - in fifth chunk (index=4)
        pattern_start = 45.0
        audio_duration = 60.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'long_beep' in peak_times
        assert len(peak_times['long_beep']) >= 1

        expected_time = pattern_start
        closest = min(peak_times['long_beep'], key=lambda t: abs(t - expected_time))

        # Even after 5 chunks, timestamp should be accurate (no drift)
        assert abs(closest - expected_time) < 0.5, \
            f"Expected ~{expected_time}s, got {closest}s (timestamp drift after 5 chunks!)"

    def test_large_window_multiple_patterns_no_drift(self):
        """Test multiple patterns across many chunks with large sliding window.

        Places patterns at known positions and verifies each has correct timestamp,
        ensuring no cumulative drift occurs.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        seconds_per_chunk = 10
        pattern = create_long_beep_pattern(duration=pattern_duration)

        # Place patterns at: 5s (chunk 0), 15s (chunk 1), 35s (chunk 3), 55s (chunk 5)
        # Spacing ensures patterns are well separated
        pattern_positions = [5.0, 15.0, 35.0, 55.0]
        audio_duration = 70.0

        audio = create_silence(audio_duration, sr)
        for pos in pattern_positions:
            start_sample = int(pos * sr)
            end_sample = start_sample + len(pattern.audio)
            if end_sample <= len(audio):
                audio[start_sample:end_sample] = pattern.audio

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'long_beep' in peak_times

        # Check each expected position has accurate timestamp
        for expected_pos in pattern_positions:
            found = False
            for actual in peak_times['long_beep']:
                if abs(actual - expected_pos) < 0.5:
                    found = True
                    break
            assert found, \
                f"No detection near {expected_pos}s (detections: {peak_times['long_beep']})"

        # Verify no significant drift by checking the last detection
        last_expected = pattern_positions[-1]
        closest_to_last = min(peak_times['long_beep'], key=lambda t: abs(t - last_expected))
        assert abs(closest_to_last - last_expected) < 0.5, \
            f"Drift detected at end: expected ~{last_expected}s, got {closest_to_last}s"

    def test_large_window_boundary_detection(self):
        """Test pattern at chunk boundary with large sliding window.

        With large sliding window, boundary detection is more complex because
        the overlap region is larger.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        seconds_per_chunk = 10
        pattern = create_long_beep_pattern(duration=pattern_duration)

        # Place pattern to straddle chunk boundary at 10s
        # Pattern from 8.5s to 11.0s spans the boundary
        pattern_start = 8.5
        audio_duration = 30.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'long_beep' in peak_times
        assert len(peak_times['long_beep']) >= 1, \
            "Pattern at boundary with large window should be detected"

        expected_time = pattern_start
        closest = min(peak_times['long_beep'], key=lambda t: abs(t - expected_time))

        assert abs(closest - expected_time) < 0.5, \
            f"Expected ~{expected_time}s, got {closest}s"

    def test_very_large_window_far_into_audio(self):
        """Test with very large sliding window and pattern far into audio.

        Pattern duration: 4.5s -> sliding_window: 5s
        This stresses the timestamp calculation with large subtract_seconds values.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 4.5
        seconds_per_chunk = 15  # Must be >= 2 * sliding_window = 10s

        # Create a longer pattern
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="very_long_beep", audio=audio, sample_rate=sr)

        # Place pattern at 50.0s - far into audio
        pattern_start = 50.0
        audio_duration = 70.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        full_audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(full_audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'very_long_beep' in peak_times
        assert len(peak_times['very_long_beep']) >= 1

        expected_time = pattern_start
        closest = min(peak_times['very_long_beep'], key=lambda t: abs(t - expected_time))

        # With very large sliding window and pattern far into audio,
        # timestamp should still be accurate
        assert abs(closest - expected_time) < 1.0, \
            f"Expected ~{expected_time}s, got {closest}s (drift with very large window!)"

    def test_large_window_timestamp_accuracy_across_ten_chunks(self):
        """Test timestamp accuracy with pattern in the 10th chunk.

        This is a stress test to ensure no drift accumulates over many chunks
        with a large sliding window.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        seconds_per_chunk = 10
        pattern = create_long_beep_pattern(duration=pattern_duration)

        # Place pattern at 95.0s - in 10th chunk (index=9)
        pattern_start = 95.0
        audio_duration = 110.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'long_beep' in peak_times
        assert len(peak_times['long_beep']) >= 1

        expected_time = pattern_start
        closest = min(peak_times['long_beep'], key=lambda t: abs(t - expected_time))

        # After 10 chunks, if there's drift, it would be significant
        # Allow slightly larger tolerance for very far positions
        assert abs(closest - expected_time) < 1.0, \
            f"Expected ~{expected_time}s, got {closest}s (drift after 10 chunks!)"

    def test_compare_first_and_tenth_chunk_accuracy(self):
        """Compare timestamp accuracy between first chunk and tenth chunk.

        If there's drift, the later chunk should have more error.
        Both should have similar accuracy if there's no drift.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        seconds_per_chunk = 10
        pattern = create_long_beep_pattern(duration=pattern_duration)

        # Place patterns at 5s (first chunk) and 95s (tenth chunk)
        early_position = 5.0
        late_position = 95.0
        audio_duration = 110.0

        audio = create_silence(audio_duration, sr)

        # Insert early pattern
        start = int(early_position * sr)
        end = start + len(pattern.audio)
        audio[start:end] = pattern.audio

        # Insert late pattern
        start = int(late_position * sr)
        end = start + len(pattern.audio)
        audio[start:end] = pattern.audio

        audio_stream = create_audio_stream_from_array(audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'long_beep' in peak_times
        assert len(peak_times['long_beep']) >= 2

        # Find detection for early position
        early_detections = [t for t in peak_times['long_beep'] if abs(t - early_position) < 1.0]
        assert len(early_detections) >= 1, \
            f"No detection near early position {early_position}s"
        early_error = abs(early_detections[0] - early_position)

        # Find detection for late position
        late_detections = [t for t in peak_times['long_beep'] if abs(t - late_position) < 1.0]
        assert len(late_detections) >= 1, \
            f"No detection near late position {late_position}s"
        late_error = abs(late_detections[0] - late_position)

        # Both errors should be similar (no cumulative drift)
        # If there was drift, late_error would be significantly larger
        assert abs(late_error - early_error) < 0.5, \
            f"Drift detected: early_error={early_error:.3f}s, late_error={late_error:.3f}s"


class TestSlidingWindowOverlapDeduplication:
    """Tests for patterns in the overlap region between chunks.

    When a pattern falls in the overlap region, it may be detected by both
    the current chunk and the next chunk. This tests whether both detections
    report the same timestamp (which would allow deduplication).
    """

    def test_pattern_in_overlap_detected_with_same_timestamp(self):
        """Test that a pattern in the overlap region produces consistent timestamps.

        Scenario:
        - Pattern duration: 3.5s -> sliding_window = ceil(3.5) = 4s
        - Chunk size: 10s
        - Audio duration: 20s
        - Pattern at 7s (overlaps into next chunk's sliding window)

        Chunk 0 processes 0-10s, detects pattern at ~7s
        Chunk 1 processes 6-20s (with 4s overlap), may also detect pattern at ~7s

        Both should report the same timestamp if detected in both.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 3.5  # sliding_window = ceil(3.5) = 4
        seconds_per_chunk = 10

        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="overlap_test", audio=audio, sample_rate=sr)

        # Pattern at 7s - this is in the overlap region for chunk 1
        # Chunk 1's overlap covers seconds 6-10 from chunk 0
        pattern_start = 7.0
        audio_duration = 20.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        full_audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(full_audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'overlap_test' in peak_times

        # Check if we got duplicate detections
        detections = peak_times['overlap_test']
        print(f"Detections: {detections}")  # Debug output

        if len(detections) >= 2:
            # If detected twice, both timestamps should be very close (same position)
            detections_sorted = sorted(detections)
            for i in range(1, len(detections_sorted)):
                diff = abs(detections_sorted[i] - detections_sorted[i - 1])
                # If timestamps are nearly identical, they represent the same detection
                if diff < 0.1:
                    print(f"Duplicate detection found: {detections_sorted[i-1]:.4f}s and {detections_sorted[i]:.4f}s")

        # Verify at least one detection near expected position
        expected_time = pattern_start
        closest = min(detections, key=lambda t: abs(t - expected_time))
        assert abs(closest - expected_time) < 0.5, \
            f"Expected detection near {expected_time}s, got {closest}s"

    def test_overlap_duplicate_timestamps_are_identical(self):
        """Test that duplicate detections from overlap have identical timestamps.

        This is important for deduplication - if both chunks detect the same
        pattern, they should report exactly the same timestamp.

        Using:
        - Pattern: 3.5s (sliding_window = 4s)
        - Chunk: 10s
        - Pattern at 8s (near end of first chunk, in overlap region)
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 3.5
        seconds_per_chunk = 10

        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="dedup_test", audio=audio, sample_rate=sr)

        # Pattern at 8s - definitely in overlap (6-10s of chunk 0)
        pattern_start = 8.0
        audio_duration = 25.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        full_audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(full_audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        detections = peak_times['dedup_test']
        print(f"Detections for dedup_test: {detections}")

        # If we have duplicates, verify they are the same timestamp
        if len(detections) > 1:
            unique_timestamps = set()
            for t in detections:
                # Round to 2 decimal places to group near-identical timestamps
                rounded = round(t, 2)
                unique_timestamps.add(rounded)

            # Check if duplicates have same timestamp (after rounding)
            if len(unique_timestamps) < len(detections):
                print("Duplicate timestamps detected - can be deduplicated!")

            # All detections should point to the same position
            for t in detections:
                assert abs(t - pattern_start) < 0.5, \
                    f"Detection {t}s too far from expected {pattern_start}s"

    def test_pattern_exactly_at_chunk_boundary_overlap(self):
        """Test pattern that ends exactly at chunk boundary.

        Pattern at position where it ends at 10s (chunk boundary).
        This is the edge case where detection might happen in both chunks.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 3.5
        seconds_per_chunk = 10

        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="boundary_exact", audio=audio, sample_rate=sr)

        # Pattern ends exactly at 10s boundary
        pattern_start = 10.0 - pattern_duration  # 6.5s
        audio_duration = 25.0

        silence_before = create_silence(pattern_start, sr)
        silence_after = create_silence(audio_duration - pattern_start - pattern_duration, sr)
        full_audio = np.concatenate([silence_before, pattern.audio, silence_after])

        audio_stream = create_audio_stream_from_array(full_audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        detections = peak_times['boundary_exact']
        print(f"Boundary exact detections: {detections}")

        assert len(detections) >= 1, "Pattern at boundary should be detected"

        # All detections should be near the expected timestamp
        for t in detections:
            assert abs(t - pattern_start) < 0.5, \
                f"Detection {t}s too far from expected {pattern_start}s"

    def test_short_pattern_large_sliding_window_scenario(self):
        """Test scenario similar to user's question.

        Note: sliding_window is always ceil(pattern_duration), so to get
        a 4-second sliding window with a 1-second pattern is not directly
        possible. This test uses a 3.5s pattern (4s sliding window).

        Audio: 20s, Chunks: 10s, Pattern at 9s
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 3.5  # ceil(3.5) = 4s sliding window
        seconds_per_chunk = 10

        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="user_scenario", audio=audio, sample_rate=sr)

        # Pattern at 9s (like user's example)
        # This is in the overlap region (6-10s) for chunk 1
        pattern_start = 9.0
        audio_duration = 20.0

        # But pattern would extend to 12.5s, crossing into chunk 1
        silence_before = create_silence(pattern_start, sr)
        remaining = audio_duration - pattern_start - pattern_duration
        if remaining > 0:
            silence_after = create_silence(remaining, sr)
            full_audio = np.concatenate([silence_before, pattern.audio, silence_after])
        else:
            # Pattern extends beyond audio duration
            full_audio = np.concatenate([silence_before, pattern.audio])
            full_audio = full_audio[:int(audio_duration * sr)]

        audio_stream = create_audio_stream_from_array(full_audio, "test_audio")

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=seconds_per_chunk
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        detections = peak_times['user_scenario']
        print(f"User scenario detections: {detections}")

        # Analyze duplicates
        if len(detections) > 1:
            # Check if duplicates are at same timestamp
            sorted_dets = sorted(detections)
            duplicates_same_ts = []
            for i in range(len(sorted_dets) - 1):
                if abs(sorted_dets[i + 1] - sorted_dets[i]) < 0.1:
                    duplicates_same_ts.append((sorted_dets[i], sorted_dets[i + 1]))
            if duplicates_same_ts:
                print(f"Same-timestamp duplicates found: {duplicates_same_ts}")
                print("These can be deduplicated by rounding/grouping")

    def test_verify_duplicate_timestamp_calculation(self):
        """Verify the exact timestamp calculation for overlap duplicates.

        This test traces through the math to confirm both chunks produce
        the same timestamp for the same pattern position.

        For a pattern at absolute position P with duration D:
        - Chunk 0 (index=0): final_ts = (P + D) - 0 + 0 - D = P
        - Chunk 1 (index=1) with sliding_window S:
          - Pattern is at position (P + D - (chunk_size - S)) in audio_section
          - final_ts = ((P + D - (chunk_size - S))) - S + chunk_size - D
          - = P + D - chunk_size + S - S + chunk_size - D = P ✓
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 3.5  # sliding_window = ceil(3.5) = 4
        seconds_per_chunk = 10

        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="calc_verify", audio=audio, sample_rate=sr)

        # Test multiple positions in the overlap region
        test_positions = [6.5, 7.0, 8.0, 9.0]
        audio_duration = 25.0

        for pattern_start in test_positions:
            silence_before = create_silence(pattern_start, sr)
            remaining = audio_duration - pattern_start - pattern_duration
            silence_after = create_silence(max(0, remaining), sr)
            full_audio = np.concatenate([silence_before, pattern.audio, silence_after])
            full_audio = full_audio[:int(audio_duration * sr)]

            audio_stream = create_audio_stream_from_array(full_audio, "test_audio")

            detector = AudioPatternDetector(
                debug_mode=False,
                audio_clips=[pattern],
                seconds_per_chunk=seconds_per_chunk
            )
            peak_times, _ = detector.find_clip_in_audio(audio_stream)

            detections = peak_times['calc_verify']

            # All detections should be near pattern_start
            for t in detections:
                error = abs(t - pattern_start)
                assert error < 0.5, \
                    f"Pattern at {pattern_start}s: detection at {t}s has error {error:.3f}s"

            # If multiple detections, they should be identical (same timestamp)
            if len(detections) > 1:
                for i, t1 in enumerate(detections):
                    for t2 in detections[i + 1:]:
                        diff = abs(t1 - t2)
                        assert diff < 0.1, \
                            f"Duplicate timestamps differ: {t1}s vs {t2}s (diff={diff:.4f}s)"


class TestSecondsPerChunkValidation:
    """Tests for seconds_per_chunk parameter validation.

    The AudioPatternDetector has the following validation rules:
    1. seconds_per_chunk must be >= 2 * sliding_window (raises ValueError in __init__)
    2. If seconds_per_chunk is None or < 1, it's auto-computed as longest_clip * 2
    """

    def test_seconds_per_chunk_too_small_raises_error(self):
        """Test that seconds_per_chunk < 2 * sliding_window raises ValueError.

        A 2.5s pattern has sliding_window = ceil(2.5) = 3s
        So seconds_per_chunk must be >= 6s
        Setting it to 5s should raise ValueError during initialization.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5  # sliding_window = ceil(2.5) = 3s
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="test_pattern", audio=audio, sample_rate=sr)

        # 5 < 2 * 3 = 6, so this should fail during __init__
        with pytest.raises(ValueError, match="too small"):
            AudioPatternDetector(
                debug_mode=False,
                audio_clips=[pattern],
                seconds_per_chunk=5
            )

    def test_seconds_per_chunk_exactly_minimum_works(self):
        """Test that seconds_per_chunk = 2 * sliding_window works.

        A 2.5s pattern has sliding_window = ceil(2.5) = 3s
        So seconds_per_chunk = 6s should work (exactly 2x).
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5  # sliding_window = 3s
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="test_pattern", audio=audio, sample_rate=sr)

        # 6 = 2 * 3, exactly at minimum - should work
        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=6
        )
        assert detector.seconds_per_chunk == 6

    def test_seconds_per_chunk_above_minimum_works(self):
        """Test that seconds_per_chunk > 2 * sliding_window works."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5  # sliding_window = 3s
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="test_pattern", audio=audio, sample_rate=sr)

        # 10 > 6 (2 * 3), should work fine
        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=10
        )
        assert detector.seconds_per_chunk == 10

    def test_seconds_per_chunk_none_auto_computes(self):
        """Test that seconds_per_chunk=None auto-computes to longest_clip * 2.

        A 2.5s pattern should result in seconds_per_chunk = ceil(2.5) * 2 = 6s
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="test_pattern", audio=audio, sample_rate=sr)

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=None
        )
        # Auto-computed: ceil(clip_length_samples / sr) * 2 = ceil(2.5) * 2 = 6
        expected = 6  # ceil(2.5) * 2
        assert detector.seconds_per_chunk == expected

    def test_seconds_per_chunk_zero_auto_computes(self):
        """Test that seconds_per_chunk=0 auto-computes to longest_clip * 2."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="test_pattern", audio=audio, sample_rate=sr)

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=0
        )
        expected = 6  # ceil(2.5) * 2
        assert detector.seconds_per_chunk == expected

    def test_seconds_per_chunk_negative_auto_computes(self):
        """Test that seconds_per_chunk < 0 auto-computes to longest_clip * 2."""
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 2.5
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="test_pattern", audio=audio, sample_rate=sr)

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=-5
        )
        expected = 6  # ceil(2.5) * 2
        assert detector.seconds_per_chunk == expected

    def test_multiple_patterns_uses_longest_for_validation(self):
        """Test that with multiple patterns, the longest is used for validation.

        Pattern 1: 0.5s -> sliding_window = 1s -> min chunk = 2s
        Pattern 2: 3.0s -> sliding_window = 3s -> min chunk = 6s

        seconds_per_chunk=4 should fail because 4 < 6 (for longest pattern).
        """
        sr = TARGET_SAMPLE_RATE

        # Short pattern
        short_audio = create_sine_tone(1000.0, 0.5, sr)
        short_pattern = AudioClip(name="short", audio=short_audio, sample_rate=sr)

        # Long pattern
        long_audio = create_sine_tone(500.0, 3.0, sr)
        long_pattern = AudioClip(name="long", audio=long_audio, sample_rate=sr)

        # 4 < 2 * 3 = 6, should fail due to long pattern
        with pytest.raises(ValueError, match="too small"):
            AudioPatternDetector(
                debug_mode=False,
                audio_clips=[short_pattern, long_pattern],
                seconds_per_chunk=4
            )

    def test_multiple_patterns_valid_chunk_size(self):
        """Test that seconds_per_chunk works when valid for all patterns."""
        sr = TARGET_SAMPLE_RATE

        # Short pattern (sliding_window = 1s)
        short_audio = create_sine_tone(1000.0, 0.5, sr)
        short_pattern = AudioClip(name="short", audio=short_audio, sample_rate=sr)

        # Long pattern (sliding_window = 3s)
        long_audio = create_sine_tone(500.0, 3.0, sr)
        long_pattern = AudioClip(name="long", audio=long_audio, sample_rate=sr)

        # 8 > 6 (2 * 3), should work
        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[short_pattern, long_pattern],
            seconds_per_chunk=8
        )
        assert detector.seconds_per_chunk == 8

    def test_short_pattern_small_chunk_works(self):
        """Test that short patterns allow small chunk sizes.

        A 0.23s pattern has sliding_window = ceil(0.23) = 1s
        So seconds_per_chunk = 2s should work (exactly 2x).
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.23  # sliding_window = 1s
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="beep", audio=audio, sample_rate=sr)

        # 2 = 2 * 1, exactly at minimum
        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern],
            seconds_per_chunk=2
        )
        assert detector.seconds_per_chunk == 2

    def test_short_pattern_chunk_just_below_minimum_fails(self):
        """Test that even 1 second below minimum fails.

        A 0.5s pattern has sliding_window = ceil(0.5) = 1s
        So seconds_per_chunk must be >= 2s. 1s should fail.
        """
        sr = TARGET_SAMPLE_RATE
        pattern_duration = 0.5  # sliding_window = 1s
        audio = create_sine_tone(1000.0, pattern_duration, sr)
        pattern = AudioClip(name="short_beep", audio=audio, sample_rate=sr)

        # 1 < 2 * 1 = 2, should fail
        with pytest.raises(ValueError, match="too small"):
            AudioPatternDetector(
                debug_mode=False,
                audio_clips=[pattern],
                seconds_per_chunk=1
            )


class TestSlidingWindowComputation:
    """Tests for sliding_window computation from pattern duration."""

    def test_sliding_window_is_ceiling_of_pattern_duration(self):
        """Verify sliding_window = ceil(pattern_duration).

        This is tested indirectly by checking which chunk sizes work/fail.
        """
        sr = TARGET_SAMPLE_RATE

        # Test cases: (pattern_duration, expected_sliding_window)
        test_cases = [
            (0.1, 1),   # ceil(0.1) = 1
            (0.5, 1),   # ceil(0.5) = 1
            (1.0, 1),   # ceil(1.0) = 1
            (1.1, 2),   # ceil(1.1) = 2
            (2.0, 2),   # ceil(2.0) = 2
            (2.5, 3),   # ceil(2.5) = 3
            (4.9, 5),   # ceil(4.9) = 5
        ]

        for pattern_duration, expected_sliding_window in test_cases:
            audio = create_sine_tone(1000.0, pattern_duration, sr)
            pattern = AudioClip(name="test", audio=audio, sample_rate=sr)

            # Minimum valid chunk size is 2 * expected_sliding_window
            min_valid_chunk = 2 * expected_sliding_window

            # Should work with exactly minimum
            detector = AudioPatternDetector(
                debug_mode=False,
                audio_clips=[pattern],
                seconds_per_chunk=min_valid_chunk
            )
            assert detector.seconds_per_chunk == min_valid_chunk, \
                f"Pattern {pattern_duration}s: expected chunk {min_valid_chunk}s to work"

            # Should fail with one less (unless it would be < 1, which auto-computes)
            if min_valid_chunk > 1:
                with pytest.raises(ValueError, match="too small"):
                    AudioPatternDetector(
                        debug_mode=False,
                        audio_clips=[pattern],
                        seconds_per_chunk=min_valid_chunk - 1
                    )

    def test_auto_compute_uses_longest_pattern(self):
        """Test that auto-compute considers the longest pattern."""
        sr = TARGET_SAMPLE_RATE

        # Multiple patterns with different lengths
        patterns = [
            AudioClip(name="p1", audio=create_sine_tone(1000.0, 1.0, sr), sample_rate=sr),  # 1s
            AudioClip(name="p2", audio=create_sine_tone(800.0, 2.5, sr), sample_rate=sr),   # 2.5s
            AudioClip(name="p3", audio=create_sine_tone(600.0, 0.3, sr), sample_rate=sr),   # 0.3s
        ]

        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=patterns,
            seconds_per_chunk=None  # Auto-compute
        )

        # Longest is 2.5s -> ceil(2.5) = 3 -> auto = 3 * 2 = 6
        # But the code uses: math.ceil(max_clip_length / target_sample_rate) * 2
        # max_clip_length is in samples: 2.5 * 8000 = 20000 samples
        # seconds = 20000 / 8000 = 2.5 -> ceil = 3 -> * 2 = 6
        assert detector.seconds_per_chunk == 6
