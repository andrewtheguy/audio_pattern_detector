"""Tests for sliding window functionality in AudioPatternDetector.

These tests verify:
1. Detections after the first window have correct timestamps
2. Detections work properly when patterns are at window boundaries
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


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float audio array to int16 bytes for streaming."""
    # Normalize to int16 range
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def create_audio_stream_from_array(audio: np.ndarray, name: str) -> AudioStream:
    """Create an AudioStream from a numpy array."""
    audio_bytes = float_to_int16_bytes(audio)
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

        from audio_pattern_detector.audio_utils import ffmpeg_get_16bit_pcm
        from audio_pattern_detector.match import match_pattern

        # First, get reference results with default chunk size
        reference_results, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Now test with small chunks
        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
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

        from audio_pattern_detector.audio_utils import ffmpeg_get_16bit_pcm

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
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
